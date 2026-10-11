"""Versioned YAML catalogs, resolved without importing a simulator or robot SDK."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .contracts import finite_vector, positive, unit_quaternion_xyzw
from .time_scaling import SmoothPhaseSchedule
from .trajectories import ApproachAndTrackTrajectory, CircleTrajectory, LissajousTrajectory


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        document = yaml.load(stream, Loader=_UniqueLoader)
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError(f"Expected schema_version: 1 mapping: {path}")
    return document


def _keys(document: dict, allowed: set[str], required: set[str], name: str):
    if not isinstance(document, dict) or set(document) - allowed or required - set(document):
        raise ValueError(f"Invalid fields for {name}; allowed={sorted(allowed)}, required={sorted(required)}")


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a YAML boolean")
    return value


@dataclass(frozen=True)
class WorkspaceConfig:
    data: dict

    def __post_init__(self):
        data = self.data
        _keys(data, {"model", "frames", "joint_names", "joint_bounds_rad", "initial_postures_rad",
                     "cartesian_bounds_m", "reference_limits", "nominal_payload_kg", "hardware_reviewed"},
              {"model", "frames", "joint_names", "joint_bounds_rad", "initial_postures_rad",
               "cartesian_bounds_m", "reference_limits", "nominal_payload_kg", "hardware_reviewed"}, "workspace")
        model_fields = {"name", "asset_relative_to_data_root", "usd_sha256", "urdf_sha256",
                        "archived_urdf", "isaaclab_revision", "armature_kg_m2"}
        _keys(data["model"], model_fields, model_fields, "model provenance")
        for key, length in (("usd_sha256", 64), ("urdf_sha256", 64), ("isaaclab_revision", 40)):
            value = data["model"][key]
            if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid {key}")
        for key in ("asset_relative_to_data_root", "archived_urdf"):
            path = Path(data["model"][key])
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"{key} must be a safe relative path")
        positive(data["model"]["armature_kg_m2"], "armature_kg_m2", allow_zero=True)
        if data["frames"] != {"base": "fr3_link0", "tcp": "fr3_flange", "quaternion_order": "xyzw",
                              "flange_to_tcp_position_m": [0., 0., 0.],
                              "flange_to_tcp_quaternion_xyzw": [0., 0., 0., 1.]}:
            raise ValueError("M0 supports only the explicit base-frame bare-flange xyzw contract")
        if data["joint_names"] != [f"panda_joint{i}" for i in range(1, 8)]:
            raise ValueError("Expected seven ordered FR3 joint aliases")
        _keys(data["joint_bounds_rad"], {"model", "commanded", "measured_abort"},
              {"model", "commanded", "measured_abort"}, "joint_bounds_rad")
        for interval in ("model", "commanded", "measured_abort"):
            _keys(data["joint_bounds_rad"][interval], {"lower", "upper"}, {"lower", "upper"}, interval)
            lower = np.array(finite_vector(data["joint_bounds_rad"][interval]["lower"], 7, interval))
            upper = np.array(finite_vector(data["joint_bounds_rad"][interval]["upper"], 7, interval))
            if np.any(lower >= upper):
                raise ValueError(f"Unordered {interval} joint interval")
            if interval != "model":
                bounds = data["joint_bounds_rad"]["model"]
                if np.any(lower < bounds["lower"]) or np.any(upper > bounds["upper"]):
                    raise ValueError(f"{interval} bounds exceed model interval")
        if not isinstance(data["initial_postures_rad"], dict) or not data["initial_postures_rad"]:
            raise ValueError("At least one named initial posture is required")
        for name, posture in data["initial_postures_rad"].items():
            values = np.array(finite_vector(posture, 7, name))
            bounds = data["joint_bounds_rad"]["commanded"]
            if np.any(values < bounds["lower"]) or np.any(values > bounds["upper"]):
                raise ValueError(f"Initial posture {name} outside command interval")
        box = data["cartesian_bounds_m"]
        _keys(box, {"lower", "upper"}, {"lower", "upper"}, "cartesian_bounds_m")
        if np.any(np.array(finite_vector(box["lower"], 3, "box lower")) >= finite_vector(box["upper"], 3, "box upper")):
            raise ValueError("Unordered Cartesian interval")
        _keys(data["reference_limits"], {"speed_m_s", "acceleration_m_s2", "jerk_m_s3"},
              {"speed_m_s", "acceleration_m_s2", "jerk_m_s3"}, "reference_limits")
        for name, value in data["reference_limits"].items():
            positive(value, name)
        positive(data["nominal_payload_kg"], "nominal_payload_kg", allow_zero=True)
        _boolean(data["hardware_reviewed"], "hardware_reviewed")


@dataclass(frozen=True)
class ControllerProfile:
    name: str
    data: dict

    def __post_init__(self):
        _keys(self.data, {"mode", "status", "hardware_approved", "physics_hz", "reference_hz", "delay_steps",
                         "reference_hold", "stiffness", "damping", "units", "source", "audit_pending", "joint_servo", "cartesian_servo"},
              {"mode", "status", "hardware_approved", "physics_hz", "reference_hz", "delay_steps",
               "reference_hold", "stiffness", "damping", "units", "source", "audit_pending"}, self.name)
        if self.data["mode"] not in ("joint", "cartesian"):
            raise ValueError("Unsupported controller mode")
        count = 7 if self.data["mode"] == "joint" else 6
        for name in ("stiffness", "damping"):
            if np.any(np.array(finite_vector(self.data[name], count, name)) < 0):
                raise ValueError("Gains must be nonnegative")
        physics = positive(self.data["physics_hz"], "physics_hz")
        reference = positive(self.data["reference_hz"], "reference_hz")
        if not math.isclose(physics / reference, round(physics / reference), abs_tol=1e-10) or reference > physics:
            raise ValueError("reference_hz must divide physics_hz")
        delay = self.data["delay_steps"]
        if type(delay) is not int or delay < 0:
            raise ValueError("delay_steps must be a nonnegative integer")
        if self.data["reference_hold"] != "zero_order_hold":
            raise ValueError("M0 supports explicit zero_order_hold only; Franky interpolation is pending audit")
        approved = _boolean(self.data["hardware_approved"], "hardware_approved")
        expected_units = ({"stiffness": "Nm/rad", "damping": "Nm*s/rad"} if count == 7 else
                          {"translation_stiffness": "N/m", "translation_damping": "N*s/m",
                           "rotation_stiffness": "Nm/rad", "rotation_damping": "Nm*s/rad"})
        if self.data["units"] != expected_units:
            raise ValueError("Controller units do not match the selected mode")
        if not isinstance(self.data["audit_pending"], list) or any(
            not isinstance(item, str) or not item for item in self.data["audit_pending"]
        ):
            raise ValueError("audit_pending must be a list of nonempty evidence requests")
        if approved and (self.data["audit_pending"] or self.data["status"] != "hardware_reviewed"):
            raise ValueError("Hardware approval requires completed audit and hardware_reviewed status")
        if "joint_servo" in self.data:
            servo = self.data["joint_servo"]
            fields = {"position_error_clip_rad", "slew_rate_nm_s", "cutoff_hz", "effort_limits_nm", "max_measured_velocity_rad_s"}
            _keys(servo, fields, fields, "joint_servo")
            if self.data["mode"] != "joint":
                raise ValueError("joint_servo requires joint mode")
            for key in ("position_error_clip_rad", "slew_rate_nm_s", "cutoff_hz"):
                positive(servo[key], key)
            for key in ("effort_limits_nm", "max_measured_velocity_rad_s"):
                if np.any(np.array(finite_vector(servo[key], 7, key)) <= 0):
                    raise ValueError(f"{key} must be positive")
        if "cartesian_servo" in self.data:
            servo = self.data["cartesian_servo"]
            fields = {"slew_rate_nm_s", "cutoff_hz", "effort_limits_nm", "max_measured_velocity_rad_s",
                      "max_reference_linear_velocity_m_s", "max_reference_angular_velocity_rad_s"}
            _keys(servo, fields, fields, "cartesian_servo")
            if self.data["mode"] != "cartesian":
                raise ValueError("cartesian_servo requires Cartesian mode")
            for key in fields-{"effort_limits_nm", "max_measured_velocity_rad_s"}:
                positive(servo[key], key)
            for key in ("effort_limits_nm", "max_measured_velocity_rad_s"):
                if np.any(np.array(finite_vector(servo[key], 7, key)) <= 0):
                    raise ValueError(f"{key} must be positive")


@dataclass(frozen=True)
class TrajectorySpec:
    name: str
    data: dict

    def __post_init__(self):
        required = {"type", "center_m", "plane_rpy_deg", "quaternion_xyzw", "phase_rad", "turns",
                    "ramp_duration_s", "approach_duration_s", "settle_duration_s", "exit_duration_s", "return_to_start"}
        kind = self.data.get("type")
        if kind in ("joint_hold", "joint_ptp"):
            required = {"type", "target_position_rad", "hold_duration_s", "settle_duration_s",
                        "max_velocity_rad_s", "max_acceleration_rad_s2", "max_jerk_rad_s3"}
            _keys(self.data, required, required, self.name)
            finite_vector(self.data["target_position_rad"], 7, "target_position_rad")
            positive(self.data["hold_duration_s"], "hold_duration_s")
            positive(self.data["settle_duration_s"], "settle_duration_s")
            for key in ("max_velocity_rad_s", "max_acceleration_rad_s2", "max_jerk_rad_s3"):
                if np.any(np.array(finite_vector(self.data[key], 7, key)) <= 0):
                    raise ValueError(f"{key} must be positive")
            return
        extra = {"radius_m"} if kind == "circle" else {"amplitudes_m", "harmonics", "phase_offsets_rad"}
        _keys(self.data, required | extra, required | extra, self.name)
        if kind not in ("circle", "lissajous"):
            raise ValueError("M0 trajectory catalog supports circle and lissajous")
        for name in ("center_m", "plane_rpy_deg"):
            finite_vector(self.data[name], 3, name)
        unit_quaternion_xyzw(self.data["quaternion_xyzw"])
        for name in ("turns", "ramp_duration_s", "approach_duration_s", "exit_duration_s"):
            positive(self.data[name], name)
        positive(self.data["settle_duration_s"], "settle_duration_s", allow_zero=True)
        if not math.isfinite(self.data["phase_rad"]):
            raise ValueError("phase_rad must be finite")
        _boolean(self.data["return_to_start"], "return_to_start")
        if kind == "circle":
            positive(self.data["radius_m"], "radius_m")
        else:
            for name in ("amplitudes_m", "harmonics", "phase_offsets_rad"):
                values = np.array(finite_vector(self.data[name], 2, name))
                if name != "phase_offsets_rad" and np.any(values <= 0):
                    raise ValueError(f"{name} must be positive")
                if name == "harmonics" and np.any(values % 1):
                    raise ValueError("harmonics must be integers")

    def build(self, angular_speed_rad_s: float, initial_position_m, initial_quaternion_xyzw=None):
        d = self.data
        schedule = SmoothPhaseSchedule(angular_speed_rad_s, d["ramp_duration_s"], d["turns"], d["phase_rad"])
        common = dict(center_m=d["center_m"], plane_rpy_deg=d["plane_rpy_deg"],
                      quaternion_xyzw=d["quaternion_xyzw"], schedule=schedule)
        if d["type"] == "circle":
            path = CircleTrajectory(radius_m=d["radius_m"], **common)
        else:
            path = LissajousTrajectory(amplitudes_m=d["amplitudes_m"], harmonics=d["harmonics"],
                                       phase_offsets_rad=d["phase_offsets_rad"], **common)
        return ApproachAndTrackTrajectory(path, initial_position_m,
            initial_quaternion_xyzw=initial_quaternion_xyzw,
            **{key: d[key] for key in ("approach_duration_s", "settle_duration_s", "exit_duration_s", "return_to_start")})


@dataclass(frozen=True)
class MotionTrialSpec:
    trial_id: str
    trajectory: TrajectorySpec
    controller: ControllerProfile
    angular_speed_rad_s: float | None
    twist_feedforward: bool
    repeat_index: int
    initial_posture: str
    speed_scale: float = 1.0


@dataclass(frozen=True)
class MotionSuiteConfig:
    name: str
    backend: str
    workspace: WorkspaceConfig
    trajectories: dict[str, TrajectorySpec]
    controllers: dict[str, ControllerProfile]
    experiments: tuple[dict, ...]
    file_hashes: dict[str, str]
    preview_initial_position_m: tuple[float, ...]


def load_motion_suite(path: str | Path | None = None) -> MotionSuiteConfig:
    suite_path = Path(path) if path is not None else Path(__file__).parent / "config/suites/simulation_smoke.yaml"
    suite_path = suite_path.expanduser().resolve()
    suite = _load(suite_path)
    required = {"schema_version", "name", "backend", "workspace_file", "trajectories_file", "controllers_file",
                "preview_initial_position_m", "experiments"}
    _keys(suite, required, required, "suite")
    paths = {key: (suite_path.parent / suite[key]).resolve()
             for key in ("workspace_file", "trajectories_file", "controllers_file")}
    workspace_doc, trajectory_doc, controller_doc = (_load(paths[key]) for key in paths)
    _keys(workspace_doc, {"schema_version", "workspace"}, {"schema_version", "workspace"}, "workspace catalog")
    _keys(trajectory_doc, {"schema_version", "trajectories"}, {"schema_version", "trajectories"}, "trajectory catalog")
    _keys(controller_doc, {"schema_version", "controllers"}, {"schema_version", "controllers"}, "controller catalog")
    result = MotionSuiteConfig(
        suite["name"], suite["backend"], WorkspaceConfig(workspace_doc["workspace"]),
        {name: TrajectorySpec(name, value) for name, value in trajectory_doc["trajectories"].items()},
        {name: ControllerProfile(name, value) for name, value in controller_doc["controllers"].items()},
        tuple(suite["experiments"]),
        {key: hashlib.sha256(p.read_bytes()).hexdigest() for key, p in {"suite": suite_path, **paths}.items()},
        finite_vector(suite["preview_initial_position_m"], 3, "preview_initial_position_m"),
    )
    validate_suite(result)
    return result


def validate_suite(suite: MotionSuiteConfig) -> None:
    if not isinstance(suite.name, str) or not suite.name or suite.backend not in ("simulation", "fake", "hardware"):
        raise ValueError("Invalid suite name/backend")
    if not suite.experiments:
        raise ValueError("Suite must contain experiments")
    names = set()
    for experiment in suite.experiments:
        joint = experiment.get("trajectory") in suite.trajectories and suite.trajectories[experiment["trajectory"]].data["type"].startswith("joint_")
        axis = "speed_scales" if joint else "angular_speeds_rad_s"
        required = {"name", "trajectory", "controller", axis, "twist_feedforward",
                    "repeats", "initial_posture"}
        _keys(experiment, required, required, "experiment")
        name = experiment["name"]
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
            raise ValueError("Experiment names must be lowercase path-safe identifiers")
        if name in names:
            raise ValueError(f"Duplicate experiment name: {name}")
        names.add(name)
        if experiment["trajectory"] not in suite.trajectories or experiment["controller"] not in suite.controllers:
            raise ValueError("Unknown trajectory/controller name")
        if suite.controllers[experiment["controller"]].data["mode"] != ("joint" if joint else "cartesian"):
            raise ValueError("Joint/Cartesian trajectory and controller modes must match (Cartesian paths need Cartesian mode)")
        if experiment["initial_posture"] not in suite.workspace.data["initial_postures_rad"]:
            raise ValueError("Unknown initial posture")
        if type(experiment["repeats"]) is not int or experiment["repeats"] < 1:
            raise ValueError("repeats must be a positive integer")
        if any(not isinstance(experiment[key], list) or not experiment[key]
               for key in (axis, "twist_feedforward")):
            raise ValueError("Sweep axes must be nonempty")
        for speed in experiment[axis]:
            positive(speed, "angular_speed_rad_s")
            spec = suite.trajectories[experiment["trajectory"]]
            if joint:
                if speed > 1:
                    raise ValueError("Joint speed_scales must be in (0, 1]")
                bounds = suite.workspace.data["joint_bounds_rad"]["commanded"]
                target = np.array(spec.data["target_position_rad"])
                if spec.data["type"] == "joint_hold" and not np.allclose(target,
                    suite.workspace.data["initial_postures_rad"][experiment["initial_posture"]], rtol=0, atol=1e-12):
                    raise ValueError("Stationary hold target must match its configured initial posture")
                if np.any(target < bounds["lower"]) or np.any(target > bounds["upper"]):
                    raise ValueError("PTP target outside command limits")
            else:
                SmoothPhaseSchedule(speed, spec.data["ramp_duration_s"], spec.data["turns"], spec.data["phase_rad"])
        for option in experiment["twist_feedforward"]:
            _boolean(option, "twist_feedforward")
        if suite.backend == "hardware":
            if not suite.workspace.data["hardware_reviewed"] or not suite.controllers[experiment["controller"]].data["hardware_approved"]:
                raise ValueError("Hardware suite requires reviewed workspace and controller profile")


def expand_trials(suite: MotionSuiteConfig) -> tuple[MotionTrialSpec, ...]:
    validate_suite(suite)
    trials = []
    for experiment in suite.experiments:
        joint = suite.trajectories[experiment["trajectory"]].data["type"].startswith("joint_")
        cells = itertools.product(experiment["speed_scales" if joint else "angular_speeds_rad_s"], experiment["twist_feedforward"])
        for cell_index, (speed, feedforward) in enumerate(cells):
            for repeat in range(experiment["repeats"]):
                trials.append(MotionTrialSpec(f"{experiment['name']}__cell_{cell_index:03d}__repeat_{repeat:03d}",
                    suite.trajectories[experiment["trajectory"]], suite.controllers[experiment["controller"]],
                    None if joint else float(speed), feedforward, repeat, experiment["initial_posture"],
                    float(speed) if joint else 1.0))
    return tuple(trials)
