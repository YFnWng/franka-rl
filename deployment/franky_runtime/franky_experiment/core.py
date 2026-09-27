"""Pure configuration, planning, observation, and supervision logic."""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .path_runtime import PathCatalog

JOINTS = 7
MODES = {"reference", "ppo"}
TERMINAL = {"complete", "fault", "stopped"}
EXPECTED_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
FR3_MAX_VELOCITY = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]
FR3_MAX_ACCELERATION = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]
FR3_MAX_JERK = [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0]


class ConfigError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector(value: Any, size: int, name: str) -> list[float]:
    require(isinstance(value, list) and len(value) == size, f"{name} must have {size} values")
    result = [float(v) for v in value]
    require(all(math.isfinite(v) for v in result), f"{name} must be finite")
    return result


def vectors_close(left: list[float], right: list[float], *, atol: float = 1.0e-10) -> bool:
    return len(left) == len(right) and all(abs(a - b) <= atol for a, b in zip(left, right))


def resolved(base: Path, value: Any, name: str) -> Path:
    require(isinstance(value, str) and value, f"{name} is required")
    path = Path(os.path.expanduser(value))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def executable_path(base: Path, value: Any, name: str) -> Path:
    """Return an absolute executable path without dereferencing venv symlinks."""
    require(isinstance(value, str) and value, f"{name} is required")
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        path = base / path
    path = Path(os.path.abspath(path))
    require(path.is_file() and os.access(path, os.X_OK), f"{name} is not executable")
    return path


def _approval(data: dict[str, Any], require_approved: bool) -> None:
    if not require_approved:
        return
    require(data.get("status") == "approved", "status must be approved")
    require(data.get("executable") is True, "executable must be true")
    if data.get("execution_context") == "shadow":
        require(data.get("motion_authorized") is False,
                "shadow execution must explicitly set motion_authorized false")
    else:
        require(data.get("motion_authorized") is True, "motion_authorized must be true")
    approval = data.get("approval", {})
    for key in ("approved_by", "approved_at", "physical_stop_procedure", "workspace_review"):
        require(isinstance(approval.get(key), str) and approval[key].strip(), f"approval.{key} required")


def _load_common(data: dict[str, Any], source_path: Path, require_approved: bool) -> None:
    require(data.get("schema_version") == 1, "unsupported schema_version")
    require(data.get("runtime") == "franky_joint_impedance_tracking_v1",
            "runtime must be franky_joint_impedance_tracking_v1")
    require(data.get("mode") in MODES, "mode must be reference or ppo")
    require(data.get("execution_context") in {"fake", "shadow", "hardware"},
            "execution_context must be fake, shadow, or hardware")
    require(data.get("mode") == "ppo" or data.get("execution_context") != "shadow",
            "shadow execution supports PPO mode only")
    require(isinstance(data.get("session_id"), int) and data["session_id"] > 0, "positive session_id required")
    _approval(data, require_approved)

    robot = data.get("robot", {})
    require(robot.get("model") == "fr3v2.1", "robot.model must be fr3v2.1")
    require(robot.get("arm_revision") == "Arm3Rv2_02.01", "robot.arm_revision mismatch")
    require(robot.get("end_effector") == "none", "robot.end_effector must be none")
    require(float(robot.get("external_load_kg", math.nan)) == 0.0, "robot.external_load_kg must be zero")
    require(robot.get("require_identity_f_t_ee") is True, "identity F_T_EE must be required")
    require(isinstance(robot.get("host"), str) and robot["host"], "robot.host required")
    require(robot.get("control_interface") == "torque", "robot.control_interface must be torque")
    require(robot.get("controller_mode") == "franky_joint_impedance_tracking",
            "robot.controller_mode must be franky_joint_impedance_tracking")
    ee_feedback = robot.get("ee_feedback", {})
    require(ee_feedback.get("source") == "measured_joint_encoder_libfranka_fk",
            "robot.ee_feedback.source must be measured joint encoder libfranka FK")
    require(ee_feedback.get("joint_signal") == "q",
            "robot.ee_feedback.joint_signal must be measured q")
    require(ee_feedback.get("base_frame") == "fr3_link0",
            "robot.ee_feedback.base_frame must be fr3_link0")
    require(ee_feedback.get("frame") == "fr3_flange",
            "robot.ee_feedback.frame must be fr3_flange")
    require(ee_feedback.get("model_frame") == "Frame.Flange",
            "robot.ee_feedback.model_frame must be Frame.Flange")
    require(ee_feedback.get("reported_pose_role") == "audit_only",
            "robot.ee_feedback reported pose must be audit_only")
    expected = robot.get("expected_franky_version")
    require(isinstance(expected, str) and expected, "robot.expected_franky_version required")
    collision = robot.get("constructor_collision_behavior", {})
    require(collision.get("acknowledged") is True, "Franky constructor collision behavior must be acknowledged")
    require(float(collision.get("joint_torque_threshold_nm", 0)) == 20.0,
            "Franky constructor joint threshold must match source default 20 Nm")
    require(float(collision.get("cartesian_force_threshold_n", 0)) == 30.0,
            "Franky constructor force threshold must match source default 30 N")
    impedance = robot.get("impedance_controller", {})
    for key in ("stiffness_nm_rad", "damping_nms_rad", "constant_torque_offset_nm",
                "expected_error_clip_rad"):
        impedance[key] = vector(impedance.get(key), JOINTS, f"robot.impedance_controller.{key}")
    require(all(v > 0 for v in impedance["stiffness_nm_rad"]), "impedance stiffness must be positive")
    require(all(v >= 0 for v in impedance["damping_nms_rad"]), "impedance damping must be nonnegative")
    require(all(v > 0 for v in impedance["expected_error_clip_rad"]), "impedance error clip must be positive")
    require(impedance.get("compensate_coriolis") is True, "Coriolis compensation must be enabled")
    for key in ("max_delta_tau_nm_per_ms", "gains_time_constant_s", "joint_limit_activation_distance_rad",
                "joint_limit_stiffness_nm", "joint_limit_damping_nms_rad", "joint_limit_max_torque_nm"):
        require(float(impedance.get(key, 0)) > 0, f"robot.impedance_controller.{key} must be positive")
    friction = impedance.get("friction", {})
    for key in ("coulomb_nm", "viscous_nms_rad", "max_torque_nm"):
        friction[key] = vector(friction.get(key), JOINTS, f"robot.impedance_controller.friction.{key}")
    require(all(v == 0 for v in friction["coulomb_nm"] + friction["viscous_nms_rad"]),
            "friction compensation must remain disabled for initial deployment")
    require(all(v > 0 for v in friction["max_torque_nm"]), "friction max torque must be positive")
    require(float(friction.get("velocity_epsilon_rad_s", 0)) > 0, "friction velocity epsilon must be positive")
    impedance["friction"] = friction
    robot["impedance_controller"] = impedance
    stop = robot.get("torque_stop", {})
    stop["damping_nms_rad"] = vector(stop.get("damping_nms_rad"), JOINTS, "robot.torque_stop.damping_nms_rad")
    require(all(v > 0 for v in stop["damping_nms_rad"]), "torque stop damping must be positive")
    require(stop.get("compensate_coriolis") is True, "torque stop Coriolis compensation must be enabled")
    for key in ("ramp_duration_s", "velocity_epsilon_rad_s", "max_duration_s", "max_delta_tau_nm_per_ms"):
        require(float(stop.get(key, 0)) > 0, f"robot.torque_stop.{key} must be positive")
    robot["torque_stop"] = stop
    data["robot"] = robot

    mapping = data.get("action_mapping", {})
    if data["mode"] == "reference":
        mapping["default_position_rad"] = vector(mapping.get("default_position_rad"), JOINTS,
                                                  "action_mapping.default_position_rad")
        scale = mapping.get("scale_rad")
        if isinstance(scale, (int, float)) and not isinstance(scale, bool):
            scale = [float(scale)] * JOINTS
        mapping["scale_rad"] = vector(scale, JOINTS, "action_mapping.scale_rad")
        require(mapping.get("clip") is None, "reference action clipping is unsupported")
    else:
        require(mapping.get("type") == "normalized_position_increment",
                "PPO action mapping must be normalized_position_increment")
        policy_joint_count = int(mapping.get("policy_joint_count", 0))
        require(policy_joint_count == 6, "PPO policy_joint_count must be 6")
        require(mapping.get("controlled_joint_names") == EXPECTED_JOINTS[:policy_joint_count],
                "PPO controlled joint order must be FR3 joints 1--6")
        require(mapping.get("held_joint_names") == EXPECTED_JOINTS[policy_joint_count:],
                "PPO held joint set must contain only FR3 joint 7")
        require(float(mapping.get("normalized_lower", math.nan)) == -1.0 and
                float(mapping.get("normalized_upper", math.nan)) == 1.0,
                "PPO normalized action interval must be [-1, 1]")
        require(mapping.get("integration") == "forward_euler", "PPO integration must be forward_euler")
        require(mapping.get("initial_reference") == "measured_start_position",
                "PPO reference must initialize from measured start position")
        mapping["max_reference_velocity_rad_s"] = vector(
            mapping.get("max_reference_velocity_rad_s"), policy_joint_count,
            "action_mapping.max_reference_velocity_rad_s")
        mapping["max_position_increment_rad"] = vector(
            mapping.get("max_position_increment_rad"), policy_joint_count,
            "action_mapping.max_position_increment_rad")
        require(mapping.get("soft_limit_projection") is True, "PPO soft-limit projection must be enabled")
    data["action_mapping"] = mapping

    timing = data.get("timing", {})
    expected_hz = 30.0 if data["mode"] == "reference" else 50.0
    require(float(timing.get("command_hz", 0)) == expected_hz,
            f"timing.command_hz must be exactly {expected_hz:g}")
    if data["mode"] == "ppo":
        expected_increment = [value / expected_hz for value in mapping["max_reference_velocity_rad_s"]]
        require(all(abs(actual - expected) < 1.0e-12 for actual, expected in
                    zip(mapping["max_position_increment_rad"], expected_increment)),
                "PPO position increments must equal max reference velocity / command_hz")
    for key in ("state_timeout_ms", "callback_gap_ms", "command_watchdog_ms", "suite_timeout_s"):
        require(float(timing.get(key, 0)) > 0, f"timing.{key} must be positive")
    require(float(timing["command_watchdog_ms"]) > 1000.0 / expected_hz,
            "command watchdog must exceed one command period")
    require(float(timing["callback_gap_ms"]) <= float(timing["command_watchdog_ms"]),
            "callback gap cannot exceed command watchdog")
    if data["mode"] == "ppo":
        for key in ("inference_timeout_ms", "trial_timeout_s"):
            require(float(timing.get(key, 0)) > 0, f"timing.{key} must be positive")
    data["timing"] = timing

    safety = data.get("safety", {})
    for key in ("joint_lower_rad", "joint_upper_rad", "position_margin_rad", "start_tolerance_rad",
                "max_start_velocity_rad_s", "max_tracking_error_rad"):
        safety[key] = vector(safety.get(key), JOINTS, f"safety.{key}")
    require(all(lo < hi for lo, hi in zip(safety["joint_lower_rad"], safety["joint_upper_rad"])),
            "joint lower bounds must be below upper bounds")
    require(all(v > 0 for v in safety["position_margin_rad"] + safety["start_tolerance_rad"] +
                safety["max_start_velocity_rad_s"] + safety["max_tracking_error_rad"]),
            "safety margins and tolerances must be positive")
    lower = [lo + m for lo, m in zip(safety["joint_lower_rad"], safety["position_margin_rad"])]
    upper = [hi - m for hi, m in zip(safety["joint_upper_rad"], safety["position_margin_rad"])]
    require(all(lo < hi for lo, hi in zip(lower, upper)), "position margins eliminate the usable range")
    safety["soft_lower_rad"], safety["soft_upper_rad"] = lower, upper
    factor = float(safety.get("reference_derivative_limit_factor", 0))
    require(0.0 < factor <= 1.0, "safety.reference_derivative_limit_factor must be in (0,1]")
    data["safety"] = safety

    artifacts = data.get("artifacts", {})
    artifacts["root"] = str(resolved(source_path.parent, artifacts.get("root"), "artifacts.root"))
    data["artifacts"] = artifacts


def load_experiment(path: str | Path, require_approved: bool = True) -> dict[str, Any]:
    source_path = Path(path).resolve(strict=True)
    data = yaml.safe_load(source_path.read_text())
    require(isinstance(data, dict), "experiment root must be a mapping")
    _load_common(data, source_path, require_approved)

    if data["mode"] == "reference":
        source = data.get("reference", {})
        source["start_position_rad"] = vector(source.get("start_position_rad"), JOINTS,
                                               "reference.start_position_rad")
        trials = source.get("trials")
        require(isinstance(trials, list) and trials, "reference.trials must be nonempty")
        seen: set[str] = set()
        for index, trial in enumerate(trials):
            name = f"reference.trials[{index}]"
            require(isinstance(trial, dict), f"{name} must be a mapping")
            require(isinstance(trial.get("id"), str) and trial["id"] and trial["id"] not in seen,
                    f"{name}.id must be nonempty and unique")
            seen.add(trial["id"])
            require(1 <= int(trial.get("joint", 0)) <= JOINTS, f"{name}.joint must be 1..7")
            for key in ("amplitude_rad", "frequency_hz", "warmup_s", "ramp_s", "cycles", "post_hold_s"):
                require(float(trial.get(key, 0)) > 0, f"{name}.{key} must be positive")
            joint = int(trial["joint"]) - 1
            amplitude = float(trial["amplitude_rad"])
            frequency = float(trial["frequency_hz"])
            ramp = float(trial["ramp_s"])
            q0 = source["start_position_rad"][joint]
            require(q0 - amplitude >= data["safety"]["soft_lower_rad"][joint] and
                    q0 + amplitude <= data["safety"]["soft_upper_rad"][joint],
                    f"{name} exceeds soft joint bounds")
            # Conservative absolute derivative bounds for A*e(t)*sin(w*t),
            # reusing the prior ROS coordinator's quintic-envelope analysis.
            omega = 2.0 * math.pi * frequency
            e1, e2, e3 = 1.875 / ramp, 5.774 / ramp**2, 60.0 / ramp**3
            demanded = (
                amplitude * (e1 + omega),
                amplitude * (e2 + 2.0 * e1 * omega + omega**2),
                amplitude * (e3 + 3.0 * e2 * omega + 3.0 * e1 * omega**2 + omega**3),
            )
            factor = float(data["safety"]["reference_derivative_limit_factor"])
            available = (
                factor * FR3_MAX_VELOCITY[joint],
                factor * FR3_MAX_ACCELERATION[joint],
                factor * FR3_MAX_JERK[joint],
            )
            require(all(actual <= limit for actual, limit in zip(demanded, available)),
                    f"{name} conservative waveform derivative bound exceeds Franky dynamics")
        require(float(source.get("inter_trial_hold_s", 0)) >= 0, "reference.inter_trial_hold_s must be nonnegative")
        data["reference"] = source
    else:
        source = data.get("ppo", {})
        bundle = resolved(source_path.parent, source.get("bundle_path"), "ppo.bundle_path")
        manifest = bundle / "manifest.json"
        require(bundle.is_dir() and manifest.is_file(), "PPO bundle/manifest missing")
        digest = source.get("manifest_sha256")
        require(isinstance(digest, str) and len(digest) == 64, "ppo.manifest_sha256 required")
        require(sha256(manifest) == digest, "PPO manifest trust anchor mismatch")
        worker = executable_path(source_path.parent, source.get("worker_python"), "ppo.worker_python")
        source["start_position_rad"] = vector(source.get("start_position_rad"), JOINTS, "ppo.start_position_rad")
        contract_path = bundle / "policy_contract.yaml"
        require(contract_path.is_file(), "PPO policy contract missing")
        contract = yaml.safe_load(contract_path.read_text())
        require(contract.get("schema_version") == 2, "PPO contract schema_version must be 2")
        require(contract.get("contract_id") == "fr3_incremental_position_29d_v1",
                "unsupported PPO contract_id")
        require(contract.get("robot_model") == "fr3", "PPO contract robot_model must be fr3")
        require(contract.get("joint_names") == EXPECTED_JOINTS, "PPO contract joint order/model is not FR3")
        require(contract.get("frames", {}).get("base") == "fr3_link0", "PPO base frame mismatch")
        require(contract.get("frames", {}).get("tracked_body") == "fr3_flange", "PPO tracked body mismatch")
        require(abs(float(contract.get("policy_period_s", 0)) - 1.0 / 50.0) < 1e-12,
                "PPO policy period is not 50 Hz")
        action_contract = contract.get("action", {})
        require(action_contract.get("inference") == "deterministic_tanh",
                "PPO contract must use deterministic tanh inference")
        require(action_contract.get("type") == "normalized_position_increment",
                "PPO contract action type mismatch")
        policy_joint_count = data["action_mapping"]["policy_joint_count"]
        require(int(action_contract.get("size", 0)) == policy_joint_count,
                "PPO contract action size mismatch")
        require(vectors_close(
                vector(action_contract.get("max_reference_velocity_rad_s"), policy_joint_count,
                       "bundle max reference velocity"),
                data["action_mapping"]["max_reference_velocity_rad_s"]),
                "PPO reference velocity mismatch")
        require(vectors_close(
                vector(action_contract.get("max_position_increment_rad"), policy_joint_count,
                       "bundle max position increment"),
                data["action_mapping"]["max_position_increment_rad"]),
                "PPO position increment mismatch")
        require(action_contract.get("controlled_joint_names") == data["action_mapping"]["controlled_joint_names"],
                "PPO controlled joint names mismatch")
        require(action_contract.get("held_joint_names") == data["action_mapping"]["held_joint_names"],
                "PPO held joint names mismatch")
        require(action_contract.get("integration") == data["action_mapping"]["integration"],
                "PPO integration mismatch")
        require(action_contract.get("initial_reference") == data["action_mapping"]["initial_reference"],
                "PPO reference initialization mismatch")
        require(action_contract.get("soft_limit_projection") is True,
                "PPO bundle must require soft-limit projection")
        require(vectors_close(
                vector(action_contract.get("soft_lower_rad"), JOINTS, "bundle soft lower"),
                data["safety"]["soft_lower_rad"]), "PPO soft lower bounds mismatch")
        require(vectors_close(
                vector(action_contract.get("soft_upper_rad"), JOINTS, "bundle soft upper"),
                data["safety"]["soft_upper_rad"]), "PPO soft upper bounds mismatch")
        require(vectors_close(
                vector(contract.get("default_joint_position_rad"), JOINTS, "bundle q_default"),
                source["start_position_rad"]), "PPO q_default/start position mismatch")
        require(int(contract.get("observation", {}).get("size", 0)) == 29,
                "PPO observation size must be 29")
        controller = contract.get("controller", {})
        impedance = data["robot"]["impedance_controller"]
        require(all(float(value) == float(controller.get("nominal_stiffness_nm_rad", 0))
                    for value in impedance["stiffness_nm_rad"]),
                "PPO nominal stiffness mismatch")
        require(all(float(value) == float(controller.get("nominal_damping_nms_rad", 0))
                    for value in impedance["damping_nms_rad"]),
                "PPO nominal damping mismatch")
        require(float(controller.get("position_error_clip_rad", 0)) ==
                impedance["expected_error_clip_rad"][0],
                "PPO position error clip mismatch")
        require(float(controller.get("torque_slew_rate_nm_s", 0)) ==
                1000.0 * float(impedance["max_delta_tau_nm_per_ms"]),
                "PPO torque slew mismatch")
        require(controller.get("coriolis_compensation") is impedance["compensate_coriolis"],
                "PPO Coriolis compensation mismatch")
        expressions = [item.get("expression") for item in contract.get("observation_layout", [])]
        require(expressions == ["q_measured - q_default", "dq_measured",
                                "target_base - fr3_flange_base", "q_reference_normalized",
                                "previous_normalized_position_increment"],
                "PPO observation layout mismatch")
        targets, path_selection = source.get("targets"), source.get("path")
        has_targets = isinstance(targets, list) and bool(targets)
        has_path = isinstance(path_selection, dict)
        require(has_targets != has_path, "exactly one of ppo.targets and ppo.path is required")
        if has_targets:
            for index, target in enumerate(targets):
                require(isinstance(target, dict), f"ppo.targets[{index}] must be a mapping")
                require(isinstance(target.get("id"), str) and target["id"],
                        f"ppo.targets[{index}].id required")
                target["position_base_m"] = vector(target.get("position_base_m"), 3,
                                                    f"ppo.targets[{index}].position_base_m")
            require(isinstance(source.get("repetitions"), int) and source["repetitions"] > 0,
                    "ppo.repetitions must be positive")
            success = source.get("success", {})
            require(float(success.get("position_threshold_m", 0)) > 0,
                    "PPO position threshold must be positive")
            require(int(success.get("consecutive_observations", 0)) > 0,
                    "PPO consecutive observations must be positive")
            require(float(success.get("max_joint_velocity_rad_s", 0)) > 0,
                    "PPO settled velocity must be positive")
            source["execution_mode"] = "targets"
        else:
            catalog_path = resolved(source_path.parent, path_selection.get("catalog_path"),
                                    "ppo.path.catalog_path")
            trusted = path_selection.get("catalog_sha256")
            require(isinstance(trusted, str) and len(trusted) == 64,
                    "ppo.path.catalog_sha256 required")
            require(catalog_path.is_file() and sha256(catalog_path) == trusted,
                    "PPO path catalog trust anchor mismatch")
            try:
                catalog = PathCatalog.from_yaml(catalog_path)
                spec = catalog.get(path_selection.get("name"))
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigError(f"invalid PPO path catalog: {exc}") from exc
            require(isinstance(path_selection.get("repetitions"), int) and
                    path_selection["repetitions"] > 0, "ppo.path.repetitions must be positive")
            workspace_lower = vector(path_selection.get("workspace_lower_base_m"), 3,
                                     "ppo.path.workspace_lower_base_m")
            workspace_upper = vector(path_selection.get("workspace_upper_base_m"), 3,
                                     "ppo.path.workspace_upper_base_m")
            require(all(lo < hi for lo, hi in zip(workspace_lower, workspace_upper)),
                    "PPO path workspace bounds are invalid")
            require(all(workspace_lower[i] <= point[i] <= workspace_upper[i]
                        for point in spec.waypoints_m for i in range(3)),
                    "PPO path leaves reviewed workspace")
            path_selection.update({
                "catalog_path": str(catalog_path),
                "resolved_catalog_sha256": catalog.sha256,
                "resolved_catalog_version": catalog.version,
                "resolved_path": spec.to_dict(),
                "workspace_lower_base_m": workspace_lower,
                "workspace_upper_base_m": workspace_upper,
            })
            source.pop("targets", None)
            source.pop("success", None)
            source.pop("repetitions", None)
            source["execution_mode"] = "path"
        source["bundle_path"], source["worker_python"] = str(bundle), str(worker)
        data["ppo"] = source

    start = data[data["mode"]]["start_position_rad"]
    check_target(data, start, "reviewed start")
    data["_config_path"] = str(source_path)
    data["_config_sha256"] = sha256(source_path)
    return data


def check_target(config: dict[str, Any], target: list[float] | tuple[float, ...], name: str = "target") -> list[float]:
    q = vector(list(target), JOINTS, name)
    lower, upper = config["safety"]["soft_lower_rad"], config["safety"]["soft_upper_rad"]
    require(all(lower[i] <= q[i] <= upper[i] for i in range(JOINTS)), f"{name} exceeds soft joint bounds")
    return q


def action_to_target(
    config: dict[str, Any], action: list[float], reference: list[float] | tuple[float, ...] | None = None
) -> list[float]:
    mapping = config["action_mapping"]
    action_size = int(mapping.get("policy_joint_count", JOINTS))
    raw = vector(list(action), action_size, "policy action")
    if mapping.get("type") != "normalized_position_increment":
        return check_target(config, [mapping["default_position_rad"][i] + mapping["scale_rad"][i] * raw[i]
                                     for i in range(JOINTS)], "mapped policy target")
    require(reference is not None, "incremental action requires the previous position reference")
    prior = vector(list(reference), JOINTS, "previous position reference")
    require(all(-1.0 <= value <= 1.0 for value in raw), "incremental policy action exceeds [-1, 1]")
    lower, upper = config["safety"]["soft_lower_rad"], config["safety"]["soft_upper_rad"]
    increment = mapping["max_position_increment_rad"]
    target = prior.copy()
    for i in range(action_size):
        target[i] = min(upper[i], max(lower[i], prior[i] + increment[i] * raw[i]))
    return target


@dataclass(frozen=True)
class PlanPoint:
    trial_id: int
    trial_name: str
    q_target: tuple[float, ...]
    complete: bool


class ReferencePlan:
    """Old coordinator's smooth sine/quintic-ramp identification plan."""
    def __init__(self, config: dict[str, Any]):
        source = config["reference"]
        self.start = tuple(source["start_position_rad"])
        self.inter_hold = float(source["inter_trial_hold_s"])
        self.trials = source["trials"]
        total = 0.0
        for trial in self.trials:
            total += (float(trial["warmup_s"]) + 2.0 * float(trial["ramp_s"]) +
                      float(trial["cycles"]) / float(trial["frequency_hz"]) +
                      float(trial["post_hold_s"]))
            total += self.inter_hold
        self.duration = total - self.inter_hold

    @staticmethod
    def _smooth(u: float) -> float:
        u = min(1.0, max(0.0, u))
        return u**3 * (10.0 + u * (-15.0 + 6.0 * u))

    def point(self, elapsed_s: float) -> PlanPoint:
        if elapsed_s >= self.duration:
            return PlanPoint(len(self.trials), "complete", self.start, True)
        cursor = 0.0
        for index, trial in enumerate(self.trials):
            warm, ramp = float(trial["warmup_s"]), float(trial["ramp_s"])
            active = float(trial["cycles"]) / float(trial["frequency_hz"])
            duration = warm + 2.0 * ramp + active + float(trial["post_hold_s"])
            if elapsed_s < cursor + duration:
                local, offset = elapsed_s - cursor, 0.0
                if local >= warm:
                    phase_t = local - warm
                    sine = math.sin(2.0 * math.pi * float(trial["frequency_hz"]) * phase_t)
                    if phase_t < ramp:
                        envelope = self._smooth(phase_t / ramp)
                    elif phase_t < ramp + active:
                        envelope = 1.0
                    elif phase_t < 2.0 * ramp + active:
                        envelope = 1.0 - self._smooth((phase_t - ramp - active) / ramp)
                    else:
                        envelope = 0.0
                    offset = float(trial["amplitude_rad"]) * envelope * sine
                q = list(self.start)
                q[int(trial["joint"]) - 1] += offset
                return PlanPoint(index + 1, str(trial["id"]), tuple(q), False)
            cursor += duration
            if elapsed_s < cursor + self.inter_hold:
                return PlanPoint(index + 1, f"{trial['id']}:inter_hold", self.start, False)
            cursor += self.inter_hold
        return PlanPoint(len(self.trials), "complete", self.start, True)


def assemble_policy_observation(
    q, dq, target, flange, reference, previous, default, lower, upper, controlled_joint_count
) -> list[float]:
    q = vector(list(q), 7, "q"); dq = vector(list(dq), 7, "dq")
    target = vector(list(target), 3, "target"); flange = vector(list(flange), 3, "flange")
    reference = vector(list(reference), 7, "reference")
    previous = vector(list(previous), controlled_joint_count, "previous")
    default = vector(list(default), 7, "default")
    lower = vector(list(lower), 7, "lower"); upper = vector(list(upper), 7, "upper")
    reference_normalized = [2.0 * (reference[i] - lower[i]) / (upper[i] - lower[i]) - 1.0
                            for i in range(controlled_joint_count)]
    return ([q[i] - default[i] for i in range(7)] + dq +
            [target[i] - flange[i] for i in range(3)] + reference_normalized + previous)


class SessionSupervisor:
    """Sticky-fault supervisor; it never recovers or restarts a session."""
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.state = "waiting"
        self.reason = ""

    def fault(self, reason: str) -> None:
        if self.state not in TERMINAL:
            self.state, self.reason = "fault", reason

    def ready(self, snapshot: Any) -> tuple[bool, str]:
        if snapshot.has_errors:
            return False, "robot has current errors"
        if snapshot.is_in_control:
            return False, "robot already has an active control loop"
        if snapshot.mode not in {"Idle", "Move"}:
            return False, f"robot mode is {snapshot.mode}"
        start = self.config[self.config["mode"]]["start_position_rad"]
        tol = self.config["safety"]["start_tolerance_rad"]
        vmax = self.config["safety"]["max_start_velocity_rad_s"]
        if any(abs(snapshot.q[i] - start[i]) > tol[i] for i in range(JOINTS)):
            return False, "outside reviewed start tolerance"
        if any(abs(snapshot.dq[i]) > vmax[i] for i in range(JOINTS)):
            return False, "robot is not near stationary"
        try:
            check_target(self.config, snapshot.q, "measured start")
            check_target(self.config, snapshot.q_d, "desired start")
        except ConfigError as exc:
            return False, str(exc)
        self.state = "ready"
        return True, ""

    def validate_active(self, snapshot: Any) -> None:
        if self.state != "running":
            return
        if snapshot.has_errors:
            return self.fault("robot_error")
        if snapshot.mode not in {"Move", "Idle"}:
            return self.fault(f"robot_mode:{snapshot.mode}")
        try:
            check_target(self.config, snapshot.q, "measured state")
            check_target(self.config, snapshot.q_d, "desired state")
        except ConfigError as exc:
            return self.fault(str(exc))
        limit = self.config["safety"]["max_tracking_error_rad"]
        if any(abs(snapshot.q[i] - snapshot.q_d[i]) > limit[i] for i in range(JOINTS)):
            self.fault("tracking_error")

    def start(self) -> None:
        require(self.state == "ready", "session is not ready")
        self.state = "running"

    def complete(self) -> None:
        if self.state == "running":
            self.state = "complete"

    def stop(self, reason: str = "operator_stop") -> None:
        if self.state not in TERMINAL:
            self.state, self.reason = "stopped", reason
