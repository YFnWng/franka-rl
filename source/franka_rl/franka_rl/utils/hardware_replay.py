"""Load immutable hardware sessions for paired Isaac Lab replay."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml


ROUTE_TASKS = {
    "franky_joint_impedance_tracking":
        "Franka-FR3v2-FrankyImpedance-Incremental6DCirclePath-v0",
    "franky_joint_velocity_impedance_tracking":
        "Franka-FR3v2-FrankyVelocityImpedance-CirclePath-v0",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _uniform(value: Any, length: int, name: str, tolerance: float = 1.0e-9) -> float:
    vector = _finite_vector(value, length, name)
    if max(vector) - min(vector) > tolerance:
        raise ValueError(
            f"Simulator controller currently requires uniform {name}; got {vector}"
        )
    return vector[0]


@dataclass(frozen=True)
class HardwareSessionReplay:
    """Resolved, validated contract for one packaged hardware path session."""

    session_dir: Path
    session_id: int
    controller_mode: str
    task: str
    checkpoint: Path
    checkpoint_sha256: str
    bundle_dir: Path
    bundle_manifest_sha256: str
    initial_joint_position: tuple[float, ...]
    initial_joint_velocity: tuple[float, ...]
    initial_rt_sequence: int
    path_name: str
    path: dict[str, Any]
    controller: dict[str, Any]
    safety: dict[str, Any]
    action_mapping: dict[str, Any]
    hardware_summary: dict[str, Any]
    resolved_config: dict[str, Any]
    final_metadata: dict[str, Any]

    @classmethod
    def from_directory(
        cls,
        session_dir: str | Path,
        *,
        data_root: str | Path,
    ) -> "HardwareSessionReplay":
        session = Path(session_dir).expanduser().resolve()
        root = Path(data_root).expanduser().resolve()
        required = (
            "resolved_config.json",
            "final_metadata.json",
            "config.sha256",
            "samples.csv",
        )
        for name in required:
            if not (session / name).is_file():
                raise FileNotFoundError(session / name)

        config = json.loads((session / "resolved_config.json").read_text())
        metadata = json.loads((session / "final_metadata.json").read_text())
        if config.get("schema_version") != 1 or metadata.get("schema_version") != 1:
            raise ValueError("Only hardware session schema version 1 is supported")
        if metadata.get("terminal_state") != "complete":
            raise ValueError(f"Hardware session is not complete: {metadata.get('terminal_state')!r}")
        if not metadata.get("smooth_stop_ok") or metadata.get("dropped_samples") != 0:
            raise ValueError("Replay requires a clean hardware session with a smooth stop and no dropped samples")
        session_id = int(config["session_id"])
        if int(metadata["session_id"]) != session_id:
            raise ValueError("Session ID differs between resolved config and final metadata")
        recorded_config_hash = (session / "config.sha256").read_text().strip().split()[0]
        if metadata.get("config_sha256") != recorded_config_hash:
            raise ValueError("config.sha256 does not match final_metadata.json")

        robot = config["robot"]
        mode = str(robot["controller_mode"])
        if mode not in ROUTE_TASKS:
            raise ValueError(f"Unsupported hardware controller mode {mode!r}")
        if robot.get("model") != "fr3v2.1" or robot.get("end_effector") != "none":
            raise ValueError("Replay currently supports the audited bare-flange FR3v2.1 only")
        if float(robot.get("external_load_kg", 0.0)) != 0.0:
            raise ValueError("Nonzero hardware payload replay is not implemented yet")
        if int(config["timing"]["command_hz"]) != 50:
            raise ValueError("Replay tasks require the recorded 50 Hz policy rate")

        controller = robot["impedance_controller"]
        _uniform(controller["stiffness_nm_rad"], 7, "stiffness_nm_rad")
        _uniform(controller["damping_nms_rad"], 7, "damping_nms_rad")
        _uniform(controller["expected_error_clip_rad"], 7, "expected_error_clip_rad")
        _finite_vector(controller["constant_torque_offset_nm"], 7, "constant_torque_offset_nm")
        if any(abs(value) > 1.0e-12 for value in controller["constant_torque_offset_nm"]):
            raise ValueError("Nonzero constant torque offsets are not modeled in replay")
        friction = controller["friction"]
        for key in ("coulomb_nm", "viscous_nms_rad"):
            if any(abs(value) > 1.0e-12 for value in _finite_vector(friction[key], 7, key)):
                raise ValueError("Nonzero hardware friction compensation is not modeled in replay")

        ppo = config["ppo"]
        resolved_path = dict(ppo["path"]["resolved_path"])
        waypoints = resolved_path.get("waypoints_m")
        if not isinstance(waypoints, list) or len(waypoints) < 3:
            raise ValueError("Resolved hardware path lacks explicit waypoints")
        for index, point in enumerate(waypoints):
            _finite_vector(point, 3, f"waypoint[{index}]")
        _finite_vector(resolved_path["target_orientation_xyzw"], 4, "target orientation")

        bundle_name = Path(str(ppo["bundle_path"])).name
        bundle_dir = (root / "deployment_bundles" / bundle_name).resolve()
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Local deployment bundle for hardware session not found: {bundle_dir}"
            )
        manifest_hash = _sha256(manifest_path)
        if manifest_hash != ppo["manifest_sha256"]:
            raise ValueError(
                f"Bundle manifest hash mismatch: expected {ppo['manifest_sha256']}, got {manifest_hash}"
            )
        manifest = json.loads(manifest_path.read_text())
        checkpoint = Path(manifest["checkpoint"]["path"]).expanduser().resolve()
        checkpoint_hash = str(manifest["checkpoint"]["sha256"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if _sha256(checkpoint) != checkpoint_hash:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")

        initial_q, initial_dq, initial_sequence = cls._first_valid_state(session / "samples.csv")
        path_summary = metadata["path_summary"]
        if path_summary.get("path_name") != ppo["path"]["name"]:
            raise ValueError("Resolved path and final path summary names differ")
        if len(path_summary.get("outcomes", ())) != len(waypoints):
            raise ValueError("Hardware path outcomes do not cover every waypoint")

        return cls(
            session_dir=session,
            session_id=session_id,
            controller_mode=mode,
            task=ROUTE_TASKS[mode],
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_hash,
            bundle_dir=bundle_dir,
            bundle_manifest_sha256=manifest_hash,
            initial_joint_position=initial_q,
            initial_joint_velocity=initial_dq,
            initial_rt_sequence=initial_sequence,
            path_name=f"hardware_session_{session_id}",
            path=resolved_path,
            controller=controller,
            safety=config["safety"],
            action_mapping=config["action_mapping"],
            hardware_summary=path_summary,
            resolved_config=config,
            final_metadata=metadata,
        )

    @staticmethod
    def _first_valid_state(path: Path) -> tuple[tuple[float, ...], tuple[float, ...], int]:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    if row.get("command_valid", "1").strip().lower() not in ("1", "true"):
                        continue
                    q = tuple(float(row[f"q_j{index}"]) for index in range(1, 8))
                    dq = tuple(float(row[f"dq_j{index}"]) for index in range(1, 8))
                    sequence = int(row["rt_sequence"])
                except (KeyError, TypeError, ValueError):
                    continue
                if all(math.isfinite(value) for value in (*q, *dq)):
                    return q, dq, sequence
        raise ValueError(f"No valid finite measured state in {path}")

    def validate_task(self, task: str) -> None:
        if task != self.task:
            raise ValueError(
                f"Hardware session {self.session_id} requires task {self.task!r}, got {task!r}"
            )

    def validate_path_metadata(self, metadata: dict[str, Any] | None) -> None:
        if metadata is None:
            raise ValueError("Hardware replay requires a waypoint-path task")
        comparisons = (
            ("waypoints_m", self.path["waypoints_m"]),
            ("waypoint_timeout_s", self.path["waypoint_timeout_s"]),
            ("first_waypoint_timeout_s", self.path["first_waypoint_timeout_s"]),
            ("position_threshold_m", self.path["position_threshold_m"]),
            ("target_orientation_xyzw", self.path["target_orientation_xyzw"]),
        )
        for key, expected in comparisons:
            actual = metadata.get(key)
            if actual != expected:
                # YAML tuples and JSON lists compare differently; normalize via JSON.
                if json.loads(json.dumps(actual)) != json.loads(json.dumps(expected)):
                    raise ValueError(f"Configured path {key} differs from hardware session")

    def apply_controller_config(self, action_cfg: Any) -> dict[str, Any]:
        """Apply every supported recorded controller/action parameter before env creation."""
        stiffness = _uniform(self.controller["stiffness_nm_rad"], 7, "stiffness_nm_rad")
        damping = _uniform(self.controller["damping_nms_rad"], 7, "damping_nms_rad")
        error_clip = _uniform(self.controller["expected_error_clip_rad"], 7, "expected_error_clip_rad")
        action_cfg.nominal_stiffness = stiffness
        action_cfg.nominal_damping = damping
        action_cfg.gain_alpha_range = None
        action_cfg.position_error_clip = error_clip
        action_cfg.torque_slew_rate = float(self.controller["max_delta_tau_nm_per_ms"]) * 1000.0
        filter_recorded = "command_filter_cutoff_hz" in self.controller
        action_cfg.command_filter_cutoff_hz = float(
            self.controller.get("command_filter_cutoff_hz", 100.0)
        )
        action_cfg.joint_limit_activation_distance = float(
            self.controller["joint_limit_activation_distance_rad"]
        )
        action_cfg.joint_limit_stiffness = float(self.controller["joint_limit_stiffness_nm"])
        action_cfg.joint_limit_damping = float(self.controller["joint_limit_damping_nms_rad"])
        action_cfg.joint_limit_max_torque = float(self.controller["joint_limit_max_torque_nm"])
        action_cfg.compensate_coriolis = bool(self.controller["compensate_coriolis"])
        action_cfg.compensate_gravity = True

        expected = {
            "model_lower": _finite_vector(self.safety["joint_lower_rad"], 7, "joint_lower_rad"),
            "model_upper": _finite_vector(self.safety["joint_upper_rad"], 7, "joint_upper_rad"),
            "soft_lower": _finite_vector(self.safety["soft_lower_rad"], 7, "soft_lower_rad"),
            "soft_upper": _finite_vector(self.safety["soft_upper_rad"], 7, "soft_upper_rad"),
        }
        for field, value in expected.items():
            if any(abs(a - b) > 1.0e-8 for a, b in zip(getattr(action_cfg, field), value)):
                raise ValueError(f"Task {field} does not match hardware session {self.session_id}")
        recorded_velocity = _finite_vector(
            self.action_mapping["max_reference_velocity_rad_s"], 6,
            "max_reference_velocity_rad_s",
        )
        if any(abs(a - b) > 1.0e-9 for a, b in zip(action_cfg.max_reference_velocity, recorded_velocity)):
            raise ValueError("Task max_reference_velocity differs from deployed action mapping")
        return {
            "stiffness_nm_rad": stiffness,
            "damping_nms_rad": damping,
            "position_error_clip_rad": error_clip,
            "torque_slew_rate_nm_s": action_cfg.torque_slew_rate,
            "command_filter_cutoff_hz": action_cfg.command_filter_cutoff_hz,
            "command_filter_source": (
                "resolved_config" if filter_recorded else "audited_franky_backend_default"
            ),
            "compensate_coriolis": action_cfg.compensate_coriolis,
            "compensate_gravity": action_cfg.compensate_gravity,
            "gain_time_constant_s": float(self.controller["gains_time_constant_s"]),
            "gain_time_constant_note": "fixed gains; no gain transition occurs during replay",
        }

    def path_catalog_document(self) -> dict[str, Any]:
        """Use explicit recorded points; do not regenerate the circle geometrically."""
        return {
            "version": 1,
            "paths": {
                self.path_name: {
                    "type": "waypoints",
                    "description": f"Exact hardware session {self.session_id} schedule",
                    "waypoints_m": self.path["waypoints_m"],
                    "waypoint_timeout_s": self.path["waypoint_timeout_s"],
                    "first_waypoint_timeout_s": self.path["first_waypoint_timeout_s"],
                    "position_threshold_m": self.path["position_threshold_m"],
                    "target_orientation_xyzw": self.path["target_orientation_xyzw"],
                }
            },
        }

    def write_path_catalog(self, destination: str | Path) -> Path:
        destination = Path(destination).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(self.path_catalog_document(), sort_keys=False))
        return destination

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "controller_mode": self.controller_mode,
            "task": self.task,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "bundle_dir": str(self.bundle_dir),
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "initial_state_source": "first command-valid finite samples.csv row",
            "initial_rt_sequence": self.initial_rt_sequence,
            "initial_joint_position_rad": list(self.initial_joint_position),
            "initial_joint_velocity_rad_s": list(self.initial_joint_velocity),
            "effective_policy_delay_steps": 1,
            "hardware_waypoints_reached": int(self.hardware_summary["reached"]),
            "hardware_waypoints_timed_out": int(self.hardware_summary["timed_out"]),
            "hardware_path_complete": bool(self.hardware_summary["complete"]),
            "hardware_terminal_state": self.final_metadata["terminal_state"],
            "hardware_dropped_samples": int(self.final_metadata["dropped_samples"]),
            "resolved_path": self.path,
        }


class HardwareInitialStateController:
    """Replace the Isaac reset event with one exact measured hardware state."""

    def __init__(self, position: Sequence[float], velocity: Sequence[float]):
        self.position = _finite_vector(position, 7, "initial joint position")
        self.velocity = _finite_vector(velocity, 7, "initial joint velocity")

    def install(self, env: Any, reset_event_name: str = "reset_arm") -> None:
        import torch

        reset_cfg = env.event_manager.get_term_cfg(reset_event_name)
        asset_cfg = reset_cfg.params.get("asset_cfg")
        if asset_cfg is None:
            raise ValueError(f"Reset event {reset_event_name!r} has no asset_cfg")
        robot = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, slice):
            resolved_names = tuple(robot.joint_names[joint_ids])
        else:
            resolved_names = tuple(robot.joint_names[index] for index in joint_ids)
        expected_names = tuple(f"panda_joint{index}" for index in range(1, 8))
        if resolved_names != expected_names:
            raise ValueError(f"Replay joint order differs: expected {expected_names}, got {resolved_names}")
        self._torch = torch
        self._position = torch.tensor(self.position, device=env.device, dtype=torch.float32)
        self._velocity = torch.tensor(self.velocity, device=env.device, dtype=torch.float32)
        reset_cfg.func = self._reset_joints
        env.event_manager.set_term_cfg(reset_event_name, reset_cfg)

    def _reset_joints(self, env: Any, env_ids: Sequence[int], asset_cfg: Any, **_: Any) -> None:
        torch = self._torch
        if isinstance(env_ids, slice):
            resolved_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        else:
            resolved_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
        robot = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        position = self._position.expand(len(resolved_ids), -1).clone()
        velocity = self._velocity.expand(len(resolved_ids), -1).clone()
        limits = robot.data.soft_joint_pos_limits.torch[resolved_ids[:, None], joint_ids]
        if torch.any(position < limits[..., 0]) or torch.any(position > limits[..., 1]):
            raise RuntimeError("Recorded hardware initial state lies outside simulated soft limits")
        robot.write_joint_position_to_sim_index(
            position=position, joint_ids=joint_ids, env_ids=resolved_ids
        )
        robot.write_joint_velocity_to_sim_index(
            velocity=velocity, joint_ids=joint_ids, env_ids=resolved_ids
        )


def rotation_matrix_xyz(roll: float, pitch: float, yaw: float) -> tuple[tuple[float, ...], ...]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def save_simulation_trace(
    trace: dict[str, Any],
    destination: str | Path,
    *,
    physics_dt_s: float,
    circle_path: dict[str, Any],
) -> dict[str, Any]:
    """Write a normalized 1 kHz trace and parity metrics for one replay."""
    import torch

    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    required = {
        "joint_position", "joint_velocity", "position_reference",
        "target_velocity_reference", "applied_velocity_reference",
        "normalized_action", "requested_torque", "slew_limited_torque",
        "filtered_torque", "applied_torque", "end_effector_position_base",
        "waypoint_index",
    }
    if not required.issubset(trace):
        raise ValueError(f"Simulation trace lacks {sorted(required - trace.keys())}")
    count = int(trace["joint_position"].shape[0])
    if count < 2 or any(int(value.shape[0]) != count for value in trace.values()):
        raise ValueError("Simulation trace fields have inconsistent or insufficient length")

    fieldnames = ["tick", "time_s", "waypoint_index"]
    groups = (
        ("q", "joint_position"),
        ("dq", "joint_velocity"),
        ("q_ref", "position_reference"),
        ("dq_ref_target", "target_velocity_reference"),
        ("dq_ref_applied", "applied_velocity_reference"),
        ("tau_requested", "requested_torque"),
        ("tau_slew_limited", "slew_limited_torque"),
        ("tau_filtered", "filtered_torque"),
        ("tau_applied", "applied_torque"),
    )
    for prefix, _ in groups:
        fieldnames.extend(f"{prefix}_j{index}" for index in range(1, 8))
    action_count = int(trace["normalized_action"].shape[1])
    fieldnames.extend(f"action_{index}" for index in range(1, action_count + 1))
    fieldnames.extend(("flange_x_m", "flange_y_m", "flange_z_m"))
    with (destination / "simulation_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for tick in range(count):
            row: dict[str, Any] = {
                "tick": tick,
                "time_s": tick * physics_dt_s,
                "waypoint_index": int(trace["waypoint_index"][tick, 0]),
            }
            for prefix, key in groups:
                for joint in range(7):
                    row[f"{prefix}_j{joint + 1}"] = float(trace[key][tick, joint])
            for action in range(action_count):
                row[f"action_{action + 1}"] = float(trace["normalized_action"][tick, action])
            for axis, name in enumerate(("x", "y", "z")):
                row[f"flange_{name}_m"] = float(trace["end_effector_position_base"][tick, axis])
            writer.writerow(row)

    velocity = trace["joint_velocity"].float()
    acceleration = (torch.diff(velocity, dim=0) / float(physics_dt_s)).abs()
    tracking_error = (trace["joint_position"] - trace["position_reference"]).abs().float()
    commanded_torque = trace["filtered_torque"].abs().float()
    applied_torque = trace["applied_torque"].abs().float()
    speed = velocity.abs()
    ee_position = trace["end_effector_position_base"].float()
    center = torch.tensor(circle_path["center_m"], dtype=ee_position.dtype)
    radius = float(circle_path["radius_m"])
    rpy = tuple(math.radians(float(value)) for value in circle_path["orientation_rpy_deg"])
    rotation = rotation_matrix_xyz(*rpy)
    normal = torch.tensor([rotation[row][2] for row in range(3)], dtype=ee_position.dtype)
    offset = ee_position - center
    axial = offset @ normal
    radial = torch.linalg.vector_norm(offset - axial[:, None] * normal, dim=1)
    circle_error = torch.sqrt(axial.square() + (radial - radius).square())
    circle_mask = trace["waypoint_index"][:, 0] > 0
    if not bool(torch.any(circle_mask)):
        raise ValueError("Trace never advanced beyond the initial approach waypoint")
    traversal_error = circle_error[circle_mask]

    def stats(values: Any) -> dict[str, float]:
        values = values.flatten().float()
        return {
            "mean": float(values.mean()),
            "median": float(torch.quantile(values, 0.5)),
            "p95": float(torch.quantile(values, 0.95)),
            "p99": float(torch.quantile(values, 0.99)),
            "maximum": float(values.max()),
        }

    summary = {
        "schema_version": 1,
        "sample_rate_hz": 1.0 / physics_dt_s,
        "sample_count": count,
        "duration_s": (count - 1) * physics_dt_s,
        "circle_error_excluding_initial_approach_m": stats(traversal_error),
        "circle_error_all_samples_m": stats(circle_error),
        "abs_tracking_error_rad": stats(tracking_error),
        "abs_joint_velocity_rad_s": stats(speed),
        "abs_estimated_acceleration_rad_s2": stats(acceleration),
        "abs_commanded_torque_nm": stats(commanded_torque),
        "abs_applied_torque_nm": stats(applied_torque),
    }
    (destination / "simulation_trace_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
