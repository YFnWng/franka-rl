"""Pure configuration, planning, observation, and supervision logic."""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

JOINTS = 7
MODES = {"reference", "ppo"}
TERMINAL = {"complete", "fault", "stopped"}
EXPECTED_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


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


def resolved(base: Path, value: Any, name: str) -> Path:
    require(isinstance(value, str) and value, f"{name} is required")
    path = Path(os.path.expanduser(value))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _approval(data: dict[str, Any], require_approved: bool) -> None:
    if not require_approved:
        return
    require(data.get("status") == "approved", "status must be approved")
    require(data.get("executable") is True, "executable must be true")
    require(data.get("motion_authorized") is True, "motion_authorized must be true")
    approval = data.get("approval", {})
    for key in ("approved_by", "approved_at", "physical_stop_procedure", "workspace_review"):
        require(isinstance(approval.get(key), str) and approval[key].strip(), f"approval.{key} required")


def _load_common(data: dict[str, Any], source_path: Path, require_approved: bool) -> None:
    require(data.get("schema_version") == 1, "unsupported schema_version")
    require(data.get("runtime") == "franky_joint_position_v1", "runtime must be franky_joint_position_v1")
    require(data.get("mode") in MODES, "mode must be reference or ppo")
    require(data.get("execution_context") in {"fake", "hardware"}, "execution_context must be fake or hardware")
    require(isinstance(data.get("session_id"), int) and data["session_id"] > 0, "positive session_id required")
    _approval(data, require_approved)

    robot = data.get("robot", {})
    require(robot.get("model") == "fr3v2.1", "robot.model must be fr3v2.1")
    require(robot.get("arm_revision") == "Arm3Rv2_02.01", "robot.arm_revision mismatch")
    require(robot.get("end_effector") == "none", "robot.end_effector must be none")
    require(float(robot.get("external_load_kg", math.nan)) == 0.0, "robot.external_load_kg must be zero")
    require(robot.get("require_identity_f_t_ee") is True, "identity F_T_EE must be required")
    require(isinstance(robot.get("host"), str) and robot["host"], "robot.host required")
    require(robot.get("controller_mode") == "joint_impedance",
            "robot.controller_mode must be joint_impedance")
    expected = robot.get("expected_franky_version")
    require(isinstance(expected, str) and expected, "robot.expected_franky_version required")
    collision = robot.get("constructor_collision_behavior", {})
    require(collision.get("acknowledged") is True, "Franky constructor collision behavior must be acknowledged")
    require(float(collision.get("joint_torque_threshold_nm", 0)) == 20.0,
            "Franky constructor joint threshold must match source default 20 Nm")
    require(float(collision.get("cartesian_force_threshold_n", 0)) == 30.0,
            "Franky constructor force threshold must match source default 30 N")
    dynamics = robot.get("relative_dynamics_factor", {})
    for key in ("velocity", "acceleration", "jerk"):
        value = float(dynamics.get(key, 0))
        require(0.0 < value <= 1.0, f"robot.relative_dynamics_factor.{key} must be in (0,1]")
    data["robot"] = robot

    mapping = data.get("action_mapping", {})
    mapping["default_position_rad"] = vector(mapping.get("default_position_rad"), JOINTS,
                                              "action_mapping.default_position_rad")
    scale = mapping.get("scale_rad")
    if isinstance(scale, (int, float)) and not isinstance(scale, bool):
        scale = [float(scale)] * JOINTS
    mapping["scale_rad"] = vector(scale, JOINTS, "action_mapping.scale_rad")
    require(mapping.get("clip") is None, "action clipping is unsupported; unsafe targets are rejected")
    data["action_mapping"] = mapping

    timing = data.get("timing", {})
    require(float(timing.get("command_hz", 0)) == 30.0, "timing.command_hz must be exactly 30")
    for key in ("state_timeout_ms", "callback_gap_ms", "command_watchdog_ms", "suite_timeout_s"):
        require(float(timing.get(key, 0)) > 0, f"timing.{key} must be positive")
    require(float(timing["command_watchdog_ms"]) > 1000.0 / 30.0,
            "command watchdog must exceed one 30 Hz period")
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
            q0 = source["start_position_rad"][joint]
            require(q0 - amplitude >= data["safety"]["soft_lower_rad"][joint] and
                    q0 + amplitude <= data["safety"]["soft_upper_rad"][joint],
                    f"{name} exceeds soft joint bounds")
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
        worker = resolved(source_path.parent, source.get("worker_python"), "ppo.worker_python")
        require(worker.is_file() and os.access(worker, os.X_OK), "ppo.worker_python is not executable")
        source["start_position_rad"] = vector(source.get("start_position_rad"), JOINTS, "ppo.start_position_rad")
        contract_path = bundle / "policy_contract.yaml"
        require(contract_path.is_file(), "PPO policy contract missing")
        contract = yaml.safe_load(contract_path.read_text())
        require(contract.get("robot_model") == "fr3", "PPO contract robot_model must be fr3")
        require(contract.get("joint_names") == EXPECTED_JOINTS, "PPO contract joint order/model is not FR3")
        require(contract.get("frames", {}).get("base") == "fr3_link0", "PPO base frame mismatch")
        require(contract.get("frames", {}).get("tracked_body") == "fr3_flange", "PPO tracked body mismatch")
        require(abs(float(contract.get("policy_period_s", 0)) - 1.0 / 30.0) < 1e-12,
                "PPO policy period is not 30 Hz")
        require(vector(contract.get("default_joint_position_rad"), JOINTS, "bundle default position") ==
                data["action_mapping"]["default_position_rad"], "PPO default offset mismatch")
        contract_scale = contract.get("action", {}).get("scale_rad")
        if isinstance(contract_scale, (int, float)) and not isinstance(contract_scale, bool):
            contract_scale = [float(contract_scale)] * JOINTS
        require(vector(contract_scale, JOINTS, "bundle action scale") == data["action_mapping"]["scale_rad"],
                "PPO action scale mismatch")
        require(contract.get("action", {}).get("clip") is None, "PPO action clipping unsupported")
        expressions = [item.get("expression") for item in contract.get("observation_layout", [])]
        require(expressions == ["q_measured - q_default", "dq_measured",
                                "target_base - fr3_flange_base", "previous_raw_actor_output"],
                "PPO observation layout mismatch")
        targets = source.get("targets")
        require(isinstance(targets, list) and targets, "ppo.targets must be nonempty")
        for index, target in enumerate(targets):
            require(isinstance(target.get("id"), str) and target["id"], f"ppo.targets[{index}].id required")
            target["position_base_m"] = vector(target.get("position_base_m"), 3,
                                                f"ppo.targets[{index}].position_base_m")
        require(isinstance(source.get("repetitions"), int) and source["repetitions"] > 0,
                "ppo.repetitions must be positive")
        success = source.get("success", {})
        require(float(success.get("position_threshold_m", 0)) > 0, "PPO position threshold must be positive")
        require(int(success.get("consecutive_observations", 0)) > 0,
                "PPO consecutive observations must be positive")
        require(float(success.get("max_joint_velocity_rad_s", 0)) > 0,
                "PPO settled velocity must be positive")
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


def action_to_target(config: dict[str, Any], action: list[float]) -> list[float]:
    raw = vector(list(action), JOINTS, "policy action")
    mapping = config["action_mapping"]
    return check_target(config, [mapping["default_position_rad"][i] + mapping["scale_rad"][i] * raw[i]
                                 for i in range(JOINTS)], "mapped policy target")


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


def assemble_policy_observation(q, dq, target, flange, previous, default) -> list[float]:
    q = vector(list(q), 7, "q"); dq = vector(list(dq), 7, "dq")
    target = vector(list(target), 3, "target"); flange = vector(list(flange), 3, "flange")
    previous = vector(list(previous), 7, "previous"); default = vector(list(default), 7, "default")
    return ([q[i] - default[i] for i in range(7)] + dq +
            [target[i] - flange[i] for i in range(3)] + previous)


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
