import ast
import csv
import json
import math
import time
from pathlib import Path

import pytest
import yaml

from franky_experiment.coordinator import Coordinator
from franky_experiment.core import ConfigError, ReferencePlan, action_to_target, load_experiment, sha256

HOME = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, 0.0]
LOWER = [-2.9007400166666666, -1.8360900166666667, -2.9007400166666666,
         -3.077020016666667, -2.87630335, 0.43982265, -3.05083335]
UPPER = [2.9007400166666666, 1.8360900166666667, 2.9007400166666666,
         -0.11693708333333333, 2.87630335, 4.62163335, 3.05083335]
MARGIN = [0.2900740016666667, 0.1836090016666667, 0.2900740016666667,
          0.1480041466666667, 0.287630335, 0.209090535, 0.305083335]
VERSION = "2.0.1.dev58+gf88f0e9b.libfranka.0.21.2"


def approved_reference(tmp_path, session_id=42):
    return {
        "schema_version": 1, "runtime": "franky_joint_impedance_tracking_v1",
        "execution_context": "fake", "mode": "reference", "session_id": session_id,
        "status": "approved", "executable": True, "motion_authorized": True,
        "approval": {"approved_by": "test", "approved_at": "2026-09-25T00:00:00Z",
                     "physical_stop_procedure": "fake", "workspace_review": "fake"},
        "robot": {"host": "127.0.0.1", "model": "fr3v2.1",
                  "arm_revision": "Arm3Rv2_02.01", "end_effector": "none",
                  "external_load_kg": 0.0, "require_identity_f_t_ee": True,
                  "control_interface": "torque",
                  "controller_mode": "franky_joint_impedance_tracking",
                  "expected_franky_version": VERSION,
                  "impedance_controller": {
                    "stiffness_nm_rad": [24.0, 24.0, 24.0, 24.0, 10.0, 6.0, 2.0],
                    "damping_nms_rad": [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5],
                    "compensate_coriolis": True, "constant_torque_offset_nm": [0.0] * 7,
                    "max_delta_tau_nm_per_ms": 1.0, "gains_time_constant_s": 0.1,
                    "expected_error_clip_rad": [0.5] * 7,
                    "joint_limit_activation_distance_rad": 0.1,
                    "joint_limit_stiffness_nm": 4.0, "joint_limit_damping_nms_rad": 1.0,
                    "joint_limit_max_torque_nm": 5.0,
                    "friction": {"coulomb_nm": [0.0] * 7, "viscous_nms_rad": [0.0] * 7,
                                 "max_torque_nm": [1.0] * 7, "velocity_epsilon_rad_s": 0.03}},
                  "torque_stop": {"damping_nms_rad": [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5],
                                  "ramp_duration_s": 0.2, "velocity_epsilon_rad_s": 0.02,
                                  "max_duration_s": 2.0, "compensate_coriolis": True,
                                  "max_delta_tau_nm_per_ms": 1.0},
                  "constructor_collision_behavior": {"acknowledged": True,
                    "joint_torque_threshold_nm": 20.0, "cartesian_force_threshold_n": 30.0}},
        "action_mapping": {"default_position_rad": HOME, "scale_rad": 0.5, "clip": None},
        "timing": {"command_hz": 30, "state_timeout_ms": 50.0, "callback_gap_ms": 50.0,
                   "command_watchdog_ms": 75.0, "suite_timeout_s": 2.0},
        "safety": {"reference_derivative_limit_factor": 0.2,
                   "joint_lower_rad": LOWER, "joint_upper_rad": UPPER,
                   "position_margin_rad": MARGIN, "start_tolerance_rad": [0.02] * 7,
                   "max_start_velocity_rad_s": [0.01] * 7,
                   "max_tracking_error_rad": [0.05] * 7},
        "reference": {"start_position_rad": HOME, "inter_trial_hold_s": 0.0,
                      "trials": [{"id": "j1", "joint": 1, "amplitude_rad": 0.00001,
                                  "frequency_hz": 1.0, "warmup_s": 0.02,
                                  "ramp_s": 0.02, "cycles": 0.05, "post_hold_s": 0.02}]},
        "artifacts": {"root": str(tmp_path / "runs")},
    }


def write_config(tmp_path, data, name="experiment.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_pending_hardware_template_is_not_structurally_executable():
    path = Path(__file__).parents[1] / "config" / "reference.pending.yaml"
    with pytest.raises(ConfigError):
        load_experiment(path, require_approved=False)


def test_reference_plan_reuses_quintic_sine_shape(tmp_path):
    config = load_experiment(write_config(tmp_path, approved_reference(tmp_path)))
    plan = ReferencePlan(config)
    assert plan.point(0.01).q_target == pytest.approx(HOME)
    assert not plan.point(0.05).complete
    assert plan.point(plan.duration).complete


def test_action_mapping_rejects_soft_limit_violation(tmp_path):
    config = load_experiment(write_config(tmp_path, approved_reference(tmp_path)))
    assert action_to_target(config, [0.0] * 7) == pytest.approx(HOME)
    with pytest.raises(ConfigError, match="soft joint bounds"):
        action_to_target(config, [20.0] * 7)



def test_reference_rejects_waveform_faster_than_reviewed_derivative_limits(tmp_path):
    value = approved_reference(tmp_path)
    value["reference"]["trials"][0].update({
        "amplitude_rad": math.radians(20.0), "frequency_hz": 0.25, "ramp_s": 2.0})
    with pytest.raises(ConfigError, match="derivative bound exceeds Franky dynamics"):
        load_experiment(write_config(tmp_path, value))


def test_collision_constructor_contract_is_exact(tmp_path):
    value = approved_reference(tmp_path)
    value["robot"]["constructor_collision_behavior"]["joint_torque_threshold_nm"] = 19.0
    with pytest.raises(ConfigError, match="20 Nm"):
        load_experiment(write_config(tmp_path, value))


def test_fake_reference_runtime_writes_complete_trace(tmp_path):
    config = load_experiment(write_config(tmp_path, approved_reference(tmp_path)))
    coordinator = Coordinator(config, auto_start=True)
    try:
        code = coordinator.run()
    finally:
        coordinator.close()
    assert code == 0
    run = tmp_path / "runs" / "session-42"
    final = json.loads((run / "final_metadata.json").read_text())
    assert final["terminal_state"] == "complete"
    assert final["dropped_samples"] == 0
    with (run / "samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert {row["command_valid"] for row in rows} == {"1"}
    assert max(int(row["policy_sequence"]) for row in rows) > 1



def test_constant_reference_uses_keepalive_instead_of_replanning(tmp_path):
    config = load_experiment(write_config(tmp_path, approved_reference(tmp_path, 44)))
    coordinator = Coordinator(config, auto_start=True)
    try:
        coordinator.send_target(HOME, [0.0] * 7, 1, 1, time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
        first_sequence = coordinator.sequence
        coordinator.send_target(HOME, [0.0] * 7, 1, 2, time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
        assert coordinator.sequence == first_sequence
        assert coordinator.callback_sequence == 2
    finally:
        coordinator.close()


def test_hardware_backend_has_no_recovery_call():
    source = (Path(__file__).parents[1] / "franky_experiment" / "backend.py").read_text()
    tree = ast.parse(source)
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "recover_from_errors" not in calls
    assert "automatic_error_recovery" not in calls


def approved_ppo(tmp_path, session_id=43):
    bundle = tmp_path / "fr3_bundle"
    bundle.mkdir()
    manifest = bundle / "manifest.json"
    manifest.write_text("{}\n")
    (bundle / "policy_contract.yaml").write_text(yaml.safe_dump({
        "robot_model": "fr3",
        "joint_names": [f"fr3_joint{i}" for i in range(1, 8)],
        "frames": {"base": "fr3_link0", "tracked_body": "fr3_flange"},
        "policy_period_s": 1.0 / 30.0,
        "default_joint_position_rad": HOME,
        "action": {"scale_rad": [0.5] * 7, "clip": None},
        "observation_layout": [
            {"expression": "q_measured - q_default"},
            {"expression": "dq_measured"},
            {"expression": "target_base - fr3_flange_base"},
            {"expression": "previous_raw_actor_output"},
        ],
    }, sort_keys=False))
    value = approved_reference(tmp_path, session_id)
    value["mode"] = "ppo"
    value.pop("reference")
    value["timing"].update({"inference_timeout_ms": 25.0, "trial_timeout_s": 1.0})
    value["ppo"] = {
        "bundle_path": str(bundle), "manifest_sha256": sha256(manifest),
        "worker_python": "/bin/true", "start_position_rad": HOME,
        "repetitions": 1,
        "targets": [{"id": "test_target", "position_base_m": [0.50, 0.0, 0.35]}],
        "success": {"position_threshold_m": 0.001, "consecutive_observations": 3,
                    "max_joint_velocity_rad_s": 0.01},
    }
    return value


class FakePolicyWorker:
    instances = []

    def __init__(self, python, bundle, digest):
        del python, bundle
        self.ready = {"manifest_sha256": digest, "provider": "fake"}
        self.inflight = None
        self.sent_ns = 0
        self.pending = None
        self.requests = []
        self.closed = False
        self.__class__.instances.append(self)

    def submit(self, request):
        self.requests.append(request)
        self.pending = request
        self.inflight = request["request_id"]
        self.sent_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

    def poll(self):
        if self.pending is None:
            return None
        request, self.pending = self.pending, None
        self.inflight = None
        return {"type": "result", "request_id": request["request_id"],
                "trial_id": request["trial_id"],
                "observation_sequence": request["observation_sequence"],
                "completed_monotonic_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                "action": [0.0] * 7}

    def close(self):
        self.closed = True


def test_fake_ppo_runtime_uses_24d_observation_and_worker(tmp_path):
    FakePolicyWorker.instances.clear()
    config = load_experiment(write_config(tmp_path, approved_ppo(tmp_path), "ppo.yaml"))
    coordinator = Coordinator(config, auto_start=True, worker_factory=FakePolicyWorker)
    try:
        code = coordinator.run(max_runtime_s=0.16)
    finally:
        coordinator.close()
    assert code == 0
    worker = FakePolicyWorker.instances[-1]
    assert worker.closed
    assert len(worker.requests) >= 2
    assert all(len(request["observation"]) == 24 for request in worker.requests)
    final = json.loads((tmp_path / "runs" / "session-43" / "final_metadata.json").read_text())
    assert final["terminal_state"] == "stopped"
    assert final["terminal_reason"] == "max_runtime"
    assert final["last_policy_sequence"] >= 1


def test_hardware_backend_uses_one_tracking_motion_not_joint_motion_preemption():
    source = (Path(__file__).parents[1] / "franky_experiment" / "backend.py").read_text()
    assert "JointImpedanceTrackingMotion" in source
    assert "JointReference" in source
    assert "JointMotion(" not in source
    assert "TorqueStopMotion" in source
