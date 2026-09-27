import ast
import csv
import json
import math
import time
from types import SimpleNamespace

import numpy as np
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
                  "ee_feedback": {
                      "source": "measured_joint_encoder_libfranka_fk",
                      "joint_signal": "q",
                      "base_frame": "fr3_link0",
                      "frame": "fr3_flange",
                      "model_frame": "Frame.Flange",
                      "reported_pose_role": "audit_only",
                  },
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
    assert {row["ee_feedback_source"] for row in rows} == {"fake_backend_joint_state_fk"}
    assert all(float(row["ee_feedback_x_m"]) == float(row["flange_x_m"]) for row in rows)
    assert all(float(row["ee_feedback_y_m"]) == float(row["flange_y_m"]) for row in rows)
    assert all(float(row["ee_feedback_z_m"]) == float(row["flange_z_m"]) for row in rows)
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



def test_encoder_forward_kinematics_uses_measured_q_and_flange_frame():
    from franky_experiment.backend import EncoderForwardKinematics, _encoder_pose_feedback

    flange_frame = object()
    identity = object()
    calls = []

    class Pose:
        matrix = np.array([
            [1.0, 0.0, 0.0, 0.31],
            [0.0, 1.0, 0.0, -0.02],
            [0.0, 0.0, 1.0, 0.59],
            [0.0, 0.0, 0.0, 1.0],
        ])
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])

    class Model:
        def pose(self, frame, q, f_t_ee, ee_t_k):
            calls.append((frame, np.asarray(q).copy(), f_t_ee, ee_t_k))
            return Pose()

    module = SimpleNamespace(
        Frame=SimpleNamespace(Flange=flange_frame),
        Affine=lambda: identity,
    )
    fk = EncoderForwardKinematics(SimpleNamespace(model=Model()), module)
    measured_q = np.array([0.1, -0.2, 0.3, -1.0, 0.5, 1.2, -0.4])
    reported = np.eye(4)
    reported[:3, 3] = [9.0, 8.0, 7.0]
    state = SimpleNamespace(q=measured_q, O_T_EE=reported, F_T_EE=np.eye(4))

    feedback = _encoder_pose_feedback(state, fk)

    assert len(calls) == 1
    assert calls[0][0] is flange_frame
    assert calls[0][1] == pytest.approx(measured_q)
    assert calls[0][2] is identity and calls[0][3] is identity
    assert feedback.position_m == pytest.approx((0.31, -0.02, 0.59))
    assert feedback.orientation_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert feedback.reported_position_m == pytest.approx((9.0, 8.0, 7.0))
    assert feedback.reported_position_error_m > 1.0


def test_hardware_backend_has_no_recovery_call():
    source = (Path(__file__).parents[1] / "franky_experiment" / "backend.py").read_text()
    tree = ast.parse(source)
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "recover_from_errors" not in calls
    assert "automatic_error_recovery" not in calls


def approved_ppo(tmp_path, session_id=43):
    value = approved_reference(tmp_path, session_id)
    value["robot"]["impedance_controller"]["stiffness_nm_rad"] = [100.0] * 7
    value["robot"]["impedance_controller"]["damping_nms_rad"] = [20.0] * 7
    soft_lower = [
        lower + margin
        for lower, margin in zip(
            value["safety"]["joint_lower_rad"], value["safety"]["position_margin_rad"]
        )
    ]
    soft_upper = [
        upper - margin
        for upper, margin in zip(
            value["safety"]["joint_upper_rad"], value["safety"]["position_margin_rad"]
        )
    ]
    bundle = tmp_path / f"fr3_bundle-{session_id}"
    bundle.mkdir()
    manifest = bundle / "manifest.json"
    manifest.write_text("{}\n")
    (bundle / "policy_contract.yaml").write_text(yaml.safe_dump({
        "schema_version": 2,
        "contract_id": "fr3_incremental_position_29d_v1",
        "robot_model": "fr3",
        "joint_names": [f"fr3_joint{i}" for i in range(1, 8)],
        "frames": {"base": "fr3_link0", "tracked_body": "fr3_flange"},
        "policy_period_s": 1.0 / 50.0,
        "action": {
            "type": "normalized_position_increment",
            "size": 6,
            "inference": "deterministic_tanh",
            "controlled_joint_names": [f"fr3_joint{i}" for i in range(1, 7)],
            "held_joint_names": ["fr3_joint7"],
            "integration": "forward_euler",
            "initial_reference": "measured_start_position",
            "soft_limit_projection": True,
            "soft_lower_rad": soft_lower,
            "soft_upper_rad": soft_upper,
            "max_reference_velocity_rad_s": [0.435] * 4 + [0.522] * 2,
            "max_position_increment_rad": [0.0087] * 4 + [0.01044] * 2,
        },
        "default_joint_position_rad": HOME,
        "observation": {"size": 29},
        "controller": {
            "nominal_stiffness_nm_rad": 100.0,
            "nominal_damping_nms_rad": 20.0,
            "position_error_clip_rad": 0.5,
            "torque_slew_rate_nm_s": 1000.0,
            "coriolis_compensation": True,
        },
        "observation_layout": [
            {"expression": "q_measured - q_default"},
            {"expression": "dq_measured"},
            {"expression": "target_base - fr3_flange_base"},
            {"expression": "q_reference_normalized"},
            {"expression": "previous_normalized_position_increment"},
        ],
    }, sort_keys=False))
    value["mode"] = "ppo"
    value.pop("reference")
    value["timing"].update({"command_hz": 50, "callback_gap_ms": 30.0,
                            "command_watchdog_ms": 40.0,
                            "inference_timeout_ms": 15.0, "trial_timeout_s": 1.0})
    value["action_mapping"] = {
        "type": "normalized_position_increment",
        "policy_joint_count": 6,
        "controlled_joint_names": [f"fr3_joint{i}" for i in range(1, 7)],
        "held_joint_names": ["fr3_joint7"],
        "normalized_lower": -1.0,
        "normalized_upper": 1.0,
        "integration": "forward_euler",
        "initial_reference": "measured_start_position",
        "max_reference_velocity_rad_s": [0.435] * 4 + [0.522] * 2,
        "max_position_increment_rad": [0.0087] * 4 + [0.01044] * 2,
        "soft_limit_projection": True,
    }
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
                "observation_monotonic_ns": request["observation_monotonic_ns"],
                "completed_monotonic_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                "action": [0.0] * 6}

    def close(self):
        self.closed = True


def test_worker_python_virtualenv_symlink_is_preserved(tmp_path):
    value = approved_ppo(tmp_path)
    worker_target = tmp_path / "worker-target"
    worker_target.write_text("#!/bin/sh\nexit 0\n")
    worker_target.chmod(0o755)
    worker_link = tmp_path / "worker-python"
    worker_link.symlink_to(worker_target)
    value["ppo"]["worker_python"] = str(worker_link)

    config = load_experiment(write_config(tmp_path, value, "worker-link.yaml"))

    assert config["ppo"]["worker_python"] == str(worker_link.absolute())
    assert Path(config["ppo"]["worker_python"]).is_symlink()


def test_fake_ppo_runtime_uses_29d_observation_and_worker(tmp_path):
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
    assert all(len(request["observation"]) == 29 for request in worker.requests)
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


def test_incremental_action_mapping_integrates_and_projects(tmp_path):
    config = load_experiment(write_config(tmp_path, approved_ppo(tmp_path), "incremental.yaml"))
    target = action_to_target(config, [1.0] * 6, HOME)
    expected = [HOME[i] + ([0.0087] * 4 + [0.01044] * 2)[i] for i in range(6)] + [HOME[6]]
    assert target == pytest.approx(expected)
    projected = action_to_target(config, [1.0] * 6, config["safety"]["soft_upper_rad"])
    assert projected == pytest.approx(config["safety"]["soft_upper_rad"])
    with pytest.raises(ConfigError, match=r"exceeds \[-1, 1\]"):
        action_to_target(config, [1.01] * 6, HOME)


def path_ppo(tmp_path, session_id=45, context="fake"):
    value = approved_ppo(tmp_path, session_id)
    catalog = tmp_path / f"path-catalog-{session_id}.yaml"
    catalog.write_text(yaml.safe_dump({
        "version": 1,
        "paths": {
            "stationary_three": {
                "type": "waypoints",
                "description": "test path",
                "waypoints_m": [[0.45, 0.0, 0.35]] * 3,
                "target_orientation_xyzw": [0.0, 1.0, 0.0, 0.0],
                "waypoint_timeout_s": 0.04,
                "position_threshold_m": 0.001,
            }
        },
    }, sort_keys=False))
    value["ppo"].pop("targets")
    value["ppo"].pop("success")
    value["ppo"].pop("repetitions")
    value["ppo"]["path"] = {
        "catalog_path": str(catalog),
        "catalog_sha256": sha256(catalog),
        "name": "stationary_three",
        "repetitions": 1,
        "workspace_lower_base_m": [0.40, -0.10, 0.30],
        "workspace_upper_base_m": [0.50, 0.10, 0.40],
    }
    value["execution_context"] = context
    value["motion_authorized"] = context != "shadow"
    return value, catalog


def test_transferred_yz_catalog_geometry_and_hash():
    from franky_experiment.path_runtime import PathCatalog

    path = Path(__file__).parents[2] / "path_catalogs" / "yz_circle_candidates_v1.yaml"
    assert sha256(path) == "b78b6fa5bbc788f471464c48edefee3d29c5ea0f4cab06c5a7f6d65cae067cae"
    catalog = PathCatalog.from_yaml(path)
    spec = catalog.get("circle_yz_r050_t1")
    assert spec.waypoints_m[0] == pytest.approx((0.475, 0.0, 0.5))
    assert all(point[0] == pytest.approx(0.475) for point in spec.waypoints_m)
    assert all(math.hypot(point[1], point[2] - 0.45) == pytest.approx(0.05)
               for point in spec.waypoints_m)
    assert len(spec.waypoints_m) == 24


def test_path_config_requires_exclusive_source_hash_and_workspace(tmp_path):
    value, _ = path_ppo(tmp_path)
    config = load_experiment(write_config(tmp_path, value, "path.yaml"))
    assert config["ppo"]["execution_mode"] == "path"
    assert config["ppo"]["path"]["resolved_path"]["waypoint_count"] == 3

    both, _ = path_ppo(tmp_path, 46)
    both["ppo"]["targets"] = [{"id": "forbidden", "position_base_m": [0.45, 0.0, 0.35]}]
    with pytest.raises(ConfigError, match="exactly one"):
        load_experiment(write_config(tmp_path, both, "both.yaml"))

    bad_hash, _ = path_ppo(tmp_path, 47)
    bad_hash["ppo"]["path"]["catalog_sha256"] = "0" * 64
    with pytest.raises(ConfigError, match="trust anchor"):
        load_experiment(write_config(tmp_path, bad_hash, "bad-hash.yaml"))

    bad_workspace, _ = path_ppo(tmp_path, 48)
    bad_workspace["ppo"]["path"]["workspace_upper_base_m"][0] = 0.44
    with pytest.raises(ConfigError, match="reviewed workspace"):
        load_experiment(write_config(tmp_path, bad_workspace, "bad-workspace.yaml"))


def test_fake_path_runtime_completes_and_records_waypoints(tmp_path):
    FakePolicyWorker.instances.clear()
    value, _ = path_ppo(tmp_path, 49)
    config = load_experiment(write_config(tmp_path, value, "path-run.yaml"))
    coordinator = Coordinator(config, auto_start=True, worker_factory=FakePolicyWorker)
    try:
        code = coordinator.run()
    finally:
        coordinator.close()

    assert code == 0
    run = tmp_path / "runs" / "session-49"
    final = json.loads((run / "final_metadata.json").read_text())
    assert final["terminal_state"] == "complete"
    assert final["path_summary"]["reached"] == 3
    assert final["path_summary"]["timed_out"] == 0
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "waypoint_resolved" for event in events) == 3
    assert any(event["event"] == "policy_result_discarded" for event in events)
    observation_events = [event for event in events
                          if event["event"] == "policy_observation_submitted"]
    assert observation_events
    assert all(len(event["observation"]) == 29 for event in observation_events)
    requests = FakePolicyWorker.instances[-1].requests
    assert requests
    assert all(request["trial_id"] in {1, 2, 3} for request in requests)


def test_shadow_is_read_only_and_records_state_to_reference_delay(tmp_path, monkeypatch):
    import io
    import franky_experiment.coordinator as coordinator_module
    from franky_experiment.backend import FakeBackend

    class ReadOnlyBackend(FakeBackend):
        send_attempts = 0

        def send_target(self, target, callback):
            del target, callback
            self.send_attempts += 1
            raise AssertionError("shadow attempted motion")

        def keepalive(self):
            raise AssertionError("shadow attempted keepalive")

    value, catalog = path_ppo(tmp_path, 50, context="shadow")
    document = yaml.safe_load(catalog.read_text())
    document["paths"]["stationary_three"]["waypoints_m"] = [[0.46, 0.0, 0.35]] * 3
    document["paths"]["stationary_three"]["waypoint_timeout_s"] = 0.08
    catalog.write_text(yaml.safe_dump(document, sort_keys=False))
    value["ppo"]["path"]["catalog_sha256"] = sha256(catalog)
    config = load_experiment(write_config(tmp_path, value, "shadow.yaml"))
    monkeypatch.setattr(coordinator_module, "ShadowBackend", ReadOnlyBackend)
    monkeypatch.setattr(coordinator_module.sys, "stdin", io.StringIO("start\n"))
    coordinator = coordinator_module.Coordinator(
        config, auto_start=False, worker_factory=FakePolicyWorker)
    backend = coordinator.backend
    try:
        code = coordinator.run()
    finally:
        coordinator.close()

    assert code == 0
    assert backend.send_attempts == 0
    events_path = tmp_path / "runs" / "session-50" / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    updates = [event for event in events if event["event"] == "shadow_reference_updated"]
    assert updates
    assert all(event["state_to_reference_consumption_ns"] >=
               event["inference_latency_ns"] >= 0 for event in updates)
    assert all(event["effective_delay_policy_steps"] >= 0 for event in updates)

    from summarize_shadow import summarize
    report = summarize(tmp_path / "runs" / "session-50")
    assert report["observation_size"] == 29
    assert report["action_size"] == 6
    assert report["initial_reference_max_abs_error_rad"] == 0.0
    assert report["initial_previous_action_max_abs"] == 0.0
    assert report["held_joint7_max_drift_rad"] == 0.0
    assert report["consumed_result_count"] == len(updates)


def test_shadow_approval_requires_motion_authorized_false(tmp_path):
    value, _ = path_ppo(tmp_path, 51, context="shadow")
    value["motion_authorized"] = True
    with pytest.raises(ConfigError, match="motion_authorized false"):
        load_experiment(write_config(tmp_path, value, "bad-shadow.yaml"))


def test_ee_feedback_contract_rejects_reported_cartesian_pose(tmp_path):
    value = approved_ppo(tmp_path, 52)
    value["robot"]["ee_feedback"]["source"] = "robot_reported_O_T_EE"
    with pytest.raises(ConfigError, match="measured joint encoder"):
        load_experiment(write_config(tmp_path, value, "reported-pose.yaml"))
