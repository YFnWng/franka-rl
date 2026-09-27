#!/usr/bin/env python3
"""Direct Franky coordinator for reference-identification and PPO experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import queue
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .backend import (CallbackRecord, FakeBackend, FrankyBackend, FrankyVelocityBackend,
                      FrankyVelocityImpedanceBackend, ShadowBackend, Snapshot)
from .core import (ConfigError, ReferencePlan, RUNTIME_VELOCITY,
                   RUNTIME_VELOCITY_IMPEDANCE, SessionSupervisor, TERMINAL,
                   action_to_target, action_to_velocity, assemble_policy_observation, load_experiment)
from .path_runtime import PathExecution, PathSpec


def raw_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def runtime_source_sha256() -> str:
    digest = hashlib.sha256()
    package = Path(__file__).parent
    for name in ("backend.py", "coordinator.py", "core.py", "path_runtime.py", "policy_worker.py"):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((package / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class WorkerBridge:
    """Reused newline-delimited subprocess boundary from the ROS coordinator."""
    def __init__(self, python: str, bundle: str, digest: str):
        script = str(Path(__file__).with_name("policy_worker.py"))
        self.process = subprocess.Popen(
            [python, "-I", script, "--bundle", bundle, "--manifest-sha256", digest],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        readable, _, _ = select.select([self.process.stdout], [], [], 60.0)
        if not readable:
            self.process.terminate()
            raise RuntimeError("policy worker startup timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"policy worker startup failed: {self.process.stderr.read().strip()}")
        ready = json.loads(line)
        if ready.get("type") != "ready":
            raise RuntimeError(f"policy worker failed to initialize: {ready}")
        self.ready = ready
        self.results: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self.inflight: int | None = None
        self.sent_ns = 0
        self.thread = threading.Thread(target=self._reader, name="policy-worker-reader", daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        for line in self.process.stdout:
            item = json.loads(line)
            try:
                self.results.put_nowait(item)
            except queue.Full:
                return

    def submit(self, request: dict[str, Any]) -> None:
        if self.inflight is not None:
            raise RuntimeError("policy worker already has an in-flight request")
        if self.process.poll() is not None:
            raise RuntimeError("policy worker is not running")
        self.process.stdin.write(json.dumps(request, allow_nan=False, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        self.inflight, self.sent_ns = int(request["request_id"]), raw_ns()

    def poll(self) -> dict[str, Any] | None:
        try:
            item = self.results.get_nowait()
        except queue.Empty:
            return None
        if item.get("type") != "result" or item.get("request_id") != self.inflight:
            raise RuntimeError(f"policy worker protocol error: {item}")
        self.inflight = None
        return item

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


@dataclass
class CommandMeta:
    policy_sequence: int
    observation_sequence: int
    trial_id: int
    raw_action: tuple[float, ...]
    mapped_target: tuple[float, ...]
    observation_monotonic_ns: int
    policy_completed_ns: int
    command_consumed_ns: int
    command_written_ns: int


class SampleWriter:
    VECTOR_FIELDS = (
        "raw_action", "mapped_target", "q_ref", "dq_ref", "ddq_ref", "q_d", "dq_d", "ddq_d",
        "q", "dq", "tau_command", "tau_J_d", "tau_J", "tau_ext_hat_filtered",
    )

    def __init__(self, path: Path, session_id: int):
        self.path, self.session_id = path, session_id
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=8192)
        self.dropped = 0
        self.error = ""
        self._drop_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="franky-sample-writer", daemon=True)
        self.thread.start()

    @classmethod
    def fieldnames(cls) -> list[str]:
        result = [
            "session_id", "trial_id", "rt_sequence", "policy_sequence", "state_sequence",
            "observation_sequence", "robot_time_s", "host_update_entry_ns", "host_state_received_ns",
            "host_policy_complete_ns", "host_command_consumed_ns", "host_command_written_ns", "period_ns",
        ]
        for name in cls.VECTOR_FIELDS:
            result.extend(f"{name}_j{i}" for i in range(1, 8))
        result.extend([
            "flange_x_m", "flange_y_m", "flange_z_m",
            "ee_feedback_x_m", "ee_feedback_y_m", "ee_feedback_z_m",
            "ee_feedback_qx", "ee_feedback_qy", "ee_feedback_qz", "ee_feedback_qw",
            "ee_feedback_source", "ee_feedback_base_frame", "ee_feedback_frame",
            "reported_flange_x_m", "reported_flange_y_m", "reported_flange_z_m",
            "encoder_fk_reported_position_error_m",
            "encoder_fk_reported_orientation_error_rad", "encoder_fk_compute_ns",
            "robot_mode", "control_command_success_rate", "current_errors",
            "last_motion_errors", "command_valid", "ring_fill", "dropped_samples",
        ])
        return result

    def put(self, row: dict[str, Any]) -> None:
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            with self._drop_lock:
                self.dropped += 1

    def _run(self) -> None:
        try:
            with self.path.open("x", newline="", buffering=1) as stream:
                writer = csv.DictWriter(stream, fieldnames=self.fieldnames())
                writer.writeheader()
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    item["ring_fill"] = self.queue.qsize()
                    item["dropped_samples"] = self.dropped
                    writer.writerow(item)
        except Exception as exc:  # captured and promoted to a terminal fault by the coordinator
            self.error = repr(exc)

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join(timeout=10)
        if self.thread.is_alive() and not self.error:
            self.error = "writer thread did not terminate"


class Coordinator:
    def __init__(self, config: dict[str, Any], auto_start: bool = False,
                 worker_factory=WorkerBridge):
        self.config = config
        self.runtime_source_sha256 = runtime_source_sha256()
        self.supervisor = SessionSupervisor(config)
        self.auto_start = auto_start
        self.sequence = 0
        self.state_sequence = 0
        self.callback_sequence = 0
        self.request_sequence = 0
        self.started_ns = 0
        self.trial_started_ns = 0
        self.last_callback_ns = 0
        self.last_callback_robot_time_s = 0.0
        self.first_command_ns = 0
        self.callback_started = False
        self.active_trial_id = 0
        self.latest: Snapshot | None = None
        action_size = int(config.get("action_mapping", {}).get("policy_joint_count", 7))
        self.previous_action = [0.0] * action_size
        self.ppo_reference: list[float] | None = None
        self.current_target_index = 0
        self.success_count = 0
        self.plan = ReferencePlan(config) if config["mode"] == "reference" else None
        ppo = config.get("ppo", {})
        self.targets = ([target for _ in range(ppo.get("repetitions", 1))
                         for target in ppo.get("targets", [])])
        self.path_execution: PathExecution | None = None
        if config["mode"] == "ppo" and ppo.get("execution_mode") == "path":
            selection = ppo["path"]
            self.path_execution = PathExecution(
                PathSpec(**selection["resolved_path"]),
                int(selection["repetitions"]),
                float(config["timing"]["command_hz"]),
            )
        self.worker: WorkerBridge | None = None
        self.command_lock = threading.Lock()
        self.current_meta: CommandMeta | None = None
        self.last_submitted_target: tuple[float, ...] | None = None
        self.previous_q_ref: tuple[float, ...] | None = None
        self.previous_dq_ref = (0.0,) * 7
        # Set by the 1 kHz callback from a time-coherent measured/reference
        # pair, then consumed by the 50 Hz coordinator thread. RobotState.q_d
        # is the motion-generator desired state and remains at the control-loop
        # start pose during this external torque-control mode; it is therefore
        # not the impedance tracker's position reference.
        self.tracking_violation: dict[str, Any] | None = None

        root = Path(config["artifacts"]["root"])
        self.run_dir = root / f"session-{config['session_id']}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.event_stream = (self.run_dir / "events.jsonl").open("x", buffering=1)
        resolved = {k: v for k, v in config.items() if not k.startswith("_")}
        (self.run_dir / "resolved_config.json").write_text(
            json.dumps(resolved, indent=2, allow_nan=False) + "\n")
        (self.run_dir / "config.sha256").write_text(config["_config_sha256"] + "\n")
        self.writer = SampleWriter(self.run_dir / "samples.csv", config["session_id"])
        self.backend = None
        try:
            context = config["execution_context"]
            self.backend = (FakeBackend(config) if context == "fake" else
                            ShadowBackend(config) if context == "shadow" else
                            FrankyVelocityBackend(config) if config["runtime"] == RUNTIME_VELOCITY else
                            FrankyVelocityImpedanceBackend(config)
                            if config["runtime"] == RUNTIME_VELOCITY_IMPEDANCE else
                            FrankyBackend(config))
            self.event("initialized", mode=config["mode"], execution_context=config["execution_context"],
                       config_sha256=config["_config_sha256"],
                       runtime_source_sha256=self.runtime_source_sha256)
            if config["mode"] == "ppo":
                ppo = config["ppo"]
                self.worker = worker_factory(
                    ppo["worker_python"], ppo["bundle_path"], ppo["manifest_sha256"])
                self.event("policy_worker_ready", **self.worker.ready)
        except Exception as exc:
            self.event("initialization_failed", error=repr(exc))
            if self.worker is not None:
                self.worker.close()
            if self.backend is not None:
                self.backend.close()
            self.writer.close()
            self.event_stream.close()
            raise

    def event(self, kind: str, **fields: Any) -> None:
        row = {"event": kind, "host_monotonic_ns": raw_ns(),
               "session_id": self.config["session_id"], **fields}
        self.event_stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        self.event_stream.flush()

    def fault(self, reason: str) -> None:
        prior = self.supervisor.state
        self.supervisor.fault(reason)
        if prior != self.supervisor.state:
            self.event("terminal_fault", reason=reason)

    def _read_state(self) -> Snapshot:
        started = raw_ns()
        snapshot = self.backend.read_state()
        elapsed = raw_ns() - started
        if elapsed > int(float(self.config["timing"]["state_timeout_ms"]) * 1e6):
            raise RuntimeError("state_read_timeout")
        return snapshot

    def _validate_inventory(self, snapshot: Snapshot) -> None:
        identity = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        if any(abs(snapshot.f_t_ee[r][c] - identity[r][c]) > 1e-9 for r in range(4) for c in range(4)):
            raise RuntimeError("live F_T_EE is not identity")
        if abs(snapshot.load_mass_kg) > 1e-9:
            raise RuntimeError(f"live external load is {snapshot.load_mass_kg}, expected zero")

    def _on_callback(self, record: CallbackRecord, meta: CommandMeta) -> None:
        self.callback_sequence += 1
        self.last_callback_ns = record.host_monotonic_ns
        self.last_callback_robot_time_s = record.robot_time_s
        self.callback_started = True
        dq_ref = record.dq_command
        ddq_ref = (0.0,) * 7
        self.previous_q_ref, self.previous_dq_ref = record.q_command, dq_ref
        tracking_limit = self.config["safety"]["max_tracking_error_rad"]
        tracking_error = tuple(abs(record.q[i] - record.q_command[i]) for i in range(7))
        if self.supervisor.state == "running" and self.tracking_violation is None:
            violating = [i for i in range(7) if tracking_error[i] > tracking_limit[i]]
            if violating:
                joint = max(violating, key=lambda i: tracking_error[i] / tracking_limit[i])
                self.tracking_violation = {
                    "joint": joint + 1,
                    "measured_rad": record.q[joint],
                    "reference_rad": record.q_command[joint],
                    "absolute_error_rad": tracking_error[joint],
                    "limit_rad": tracking_limit[joint],
                    "callback_sequence": self.callback_sequence,
                    "robot_time_s": record.robot_time_s,
                }
        row: dict[str, Any] = {
            "session_id": self.config["session_id"], "trial_id": meta.trial_id,
            "rt_sequence": self.callback_sequence, "policy_sequence": meta.policy_sequence,
            "state_sequence": self.state_sequence, "observation_sequence": meta.observation_sequence,
            "robot_time_s": record.robot_time_s, "host_update_entry_ns": record.host_monotonic_ns,
            "host_state_received_ns": record.host_monotonic_ns,
            "host_policy_complete_ns": meta.policy_completed_ns,
            "host_command_consumed_ns": meta.command_consumed_ns,
            "host_command_written_ns": meta.command_written_ns,
            "period_ns": int(round(record.time_step_s * 1e9)),
            # Legacy flange columns and explicit EE-feedback columns both carry
            # measured-joint FK for the contract's physical fr3_flange frame.
            "flange_x_m": record.flange_position_m[0],
            "flange_y_m": record.flange_position_m[1],
            "flange_z_m": record.flange_position_m[2],
            "ee_feedback_x_m": record.flange_position_m[0],
            "ee_feedback_y_m": record.flange_position_m[1],
            "ee_feedback_z_m": record.flange_position_m[2],
            "ee_feedback_qx": record.flange_orientation_xyzw[0],
            "ee_feedback_qy": record.flange_orientation_xyzw[1],
            "ee_feedback_qz": record.flange_orientation_xyzw[2],
            "ee_feedback_qw": record.flange_orientation_xyzw[3],
            "ee_feedback_source": record.ee_feedback_source,
            "ee_feedback_base_frame": record.ee_feedback_base_frame,
            "ee_feedback_frame": record.ee_feedback_frame,
            "reported_flange_x_m": record.reported_flange_position_m[0],
            "reported_flange_y_m": record.reported_flange_position_m[1],
            "reported_flange_z_m": record.reported_flange_position_m[2],
            "encoder_fk_reported_position_error_m":
                record.encoder_fk_reported_position_error_m,
            "encoder_fk_reported_orientation_error_rad":
                record.encoder_fk_reported_orientation_error_rad,
            "encoder_fk_compute_ns": record.encoder_fk_compute_ns,
            "robot_mode": record.mode,
            "control_command_success_rate": record.command_success_rate,
            "current_errors": record.current_errors, "last_motion_errors": record.last_motion_errors,
            "command_valid": 1,
        }
        values = {
            "raw_action": meta.raw_action, "mapped_target": meta.mapped_target,
            "q_ref": record.q_command, "dq_ref": dq_ref, "ddq_ref": ddq_ref,
            "q_d": record.q_d, "dq_d": record.dq_d, "ddq_d": record.ddq_d,
            "q": record.q, "dq": record.dq, "tau_command": record.tau_command,
            "tau_J_d": record.tau_joint_desired,
            "tau_J": record.tau_joint, "tau_ext_hat_filtered": record.tau_external,
        }
        for name, vector in values.items():
            padded = tuple(vector) + (0.0,) * (7 - len(vector))
            row.update({f"{name}_j{i + 1}": padded[i] for i in range(7)})
        self.writer.put(row)

    def send_target(self, target: list[float], raw_action: list[float], trial_id: int,
                    observation_sequence: int, completed_ns: int,
                    observation_ns: int | None = None) -> None:
        from .core import check_target
        q_target = check_target(self.config, target)
        q_target_tuple = tuple(q_target)
        if (self.config["runtime"] != RUNTIME_VELOCITY and
                self.last_submitted_target == q_target_tuple and
                self.config["execution_context"] != "shadow"):
            self.backend.keepalive()
            self.event("target_held", trial_id=trial_id, policy_sequence=self.sequence,
                       observation_sequence=observation_sequence, target=q_target)
            return
        self.sequence += 1
        consumed = raw_ns()
        # Record the call boundary before entering Franky. Its callback executor may
        # deliver a sample before Robot.move() returns, including in test backends.
        # This is submission time rather than API-completion time.
        submitted = raw_ns()
        if self.first_command_ns == 0:
            self.first_command_ns = submitted
        observed = completed_ns if observation_ns is None else observation_ns
        meta = CommandMeta(self.sequence, observation_sequence, trial_id, tuple(raw_action),
                           tuple(q_target), observed, completed_ns, consumed, submitted)

        if self.config["execution_context"] == "shadow":
            self.current_meta = meta
            self.last_submitted_target = q_target_tuple
            self.event("shadow_reference_updated", trial_id=trial_id,
                       policy_sequence=self.sequence,
                       observation_sequence=observation_sequence,
                       observation_monotonic_ns=observed,
                       policy_completed_ns=completed_ns,
                       command_consumed_ns=consumed,
                       inference_latency_ns=max(0, completed_ns - observed),
                       completion_to_consumption_ns=max(0, consumed - completed_ns),
                       state_to_reference_consumption_ns=max(0, consumed - observed),
                       effective_delay_policy_steps=(max(0, consumed - observed) /
                           (1e9 / float(self.config["timing"]["command_hz"]))),
                       raw_action=raw_action, target=q_target)
            return

        def callback(record: CallbackRecord, command_meta: CommandMeta = meta) -> None:
            self._on_callback(record, command_meta)

        with self.command_lock:
            self.current_meta = meta
            self.backend.send_target(q_target, callback)
            self.last_submitted_target = q_target_tuple
        returned = raw_ns()
        event_name = ("velocity_target_updated" if self.config["runtime"] == RUNTIME_VELOCITY
                      else "impedance_reference_updated")
        self.event(event_name, trial_id=trial_id, policy_sequence=self.sequence,
                   observation_sequence=observation_sequence, command_consumed_ns=consumed,
                   command_submitted_ns=submitted, command_api_returned_ns=returned,
                   target=q_target)

    def send_velocity(self, velocity: list[float], raw_action: list[float], trial_id: int,
                      observation_sequence: int, completed_ns: int,
                      observation_ns: int | None = None) -> None:
        if self.config["runtime"] != RUNTIME_VELOCITY_IMPEDANCE:
            raise RuntimeError("velocity command used outside velocity-impedance runtime")
        if len(velocity) != 7 or not all(math.isfinite(value) for value in velocity):
            raise RuntimeError("velocity command must contain seven finite values")
        limits = self.config["action_mapping"]["max_reference_velocity_rad_s"]
        if any(abs(velocity[i]) > limits[i] + 1.0e-12 for i in range(len(limits))):
            raise RuntimeError("velocity command exceeds policy contract")
        if any(abs(value) > 1.0e-12 for value in velocity[len(limits):]):
            raise RuntimeError("held joint received a nonzero velocity command")
        self.sequence += 1
        consumed = raw_ns()
        submitted = raw_ns()
        if self.first_command_ns == 0:
            self.first_command_ns = submitted
        observed = completed_ns if observation_ns is None else observation_ns
        reference = tuple(self.ppo_reference or self.config["ppo"]["start_position_rad"])
        meta = CommandMeta(self.sequence, observation_sequence, trial_id, tuple(raw_action),
                           reference, observed, completed_ns, consumed, submitted)
        if self.config["execution_context"] == "shadow":
            dt = 1.0 / float(self.config["timing"]["command_hz"])
            lower, upper = self.config["safety"]["soft_lower_rad"], self.config["safety"]["soft_upper_rad"]
            self.ppo_reference = [min(upper[i], max(lower[i], reference[i] + velocity[i] * dt))
                                  for i in range(7)]
            self.event("shadow_velocity_updated", trial_id=trial_id, policy_sequence=self.sequence,
                       observation_sequence=observation_sequence, velocity=velocity,
                       reference=self.ppo_reference)
            return

        def callback(record: CallbackRecord, command_meta: CommandMeta = meta) -> None:
            self._on_callback(record, command_meta)

        with self.command_lock:
            self.current_meta = meta
            self.backend.send_velocity(velocity, callback)
        self.event("velocity_impedance_target_updated", trial_id=trial_id,
                   policy_sequence=self.sequence, observation_sequence=observation_sequence,
                   command_consumed_ns=consumed, command_submitted_ns=submitted,
                   velocity=velocity)

    def _reference_tick(self, now: int) -> None:
        point = self.plan.point((now - self.started_ns) / 1e9)
        if point.complete:
            self.supervisor.complete()
            self.event("session_complete")
            return
        if point.trial_id != self.active_trial_id:
            self.active_trial_id = point.trial_id
            self.event("trial_start", trial_id=point.trial_id, trial_name=point.trial_name)
        default, scale = self.config["action_mapping"]["default_position_rad"], self.config["action_mapping"]["scale_rad"]
        raw_action = [(point.q_target[i] - default[i]) / scale[i] for i in range(7)]
        self.send_target(list(point.q_target), raw_action, point.trial_id, self.state_sequence, now)

    def _active_ppo_target(self) -> list[float] | None:
        if self.path_execution is not None:
            return None if self.path_execution.complete else list(self.path_execution.target)
        if self.current_target_index >= len(self.targets):
            return None
        return list(self.targets[self.current_target_index]["position_base_m"])

    def _active_ppo_trial_id(self) -> int:
        return self.path_execution.trial_id if self.path_execution is not None else self.current_target_index + 1

    def _ppo_success(self, snapshot: Snapshot, now: int) -> None:
        if self.path_execution is not None:
            result = self.path_execution.update(snapshot.flange_position_m)
            if result is None:
                return
            self.event("waypoint_resolved", **result)
            self.trial_started_ns = now
            if (result["outcome"] == "timed_out" and
                    self.config["ppo"]["path"]["abort_on_timeout"]):
                self.event("path_aborted", reason="waypoint_timeout", **result)
                self.fault("waypoint_timeout")
                return
            if self.path_execution.complete:
                summary = self.path_execution.summary()
                self.event("path_complete", **summary)
                self.supervisor.complete()
            else:
                self.event("waypoint_start", trial_id=self.path_execution.trial_id,
                           traversal_index=self.path_execution.traversal_index,
                           waypoint_index=self.path_execution.waypoint_index,
                           target_position_base_m=self.path_execution.target)
            return
        if self.sequence == 0 or self.current_target_index >= len(self.targets):
            return
        ppo = self.config["ppo"]
        target = self.targets[self.current_target_index]["position_base_m"]
        distance = sum((target[i] - snapshot.flange_position_m[i])**2 for i in range(3))**0.5
        settled = max(abs(v) for v in snapshot.dq) <= float(ppo["success"]["max_joint_velocity_rad_s"])
        self.success_count = self.success_count + 1 if (
            distance <= float(ppo["success"]["position_threshold_m"]) and settled) else 0
        if self.success_count >= int(ppo["success"]["consecutive_observations"]):
            self.event("trial_success", trial_id=self.current_target_index + 1,
                       target_id=self.targets[self.current_target_index]["id"], position_error_m=distance)
            self.current_target_index += 1
            self.success_count = 0
            self.trial_started_ns = now
            if self.current_target_index >= len(self.targets):
                self.supervisor.complete()
                self.event("session_complete")

    def _ppo_tick(self, now: int, snapshot: Snapshot) -> None:
        assert self.worker is not None
        timeout_ns = int(float(self.config["timing"]["inference_timeout_ms"]) * 1e6)
        result = self.worker.poll()
        if result is not None:
            trial_id = self._active_ppo_trial_id()
            result_trial = int(result["trial_id"])
            if result_trial < trial_id:
                # Success can be observed while the preceding trial still has one
                # inference in flight. Consume and record that result, but never send it
                # to the next target.
                self.event("policy_result_discarded", result_trial_id=result_trial,
                           active_trial_id=trial_id, reason="completed_previous_trial")
            elif result_trial > trial_id:
                raise RuntimeError("future policy result")
            else:
                action = [float(v) for v in result["action"]]
                if self.ppo_reference is None:
                    raise RuntimeError("PPO reference was not initialized")
                if self.config["runtime"] == RUNTIME_VELOCITY_IMPEDANCE:
                    velocity = action_to_velocity(self.config, action)
                    self.send_velocity(velocity, action, trial_id, int(result["observation_sequence"]),
                                       int(result["completed_monotonic_ns"]),
                                       int(result["observation_monotonic_ns"]))
                else:
                    target = action_to_target(self.config, action, self.ppo_reference)
                    self.send_target(target, action, trial_id, int(result["observation_sequence"]),
                                     int(result["completed_monotonic_ns"]),
                                     int(result["observation_monotonic_ns"]))
                    self.ppo_reference = target
                self.previous_action = action
        if self.worker.inflight is not None:
            if now - self.worker.sent_ns > timeout_ns:
                raise RuntimeError("inference_deadline")
            return
        target = self._active_ppo_target()
        if target is None:
            return
        if self.ppo_reference is None:
            raise RuntimeError("PPO reference was not initialized")
        if (self.config["runtime"] == RUNTIME_VELOCITY_IMPEDANCE and
                self.config["execution_context"] != "shadow"):
            self.ppo_reference = list(self.backend.current_reference())
        observation = assemble_policy_observation(
            snapshot.q, snapshot.dq, target, snapshot.flange_position_m,
            self.ppo_reference, self.previous_action, self.config["ppo"]["start_position_rad"],
            self.config["safety"]["soft_lower_rad"], self.config["safety"]["soft_upper_rad"],
            self.config["action_mapping"]["policy_joint_count"])
        self.request_sequence += 1
        request = {
            "request_id": self.request_sequence, "trial_id": self._active_ppo_trial_id(),
            "observation_sequence": self.state_sequence,
            "observation_monotonic_ns": snapshot.host_monotonic_ns,
            "observation": observation,
        }
        self.event("policy_observation_submitted",
                   request_id=request["request_id"], trial_id=request["trial_id"],
                   observation_sequence=request["observation_sequence"],
                   observation_monotonic_ns=request["observation_monotonic_ns"],
                   observation=observation, target_position_base_m=target,
                   measured_q_rad=snapshot.q, measured_dq_rad_s=snapshot.dq,
                   flange_position_base_m=snapshot.flange_position_m,
                   ee_feedback_position_base_m=snapshot.flange_position_m,
                   ee_feedback_orientation_base_xyzw=snapshot.flange_orientation_xyzw,
                   ee_feedback_source=snapshot.ee_feedback_source,
                   ee_feedback_base_frame=snapshot.ee_feedback_base_frame,
                   ee_feedback_frame=snapshot.ee_feedback_frame,
                   reported_flange_position_base_m=snapshot.reported_flange_position_m,
                   encoder_fk_reported_position_error_m=
                       snapshot.encoder_fk_reported_position_error_m,
                   encoder_fk_reported_orientation_error_rad=
                       snapshot.encoder_fk_reported_orientation_error_rad,
                   encoder_fk_compute_ns=snapshot.encoder_fk_compute_ns,
                   current_reference_rad=self.ppo_reference,
                   previous_action=self.previous_action)
        self.worker.submit(request)

    def run(self, max_runtime_s: float | None = None) -> int:
        snapshot = self._read_state()
        self.latest = snapshot
        self._validate_inventory(snapshot)
        ok, reason = self.supervisor.ready(snapshot)
        if not ok:
            raise RuntimeError(f"start preflight rejected: {reason}")
        self.event("ready", snapshot=asdict(snapshot))
        if self.config["mode"] == "ppo":
            # Match simulation reset: the incremental integrator starts at the
            # measured joint position, not at an unrelated fixed offset.
            self.ppo_reference = list(snapshot.q)
            if (self.config["runtime"] in {RUNTIME_VELOCITY, RUNTIME_VELOCITY_IMPEDANCE}
                    and self.config["execution_context"] != "shadow"):
                self.backend.synchronize_reference(self.ppo_reference)
                self.event("velocity_reference_synchronized", reference=self.ppo_reference,
                           source="preflight_snapshot_q")
        if self.auto_start:
            if self.config["execution_context"] != "fake":
                raise RuntimeError("--auto-start is allowed only for fake execution")
        else:
            print(f"READY session {self.config['session_id']}; type 'start' then Enter, or 'stop' to exit", flush=True)
            while True:
                command = sys.stdin.readline()
                if command == "":
                    raise RuntimeError("stdin closed before explicit start")
                command = command.strip().lower()
                if command == "start":
                    break
                if command == "stop":
                    self.supervisor.stop("operator_stop_before_start")
                    return 0
                print("expected 'start' or 'stop'", flush=True)
        self.supervisor.start()
        self.started_ns = self.trial_started_ns = raw_ns()
        self.event("operator_start" if not self.auto_start else "fake_auto_start")
        if self.path_execution is not None:
            self.event("path_start", path=self.path_execution.spec.to_dict(),
                       repetitions=self.path_execution.repetitions)
            self.event("waypoint_start", trial_id=self.path_execution.trial_id,
                       traversal_index=0, waypoint_index=0,
                       target_position_base_m=self.path_execution.target)
        period_ns = int(round(1e9 / float(self.config["timing"]["command_hz"])))
        next_tick = raw_ns()
        while self.supervisor.state == "running":
            now = raw_ns()
            if max_runtime_s is not None and (now - self.started_ns) / 1e9 >= max_runtime_s:
                self.supervisor.stop("max_runtime")
                break
            if now < next_tick:
                time.sleep((next_tick - now) / 1e9)
                now = raw_ns()
            next_tick += period_ns
            if now - self.started_ns > int(float(self.config["timing"]["suite_timeout_s"]) * 1e9):
                self.fault("suite_timeout"); break
            if self.writer.error:
                self.fault(f"writer_error:{self.writer.error}"); break
            if self.writer.dropped:
                self.fault("sample_queue_overflow"); break
            try:
                snapshot = self._read_state()
            except Exception as exc:
                self.fault(str(exc)); break
            self.latest = snapshot
            self.state_sequence += 1
            self.supervisor.validate_active(snapshot)
            if self.supervisor.state != "running":
                break
            if self.tracking_violation is not None:
                self.event("tracking_error", **self.tracking_violation)
                self.fault("tracking_error")
                break
            if self.config["execution_context"] != "shadow":
                callback_limit_ms = float(self.config["timing"]["callback_gap_ms"])
                callback_reference = self.last_callback_ns if self.callback_started else self.first_command_ns
                if callback_reference and now - callback_reference > int(callback_limit_ms * 1e6):
                    self.fault("callback_gap"); break
                if (self.callback_started and
                        snapshot.robot_time_s - self.last_callback_robot_time_s > callback_limit_ms / 1000.0):
                    self.fault("callback_queue_lag"); break
            try:
                if self.config["mode"] == "reference":
                    self._reference_tick(now)
                else:
                    if (self.path_execution is None and
                            now - self.trial_started_ns >
                            int(float(self.config["timing"]["trial_timeout_s"]) * 1e9)):
                        raise RuntimeError("trial_timeout")
                    self._ppo_success(snapshot, now)
                    if self.supervisor.state == "running":
                        self._ppo_tick(now, snapshot)
            except Exception as exc:
                self.fault(str(exc))
        return 0 if self.supervisor.state in {"complete", "stopped"} else 3

    def close(self) -> None:
        stop_ok, stop_error = False, ""
        try:
            stop_ok = self.backend.smooth_stop()
            if not stop_ok:
                stop_error = "smooth stop timed out"
        except Exception as exc:
            stop_error = repr(exc)
        if stop_error:
            self.event("stop_error", error=stop_error)
            if self.supervisor.state not in TERMINAL:
                self.supervisor.fault("stop_error")
        else:
            self.event("smooth_stop_complete", joined=stop_ok)
        if self.worker is not None:
            self.worker.close()
        self.backend.close()
        self.writer.close()
        if self.writer.error:
            self.event("writer_incomplete", error=self.writer.error)
        self.event("coordinator_shutdown", terminal_state=self.supervisor.state,
                   terminal_reason=self.supervisor.reason)
        final = {
            "schema_version": 1, "session_id": self.config["session_id"], "mode": self.config["mode"],
            "execution_context": self.config["execution_context"], "terminal_state": self.supervisor.state,
            "terminal_reason": self.supervisor.reason, "last_policy_sequence": self.sequence,
            "last_state_sequence": self.state_sequence, "last_callback_sequence": self.callback_sequence,
            "dropped_samples": self.writer.dropped, "writer_error": self.writer.error,
            "config_sha256": self.config["_config_sha256"],
            "runtime_source_sha256": self.runtime_source_sha256, "smooth_stop_ok": stop_ok,
            "smooth_stop_error": stop_error,
            "ee_feedback_source": (self.latest.ee_feedback_source
                                   if self.latest is not None else None),
            "ee_feedback_base_frame": (self.latest.ee_feedback_base_frame
                                       if self.latest is not None else None),
            "ee_feedback_frame": (self.latest.ee_feedback_frame
                                  if self.latest is not None else None),
            "path_summary": self.path_execution.summary() if self.path_execution is not None else None,
        }
        (self.run_dir / "final_metadata.json").write_text(json.dumps(final, indent=2, allow_nan=False) + "\n")
        self.event_stream.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--auto-start", action="store_true", help="fake execution only")
    parser.add_argument("--max-runtime-s", type=float, help="test bound; fake execution only")
    args = parser.parse_args(argv)
    coordinator: Coordinator | None = None
    try:
        config = load_experiment(args.config, require_approved=not args.validate_only)
        if args.validate_only:
            print(json.dumps({
                "valid_structure": True,
                "approval_ready": all((config.get("status") == "approved",
                                       config.get("executable") is True,
                                       config.get("motion_authorized") is
                                       (config.get("execution_context") != "shadow"))),
                "mode": config["mode"], "config_sha256": config["_config_sha256"],
            }, indent=2))
            return 0
        if not args.execute:
            raise ConfigError("an approved runtime still requires explicit --execute")
        if args.max_runtime_s is not None and config["execution_context"] != "fake":
            raise ConfigError("--max-runtime-s is allowed only for fake execution")
        coordinator = Coordinator(config, auto_start=args.auto_start)
        stop_requested = False

        def request_stop(signum, frame):
            nonlocal stop_requested
            del signum, frame
            stop_requested = True
            if coordinator is not None:
                coordinator.supervisor.stop("operator_signal")

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        result = coordinator.run(max_runtime_s=args.max_runtime_s)
        coordinator.close()
        print(f"SESSION {coordinator.supervisor.state}: "
              f"{coordinator.supervisor.reason or 'normal completion'}", flush=True)
        coordinator = None
        return result
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"coordinator rejected: {exc}", file=sys.stderr)
        return 2
    finally:
        if coordinator is not None:
            coordinator.close()


if __name__ == "__main__":
    raise SystemExit(main())
