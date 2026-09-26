"""Franky joint-impedance tracking and deterministic fake backends."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class Snapshot:
    host_monotonic_ns: int
    robot_time_s: float
    q: tuple[float, ...]
    dq: tuple[float, ...]
    q_d: tuple[float, ...]
    dq_d: tuple[float, ...]
    ddq_d: tuple[float, ...]
    flange_position_m: tuple[float, ...]
    f_t_ee: tuple[tuple[float, ...], ...]
    load_mass_kg: float
    mode: str
    has_errors: bool
    current_errors: str
    last_motion_errors: str
    is_in_control: bool
    command_success_rate: float


@dataclass(frozen=True)
class CallbackRecord:
    host_monotonic_ns: int
    robot_time_s: float
    time_step_s: float
    relative_time_s: float
    absolute_time_s: float
    q_command: tuple[float, ...]
    dq_command: tuple[float, ...]
    tau_command: tuple[float, ...]
    q: tuple[float, ...]
    dq: tuple[float, ...]
    q_d: tuple[float, ...]
    dq_d: tuple[float, ...]
    ddq_d: tuple[float, ...]
    tau_joint: tuple[float, ...]
    tau_joint_desired: tuple[float, ...]
    tau_external: tuple[float, ...]
    flange_position_m: tuple[float, ...]
    mode: str
    current_errors: str
    last_motion_errors: str
    command_success_rate: float


def _tuple(value: Any) -> tuple[float, ...]:
    return tuple(float(v) for v in np.asarray(value).reshape(-1))


def _matrix(value: Any) -> np.ndarray:
    matrix = value.matrix
    return np.asarray(matrix() if callable(matrix) else matrix, dtype=float)


def _snapshot(robot: Any, state: Any) -> Snapshot:
    return Snapshot(
        host_monotonic_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
        robot_time_s=float(state.time.to_sec()),
        q=_tuple(state.q), dq=_tuple(state.dq), q_d=_tuple(state.q_d),
        dq_d=_tuple(state.dq_d), ddq_d=_tuple(state.ddq_d),
        flange_position_m=tuple(float(v) for v in _matrix(state.O_T_EE)[:3, 3]),
        f_t_ee=tuple(tuple(float(v) for v in row) for row in _matrix(state.F_T_EE)),
        load_mass_kg=float(state.m_load), mode=state.robot_mode.name,
        has_errors=bool(robot.has_errors), current_errors=repr(state.current_errors),
        last_motion_errors=repr(state.last_motion_errors), is_in_control=bool(robot.is_in_control),
        command_success_rate=float(state.control_command_success_rate),
    )


class FrankyBackend:
    """One long-lived Franky torque motion with atomically replaced q/dq references."""
    def __init__(self, config: dict[str, Any]):
        from franky import RealtimeConfig, Robot

        expected = config["robot"]["expected_franky_version"]
        actual = metadata.version("franky-control")
        if actual != expected:
            raise RuntimeError(f"Franky version mismatch: expected {expected}, got {actual}")
        collision = config["robot"]["constructor_collision_behavior"]
        self._franky = __import__("franky")
        self.robot = Robot(
            config["robot"]["host"],
            default_torque_threshold=float(collision["joint_torque_threshold_nm"]),
            default_force_threshold=float(collision["cartesian_force_threshold_n"]),
            realtime_config=RealtimeConfig.Enforce,
        )
        self.impedance = config["robot"]["impedance_controller"]
        self.stop_config = config["robot"]["torque_stop"]
        self.soft_lower = tuple(config["safety"]["soft_lower_rad"])
        self.soft_upper = tuple(config["safety"]["soft_upper_rad"])
        self.watchdog_ms = int(math.ceil(float(config["timing"]["command_watchdog_ms"])))
        self._motion_lock = threading.Lock()
        self._motion = None
        self._active_callback: Callable[[CallbackRecord], None] | None = None
        self._reference = _tuple(self.robot.state.q)
        self._velocity_reference = (0.0,) * 7
        self._last_target_ns = 0
        self._watchdog_tripped = False
        self._watchdog_error = ""
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="franky-command-watchdog", daemon=True)
        self._watchdog_thread.start()

    def _torque_stop_motion(self):
        s = self.stop_config
        return self._franky.TorqueStopMotion(
            damping=s["damping_nms_rad"], ramp_duration=float(s["ramp_duration_s"]),
            velocity_epsilon=float(s["velocity_epsilon_rad_s"]),
            max_duration=float(s["max_duration_s"]),
            compensate_coriolis=bool(s["compensate_coriolis"]),
            max_delta_tau=float(s["max_delta_tau_nm_per_ms"]),
        )

    def _watchdog_loop(self) -> None:
        interval_s = min(0.005, self.watchdog_ms / 4000.0)
        while not self._watchdog_stop.wait(interval_s):
            try:
                with self._motion_lock:
                    if (self._last_target_ns == 0 or
                            time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - self._last_target_ns
                            <= self.watchdog_ms * 1_000_000):
                        continue
                    self._watchdog_tripped = True
                    if self.robot.is_in_control:
                        self.robot.move(self._torque_stop_motion(), asynchronous=True,
                                        limit_rate=False, cutoff_frequency=100.0)
                    return
            except Exception as exc:
                self._watchdog_error = repr(exc)
                self._watchdog_tripped = True
                return

    def _check_watchdog(self) -> None:
        if self._watchdog_error:
            raise RuntimeError(f"command_watchdog_stop_error:{self._watchdog_error}")
        if self._watchdog_tripped:
            raise RuntimeError("command_watchdog")

    def read_state(self) -> Snapshot:
        self._check_watchdog()
        return _snapshot(self.robot, self.robot.state)

    def keepalive(self) -> None:
        with self._motion_lock:
            self._check_watchdog()
            if not self.robot.is_in_control:
                raise RuntimeError("control ended while holding impedance reference")
            self._last_target_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

    def _create_motion(self):
        f, c = self._franky, self.impedance
        friction = f.FrictionCompensationParams(
            coulomb=c["friction"]["coulomb_nm"],
            viscous=c["friction"]["viscous_nms_rad"],
            max_torque=c["friction"]["max_torque_nm"],
            velocity_epsilon=float(c["friction"]["velocity_epsilon_rad_s"]),
        )
        motion = f.JointImpedanceTrackingMotion(
            stiffness=c["stiffness_nm_rad"], damping=c["damping_nms_rad"],
            constant_torque_offset=c["constant_torque_offset_nm"],
            lower_joint_limits=self.soft_lower, upper_joint_limits=self.soft_upper,
            compensate_coriolis=bool(c["compensate_coriolis"]),
            max_delta_tau=float(c["max_delta_tau_nm_per_ms"]),
            joint_limit_activation_distance=float(c["joint_limit_activation_distance_rad"]),
            joint_limit_stiffness=float(c["joint_limit_stiffness_nm"]),
            joint_limit_damping=float(c["joint_limit_damping_nms_rad"]),
            joint_limit_max_torque=float(c["joint_limit_max_torque_nm"]),
            friction=friction, gains_time_constant=float(c["gains_time_constant_s"]),
        )
        resolved_error_clip = _tuple(motion.params.error_clip)
        expected_error_clip = tuple(float(v) for v in c["expected_error_clip_rad"])
        if resolved_error_clip != expected_error_clip:
            raise RuntimeError(
                f"Franky error clip mismatch: expected {expected_error_clip}, got {resolved_error_clip}")
        return motion

    def _on_update(self, state, time_step, relative_time, absolute_time, command) -> None:
        callback = self._active_callback
        if callback is None:
            return
        callback(CallbackRecord(
            host_monotonic_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
            robot_time_s=float(state.time.to_sec()), time_step_s=float(time_step.to_sec()),
            relative_time_s=float(relative_time.to_sec()), absolute_time_s=float(absolute_time.to_sec()),
            q_command=self._reference, dq_command=self._velocity_reference,
            tau_command=_tuple(command.tau_J), q=_tuple(state.q), dq=_tuple(state.dq),
            q_d=_tuple(state.q_d), dq_d=_tuple(state.dq_d), ddq_d=_tuple(state.ddq_d),
            tau_joint=_tuple(state.tau_J), tau_joint_desired=_tuple(state.tau_J_d),
            tau_external=_tuple(state.tau_ext_hat_filtered),
            flange_position_m=tuple(float(v) for v in _matrix(state.O_T_EE)[:3, 3]),
            mode=state.robot_mode.name, current_errors=repr(state.current_errors),
            last_motion_errors=repr(state.last_motion_errors),
            command_success_rate=float(state.control_command_success_rate),
        ))

    def send_target(self, target: list[float], callback: Callable[[CallbackRecord], None]) -> None:
        f = self._franky
        q = tuple(float(v) for v in target)
        dq = (0.0,) * 7
        with self._motion_lock:
            self._check_watchdog()
            self._active_callback = callback
            if self._motion is None:
                self._motion = self._create_motion()
                seed = _tuple(self.robot.state.q)
                self._reference, self._velocity_reference = seed, dq
                self._motion.set_reference(f.JointReference(q=seed, dq=dq))
                self._motion.register_callback(self._on_update)
                self._last_target_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                self.robot.move(self._motion, asynchronous=True,
                                limit_rate=False, cutoff_frequency=100.0)
            self._reference, self._velocity_reference = q, dq
            self._motion.set_reference(f.JointReference(q=q, dq=dq))
            self._last_target_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

    def smooth_stop(self, timeout_s: float = 5.0) -> bool:
        self._watchdog_stop.set()
        self._watchdog_thread.join(timeout=0.1)
        with self._motion_lock:
            if not self.robot.is_in_control:
                return True
            if not self._watchdog_tripped:
                self.robot.move(self._torque_stop_motion(), asynchronous=True,
                                limit_rate=False, cutoff_frequency=100.0)
        return bool(self.robot.join_motion(timeout_s))

    def close(self) -> None:
        self._watchdog_stop.set()
        self._watchdog_thread.join(timeout=0.1)
        self._motion = None
        self.robot = None


class FakeBackend:
    """Deterministic fake of the long-lived impedance tracker."""
    def __init__(self, config: dict[str, Any]):
        start = config[config["mode"]]["start_position_rad"]
        self.q = np.asarray(start, dtype=float)
        self.dq = np.zeros(7)
        self.q_d = self.q.copy()
        self.target = self.q.copy()
        self.stiffness = np.asarray(config["robot"]["impedance_controller"]["stiffness_nm_rad"])
        self.damping = np.asarray(config["robot"]["impedance_controller"]["damping_nms_rad"])
        self.started_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        self.in_control = False
        self.closed = False
        self.active_callback: Callable[[CallbackRecord], None] | None = None

    def read_state(self) -> Snapshot:
        prior = self.q.copy()
        self.q += 0.35 * (self.target - self.q)
        self.dq = (self.q - prior) / 0.001
        self.q_d = self.q.copy()
        now = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        return Snapshot(
            host_monotonic_ns=now, robot_time_s=(now - self.started_ns) / 1e9,
            q=_tuple(self.q), dq=_tuple(self.dq), q_d=_tuple(self.q_d), dq_d=_tuple(self.dq),
            ddq_d=(0.0,) * 7, flange_position_m=(0.45, 0.0, 0.35),
            f_t_ee=((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            load_mass_kg=0.0, mode="Move" if self.in_control else "Idle", has_errors=False,
            current_errors="[]", last_motion_errors="[]", is_in_control=self.in_control,
            command_success_rate=1.0 if self.in_control else 0.0,
        )

    def _emit_callback(self, callback: Callable[[CallbackRecord], None]) -> None:
        state = self.read_state()
        tau = self.stiffness * (self.target - self.q) - self.damping * self.dq
        callback(CallbackRecord(
            host_monotonic_ns=state.host_monotonic_ns, robot_time_s=state.robot_time_s,
            time_step_s=0.001, relative_time_s=0.0, absolute_time_s=state.robot_time_s,
            q_command=_tuple(self.target), dq_command=(0.0,) * 7, tau_command=_tuple(tau),
            q=state.q, dq=state.dq, q_d=state.q_d, dq_d=state.dq_d, ddq_d=state.ddq_d,
            tau_joint=_tuple(tau), tau_joint_desired=_tuple(tau), tau_external=(0.0,) * 7,
            flange_position_m=state.flange_position_m, mode="Move", current_errors="[]",
            last_motion_errors="[]", command_success_rate=1.0,
        ))

    def keepalive(self) -> None:
        if not self.in_control or self.active_callback is None:
            raise RuntimeError("fake control ended while holding impedance reference")
        self._emit_callback(self.active_callback)

    def send_target(self, target: list[float], callback: Callable[[CallbackRecord], None]) -> None:
        self.target = np.asarray(target, dtype=float)
        self.in_control = True
        self.active_callback = callback
        self._emit_callback(callback)

    def smooth_stop(self, timeout_s: float = 5.0) -> bool:
        del timeout_s
        self.in_control = False
        return True

    def close(self) -> None:
        self.closed = True
