"""Immutable physical references with explicit frames, units and quaternion order."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def finite_vector(value, length: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {length} finite scalars")
    return tuple(float(v) for v in array)


def positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return result


def unit_quaternion_xyzw(value) -> tuple[float, ...]:
    quaternion = finite_vector(value, 4, "quaternion_xyzw")
    if not math.isclose(float(np.linalg.norm(quaternion)), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("quaternion_xyzw must have unit norm; normalization is not implicit")
    return quaternion


@dataclass(frozen=True)
class JointReference:
    position_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    time_s: float = 0.0

    def __post_init__(self):
        for name in ("position_rad", "velocity_rad_s"):
            object.__setattr__(self, name, finite_vector(getattr(self, name), 7, name))
        object.__setattr__(self, "time_s", positive(self.time_s, "time_s", allow_zero=True))

    def as_action(self) -> np.ndarray:
        return np.asarray(self.position_rad + self.velocity_rad_s)


@dataclass(frozen=True)
class CartesianReference:
    position_m: tuple[float, ...]
    quaternion_xyzw: tuple[float, ...]
    linear_velocity_m_s: tuple[float, ...] = (0.0, 0.0, 0.0)
    angular_velocity_rad_s: tuple[float, ...] = (0.0, 0.0, 0.0)
    time_s: float = 0.0
    base_frame: str = "fr3_link0"
    tcp_frame: str = "fr3_flange"

    def __post_init__(self):
        for name in ("position_m", "linear_velocity_m_s", "angular_velocity_rad_s"):
            object.__setattr__(self, name, finite_vector(getattr(self, name), 3, name))
        object.__setattr__(self, "quaternion_xyzw", unit_quaternion_xyzw(self.quaternion_xyzw))
        object.__setattr__(self, "time_s", positive(self.time_s, "time_s", allow_zero=True))
        if any(not isinstance(value, str) or not value for value in (self.base_frame, self.tcp_frame)):
            raise ValueError("Explicit base_frame and tcp_frame are required")

    def as_action(self, *, twist_feedforward: bool = True) -> np.ndarray:
        twist = self.linear_velocity_m_s + self.angular_velocity_rad_s if twist_feedforward else (0.0,) * 6
        return np.asarray(self.position_m + self.quaternion_xyzw + twist)


@dataclass(frozen=True)
class TrajectorySample:
    reference: CartesianReference
    acceleration_m_s2: tuple[float, ...]
    jerk_m_s3: tuple[float, ...]
    phase: str = "track"

    def __post_init__(self):
        for name in ("acceleration_m_s2", "jerk_m_s3"):
            object.__setattr__(self, name, finite_vector(getattr(self, name), 3, name))


@dataclass(frozen=True)
class RobotSample:
    """Reference-independent measured state; absent torque signals remain None."""

    host_monotonic_ns: int
    robot_time_s: float
    position_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    tcp_pose: CartesianReference
    commanded_torque_nm: tuple[float, ...] | None = None
    measured_torque_nm: tuple[float, ...] | None = None
    external_torque_nm: tuple[float, ...] | None = None
    applied_reference_sequence: int | None = None

    def __post_init__(self):
        if isinstance(self.host_monotonic_ns, bool) or not isinstance(self.host_monotonic_ns, int):
            raise ValueError("host_monotonic_ns must be an integer")
        if self.host_monotonic_ns < 0:
            raise ValueError("host_monotonic_ns must be nonnegative")
        object.__setattr__(self, "robot_time_s", positive(self.robot_time_s, "robot_time_s", allow_zero=True))
        if not isinstance(self.tcp_pose, CartesianReference):
            raise TypeError("tcp_pose must be an explicitly framed CartesianReference")
        if self.applied_reference_sequence is not None and (
            type(self.applied_reference_sequence) is not int or self.applied_reference_sequence < 0
        ):
            raise ValueError("applied_reference_sequence must be a nonnegative integer or None")
        for name in ("position_rad", "velocity_rad_s", "commanded_torque_nm", "measured_torque_nm",
                     "external_torque_nm"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, finite_vector(value, 7, name))


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    status: str
    duration_s: float
    sample_count: int
    stop_reason: str | None = None

    def __post_init__(self):
        if not self.trial_id or self.status not in ("complete", "fault", "unsupported"):
            raise ValueError("Invalid trial id/status")
        positive(self.duration_s, "duration_s", allow_zero=True)
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 0:
            raise ValueError("sample_count must be a nonnegative integer")
        if self.status != "complete" and not self.stop_reason:
            raise ValueError("A failed/unsupported trial requires a stop_reason")


def validate_reference(reference, *, lower_rad=None, upper_rad=None, max_velocity_rad_s=None) -> None:
    if isinstance(reference, JointReference):
        if (lower_rad is None) != (upper_rad is None):
            raise ValueError("Both joint bounds must be supplied")
        if lower_rad is not None:
            lower = np.asarray(finite_vector(lower_rad, 7, "lower_rad"))
            upper = np.asarray(finite_vector(upper_rad, 7, "upper_rad"))
            if np.any(lower >= upper) or np.any(reference.position_rad < lower) or np.any(reference.position_rad > upper):
                raise ValueError("Joint reference outside configured position interval")
        if max_velocity_rad_s is not None:
            maximum = np.asarray(finite_vector(max_velocity_rad_s, 7, "max_velocity_rad_s"))
            if np.any(maximum <= 0) or np.any(np.abs(reference.velocity_rad_s) > maximum):
                raise ValueError("Joint reference exceeds configured velocity bound")
    elif not isinstance(reference, CartesianReference):
        raise TypeError("Expected JointReference or CartesianReference")
