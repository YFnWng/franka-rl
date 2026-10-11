"""Local offline Ruckig state-to-state planning; no cloud waypoint API."""
from dataclasses import dataclass
import importlib.metadata
import math
import numpy as np

from .contracts import JointReference, finite_vector, positive, validate_reference


@dataclass
class JointTrajectory:
    trajectory: object
    lower: tuple
    upper: tuple
    max_velocity: tuple
    max_acceleration: tuple
    max_jerk: tuple
    version: str

    @property
    def duration_s(self):
        return self.trajectory.duration

    def evaluate(self, time_s):
        t = float(time_s)
        if not math.isfinite(t) or t < 0 or t > self.duration_s:
            raise ValueError("Time outside joint trajectory")
        q, dq, _ = self.trajectory.at_time(t)
        return JointReference(q, dq, t)


class RuckigJointPlanner:
    def __init__(self, lower, upper, max_velocity, max_acceleration, max_jerk):
        self.lower, self.upper = (finite_vector(x, 7, "bounds") for x in (lower, upper))
        self.limits = tuple(finite_vector(x, 7, "limits") for x in (max_velocity, max_acceleration, max_jerk))
        if np.any(np.array(self.lower) >= self.upper) or any(np.any(np.array(x) <= 0) for x in self.limits):
            raise ValueError("Invalid planner bounds/limits")

    def plan(self, start, end):
        # Lazy dependency so portable contract imports still work without Ruckig.
        import ruckig
        version = importlib.metadata.version("ruckig")
        if version != "0.17.3":
            raise RuntimeError(f"Expected pinned Ruckig 0.17.3, found {version}")
        for q in (start, end):
            validate_reference(JointReference(q, [0.]*7), lower_rad=self.lower, upper_rad=self.upper)
        inp, trajectory = ruckig.InputParameter(7), ruckig.Trajectory(7)
        inp.current_position, inp.target_position = list(start), list(end)
        inp.current_velocity = inp.current_acceleration = [0.]*7
        inp.target_velocity = inp.target_acceleration = [0.]*7
        inp.max_velocity, inp.max_acceleration, inp.max_jerk = (list(x) for x in self.limits)
        result = ruckig.Ruckig(7).calculate(inp, trajectory)
        if result not in (ruckig.Result.Working, ruckig.Result.Finished):
            raise RuntimeError(f"Ruckig planning failed: {result}")
        return JointTrajectory(trajectory, self.lower, self.upper, *self.limits, version)


def plan_joint_ptp(start, end, *, lower, upper, max_velocity, max_acceleration, max_jerk):
    return RuckigJointPlanner(lower, upper, max_velocity, max_acceleration, max_jerk).plan(start, end)


def validate_joint_trajectory(path, sample_hz=1000.):
    rate = positive(sample_hz, "sample_hz")
    times = (np.array([0.]) if path.duration_s == 0 else
             np.linspace(0, path.duration_s, max(2, math.ceil(path.duration_s * rate)+1)))
    states = np.array([path.trajectory.at_time(float(t)) for t in times])
    q, dq, ddq = states[:, 0], states[:, 1], states[:, 2]
    if not np.isfinite(states).all() or np.any(q < np.array(path.lower)-1e-10) or np.any(q > np.array(path.upper)+1e-10):
        raise ValueError("Joint trajectory outside interval or nonfinite")
    jerk = np.diff(ddq, axis=0) / np.diff(times)[:, None]
    for values, limit in ((dq, path.max_velocity), (ddq, path.max_acceleration), (jerk, path.max_jerk)):
        if np.any(np.abs(values) > np.array(limit) * (1+1e-6)):
            raise ValueError("Joint trajectory violates configured derivative limit")
    return {"passed": True, "planner": "ruckig", "version": path.version,
            "duration_s": path.duration_s, "sample_count": len(times),
            "method": "Ruckig jerk constraints plus sampled position/velocity/acceleration/finite-difference jerk checks"}
