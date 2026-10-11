"""NumPy joint torque reference; explicit simulator profile, no robot SDK imports."""
import math
import numpy as np

from .contracts import finite_vector, positive


class TorqueCommandProcessor:
    """Slew non-gravity command, low-pass it, add gravity, then enforce effort bounds.

    The non-gravity slew/filter order follows the existing archived Franky joint
    route. Gravity is supplied by PhysX in simulation (libfranka compensates it on
    hardware). The final clamp is a simulator effort bound, not a claimed Franky
    implementation detail. Reset deliberately seeds non-gravity state at zero.
    """
    def __init__(self, *, dt=.001, slew_rate_nm_s=1000., cutoff_hz=100., effort_limits_nm=(87.,)*4+(12.,)*3):
        self.dt = positive(dt, "dt")
        self.step = positive(slew_rate_nm_s, "slew_rate_nm_s") * self.dt
        self.alpha = self.dt / (self.dt + 1 / (2 * math.pi * positive(cutoff_hz, "cutoff_hz")))
        self.effort = np.asarray(finite_vector(effort_limits_nm, 7, "effort_limits_nm"))
        if np.any(self.effort <= 0):
            raise ValueError("effort limits must be positive")
        self.reset()

    def reset(self):
        self.limited = np.zeros(7)
        self.filtered = np.zeros(7)

    def compute(self, requested, gravity):
        requested = np.asarray(finite_vector(requested, 7, "requested"))
        gravity = np.asarray(finite_vector(gravity, 7, "gravity"))
        self.limited += np.clip(requested - self.limited, -self.step, self.step)
        self.filtered += self.alpha * (self.limited - self.filtered)
        unclipped = self.filtered + gravity
        applied = np.clip(unclipped, -self.effort, self.effort)
        return {"requested": requested.copy(), "limited": self.limited.copy(), "filtered": self.filtered.copy(),
                "gravity": gravity.copy(), "unclipped": unclipped, "applied": applied}


class JointImpedanceController:
    def __init__(self, stiffness, damping, *, position_error_clip=.5, **processing):
        self.stiffness = np.asarray(finite_vector(stiffness, 7, "stiffness"))
        self.damping = np.asarray(finite_vector(damping, 7, "damping"))
        if np.any(self.stiffness < 0) or np.any(self.damping < 0):
            raise ValueError("gains must be nonnegative")
        self.error_clip = positive(position_error_clip, "position_error_clip")
        self.processor = TorqueCommandProcessor(**processing)

    def reset(self):
        self.processor.reset()

    def compute(self, q, dq, q_ref, dq_ref, coriolis, gravity):
        q, dq, q_ref, dq_ref, coriolis, gravity = (
            np.asarray(finite_vector(value, 7, name)) for name, value in
            zip(("q", "dq", "q_ref", "dq_ref", "coriolis", "gravity"), (q, dq, q_ref, dq_ref, coriolis, gravity)))
        pd = self.stiffness * np.clip(q_ref - q, -self.error_clip, self.error_clip) + self.damping * (dq_ref - dq)
        return {"pd": pd, "coriolis": coriolis, **self.processor.compute(pd + coriolis, gravity)}
