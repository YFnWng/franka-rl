"""Finite base-frame line fixture; no simulator or hardware dependencies."""
import numpy as np
from .contracts import finite_vector, positive
from .kinematics import nullspace_projector
from .impedance import compute_posture_torque


class LineVirtualFixture:
    def __init__(self, origin, direction, lower=-.03, upper=.03, *, dt=.02, cutoff_hz=5., max_speed=.03):
        self.origin = np.array(finite_vector(origin,3,"origin"))
        self.direction = np.array(finite_vector(direction,3,"direction"))
        if not np.isclose(np.linalg.norm(self.direction),1.,atol=1e-8):
            raise ValueError("Fixture direction must be a unit vector")
        if not np.isfinite([lower,upper]).all() or not lower < upper:
            raise ValueError("Fixture segment must be finite and ordered")
        self.lower,self.upper = lower,upper
        self.max_speed = positive(max_speed,"max_speed")
        self.alpha = positive(dt,"dt")/(dt+1/(2*np.pi*positive(cutoff_hz,"cutoff_hz")))
        self.velocity = 0.

    def reset(self):
        self.velocity = 0.

    def evaluate(self, position, velocity):
        p = np.array(finite_vector(position,3,"position"))
        v = np.array(finite_vector(velocity,3,"velocity"))
        s = float((p-self.origin)@self.direction)
        self.velocity += self.alpha*(float(v@self.direction)-self.velocity)
        speed = np.clip(self.velocity,-self.max_speed,self.max_speed)
        if (s <= self.lower and speed < 0) or (s >= self.upper and speed > 0):
            speed = 0.
        return self.origin+np.clip(s,self.lower,self.upper)*self.direction, speed*self.direction


def load_demo_b(path):
    from pathlib import Path
    from .config import _load, _keys
    path = Path(path).resolve()
    data = _load(path)
    fields = {"schema_version","motion_suite","direction_base","segment_m","force_n","hold_s","force_duration_s",
              "release_s","cutoff_hz","max_reference_speed_m_s","posture_offset_rad","transition_s","posture_settle_s",
              "posture_stiffness","posture_damping","weak_stiffness","projector_rcond","experiments"}
    _keys(data,fields,fields,"demo B")
    LineVirtualFixture([0,0,0],data["direction_base"],*finite_vector(data["segment_m"],2,"segment"),
        cutoff_hz=data["cutoff_hz"],max_speed=data["max_reference_speed_m_s"])
    for key in fields-{"schema_version","motion_suite","direction_base","segment_m","experiments"}:
        positive(data[key],key)
    if data["force_n"] > 2 or data["posture_offset_rad"] > .3 or data["projector_rcond"] >= 1:
        raise ValueError("Demo B bounded simulation limits exceeded")
    allowed = {"fixture_axial","fixture_transverse","posture_a","posture_b","posture_weak"}
    if not isinstance(data["experiments"],list) or not data["experiments"] or len(set(data["experiments"])) != len(data["experiments"]) or not set(data["experiments"]) <= allowed:
        raise ValueError("Invalid/duplicate Demo B experiments")
    data["motion_suite"] = str((path.parent/data["motion_suite"]).resolve())
    return data
