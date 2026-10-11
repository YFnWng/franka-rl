"""Base-frame finite line and endpoint-free curve fixtures; no simulator imports."""
import numpy as np
from .contracts import finite_vector, positive
from .kinematics import nullspace_projector
from .impedance import compute_posture_torque


class CurveVirtualFixture:
    """Analytic infinite Z line or closed YZ circle, expressed in robot base."""
    def __init__(self,kind,origin,radius=.05):
        if kind not in ("z_line","yz_circle"):
            raise ValueError("Expected z_line or yz_circle")
        self.kind = kind
        self.origin = np.array(finite_vector(origin,3,"curve origin"))
        self.radius = positive(radius,"radius")

    def project(self,position):
        p = np.array(finite_vector(position,3,"position"))
        if self.kind == "z_line":
            return np.r_[self.origin[:2],p[2]],np.array([0.,0.,1.])
        radial = p[1:]-self.origin[1:]
        distance = np.linalg.norm(radial)
        if distance <= 1e-6:
            raise ValueError("YZ circle projection undefined near its center")
        u = radial/distance
        return self.origin+np.r_[0.,self.radius*u],np.r_[0.,-u[1],u[0]]

    def force(self,position,velocity,stiffness=100.,normal_damping=20.,tangent_damping=20.):
        for value in (stiffness,normal_damping,tangent_damping):
            positive(value,"fixture gain")
        p = np.array(finite_vector(position,3,"position"))
        v = np.array(finite_vector(velocity,3,"velocity"))
        target,tangent = self.project(p)
        normal = np.eye(3)-np.outer(tangent,tangent)
        return -stiffness*(normal@(p-target))-normal_damping*(normal@v)-tangent_damping*tangent*(tangent@v)


def load_curve_fixture(path):
    from pathlib import Path
    from .config import _load,_keys
    path = Path(path).resolve()
    data = _load(path)
    fields = {"schema_version","motion_suite","line_origin_m","circle_center_m","circle_radius_m",
              "workspace_lower_m","workspace_upper_m","normal_stiffness","normal_damping","tangent_damping",
              "force_n","hold_s","push_s","release_s","experiments"}
    _keys(data,fields,fields,"curve fixture")
    CurveVirtualFixture("z_line",data["line_origin_m"])
    CurveVirtualFixture("yz_circle",data["circle_center_m"],data["circle_radius_m"])
    lower = np.array(finite_vector(data["workspace_lower_m"],3,"workspace lower"))
    upper = np.array(finite_vector(data["workspace_upper_m"],3,"workspace upper"))
    if np.any(lower>=upper):
        raise ValueError("Unordered curve workspace")
    for key in ("normal_stiffness","normal_damping","tangent_damping","force_n","hold_s","push_s","release_s"):
        positive(data[key],key)
    if data["force_n"] > 2 or data["normal_stiffness"] > 200 or data["tangent_damping"] > data["normal_damping"]:
        raise ValueError("Curve fixture simulation limits exceeded")
    allowed = {"line_tangent","line_normal","circle_tangent","circle_normal"}
    if not isinstance(data["experiments"],list) or not data["experiments"] or not set(data["experiments"])<=allowed or len(set(data["experiments"]))!=len(data["experiments"]):
        raise ValueError("Invalid curve experiments")
    data["motion_suite"] = str((path.parent/data["motion_suite"]).resolve())
    return data


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
    if data["force_n"] > 2 or data["posture_offset_rad"] > .6 or data["projector_rcond"] >= 1:
        raise ValueError("Demo B bounded simulation limits exceeded")
    allowed = {"fixture_axial","fixture_transverse","posture_a","posture_b","posture_weak"}
    if not isinstance(data["experiments"],list) or not data["experiments"] or len(set(data["experiments"])) != len(data["experiments"]) or not set(data["experiments"]) <= allowed:
        raise ValueError("Invalid/duplicate Demo B experiments")
    data["motion_suite"] = str((path.parent/data["motion_suite"]).resolve())
    return data
