"""Batched simulator joint controller, algebraically matched to franka_motion."""
import math
import torch
from isaaclab.utils.math import compute_pose_error


def pose_error_base_torch(pose, reference):
    p, r = compute_pose_error(pose[:, :3], pose[:, 3:7], reference[:, :3], reference[:, 3:7], rot_error_type="axis_angle")
    return torch.cat((p, r), dim=-1)


class BatchedCartesianImpedanceController:
    """Same SO(3) log and torque processing as the portable NumPy candidate."""
    def __init__(self, count, device, stiffness, damping, **processing):
        self.k = torch.tensor(stiffness, device=device)
        self.d = torch.tensor(damping, device=device)
        self.processor = BatchedJointImpedanceController(count, device, (0.,)*7, (0.,)*7, **processing)
        self.components = self.processor.components
        self.error = torch.zeros((count, 6), device=device)
        self.wrench = torch.zeros_like(self.error)

    def reset(self, ids):
        self.processor.reset(ids)
        self.error[ids] = 0.
        self.wrench[ids] = 0.

    def compute(self, pose, twist, reference, jacobian, coriolis, gravity):
        self.error.copy_(pose_error_base_torch(pose, reference))
        self.wrench.copy_(self.k*self.error+self.d*(reference[:, 7:]-twist))
        c = self.components
        c["pd"].copy_((jacobian.transpose(1,2)@self.wrench.unsqueeze(-1)).squeeze(-1))
        c["coriolis"].copy_(coriolis)
        c["gravity"].copy_(gravity)
        c["requested"].copy_(c["pd"]+coriolis)
        c["limited"].add_((c["requested"]-c["limited"]).clamp(-self.processor.step,self.processor.step))
        c["filtered"].add_(self.processor.alpha*(c["limited"]-c["filtered"]))
        c["unclipped"].copy_(c["filtered"]+gravity)
        c["applied"].copy_(torch.maximum(torch.minimum(c["unclipped"],self.processor.effort),-self.processor.effort))
        return c["applied"]


class BatchedJointImpedanceController:
    def __init__(self, count, device, stiffness, damping, *, dt=.001,
                 position_error_clip=.5, slew_rate_nm_s=1000., cutoff_hz=100.,
                 effort_limits_nm=(87.,)*4+(12.,)*3):
        self.k = torch.tensor(stiffness, device=device)
        self.d = torch.tensor(damping, device=device)
        self.error_clip = position_error_clip
        self.step = slew_rate_nm_s * dt
        self.alpha = dt / (dt + 1/(2*math.pi*cutoff_hz))
        self.effort = torch.tensor(effort_limits_nm, device=device)
        self.components = {key: torch.zeros((count, 7), device=device) for key in
            ("pd", "coriolis", "gravity", "requested", "limited", "filtered", "unclipped", "applied")}

    def reset(self, ids):
        for value in self.components.values():
            value[ids] = 0.

    def compute(self, q, dq, q_ref, dq_ref, coriolis, gravity):
        c = self.components
        c["pd"].copy_(self.k * (q_ref-q).clamp(-self.error_clip, self.error_clip) + self.d * (dq_ref-dq))
        c["coriolis"].copy_(coriolis)
        c["gravity"].copy_(gravity)
        c["requested"].copy_(c["pd"] + coriolis)
        c["limited"].add_((c["requested"]-c["limited"]).clamp(-self.step, self.step))
        c["filtered"].add_(self.alpha * (c["limited"]-c["filtered"]))
        c["unclipped"].copy_(c["filtered"] + gravity)
        c["applied"].copy_(torch.maximum(torch.minimum(c["unclipped"], self.effort), -self.effort))
        return c["applied"]
