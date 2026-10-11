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
        return self.process_wrench(jacobian,coriolis,gravity)

    def process_wrench(self,jacobian,coriolis,gravity):
        c = self.components
        c["pd"].copy_((jacobian.transpose(1,2)@self.wrench.unsqueeze(-1)).squeeze(-1))
        c["coriolis"].copy_(coriolis)
        c["gravity"].copy_(gravity)
        c["requested"].copy_(c["pd"]+coriolis)
        if "posture" in c:
            c["requested"].add_(c["posture"])
        c["limited"].add_((c["requested"]-c["limited"]).clamp(-self.processor.step,self.processor.step))
        c["filtered"].add_(self.processor.alpha*(c["limited"]-c["filtered"]))
        c["unclipped"].copy_(c["filtered"]+gravity)
        c["applied"].copy_(torch.maximum(torch.minimum(c["unclipped"],self.processor.effort),-self.processor.effort))
        return c["applied"]


def curve_projection_torch(position,kind,origin,radius):
    if kind == "z_line":
        target = position.clone()
        target[:,:2] = origin[:2]
        tangent = torch.zeros_like(position)
        tangent[:,2] = 1.
        return target,tangent
    if kind != "yz_circle":
        raise ValueError(f"Unsupported curve: {kind}")
    radial = position[:,1:]-origin[1:]
    distance = torch.linalg.vector_norm(radial,dim=-1,keepdim=True)
    u = radial/distance.clamp_min(1e-6)
    target = origin.expand_as(position).clone()
    target[:,1:] += radius*u
    tangent = torch.cat((torch.zeros_like(distance),-u[:,1:],u[:,:1]),dim=-1)
    target = torch.where(distance>1e-6,target,torch.full_like(target,float("nan")))
    return target,tangent


class BatchedCurveImpedanceController(BatchedCartesianImpedanceController):
    """1 kHz normal spring/damper and positive tangent damping; fixed orientation."""
    def __init__(self,count,device,stiffness,damping,**processing):
        super().__init__(count,device,stiffness,damping,**processing)
        self.kind = "z_line"
        self.origin = torch.zeros(3,device=device)
        self.radius = .05
        self.normal_stiffness,self.normal_damping,self.tangent_damping = 100.,20.,20.
        self.reference = torch.zeros((count,13),device=device)

    def reset(self,ids):
        super().reset(ids)
        self.reference[ids] = 0.

    def compute(self,pose,twist,reference,jacobian,coriolis,gravity):
        target,tangent = curve_projection_torch(pose[:,:3],self.kind,self.origin,self.radius)
        self.reference.copy_(reference)
        self.reference[:,:3] = target
        self.reference[:,3:7] = 0.
        self.reference[:,3] = 1.
        self.reference[:,7:] = 0.
        self.error.copy_(pose_error_base_torch(pose,self.reference))
        v = twist[:,:3]
        axial_v = (v*tangent).sum(-1,keepdim=True)*tangent
        e = self.error[:,:3]
        normal_e = e-(e*tangent).sum(-1,keepdim=True)*tangent
        self.wrench[:,:3] = self.normal_stiffness*normal_e-self.normal_damping*(v-axial_v)-self.tangent_damping*axial_v
        self.wrench[:,3:] = self.k[3:]*self.error[:,3:]-self.d[3:]*twist[:,3:]
        return self.process_wrench(jacobian,coriolis,gravity)


def nullspace_projector_torch(jacobian, rcond=1e-6):
    jt = jacobian.transpose(-1,-2)
    return torch.eye(7,device=jacobian.device,dtype=jacobian.dtype)-jt@torch.linalg.pinv(jt,rtol=rcond)


def line_fixture_reference_torch(position,velocity,origin,axis,filtered,alpha,lower,upper,max_speed):
    s = ((position-origin)*axis).sum(-1)
    filtered = filtered+alpha*((velocity*axis).sum(-1)-filtered)
    speed = filtered.clamp(-max_speed,max_speed)
    outward = ((s<=lower)&(speed<0))|((s>=upper)&(speed>0))
    return origin+s.clamp(lower,upper)[...,None]*axis,torch.where(outward,0.,speed)[...,None]*axis,filtered


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
