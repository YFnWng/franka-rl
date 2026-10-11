"""Physical references, explicit torque, bounded telemetry and latched fault samples."""
import torch
import warp as wp
from isaaclab.managers.action_manager import ActionTerm
from .controllers import BatchedJointImpedanceController, BatchedCartesianImpedanceController
from .observations import tcp_pose_base, tcp_twist_base
from isaaclab.utils.math import matrix_from_quat


class JointMotionAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ids, names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if len(self.ids) != 7 or abs(env.physics_dt-.001) > 1e-10:
            raise ValueError("Joint motion requires seven ordered joints and 1 kHz physics")
        self.dynamics_ids = torch.tensor(self.ids, device=self.device) + self._asset.num_base_dofs
        self.flange_id = self._asset.find_bodies("fr3_flange")[0][0]
        self._raw = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._processed = torch.zeros_like(self._raw)
        self.fault_code = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.first_fault_tick = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.first_fault_code = torch.zeros_like(self.fault_code)
        self.contact_force = torch.zeros(self.num_envs, device=self.device)
        self.first_fault_contact_force = torch.zeros_like(self.contact_force)
        self.first_fault_q = torch.zeros((self.num_envs, 7), device=self.device)
        self.first_fault_dq = torch.zeros_like(self.first_fault_q)
        self.first_fault_reference = torch.zeros_like(self._raw)
        self.fault_hold_q = torch.zeros_like(self.first_fault_q)
        self.bounds = {key: torch.tensor(getattr(cfg, key), device=self.device) for key in
            ("lower", "upper", "abort_lower", "abort_upper", "max_reference_velocity", "max_measured_velocity")}
        self.controller = BatchedJointImpedanceController(self.num_envs, self.device, cfg.stiffness, cfg.damping,
            dt=env.physics_dt, position_error_clip=cfg.position_error_clip, slew_rate_nm_s=cfg.slew_rate_nm_s,
            cutoff_hz=cfg.cutoff_hz, effort_limits_nm=cfg.effort_limits_nm)
        self.trace = None
        self.trace_index = 0

    @property
    def action_dim(self):
        return 14

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._processed

    def _latch(self, code, q, dq):
        fresh = (self.fault_code == 0) & (code != 0)
        self.first_fault_tick[fresh] = self._env._sim_step_counter
        self.first_fault_code[fresh] = code[fresh]
        self.first_fault_contact_force[fresh] = self.contact_force[fresh]
        self.first_fault_q[fresh] = q[fresh]
        self.first_fault_dq[fresh] = dq[fresh]
        self.first_fault_reference[fresh] = self._raw[fresh]
        self.fault_hold_q[fresh] = torch.nan_to_num(q[fresh])
        self.fault_code.bitwise_or_(code)

    def check_state(self):
        q = self._asset.data.joint_pos.torch[:, self.ids]
        dq = self._asset.data.joint_vel.torch[:, self.ids]
        b = self.bounds
        code = (((q < b["abort_lower"]) | (q > b["abort_upper"])).any(1).int()
            + 2*(dq.abs() > b["max_measured_velocity"]).any(1).int()
            + 4*(~torch.isfinite(torch.cat((q, dq), dim=1)).all(1)).int())
        forces = self._env.scene["arm_contacts"].data.net_forces_w.torch
        self.contact_force.copy_(torch.linalg.vector_norm(forces, dim=-1).amax(1))
        code += 8*(self.contact_force > self.cfg.contact_threshold_n).int()
        self._latch(code, q, dq)
        return q, dq

    def process_actions(self, actions):
        self._raw.copy_(actions)
        b = self.bounds
        invalid = (~torch.isfinite(actions).all(1) | (actions[:, :7] < b["lower"]).any(1)
            | (actions[:, :7] > b["upper"]).any(1) | (actions[:, 7:].abs() > b["max_reference_velocity"]+1e-7).any(1))
        q, dq = self.check_state()
        self._latch(16*invalid.int(), q, dq)
        # Invalid references are rejected, not mapped or clipped into another motion.
        self._processed.copy_(torch.where(invalid[:, None], self._processed, actions))

    def allocate_trace(self, ticks):
        if ticks < 1 or ticks > 120000:
            raise ValueError("M1 bounded trace supports 1..120000 physics samples")
        self.trace = {key: torch.empty((ticks, self.num_envs, 7), device=self.device)
            for key in ("q", "dq", "q_ref", "dq_ref", *self.controller.components)}
        self.trace["fault_code"] = torch.empty((ticks, self.num_envs), device=self.device, dtype=torch.int32)
        self.trace["contact_force_n"] = torch.empty((ticks, self.num_envs), device=self.device)
        self.trace_index = 0

    def apply_actions(self):
        q, dq = self.check_state()
        fault = self.fault_code != 0
        q_ref = torch.where(fault[:, None], self.fault_hold_q, self._processed[:, :7])
        dq_ref = torch.where(fault[:, None], 0., self._processed[:, 7:])
        raw_c = self._asset.root_view.get_coriolis_and_centrifugal_compensation_forces()
        coriolis = (wp.to_torch(raw_c) if isinstance(raw_c, wp.array) else raw_c)[:, self.dynamics_ids]
        gravity = self._asset.data.gravity_compensation_forces.torch[:, self.dynamics_ids]
        torque = self.controller.compute(torch.nan_to_num(q), torch.nan_to_num(dq), q_ref, dq_ref,
                                         torch.nan_to_num(coriolis), torch.nan_to_num(gravity))
        self._latch(4*(~torch.isfinite(torque).all(1)).int(), q, dq)
        if self.trace is not None:
            if self.trace_index >= len(self.trace["q"]):
                raise RuntimeError("Physics telemetry capacity exhausted")
            i = self.trace_index
            for key, value in {"q": q, "dq": dq, "q_ref": q_ref, "dq_ref": dq_ref,
                               **self.controller.components, "fault_code": self.fault_code,
                               "contact_force_n": self.contact_force}.items():
                self.trace[key][i].copy_(value)
            self.trace_index += 1
        self._asset.set_joint_effort_target_index(target=torch.nan_to_num(torque), joint_ids=self.ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.
        self._processed[ids, :7] = self._asset.data.joint_pos.torch[ids][:, self.ids]
        self._processed[ids, 7:] = 0.
        self.fault_code[ids] = 0
        self.first_fault_tick[ids] = -1
        self.first_fault_code[ids] = 0
        self.first_fault_contact_force[ids] = 0.
        self.first_fault_q[ids] = 0.
        self.first_fault_dq[ids] = 0.
        self.first_fault_reference[ids] = 0.
        self.fault_hold_q[ids] = self._processed[ids, :7]
        self.controller.reset(ids)


class CartesianMotionAction(JointMotionAction):
    """Physical 13D base-frame flange pose/twist; no hidden IK/planner in servo."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        processing = dict(dt=env.physics_dt, slew_rate_nm_s=cfg.slew_rate_nm_s,
                          cutoff_hz=cfg.cutoff_hz, effort_limits_nm=cfg.effort_limits_nm)
        self.controller = BatchedCartesianImpedanceController(self.num_envs, self.device, cfg.stiffness, cfg.damping, **processing)
        self.stop_controller = BatchedJointImpedanceController(self.num_envs, self.device, (0.,)*7, (20.,)*7, **processing)
        self.cart_lower = torch.tensor(cfg.cartesian_lower, device=self.device)
        self.cart_upper = torch.tensor(cfg.cartesian_upper, device=self.device)

    @property
    def action_dim(self):
        return 13

    def jacobian_base(self):
        value = self._asset.data.body_link_jacobian_w.torch
        # Fixed-base PhysX excludes root body from Jacobian rows. Shifted TCPs
        # are deliberately unsupported until their pose/Jacobian parity exists.
        jacobian = value[:, self.flange_id-1, :, self.dynamics_ids]
        rotation = matrix_from_quat(self._asset.data.root_quat_w.torch).transpose(1,2)
        return torch.cat((rotation@jacobian[:, :3], rotation@jacobian[:, 3:]), dim=1)

    def process_actions(self, actions):
        self._raw.copy_(actions)
        invalid = (~torch.isfinite(actions).all(1)
            | (actions[:,:3] < self.cart_lower-1e-6).any(1) | (actions[:,:3] > self.cart_upper+1e-6).any(1)
            | ((torch.linalg.vector_norm(actions[:,3:7], dim=1)-1).abs() > 1e-4)
            | (torch.linalg.vector_norm(actions[:,7:10], dim=1) > self.cfg.max_reference_linear_velocity+1e-7)
            | (torch.linalg.vector_norm(actions[:,10:13], dim=1) > self.cfg.max_reference_angular_velocity+1e-7))
        q, dq = self.check_state()
        self._latch(16*invalid.int(), q, dq)
        self._processed.copy_(torch.where(invalid[:,None], self._processed, actions))

    def allocate_trace(self, ticks):
        super().allocate_trace(ticks)
        self.trace.pop("q_ref")
        self.trace.pop("dq_ref")
        for key, size in (("pose",7),("twist",6),("reference",13),("pose_error",6),("wrench",6),("kinematic_twist",6)):
            self.trace[key] = torch.empty((ticks,self.num_envs,size), device=self.device)

    def apply_actions(self):
        q, dq = self.check_state()
        pose, twist = tcp_pose_base(self._env), tcp_twist_base(self._env)
        raw_c = self._asset.root_view.get_coriolis_and_centrifugal_compensation_forces()
        coriolis = (wp.to_torch(raw_c) if isinstance(raw_c,wp.array) else raw_c)[:,self.dynamics_ids]
        gravity = self._asset.data.gravity_compensation_forces.torch[:,self.dynamics_ids]
        jacobian = self.jacobian_base()
        torque = self.controller.compute(pose,twist,self._processed,jacobian,coriolis,gravity)
        # Faulted clones switch to a gravity-compensated velocity damper, with
        # independent resettable processing state. This is not a hardware stop.
        stop = self.stop_controller.compute(torch.nan_to_num(q),torch.nan_to_num(dq),self.fault_hold_q,
            torch.zeros_like(dq),torch.nan_to_num(coriolis),torch.nan_to_num(gravity))
        torque = torch.where((self.fault_code!=0)[:,None],stop,torque)
        self._latch(4*(~torch.isfinite(torque).all(1)).int(),q,dq)
        if self.trace is not None:
            if self.trace_index >= len(self.trace["q"]):
                raise RuntimeError("Physics telemetry capacity exhausted")
            values = {"q":q,"dq":dq,"pose":pose,"twist":twist,"reference":self._processed,
                "pose_error":self.controller.error,"wrench":self.controller.wrench,
                "kinematic_twist":(jacobian@dq.unsqueeze(-1)).squeeze(-1),
                **self.controller.components,"fault_code":self.fault_code,"contact_force_n":self.contact_force}
            values["applied"] = torque
            for key,value in values.items():
                self.trace[key][self.trace_index].copy_(value)
            self.trace_index += 1
        self._asset.set_joint_effort_target_index(target=torch.nan_to_num(torque),joint_ids=self.ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        # Joint base handles common fault metadata; restore Cartesian reference
        # immediately from measured FK, never from a nominal preview pose.
        super().reset(env_ids)
        self._processed[ids,:7] = tcp_pose_base(self._env)[ids]
        self._processed[ids,7:] = 0.
        self.fault_hold_q[ids] = self._asset.data.joint_pos.torch[ids][:,self.ids]
        self.stop_controller.reset(ids)
