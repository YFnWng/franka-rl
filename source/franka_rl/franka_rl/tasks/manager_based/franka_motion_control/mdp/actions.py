"""Physical references, explicit torque, bounded telemetry and latched fault samples."""
import torch
import warp as wp
from isaaclab.managers.action_manager import ActionTerm
from .controllers import BatchedJointImpedanceController


class JointMotionAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ids, names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if len(self.ids) != 7 or abs(env.physics_dt-.001) > 1e-10:
            raise ValueError("Joint motion requires seven ordered joints and 1 kHz physics")
        self.dynamics_ids = torch.tensor(self.ids, device=self.device) + self._asset.num_base_dofs
        self.flange_id = self._asset.find_bodies("fr3_flange")[0][0]
        self._raw = torch.zeros((self.num_envs, 14), device=self.device)
        self._processed = torch.zeros_like(self._raw)
        self.fault_code = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.first_fault_tick = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.first_fault_code = torch.zeros_like(self.fault_code)
        self.contact_force = torch.zeros(self.num_envs, device=self.device)
        self.first_fault_contact_force = torch.zeros_like(self.contact_force)
        self.first_fault_q = torch.zeros((self.num_envs, 7), device=self.device)
        self.first_fault_dq = torch.zeros_like(self.first_fault_q)
        self.first_fault_reference = torch.zeros((self.num_envs, 14), device=self.device)
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
