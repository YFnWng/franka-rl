"""Opt-in Isaac Lab v3.0.0-beta2.patch1 action term (Isaac Sim 6.0.1).

This module imports Isaac/Torch only when explicitly requested. Full runtime
validation must run on the simulation workstation; the RT host has no Isaac.
"""
from dataclasses import MISSING
import numpy as np
import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass
from . import load_config
from .simulation import SimulationBatch


class GovernedJointPositionAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if abs(env.physics_dt - 1/3000) > 1e-12 or env.cfg.decimation != 100:
            raise ValueError("governor reference adapter requires physics_dt=1/3000 and decimation=100")
        if getattr(env, '_physics_handles_decimation', False):
            raise ValueError("backend must call apply_actions once per physics step")
        self._joint_ids, names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if len(names) != 7 or names != cfg.joint_names:
            raise ValueError("exact seven-joint policy ordering required")
        core, self.config_sha256 = load_config(cfg.governor_config, allow_simulation=cfg.allow_simulation_fixture)
        defaults = self._asset.data.default_joint_pos.torch[:, self._joint_ids].detach().cpu().numpy()
        if not np.allclose(defaults, np.asarray(core.default_position, dtype=np.float32), rtol=0, atol=1e-7):
            raise ValueError("asset default positions do not match governor policy contract")
        self._governor = SimulationBatch(core, env.num_envs)
        self._raw = torch.zeros((env.num_envs, 7), device=env.device)
        self._processed = self._raw.clone()
        self._needs_reset = np.ones(env.num_envs, dtype=bool)
        self._physics_tick = 0
        self._pending = None
        self._pending_time = 0
        self._pending_rows = np.zeros(env.num_envs, dtype=bool)
        self._failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    @property
    def action_dim(self): return 7
    @property
    def raw_actions(self): return self._raw
    @property
    def processed_actions(self): return self._processed
    @property
    def failed(self): return self._failed

    def process_actions(self, actions):
        if actions.shape != self._raw.shape:
            raise ValueError("governor action shape mismatch")
        if self._pending is not None:
            raise RuntimeError("policy overwrote an unconsumed governor message")
        self._raw[:] = actions
        self._pending = actions.detach().cpu().numpy().copy()
        self._pending_rows[:] = True
        self._pending_time = self._physics_tick * 1000000 // 3

    def apply_actions(self):
        if getattr(self._env, "_physics_handles_decimation", False):
            raise ValueError("backend decimation bypasses per-physics governor updates")
        # Common integer timeline: 3 physics steps per governor tick, 100 per policy.
        if self._physics_tick % 3 == 0:
            now = self._physics_tick // 3 * 1000000
            q = self._asset.data.joint_pos.torch[:, self._joint_ids].detach().cpu().numpy()
            dq = self._asset.data.joint_vel.torch[:, self._joint_ids].detach().cpu().numpy()
            ids = np.flatnonzero(self._needs_reset).tolist()
            if ids:
                # Reset at this boundary; the first reference advance is next tick.
                self._governor.reset(ids, q, dq, now)
                self._needs_reset[ids] = False
            # Freshly reset rows cannot step at the reset timestamp. Advance all
            # other rows individually; the usual all-active path is one native batch.
            if not ids:
                self._governor.step(now, q, dq)
            else:
                from . import Feedback
                for i in range(self._env.num_envs):
                    if i in ids: continue
                    o = self._governor.outputs[i]
                    f = Feedback(); f.session = int(self._governor.sessions[i]); f.observed_ns = now
                    f.q, f.dq = q[i].tolist(), dq[i].tolist()
                    f.desired_q, f.desired_dq, f.desired_ddq = o.q, o.dq, o.ddq
                    self._governor.outputs[i] = self._step_one(i, now, f)
            if self._pending is not None:
                # Arrival at this boundary: submit now, consumed by next 1 ms step.
                # Same ordering applies at t=0 and between 30 Hz policy boundaries.
                from . import Message, Status
                for i in range(self._env.num_envs):
                    if not self._pending_rows[i] or self._governor.outputs[i].status != Status.RUNNING:
                        continue
                    m = Message();m.session=int(self._governor.sessions[i]);m.sequence=int(self._governor.sequences[i])+1
                    m.action=self._pending[i].tolist();m.observation_ns=self._pending_time;m.completed_ns=self._pending_time
                    if self._governor.batch.submit(i,m,now):self._governor.sequences[i]+=1
                    self._governor.outputs[i]=self._governor.batch.output(i)
                self._pending = None
                self._pending_rows[:] = False
            target = self._governor.positions
            self._processed[:] = torch.as_tensor(target, device=self._env.device, dtype=self._processed.dtype)
            self._failed[:] = torch.as_tensor(self._governor.failed, device=self._env.device)
        elif self._needs_reset.any():
            # Until the next 1 kHz boundary, newly reset rows hold their new measured
            # pose rather than a reference belonging to the previous episode.
            ids = np.flatnonzero(self._needs_reset).tolist()
            self._processed[ids] = self._asset.data.joint_pos.torch[ids][:, self._joint_ids]
        self._asset.set_joint_position_target_index(target=self._processed, joint_ids=self._joint_ids)
        self._physics_tick += 1

    def _step_one(self, i, now, feedback):
        return self._governor.batch.step_one(i, now, feedback)

    def reset(self, env_ids=None):
        ids = list(range(self._env.num_envs)) if env_ids is None else [int(i) for i in env_ids]
        self._needs_reset[ids] = True
        self._raw[ids] = 0
        self._failed[ids] = False
        self._pending_rows[ids] = False


@configclass
class GovernedJointPositionActionCfg(ActionTermCfg):
    class_type: type = GovernedJointPositionAction
    joint_names: list[str] = MISSING
    governor_config: str = MISSING
    allow_simulation_fixture: bool = False


def governor_failed(env, action_name='arm_action'):
    """Episode termination; the standalone core separately tests full stopping trajectories."""
    return env.action_manager.get_term(action_name).failed
