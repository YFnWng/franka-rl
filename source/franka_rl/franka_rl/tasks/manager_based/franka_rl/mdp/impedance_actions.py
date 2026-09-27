"""Deployment-oriented joint-position action with an explicit torque servo."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers.action_manager import ActionTerm

from .impedance_controller import (
    FrankyImpedanceController,
    full_range_position_command,
    incremental_position_command,
    integrate_velocity_reference,
)

if TYPE_CHECKING:
    from .impedance_actions_cfg import (
        FrankyImpedanceActionCfg,
        FrankyIncremental6DImpedanceActionCfg,
        FrankyIncrementalImpedanceActionCfg,
        FrankyVelocityReference6DImpedanceActionCfg,
    )


class FrankyImpedanceAction(ActionTerm):
    """Map full-range joint goals to direct held Franky impedance references."""

    cfg: FrankyImpedanceActionCfg

    def __init__(self, cfg: FrankyImpedanceActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=cfg.preserve_order)
        if len(self._joint_ids) != 7:
            raise ValueError(f"Franky impedance action requires seven arm joints, got {self._joint_names}")
        self._joint_ids_tensor = torch.tensor(self._joint_ids, device=self.device, dtype=torch.long)
        self._dynamics_ids = self._joint_ids_tensor + self._asset.num_base_dofs
        self._physics_dt = float(env.cfg.sim.dt)
        if abs(self._physics_dt - 0.001) > 1.0e-9:
            raise ValueError("Franky impedance action requires a 1 kHz physics/controller rate")

        shape = (self.num_envs, len(self._joint_ids))
        self._raw_actions = torch.zeros(shape, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._mapped_target = torch.zeros_like(self._raw_actions)
        self._previous_mapped_target = torch.zeros_like(self._raw_actions)
        self._action_clipping = torch.zeros(self.num_envs, device=self.device)
        self._peak_measured_speed = torch.zeros_like(self._raw_actions)
        self._window_peak_measured_speed = torch.zeros_like(self._raw_actions)
        self._peak_applied_torque = torch.zeros_like(self._raw_actions)
        self._faulted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._diagnostic_trace: dict[str, torch.Tensor] | None = None
        self._diagnostic_trace_index = 0
        self._diagnostic_env_id = 0
        self._diagnostic_trace_active = False

        dtype = self._raw_actions.dtype
        self._controller = FrankyImpedanceController(
            self.num_envs,
            len(self._joint_ids),
            self.device,
            dtype,
            dt=self._physics_dt,
            nominal_stiffness=cfg.nominal_stiffness,
            gain_alpha_range=cfg.gain_alpha_range,
            position_error_clip=cfg.position_error_clip,
            torque_slew_rate=cfg.torque_slew_rate,
            filter_cutoff_hz=cfg.command_filter_cutoff_hz,
            lower=cfg.model_lower,
            upper=cfg.model_upper,
            limit_activation_distance=cfg.joint_limit_activation_distance,
            limit_stiffness=cfg.joint_limit_stiffness,
            limit_damping=cfg.joint_limit_damping,
            limit_max_torque=cfg.joint_limit_max_torque,
        )
        self._soft_lower = torch.tensor(cfg.soft_lower, device=self.device, dtype=dtype)
        self._soft_upper = torch.tensor(cfg.soft_upper, device=self.device, dtype=dtype)
        self._action_center = 0.5 * (self._soft_lower + self._soft_upper)
        self._action_half_range = 0.5 * (self._soft_upper - self._soft_lower)

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Clipped normalized command whose full range maps to the soft limits."""

        return self._processed_actions

    @property
    def delay_fill_action(self) -> torch.Tensor:
        """Normalized command that holds each environment's current reset pose."""

        return self._processed_actions

    @property
    def mapped_target(self) -> torch.Tensor:
        return self._mapped_target

    @property
    def previous_mapped_target(self) -> torch.Tensor:
        return self._previous_mapped_target

    @property
    def command_difference(self) -> torch.Tensor:
        return self._mapped_target - self._previous_mapped_target

    @property
    def action_clipping(self) -> torch.Tensor:
        return self._action_clipping

    @property
    def reference_position(self) -> torch.Tensor:
        """Direct position reference held by the 1 kHz torque controller."""

        return self._mapped_target

    @property
    def applied_torque(self) -> torch.Tensor:
        return self._controller.applied_torque

    @property
    def gain_alpha(self) -> torch.Tensor:
        return self._controller.gain_alpha

    @property
    def peak_measured_speed(self) -> torch.Tensor:
        return self._peak_measured_speed

    @property
    def window_peak_measured_speed(self) -> torch.Tensor:
        return self._window_peak_measured_speed

    @property
    def peak_applied_torque(self) -> torch.Tensor:
        return self._peak_applied_torque

    @property
    def faulted(self) -> torch.Tensor:
        return self._faulted

    def start_diagnostic_trace(self, max_ticks: int, env_id: int = 0) -> None:
        """Enable a bounded, GPU-resident 1 kHz velocity-controller trace."""

        if not hasattr(self, "_target_reference_velocity") or self.action_dim != 6:
            raise TypeError("Diagnostic tracing is only supported by the 6D velocity-reference action")
        if max_ticks < 1:
            raise ValueError("max_ticks must be positive")
        if env_id < 0 or env_id >= self.num_envs:
            raise ValueError(f"env_id must be in [0, {self.num_envs}), got {env_id}")
        joint_shape = (max_ticks, 7)
        self._diagnostic_trace = {
            "joint_position": torch.empty(joint_shape, device=self.device),
            "joint_velocity": torch.empty(joint_shape, device=self.device),
            "position_reference": torch.empty(joint_shape, device=self.device),
            "target_velocity_reference": torch.empty(joint_shape, device=self.device),
            "applied_velocity_reference": torch.empty(joint_shape, device=self.device),
            "normalized_action": torch.empty((max_ticks, 6), device=self.device),
            "requested_torque": torch.empty(joint_shape, device=self.device),
            "slew_limited_torque": torch.empty(joint_shape, device=self.device),
            "filtered_torque": torch.empty(joint_shape, device=self.device),
            "applied_torque": torch.empty(joint_shape, device=self.device),
        }
        self._diagnostic_trace_index = 0
        self._diagnostic_env_id = env_id
        self._diagnostic_trace_active = True

    def stop_diagnostic_trace(self) -> None:
        """Stop appending samples while retaining the trace for export."""

        self._diagnostic_trace_active = False

    def diagnostic_trace(self) -> dict[str, torch.Tensor]:
        """Return the recorded trace prefix as detached CPU tensors."""

        if self._diagnostic_trace is None:
            raise RuntimeError("Diagnostic trace has not been started")
        count = min(self._diagnostic_trace_index, next(iter(self._diagnostic_trace.values())).shape[0])
        return {name: value[:count].detach().cpu() for name, value in self._diagnostic_trace.items()}

    def _record_diagnostic_tick(
        self,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        applied_velocity_reference: torch.Tensor,
    ) -> None:
        trace = self._diagnostic_trace
        if trace is None or not self._diagnostic_trace_active:
            return
        index = self._diagnostic_trace_index
        if index >= next(iter(trace.values())).shape[0]:
            self._diagnostic_trace_active = False
            return
        env_id = self._diagnostic_env_id
        trace["joint_position"][index].copy_(joint_position[env_id])
        trace["joint_velocity"][index].copy_(joint_velocity[env_id])
        trace["position_reference"][index].copy_(self._mapped_target[env_id])
        trace["target_velocity_reference"][index].zero_()
        trace["target_velocity_reference"][index, :6].copy_(self._target_reference_velocity[env_id])
        trace["applied_velocity_reference"][index].copy_(applied_velocity_reference[env_id])
        trace["normalized_action"][index].copy_(self._processed_actions[env_id])
        trace["requested_torque"][index].copy_(self._controller.requested_torque[env_id])
        trace["slew_limited_torque"][index].copy_(self._controller.previous_limited_torque[env_id])
        trace["filtered_torque"][index].copy_(self._controller.filtered_torque[env_id])
        trace["applied_torque"][index].copy_(self._controller.applied_torque[env_id])
        self._diagnostic_trace_index += 1

    def process_actions(self, actions: torch.Tensor) -> None:
        self._window_peak_measured_speed.zero_()
        self._raw_actions.copy_(actions)
        finite = torch.isfinite(actions).all(dim=1)
        self._faulted |= ~finite
        safe = torch.where(torch.isfinite(actions), actions, 0.0)
        self._previous_mapped_target.copy_(self._mapped_target)
        bounded, mapped = full_range_position_command(safe, self._soft_lower, self._soft_upper)
        self._processed_actions.copy_(bounded)
        self._action_clipping.copy_(torch.abs(safe - bounded).amax(dim=1))
        self._mapped_target.copy_(mapped)

    def apply_actions(self) -> None:
        joint_position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        joint_velocity = self._asset.data.joint_vel.torch[:, self._joint_ids]
        absolute_velocity = torch.abs(joint_velocity)
        self._peak_measured_speed.copy_(torch.maximum(self._peak_measured_speed, absolute_velocity))
        self._window_peak_measured_speed.copy_(torch.maximum(self._window_peak_measured_speed, absolute_velocity))

        if self.cfg.compensate_coriolis:
            coriolis_raw = self._asset.root_view.get_coriolis_and_centrifugal_compensation_forces()
            coriolis_all = wp.to_torch(coriolis_raw) if isinstance(coriolis_raw, wp.array) else coriolis_raw
            coriolis = coriolis_all[:, self._dynamics_ids]
        else:
            coriolis = torch.zeros_like(joint_position)
        if self.cfg.compensate_gravity:
            gravity = self._asset.data.gravity_compensation_forces.torch[:, self._dynamics_ids]
        else:
            gravity = torch.zeros_like(joint_position)

        torque = self._controller.compute(
            joint_position,
            joint_velocity,
            self._mapped_target,
            coriolis,
            gravity,
        )
        self._peak_applied_torque.copy_(torch.maximum(self._peak_applied_torque, torch.abs(torque)))
        self._asset.set_joint_effort_target_index(target=torque, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._action_clipping[env_ids] = 0.0
        self._faulted[env_ids] = False
        self._peak_measured_speed[env_ids] = 0.0
        self._window_peak_measured_speed[env_ids] = 0.0
        self._peak_applied_torque[env_ids] = 0.0
        joint_position = self._asset.data.joint_pos.torch[env_ids][:, self._joint_ids]
        self._mapped_target[env_ids] = joint_position
        self._previous_mapped_target[env_ids] = joint_position
        self._processed_actions[env_ids] = torch.clamp(
            (joint_position - self._action_center) / self._action_half_range,
            -1.0,
            1.0,
        )
        self._controller.reset(env_ids)


class FrankyIncrementalImpedanceAction(FrankyImpedanceAction):
    """Integrate bounded 50 Hz position increments into a held joint reference."""

    cfg: FrankyIncrementalImpedanceActionCfg

    def __init__(self, cfg: FrankyIncrementalImpedanceActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._policy_dt = float(env.step_dt)
        if abs(self._policy_dt - 0.02) > 1.0e-9:
            raise ValueError("Incremental Franky action requires a 50 Hz policy rate")
        self._max_reference_velocity = self._raw_actions.new_tensor(cfg.max_reference_velocity)
        self._max_reference_acceleration = self._raw_actions.new_tensor(cfg.max_reference_acceleration)
        if self._max_reference_velocity.shape != (self.action_dim,):
            raise ValueError("max_reference_velocity must have seven values")
        if self._max_reference_acceleration.shape != (self.action_dim,):
            raise ValueError("max_reference_acceleration must have seven values")
        if torch.any(self._max_reference_velocity <= 0.0):
            raise ValueError("max_reference_velocity must be positive")
        if torch.any(self._max_reference_acceleration <= 0.0):
            raise ValueError("max_reference_acceleration must be positive")
        self._reference_velocity = torch.zeros_like(self._raw_actions)
        self._previous_reference_velocity = torch.zeros_like(self._raw_actions)
        self._reference_projection = torch.zeros_like(self._raw_actions)
        self._delay_fill = torch.zeros_like(self._raw_actions)

    @property
    def delay_fill_action(self) -> torch.Tensor:
        """A delayed missing incremental action holds the current reference."""

        return self._delay_fill

    @property
    def normalized_reference_position(self) -> torch.Tensor:
        return torch.clamp((self._mapped_target - self._action_center) / self._action_half_range, -1.0, 1.0)

    @property
    def reference_velocity(self) -> torch.Tensor:
        return self._reference_velocity

    @property
    def previous_reference_velocity(self) -> torch.Tensor:
        return self._previous_reference_velocity

    @property
    def max_reference_velocity(self) -> torch.Tensor:
        return self._max_reference_velocity

    @property
    def max_reference_acceleration(self) -> torch.Tensor:
        return self._max_reference_acceleration

    @property
    def reference_projection(self) -> torch.Tensor:
        return self._reference_projection

    def process_actions(self, actions: torch.Tensor) -> None:
        self._window_peak_measured_speed.zero_()
        self._raw_actions.copy_(actions)
        finite = torch.isfinite(actions).all(dim=1)
        self._faulted |= ~finite
        safe = torch.where(torch.isfinite(actions), actions, 0.0)
        self._previous_mapped_target.copy_(self._mapped_target)
        self._previous_reference_velocity.copy_(self._reference_velocity)
        bounded, target, increment, projection = incremental_position_command(
            safe, self._mapped_target, self._max_reference_velocity, self._policy_dt,
            self._soft_lower, self._soft_upper,
        )
        self._processed_actions.copy_(bounded)
        self._action_clipping.copy_(torch.abs(safe - bounded).amax(dim=1))
        self._mapped_target.copy_(target)
        self._reference_velocity.copy_(increment / self._policy_dt)
        self._reference_projection.copy_(projection)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._processed_actions[env_ids] = 0.0
        self._reference_velocity[env_ids] = 0.0
        self._previous_reference_velocity[env_ids] = 0.0
        self._reference_projection[env_ids] = 0.0


class FrankyIncremental6DImpedanceAction(FrankyImpedanceAction):
    """Integrate six position increments and hold the joint-7 reset reference."""

    cfg: FrankyIncremental6DImpedanceActionCfg

    def __init__(self, cfg: FrankyIncremental6DImpedanceActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._policy_dt = float(env.step_dt)
        if abs(self._policy_dt - 0.02) > 1.0e-9:
            raise ValueError("Incremental Franky action requires a 50 Hz policy rate")
        policy_shape = (self.num_envs, 6)
        self._raw_actions = torch.zeros(policy_shape, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._max_reference_velocity = self._raw_actions.new_tensor(cfg.max_reference_velocity)
        self._max_reference_acceleration = self._raw_actions.new_tensor(cfg.max_reference_acceleration)
        if self._max_reference_velocity.shape != (6,):
            raise ValueError("max_reference_velocity must have six values")
        if self._max_reference_acceleration.shape != (6,):
            raise ValueError("max_reference_acceleration must have six values")
        if torch.any(self._max_reference_velocity <= 0.0):
            raise ValueError("max_reference_velocity must be positive")
        if torch.any(self._max_reference_acceleration <= 0.0):
            raise ValueError("max_reference_acceleration must be positive")
        self._reference_velocity = torch.zeros_like(self._raw_actions)
        self._previous_reference_velocity = torch.zeros_like(self._raw_actions)
        self._reference_projection = torch.zeros_like(self._raw_actions)
        self._delay_fill = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def delay_fill_action(self) -> torch.Tensor:
        return self._delay_fill

    @property
    def normalized_reference_position(self) -> torch.Tensor:
        return torch.clamp(
            (self._mapped_target[:, :6] - self._action_center[:6]) / self._action_half_range[:6],
            -1.0,
            1.0,
        )

    @property
    def reference_velocity(self) -> torch.Tensor:
        return self._reference_velocity

    @property
    def previous_reference_velocity(self) -> torch.Tensor:
        return self._previous_reference_velocity

    @property
    def max_reference_velocity(self) -> torch.Tensor:
        return self._max_reference_velocity

    @property
    def max_reference_acceleration(self) -> torch.Tensor:
        return self._max_reference_acceleration

    @property
    def reference_projection(self) -> torch.Tensor:
        return self._reference_projection

    def process_actions(self, actions: torch.Tensor) -> None:
        self._window_peak_measured_speed.zero_()
        self._raw_actions.copy_(actions)
        finite = torch.isfinite(actions).all(dim=1)
        self._faulted |= ~finite
        safe = torch.where(torch.isfinite(actions), actions, 0.0)
        self._previous_mapped_target.copy_(self._mapped_target)
        self._previous_reference_velocity.copy_(self._reference_velocity)
        bounded, target, increment, projection = incremental_position_command(
            safe,
            self._mapped_target[:, :6],
            self._max_reference_velocity,
            self._policy_dt,
            self._soft_lower[:6],
            self._soft_upper[:6],
        )
        self._processed_actions.copy_(bounded)
        self._action_clipping.copy_(torch.abs(safe - bounded).amax(dim=1))
        self._mapped_target[:, :6].copy_(target)
        self._reference_velocity.copy_(increment / self._policy_dt)
        self._reference_projection.copy_(projection)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._action_clipping[env_ids] = 0.0
        self._faulted[env_ids] = False
        self._peak_measured_speed[env_ids] = 0.0
        self._window_peak_measured_speed[env_ids] = 0.0
        self._peak_applied_torque[env_ids] = 0.0
        joint_position = self._asset.data.joint_pos.torch[env_ids][:, self._joint_ids]
        self._mapped_target[env_ids] = joint_position
        self._previous_mapped_target[env_ids] = joint_position
        self._reference_velocity[env_ids] = 0.0
        self._previous_reference_velocity[env_ids] = 0.0
        self._reference_projection[env_ids] = 0.0
        self._controller.reset(env_ids)


class FrankyVelocityReference6DImpedanceAction(FrankyImpedanceAction):
    """Track six velocity references with a 1 kHz-integrated position state."""

    cfg: FrankyVelocityReference6DImpedanceActionCfg

    def __init__(self, cfg: FrankyVelocityReference6DImpedanceActionCfg, env) -> None:
        super().__init__(cfg, env)
        if abs(float(env.step_dt) - 0.02) > 1.0e-9:
            raise ValueError("Velocity-reference Franky action requires a 50 Hz policy rate")
        policy_shape = (self.num_envs, 6)
        self._raw_actions = torch.zeros(policy_shape, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._max_reference_velocity = self._raw_actions.new_tensor(cfg.max_reference_velocity)
        self._max_reference_acceleration = self._raw_actions.new_tensor(cfg.max_reference_acceleration)
        if self._max_reference_velocity.shape != (6,):
            raise ValueError("max_reference_velocity must have six values")
        if self._max_reference_acceleration.shape != (6,):
            raise ValueError("max_reference_acceleration must have six values")
        if torch.any(self._max_reference_velocity <= 0.0):
            raise ValueError("max_reference_velocity must be positive")
        if torch.any(self._max_reference_acceleration <= 0.0):
            raise ValueError("max_reference_acceleration must be positive")
        self._target_reference_velocity = torch.zeros_like(self._raw_actions)
        self._reference_velocity = torch.zeros_like(self._raw_actions)
        self._previous_reference_velocity = torch.zeros_like(self._raw_actions)
        self._reference_projection = torch.zeros_like(self._raw_actions)
        self._delay_fill = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def delay_fill_action(self) -> torch.Tensor:
        return self._delay_fill

    @property
    def normalized_reference_position(self) -> torch.Tensor:
        return torch.clamp(
            (self._mapped_target[:, :6] - self._action_center[:6]) / self._action_half_range[:6],
            -1.0,
            1.0,
        )

    @property
    def reference_velocity(self) -> torch.Tensor:
        return self._reference_velocity

    @property
    def previous_reference_velocity(self) -> torch.Tensor:
        return self._previous_reference_velocity

    @property
    def max_reference_velocity(self) -> torch.Tensor:
        return self._max_reference_velocity

    @property
    def max_reference_acceleration(self) -> torch.Tensor:
        return self._max_reference_acceleration

    @property
    def reference_projection(self) -> torch.Tensor:
        return self._reference_projection

    def process_actions(self, actions: torch.Tensor) -> None:
        self._window_peak_measured_speed.zero_()
        self._raw_actions.copy_(actions)
        finite = torch.isfinite(actions).all(dim=1)
        self._faulted |= ~finite
        safe = torch.where(torch.isfinite(actions), actions, 0.0)
        bounded = torch.clamp(safe, -1.0, 1.0)
        self._previous_mapped_target.copy_(self._mapped_target)
        self._previous_reference_velocity.copy_(self._reference_velocity)
        self._processed_actions.copy_(bounded)
        self._action_clipping.copy_(torch.abs(safe - bounded).amax(dim=1))
        self._target_reference_velocity.copy_(bounded * self._max_reference_velocity)
        self._reference_projection.zero_()

    def apply_actions(self) -> None:
        target, applied_velocity, projection = integrate_velocity_reference(
            self._target_reference_velocity,
            self._mapped_target[:, :6],
            self._physics_dt,
            self._soft_lower[:6],
            self._soft_upper[:6],
        )
        self._mapped_target[:, :6].copy_(target)
        self._reference_velocity.copy_(applied_velocity)
        self._reference_projection.add_(projection)

        joint_position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        joint_velocity = self._asset.data.joint_vel.torch[:, self._joint_ids]
        absolute_velocity = torch.abs(joint_velocity)
        self._peak_measured_speed.copy_(torch.maximum(self._peak_measured_speed, absolute_velocity))
        self._window_peak_measured_speed.copy_(torch.maximum(self._window_peak_measured_speed, absolute_velocity))

        if self.cfg.compensate_coriolis:
            coriolis_raw = self._asset.root_view.get_coriolis_and_centrifugal_compensation_forces()
            coriolis_all = wp.to_torch(coriolis_raw) if isinstance(coriolis_raw, wp.array) else coriolis_raw
            coriolis = coriolis_all[:, self._dynamics_ids]
        else:
            coriolis = torch.zeros_like(joint_position)
        if self.cfg.compensate_gravity:
            gravity = self._asset.data.gravity_compensation_forces.torch[:, self._dynamics_ids]
        else:
            gravity = torch.zeros_like(joint_position)

        velocity_reference = torch.zeros_like(joint_velocity)
        velocity_reference[:, :6] = self._reference_velocity
        torque = self._controller.compute(
            joint_position,
            joint_velocity,
            self._mapped_target,
            coriolis,
            gravity,
            velocity_reference=velocity_reference,
        )
        self._record_diagnostic_tick(joint_position, joint_velocity, velocity_reference)
        self._peak_applied_torque.copy_(torch.maximum(self._peak_applied_torque, torch.abs(torque)))
        self._asset.set_joint_effort_target_index(target=torque, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._action_clipping[env_ids] = 0.0
        self._faulted[env_ids] = False
        self._peak_measured_speed[env_ids] = 0.0
        self._window_peak_measured_speed[env_ids] = 0.0
        self._peak_applied_torque[env_ids] = 0.0
        joint_position = self._asset.data.joint_pos.torch[env_ids][:, self._joint_ids]
        self._mapped_target[env_ids] = joint_position
        self._previous_mapped_target[env_ids] = joint_position
        self._target_reference_velocity[env_ids] = 0.0
        self._reference_velocity[env_ids] = 0.0
        self._previous_reference_velocity[env_ids] = 0.0
        self._reference_projection[env_ids] = 0.0
        self._controller.reset(env_ids)
