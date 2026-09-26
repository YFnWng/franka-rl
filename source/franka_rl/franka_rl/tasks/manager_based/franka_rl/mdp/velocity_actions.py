"""Bounded joint-velocity actions for the FR3 deployment-oriented task.

The policy produces normalized target velocities at the policy rate. This
term applies a conservative acceleration limit and a position-dependent
braking envelope before forwarding velocity targets to PhysX. It also adds
the gravity effort computed by PhysX, allowing the implicit drive to be
configured as a pure velocity servo (zero stiffness).

The acceleration and servo parameters are provisional until the matching
Franky ``JointVelocityMotion`` response is measured on the real robot.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointVelocityAction

if TYPE_CHECKING:
    from .velocity_actions_cfg import GovernedJointVelocityActionCfg


class GovernedJointVelocityAction(JointVelocityAction):
    """Apply normalized joint-velocity targets through conservative limits."""

    cfg: "GovernedJointVelocityActionCfg"

    def __init__(self, cfg: "GovernedJointVelocityActionCfg", env):
        super().__init__(cfg, env)
        self._policy_dt = float(env.step_dt)
        self._normalized_clip = float(cfg.normalized_clip)
        self._max_acceleration = float(cfg.max_acceleration)
        self._braking_acceleration = float(cfg.braking_acceleration)

        if self._normalized_clip <= 0.0:
            raise ValueError("normalized_clip must be positive")
        if self._max_acceleration <= 0.0:
            raise ValueError("max_acceleration must be positive")
        if self._braking_acceleration <= 0.0:
            raise ValueError("braking_acceleration must be positive")
        if cfg.joint_limit_margin < 0.0:
            raise ValueError("joint_limit_margin must be non-negative")

        self._unclipped_actions = torch.zeros_like(self._raw_actions)
        self._governed_actions = torch.zeros_like(self._raw_actions)
        self._previous_governed_actions = torch.zeros_like(self._raw_actions)
        self._applied_actions = torch.zeros_like(self._raw_actions)
        self._faulted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._velocity_target_scale = torch.ones(
            (self.num_envs, 1), dtype=self._raw_actions.dtype, device=self.device
        )
        self._acceleration_scale = torch.ones_like(self._velocity_target_scale)
        self._validate_scale_range(
            cfg.velocity_target_scale_range, "velocity_target_scale_range"
        )
        self._validate_scale_range(cfg.acceleration_scale_range, "acceleration_scale_range")

        if isinstance(self._joint_ids, slice):
            self._joint_ids_tensor = torch.arange(
                self._asset.num_joints, dtype=torch.long, device=self.device
            )
        else:
            self._joint_ids_tensor = torch.as_tensor(
                self._joint_ids, dtype=torch.long, device=self.device
            )

    @property
    def unclipped_actions(self) -> torch.Tensor:
        """Raw policy output before finite checking and clipping."""

        return self._unclipped_actions

    @property
    def accepted_actions(self) -> torch.Tensor:
        """Finite normalized policy action accepted by the processor."""

        return self._raw_actions

    @property
    def governed_actions(self) -> torch.Tensor:
        """Velocity target after policy-rate acceleration limiting."""

        return self._governed_actions

    @property
    def previous_governed_actions(self) -> torch.Tensor:
        return self._previous_governed_actions

    @property
    def applied_actions(self) -> torch.Tensor:
        """Velocity target after the current position braking envelope."""

        return self._applied_actions

    @property
    def faulted(self) -> torch.Tensor:
        """Per-environment latch for non-finite policy commands."""

        return self._faulted

    @property
    def velocity_target_scale(self) -> torch.Tensor:
        """Realized per-environment multiplier on the target velocity."""

        return self._velocity_target_scale

    @property
    def acceleration_scale(self) -> torch.Tensor:
        """Realized per-environment multiplier on the acceleration limit."""

        return self._acceleration_scale

    @staticmethod
    def _validate_scale_range(value: tuple[float, float] | None, name: str) -> None:
        if value is None:
            return
        if len(value) != 2 or value[0] <= 0.0 or value[0] > value[1]:
            raise ValueError(
                f"{name} must be an ordered, strictly positive two-value range"
            )

    def process_actions(self, actions: torch.Tensor) -> None:
        self._unclipped_actions.copy_(actions)
        finite = torch.isfinite(actions).all(dim=1)
        self._faulted |= ~finite

        safe_actions = torch.where(torch.isfinite(actions), actions, 0.0)
        safe_actions = torch.clamp(
            safe_actions, -self._normalized_clip, self._normalized_clip
        )
        self._raw_actions.copy_(safe_actions)

        requested_velocity = (
            safe_actions * self._scale * self._velocity_target_scale + self._offset
        )
        max_delta = self._max_acceleration * self._policy_dt * self._acceleration_scale
        self._previous_governed_actions.copy_(self._governed_actions)
        delta = torch.clamp(
            requested_velocity - self._governed_actions,
            min=-max_delta,
            max=max_delta,
        )
        self._governed_actions.add_(delta)
        self._processed_actions.copy_(self._governed_actions)

    def apply_actions(self) -> None:
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        lower = limits[..., 0] + self.cfg.joint_limit_margin
        upper = limits[..., 1] - self.cfg.joint_limit_margin

        # Stopping-distance envelope: v^2 <= 2 a d.
        lower_distance = torch.clamp(joint_pos - lower, min=0.0)
        upper_distance = torch.clamp(upper - joint_pos, min=0.0)
        lower_velocity = -torch.sqrt(2.0 * self._braking_acceleration * lower_distance)
        upper_velocity = torch.sqrt(2.0 * self._braking_acceleration * upper_distance)
        self._applied_actions.copy_(
            torch.maximum(
                torch.minimum(self._governed_actions, upper_velocity),
                lower_velocity,
            )
        )

        self._asset.set_joint_velocity_target_index(
            target=self._applied_actions, joint_ids=self._joint_ids
        )

        if self.cfg.enable_gravity_compensation:
            gravity_ids = self._joint_ids_tensor + self._asset.num_base_dofs
            gravity = self._asset.data.gravity_compensation_forces.torch[:, gravity_ids]
            self._asset.set_joint_effort_target_index(
                target=gravity, joint_ids=self._joint_ids
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._unclipped_actions[env_ids] = 0.0
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._governed_actions[env_ids] = 0.0
        self._previous_governed_actions[env_ids] = 0.0
        self._applied_actions[env_ids] = 0.0
        self._faulted[env_ids] = False
        self._sample_scale(
            self._velocity_target_scale,
            env_ids,
            self.cfg.velocity_target_scale_range,
        )
        self._sample_scale(
            self._acceleration_scale,
            env_ids,
            self.cfg.acceleration_scale_range,
        )

    @staticmethod
    def _sample_scale(
        buffer: torch.Tensor,
        env_ids: Sequence[int] | slice,
        value_range: tuple[float, float] | None,
    ) -> None:
        if value_range is None:
            buffer[env_ids] = 1.0
            return
        selected = buffer[env_ids]
        samples = torch.empty_like(selected).uniform_(*value_range)
        buffer[env_ids] = samples

