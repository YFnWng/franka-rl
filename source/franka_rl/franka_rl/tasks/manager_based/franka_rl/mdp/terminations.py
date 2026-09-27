from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from .observations import ee_position_error_b

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


class SustainedPositionSuccess(ManagerTermBase):
    """Terminate after remaining close to the target for several steps."""

    def __init__(
        self,
        cfg: TerminationTermCfg,
        env: ManagerBasedRLEnv,
    ):
        super().__init__(cfg, env)

        self._consecutive_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset counters only for environments whose episodes ended."""

        if env_ids is None:
            env_ids = slice(None)

        self._consecutive_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        distance_threshold: float,
        required_steps: int,
    ) -> torch.Tensor:
        position_error = ee_position_error_b(
            env,
            command_name=command_name,
            asset_cfg=asset_cfg,
        )

        distance = torch.linalg.vector_norm(position_error, dim=1)
        within_threshold = distance < distance_threshold

        # If within the threshold: increment.
        # Otherwise: reset to zero.
        self._consecutive_steps.mul_(within_threshold)
        self._consecutive_steps.add_(within_threshold)

        return self._consecutive_steps >= required_steps


class EvaluationStateMetrics(ManagerTermBase):
    """Capture post-physics trajectory metrics before automatic episode reset.

    The term never terminates an environment.  Evaluation code reads the
    cached tensors after ``env.step()``; keeping the cache unchanged in
    :meth:`reset` preserves the terminal sample even though manager-based
    environments reset completed environments inside that call.
    """

    def __init__(
        self,
        cfg: TerminationTermCfg,
        env: ManagerBasedRLEnv,
    ):
        super().__init__(cfg, env)
        self.position_error_m = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.joint_limit_margin_rad = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.joint_velocity_ratio = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.command_difference_norm_rad = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.action_clipping = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.applied_torque_norm_nm = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._never_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Retain the last pre-reset sample for the evaluator."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        hand_asset_cfg: SceneEntityCfg,
        joint_asset_cfg: SceneEntityCfg,
        max_joint_velocity: tuple[float, ...] | None = None,
        action_name: str | None = None,
    ) -> torch.Tensor:
        position_error = ee_position_error_b(
            env,
            command_name=command_name,
            asset_cfg=hand_asset_cfg,
        )
        self.position_error_m.copy_(torch.linalg.vector_norm(position_error, dim=1))

        robot: Articulation = env.scene[joint_asset_cfg.name]
        joint_pos = robot.data.joint_pos.torch[:, joint_asset_cfg.joint_ids]
        limits = robot.data.soft_joint_pos_limits.torch[:, joint_asset_cfg.joint_ids]
        lower_margin = joint_pos - limits[..., 0]
        upper_margin = limits[..., 1] - joint_pos
        self.joint_limit_margin_rad.copy_(torch.minimum(lower_margin, upper_margin).amin(dim=1))

        if max_joint_velocity is None:
            self.joint_velocity_ratio.fill_(float("nan"))
        else:
            joint_velocity = robot.data.joint_vel.torch[:, joint_asset_cfg.joint_ids]
            velocity_limit = joint_velocity.new_tensor(max_joint_velocity)
            action = env.action_manager.get_term(action_name) if action_name else None
            peak_speed = getattr(action, "peak_measured_speed", None)
            measured_speed = torch.abs(joint_velocity) if peak_speed is None else peak_speed
            self.joint_velocity_ratio.copy_((measured_speed / velocity_limit).amax(dim=1))

        if action_name is None:
            self.command_difference_norm_rad.fill_(float("nan"))
            self.action_clipping.fill_(float("nan"))
            self.applied_torque_norm_nm.fill_(float("nan"))
        else:
            action = env.action_manager.get_term(action_name)
            command_difference = getattr(action, "command_difference", None)
            action_clipping = getattr(action, "action_clipping", None)
            applied_torque = getattr(action, "peak_applied_torque", None)
            if command_difference is None:
                self.command_difference_norm_rad.fill_(float("nan"))
            else:
                self.command_difference_norm_rad.copy_(torch.linalg.vector_norm(command_difference, dim=1))
            if action_clipping is None:
                self.action_clipping.fill_(float("nan"))
            else:
                self.action_clipping.copy_(action_clipping)
            if applied_torque is None:
                self.applied_torque_norm_nm.fill_(float("nan"))
            else:
                self.applied_torque_norm_nm.copy_(torch.linalg.vector_norm(applied_torque, dim=1))

        return self._never_done


def non_finite_joint_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate environments containing NaN or infinite arm state."""

    robot: Articulation = env.scene[asset_cfg.name]

    joint_pos = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    joint_vel = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]

    position_is_finite = torch.isfinite(joint_pos).all(dim=1)
    velocity_is_finite = torch.isfinite(joint_vel).all(dim=1)

    return ~(position_is_finite & velocity_is_finite)
