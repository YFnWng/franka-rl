"""Observations, rewards, and terminations for governed velocity actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import ee_position_error_b

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def accepted_velocity_action(
    env: ManagerBasedRLEnv,
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Return the bounded normalized action accepted by the action term."""

    action_term = env.action_manager.get_term(action_name)
    if not hasattr(action_term, "accepted_actions"):
        raise TypeError(f"Action term {action_name!r} does not expose accepted_actions")
    return action_term.accepted_actions


def governed_velocity_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Penalize the physical joint velocity target sent to the simulator."""

    action_term = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(action_term.applied_actions), dim=1)


def governed_velocity_rate_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Penalize policy-rate changes in the governed velocity target."""

    action_term = env.action_manager.get_term(action_name)
    delta = action_term.governed_actions - action_term.previous_governed_actions
    return torch.sum(torch.square(delta), dim=1)


def near_target_joint_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    hand_asset_cfg: SceneEntityCfg,
    joint_asset_cfg: SceneEntityCfg,
    sigma: float,
) -> torch.Tensor:
    """Penalize measured motion primarily when the end effector is near target."""

    position_error = ee_position_error_b(
        env, command_name=command_name, asset_cfg=hand_asset_cfg
    )
    gate = torch.exp(-torch.sum(torch.square(position_error), dim=1) / sigma)
    robot: Articulation = env.scene[joint_asset_cfg.name]
    joint_vel = robot.data.joint_vel.torch[:, joint_asset_cfg.joint_ids]
    return gate * torch.sum(torch.square(joint_vel), dim=1)


def velocity_action_fault(
    env: ManagerBasedRLEnv,
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Terminate an episode after a non-finite velocity command is rejected."""

    return env.action_manager.get_term(action_name).faulted
