"""Observations and safety terms for the Franky impedance action."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def held_position_command(env: ManagerBasedRLEnv, action_name: str = "arm_action") -> torch.Tensor:
    """Return the direct held position command in normalized policy coordinates."""

    return env.action_manager.get_term(action_name).processed_actions


def position_command_difference_l2(env: ManagerBasedRLEnv, action_name: str = "arm_action") -> torch.Tensor:
    """Penalize consecutive held joint-position commands in physical radians."""

    difference = env.action_manager.get_term(action_name).command_difference
    return torch.sum(torch.square(difference), dim=1)


def normalized_command_step_violation_l2(
    env: ManagerBasedRLEnv,
    max_velocity: tuple[float, ...],
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Penalize command increments above the 20-percent velocity envelope."""

    difference = env.action_manager.get_term(action_name).command_difference
    velocity = difference.new_tensor(max_velocity)
    allowed_step = velocity * float(env.step_dt)
    excess = torch.relu(torch.abs(difference) / allowed_step - 1.0)
    return torch.mean(torch.square(excess), dim=1)


def normalized_measured_velocity_violation_l2(
    env: ManagerBasedRLEnv,
    max_velocity: tuple[float, ...],
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Penalize 1 kHz peak measured velocity above the deployment envelope."""

    peak_velocity = env.action_manager.get_term(action_name).window_peak_measured_speed
    velocity = peak_velocity.new_tensor(max_velocity)
    excess = torch.relu(peak_velocity / velocity - 1.0)
    return torch.mean(torch.square(excess), dim=1)


def action_clipping_overshoot_l2(env: ManagerBasedRLEnv, action_name: str = "arm_action") -> torch.Tensor:
    """Penalize only the portion of a raw policy command outside [-1, 1]."""

    action = env.action_manager.get_term(action_name)
    overshoot = action.raw_actions - action.processed_actions
    return torch.mean(torch.square(overshoot), dim=1)


def impedance_action_fault(env: ManagerBasedRLEnv, action_name: str = "arm_action") -> torch.Tensor:
    """Terminate after a non-finite policy command."""

    return env.action_manager.get_term(action_name).faulted


def soft_joint_position_violation(
    env: ManagerBasedRLEnv,
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Check the same reviewed operating interval used by the reference governor."""

    robot: Articulation = env.scene[asset_cfg.name]
    position = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    return ((position < position.new_tensor(lower)) | (position > position.new_tensor(upper))).any(dim=1)


def measured_joint_velocity_violation(
    env: ManagerBasedRLEnv,
    max_velocity: tuple[float, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Check measured speed against the reviewed training safety cap."""

    robot: Articulation = env.scene[asset_cfg.name]
    velocity = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return (torch.abs(velocity) > velocity.new_tensor(max_velocity)).any(dim=1)


def normalized_position_reference(env: ManagerBasedRLEnv, action_name: str = "arm_action") -> torch.Tensor:
    """Return the integrated held reference in soft-range coordinates."""

    return env.action_manager.get_term(action_name).normalized_reference_position


def incremental_position_action(env: ManagerBasedRLEnv, action_name: str = "arm_action") -> torch.Tensor:
    """Return the last bounded dimensionless position-increment action."""

    return env.action_manager.get_term(action_name).processed_actions


def normalized_reference_acceleration_l2(
    env: ManagerBasedRLEnv, action_name: str = "arm_action"
) -> torch.Tensor:
    """Penalize physical reference-velocity changes normalized by reviewed limits."""

    action = env.action_manager.get_term(action_name)
    acceleration = (action.reference_velocity - action.previous_reference_velocity) / float(env.step_dt)
    return torch.mean(torch.square(acceleration / action.max_reference_acceleration), dim=1)


def reference_projection_overshoot_l2(
    env: ManagerBasedRLEnv, action_name: str = "arm_action"
) -> torch.Tensor:
    """Penalize commands that push the integrated reference past a soft limit."""

    projection = env.action_manager.get_term(action_name).reference_projection
    return torch.mean(torch.square(projection), dim=1)


def near_target_reference_velocity_l2(
    env: ManagerBasedRLEnv,
    threshold: float,
    action_name: str = "arm_action",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """Settle the reference velocity once the flange is near its target."""

    from .observations import ee_position_error_b

    error = torch.linalg.vector_norm(ee_position_error_b(env, command_name, asset_cfg), dim=1)
    action = env.action_manager.get_term(action_name)
    cost = torch.mean(torch.square(action.reference_velocity / action.max_reference_velocity), dim=1)
    return (error < threshold).to(cost.dtype) * cost
