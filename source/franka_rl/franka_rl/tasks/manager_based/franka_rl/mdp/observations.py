from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def ee_position_error_b(
    env: ManagerBasedEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """End-effector position error expressed in the robot base frame."""

    robot: Articulation = env.scene[asset_cfg.name]

    # The command generator already stores its position in the robot base frame.
    command = env.command_manager.get_command(command_name)
    target_pos_b = command[:, :3]

    # The articulation reports body and root poses in world coordinates.
    hand_pos_w = robot.data.body_pos_w.torch[:, asset_cfg.body_ids[0]]
    root_pos_w = robot.data.root_pos_w.torch
    root_quat_w = robot.data.root_quat_w.torch

    # Convert the current hand position from world to robot-base coordinates.
    hand_pos_b, _ = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        hand_pos_w,
    )

    return target_pos_b - hand_pos_b


def z_axis_error_from_quaternions(
    current_quat_xyzw: torch.Tensor,
    target_quat_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Return the minimal z-axis alignment rotation about current tip x/y.

    The target z-axis is expressed in the current tip frame. The returned two
    components are the x and y components of the shortest axis-angle rotation
    that aligns the current tip z-axis with the target z-axis. Rotation about
    z is intentionally unobserved because it does not change the tip z-axis.
    """

    z_axis = torch.zeros(
        (*target_quat_xyzw.shape[:-1], 3),
        dtype=target_quat_xyzw.dtype,
        device=target_quat_xyzw.device,
    )
    z_axis[..., 2] = 1.0
    target_z_b = quat_apply(target_quat_xyzw, z_axis)
    target_z_tip = quat_apply_inverse(current_quat_xyzw, target_z_b)

    sine = torch.linalg.vector_norm(target_z_tip[..., :2], dim=-1)
    cosine = torch.clamp(target_z_tip[..., 2], min=-1.0, max=1.0)
    angle = torch.atan2(sine, cosine)
    scale = angle / torch.clamp(sine, min=1.0e-8)
    error = torch.stack((-target_z_tip[..., 1], target_z_tip[..., 0]), dim=-1) * scale.unsqueeze(-1)

    # Exactly anti-parallel axes have no unique shortest rotation axis. Pick
    # current tip +x deterministically; sampled training targets stay far from
    # this singular case, but the fallback keeps the observation finite.
    anti_parallel = (sine <= 1.0e-8) & (cosine < 0.0)
    fallback = torch.zeros_like(error)
    fallback[..., 0] = torch.pi
    return torch.where(anti_parallel.unsqueeze(-1), fallback, error)


def ee_z_axis_error_b(
    env: ManagerBasedEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Two-angle tip z-axis error expressed about current tip x/y axes."""

    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_quat_b = command[:, 3:7]

    root_pos_w = robot.data.root_pos_w.torch
    root_quat_w = robot.data.root_quat_w.torch
    hand_pos_w = robot.data.body_pos_w.torch[:, asset_cfg.body_ids[0]]
    hand_quat_w = robot.data.body_quat_w.torch[:, asset_cfg.body_ids[0]]
    _, hand_quat_b = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        hand_pos_w,
        hand_quat_w,
    )
    return z_axis_error_from_quaternions(hand_quat_b, target_quat_b)
