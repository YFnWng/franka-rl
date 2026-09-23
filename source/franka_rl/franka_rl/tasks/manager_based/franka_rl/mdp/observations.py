from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

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