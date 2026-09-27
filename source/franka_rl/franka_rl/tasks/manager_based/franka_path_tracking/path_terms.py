"""Termination terms for waypoint-path evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def waypoint_path_succeeded(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate successfully after every waypoint was reached before timeout."""

    return env.command_manager.get_term(command_name).path_succeeded


def waypoint_path_failed(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate after path completion if at least one waypoint timed out."""

    return env.command_manager.get_term(command_name).path_failed
