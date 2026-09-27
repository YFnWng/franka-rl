"""Deterministic waypoint-path command for policy evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .path_command_cfg import WaypointPathCommandCfg


class WaypointPathCommand(UniformPoseCommand):
    """Advance through a fixed path when a waypoint is reached or times out."""

    cfg: WaypointPathCommandCfg

    def __init__(self, cfg: WaypointPathCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        waypoints = torch.tensor(cfg.waypoints_m, device=self.device, dtype=torch.float32)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or waypoints.shape[0] < 3:
            raise ValueError("WaypointPathCommand requires at least three 3D waypoints")
        if (
            cfg.waypoint_timeout_s <= 0.0
            or cfg.first_waypoint_timeout_s <= 0.0
            or cfg.position_threshold_m <= 0.0
        ):
            raise ValueError("waypoint timeout and position threshold must be positive")
        # UniformPoseCommand owns the resampling clock. Give it the longer
        # first-waypoint budget; later waypoints are advanced explicitly by
        # the per-waypoint step counter below.
        command_timeout_s = max(cfg.waypoint_timeout_s, cfg.first_waypoint_timeout_s)
        expected_timeout = (command_timeout_s, command_timeout_s)
        if cfg.resampling_time_range != expected_timeout:
            raise ValueError(
                "resampling_time_range must equal the fixed waypoint timeout "
                f"{expected_timeout}, got {cfg.resampling_time_range}"
            )

        self._waypoints = waypoints
        self._waypoint_count = int(waypoints.shape[0])
        self._waypoint_timeout_steps = max(
            1,
            math.ceil(cfg.waypoint_timeout_s / env.step_dt - 1.0e-9),
        )
        self._first_waypoint_timeout_steps = max(
            1,
            math.ceil(cfg.first_waypoint_timeout_s / env.step_dt - 1.0e-9),
        )
        self._fixed_quaternion = torch.tensor(
            cfg.fixed_quaternion_xyzw, device=self.device, dtype=torch.float32
        )
        if self._fixed_quaternion.shape != (4,):
            raise ValueError("fixed_quaternion_xyzw must contain four values")
        self._fixed_quaternion /= torch.linalg.vector_norm(self._fixed_quaternion)

        self.waypoint_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.waypoint_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.waypoints_reached = torch.zeros_like(self.waypoint_steps)
        self.waypoints_timed_out = torch.zeros_like(self.waypoint_steps)
        self.completed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._reached_now = torch.zeros_like(self.completed)
        self._current_min_error = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )

        shape = (self.num_envs, self._waypoint_count)
        self.waypoint_outcomes = torch.zeros(shape, dtype=torch.int8, device=self.device)
        self.waypoint_elapsed_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.waypoint_final_error_m = torch.full(shape, float("nan"), device=self.device)
        self.waypoint_min_error_m = torch.full(shape, float("nan"), device=self.device)

        self.last_episode_valid = torch.zeros_like(self.completed)
        self.last_episode_completed = torch.zeros_like(self.completed)
        self.last_episode_waypoints_reached = torch.zeros_like(self.waypoint_steps)
        self.last_episode_waypoints_timed_out = torch.zeros_like(self.waypoint_steps)
        self.last_episode_outcomes = torch.zeros_like(self.waypoint_outcomes)
        self.last_episode_elapsed_steps = torch.zeros_like(self.waypoint_elapsed_steps)
        self.last_episode_final_error_m = torch.full_like(
            self.waypoint_final_error_m, float("nan")
        )
        self.last_episode_min_error_m = torch.full_like(
            self.waypoint_min_error_m, float("nan")
        )

        self.metrics["path_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["waypoint_reach_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["waypoint_timeout_rate"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def waypoint_count(self) -> int:
        return self._waypoint_count

    @property
    def path_succeeded(self) -> torch.Tensor:
        return self.completed & (self.waypoints_timed_out == 0)

    @property
    def path_failed(self) -> torch.Tensor:
        return self.completed & (self.waypoints_timed_out > 0)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        had_episode = self.waypoint_index[ids] >= 0
        self.last_episode_valid[ids] = had_episode
        self.last_episode_completed[ids] = self.completed[ids]
        self.last_episode_waypoints_reached[ids] = self.waypoints_reached[ids]
        self.last_episode_waypoints_timed_out[ids] = self.waypoints_timed_out[ids]
        self.last_episode_outcomes[ids] = self.waypoint_outcomes[ids]
        self.last_episode_elapsed_steps[ids] = self.waypoint_elapsed_steps[ids]
        self.last_episode_final_error_m[ids] = self.waypoint_final_error_m[ids]
        self.last_episode_min_error_m[ids] = self.waypoint_min_error_m[ids]

        self.waypoint_index[ids] = -1
        self.waypoint_steps[ids] = 0
        self.waypoints_reached[ids] = 0
        self.waypoints_timed_out[ids] = 0
        self.completed[ids] = False
        self._reached_now[ids] = False
        self._current_min_error[ids] = float("inf")
        self.waypoint_outcomes[ids] = 0
        self.waypoint_elapsed_steps[ids] = 0
        self.waypoint_final_error_m[ids] = float("nan")
        self.waypoint_min_error_m[ids] = float("nan")
        return CommandTerm.reset(self, ids)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        active = (self.waypoint_index >= 0) & ~self.completed
        self.waypoint_steps[active] += 1
        self._current_min_error[active] = torch.minimum(
            self._current_min_error[active], self.metrics["position_error"][active]
        )
        self._reached_now.copy_(
            active & (self.metrics["position_error"] <= self.cfg.position_threshold_m)
        )
        timeout_steps = torch.where(
            self.waypoint_index == 0,
            self._first_waypoint_timeout_steps,
            self._waypoint_timeout_steps,
        )
        timed_out_now = active & ~self._reached_now & (self.waypoint_steps >= timeout_steps)
        self.time_left[self._reached_now | timed_out_now] = 0.0
        self._update_path_metrics()

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        initial = self.waypoint_index[ids] < 0
        transition_ids = ids[~initial]

        if transition_ids.numel() > 0:
            columns = self.waypoint_index[transition_ids]
            reached = self._reached_now[transition_ids]
            self.waypoint_outcomes[transition_ids, columns] = torch.where(
                reached,
                torch.ones_like(columns, dtype=torch.int8),
                torch.full_like(columns, 2, dtype=torch.int8),
            )
            self.waypoint_elapsed_steps[transition_ids, columns] = self.waypoint_steps[
                transition_ids
            ]
            self.waypoint_final_error_m[transition_ids, columns] = self.metrics[
                "position_error"
            ][transition_ids]
            self.waypoint_min_error_m[transition_ids, columns] = self._current_min_error[
                transition_ids
            ]
            self.waypoints_reached[transition_ids] += reached.long()
            self.waypoints_timed_out[transition_ids] += (~reached).long()

        next_index = self.waypoint_index[ids] + 1
        finished = next_index >= self._waypoint_count
        self.completed[ids] |= finished
        next_index = torch.clamp(next_index, max=self._waypoint_count - 1)
        self.waypoint_index[ids] = next_index
        self.waypoint_steps[ids] = 0
        self._current_min_error[ids] = float("inf")
        self._reached_now[ids] = False
        self.pose_command_b[ids, :3] = self._waypoints[next_index]
        self.pose_command_b[ids, 3:] = self._fixed_quaternion
        self.time_left[ids[finished]] = 1.0e9
        self._update_path_metrics()

    def _update_path_metrics(self) -> None:
        resolved = self.waypoints_reached + self.waypoints_timed_out
        denominator = torch.clamp(resolved, min=1).float()
        self.metrics["path_progress"].copy_(resolved.float() / self._waypoint_count)
        self.metrics["waypoint_reach_rate"].copy_(
            self.waypoints_reached.float() / denominator
        )
        self.metrics["waypoint_timeout_rate"].copy_(
            self.waypoints_timed_out.float() / denominator
        )
