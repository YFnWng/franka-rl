"""Deterministic command replay installed after Isaac Sim has started."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch


class TargetReplayController:
    """Replay paired targets and episode-start joint states.

    Schema version 1 files replay targets only. Version 2 files additionally
    replace the configured joint-reset event with normalized state samples.
    Both callbacks share one per-environment episode cursor. Isaac Lab applies
    reset events before resetting the command manager, so the state callback
    reads episode ``k`` and the target callback advances the cursor afterwards.
    """

    def __init__(self, path: Path | str, expected_sha256: str | None = None):
        self.path = Path(path).expanduser().resolve()
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError(
                f"Target-set hash mismatch for {self.path}: expected "
                f"{expected_sha256}, got {self.sha256}."
            )
        self.document = json.loads(raw)
        if self.document.get("schema_version") not in (1, 2):
            raise ValueError(f"Unsupported target-set schema in {self.path}.")

        self._term: Any | None = None
        self._targets: torch.Tensor | None = None
        self._episode: torch.Tensor | None = None
        self._orientation: torch.Tensor | None = None
        self._joint_position_samples: torch.Tensor | None = None
        self._joint_velocity_samples: torch.Tensor | None = None
        self._joint_names: tuple[str, ...] = ()

    @property
    def schema_version(self) -> int:
        return int(self.document["schema_version"])

    @property
    def replays_initial_joint_state(self) -> bool:
        return self._joint_position_samples is not None

    def install(
        self,
        env: Any,
        command_name: str,
        reset_event_name: str = "reset_arm",
    ) -> None:
        """Install replay callbacks in an initialized environment."""
        # Deliberately local: this method runs inside launch_simulation, after
        # SimulationApp is initialized.
        from isaaclab.utils.math import quat_from_euler_xyz, quat_unique

        term = env.command_manager.get_term(command_name)
        num_envs = env.num_envs
        if self.document.get("num_envs") != num_envs:
            raise ValueError(
                f"Target set has {self.document.get('num_envs')} environments; "
                f"the scene has {num_envs}."
            )

        targets = torch.tensor(self.document.get("targets"), dtype=torch.float32, device=env.device)
        if targets.ndim != 3 or targets.shape[0] != num_envs or targets.shape[2] != 3:
            raise ValueError(
                f"Target positions must have shape ({num_envs}, episodes, 3), "
                f"got {tuple(targets.shape)}."
            )
        for axis, name in enumerate(("pos_x", "pos_y", "pos_z")):
            low, high = getattr(term.cfg.ranges, name)
            values = targets[..., axis]
            if torch.any(values < low) or torch.any(values > high):
                raise ValueError(
                    f"Target-set {name} values fall outside environment range [{low}, {high}]."
                )

        euler = torch.tensor(
            [
                self._fixed_angle(term, "roll"),
                self._fixed_angle(term, "pitch"),
                self._fixed_angle(term, "yaw"),
            ],
            dtype=torch.float32,
            device=env.device,
        )
        quat = quat_from_euler_xyz(euler[0:1], euler[1:2], euler[2:3])[0]
        if term.cfg.make_quat_unique:
            quat = quat_unique(quat.unsqueeze(0))[0]

        self._term = term
        self._targets = targets
        self._episode = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        self._orientation = quat
        term._resample_command = self._resample_command

        initial_state = self.document.get("initial_joint_state")
        if self.schema_version == 1:
            return
        if not isinstance(initial_state, dict):
            raise ValueError("Schema version 2 replay set lacks initial_joint_state.")

        joint_names = initial_state.get("joint_names")
        if not isinstance(joint_names, list) or not joint_names or not all(
            isinstance(name, str) for name in joint_names
        ):
            raise ValueError("Replay joint_names must be a non-empty list of strings.")

        reset_cfg = env.event_manager.get_term_cfg(reset_event_name)
        asset_cfg = reset_cfg.params.get("asset_cfg")
        if asset_cfg is None:
            raise ValueError(f"Reset event {reset_event_name!r} has no asset_cfg.")
        robot = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, slice):
            resolved_names = tuple(robot.joint_names[joint_ids])
        else:
            resolved_names = tuple(robot.joint_names[index] for index in joint_ids)
        if tuple(joint_names) != resolved_names:
            raise ValueError(
                "Replay joint names do not match reset event: "
                f"expected {resolved_names}, got {tuple(joint_names)}."
            )

        expected_shape = (num_envs, targets.shape[1], len(joint_names))
        position_samples = torch.tensor(
            initial_state.get("position_unit_samples"),
            dtype=torch.float32,
            device=env.device,
        )
        velocity_samples = torch.tensor(
            initial_state.get("velocity_unit_samples"),
            dtype=torch.float32,
            device=env.device,
        )
        if tuple(position_samples.shape) != expected_shape:
            raise ValueError(
                f"Joint-position samples must have shape {expected_shape}, "
                f"got {tuple(position_samples.shape)}."
            )
        if tuple(velocity_samples.shape) != expected_shape:
            raise ValueError(
                f"Joint-velocity samples must have shape {expected_shape}, "
                f"got {tuple(velocity_samples.shape)}."
            )
        if torch.any((position_samples < 0.0) | (position_samples > 1.0)):
            raise ValueError("Joint-position unit samples must lie in [0, 1].")
        if torch.any((velocity_samples < 0.0) | (velocity_samples > 1.0)):
            raise ValueError("Joint-velocity unit samples must lie in [0, 1].")

        self._joint_names = tuple(joint_names)
        self._joint_position_samples = position_samples
        self._joint_velocity_samples = velocity_samples
        reset_cfg.func = self._reset_joints
        env.event_manager.set_term_cfg(reset_event_name, reset_cfg)

    @staticmethod
    def _fixed_angle(term: Any, name: str) -> float:
        value_range = getattr(term.cfg.ranges, name)
        if value_range[0] != value_range[1]:
            raise ValueError(
                f"Target replay requires a fixed {name} range; got {value_range}."
            )
        return float(value_range[0])

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if self._term is None or self._targets is None or self._episode is None or self._orientation is None:
            raise RuntimeError("Target replay is not installed.")

        if isinstance(env_ids, slice):
            resolved_ids = torch.arange(len(self._episode), device=self._episode.device)[env_ids]
        else:
            resolved_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._episode.device)
        episode_ids = self._episode[resolved_ids]
        capacity = self._targets.shape[1]
        if torch.any(episode_ids >= capacity):
            offending = resolved_ids[episode_ids >= capacity].tolist()
            raise RuntimeError(
                f"Target replay exhausted its {capacity} targets for environments {offending}."
            )

        self._term.pose_command_b[resolved_ids, :3] = self._targets[resolved_ids, episode_ids]
        self._term.pose_command_b[resolved_ids, 3:] = self._orientation
        self._episode[resolved_ids] += 1

    def _reset_joints(
        self,
        env: Any,
        env_ids: Sequence[int],
        position_range: tuple[float, float],
        velocity_range: tuple[float, float],
        asset_cfg: Any,
    ) -> None:
        if (
            self._episode is None
            or self._joint_position_samples is None
            or self._joint_velocity_samples is None
        ):
            raise RuntimeError("Initial-state replay is not installed.")

        if isinstance(env_ids, slice):
            resolved_ids = torch.arange(
                len(self._episode), device=self._episode.device
            )[env_ids]
        else:
            resolved_ids = torch.as_tensor(
                env_ids, dtype=torch.long, device=self._episode.device
            )
        episode_ids = self._episode[resolved_ids]
        capacity = self._joint_position_samples.shape[1]
        if torch.any(episode_ids >= capacity):
            offending = resolved_ids[episode_ids >= capacity].tolist()
            raise RuntimeError(
                f"Initial-state replay exhausted its {capacity} states for "
                f"environments {offending}."
            )

        robot = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        all_joints = isinstance(joint_ids, slice) and joint_ids == slice(None)
        iter_env_ids = resolved_ids if all_joints else resolved_ids[:, None]

        position_low, position_high = position_range
        position_unit = self._joint_position_samples[resolved_ids, episode_ids]
        joint_pos = robot.data.default_joint_pos.torch[
            iter_env_ids, joint_ids
        ].clone()
        joint_pos += position_low + position_unit * (
            position_high - position_low
        )

        velocity_low, velocity_high = velocity_range
        velocity_unit = self._joint_velocity_samples[resolved_ids, episode_ids]
        joint_vel = robot.data.default_joint_vel.torch[
            iter_env_ids, joint_ids
        ].clone()
        joint_vel += velocity_low + velocity_unit * (
            velocity_high - velocity_low
        )

        position_limits = robot.data.soft_joint_pos_limits.torch[
            iter_env_ids, joint_ids
        ]
        joint_pos.clamp_(position_limits[..., 0], position_limits[..., 1])
        velocity_limits = robot.data.soft_joint_vel_limits.torch[
            iter_env_ids, joint_ids
        ]
        joint_vel.clamp_(-velocity_limits, velocity_limits)

        robot.write_joint_position_to_sim_index(
            position=joint_pos,
            joint_ids=joint_ids,
            env_ids=resolved_ids,
        )
        robot.write_joint_velocity_to_sim_index(
            velocity=joint_vel,
            joint_ids=joint_ids,
            env_ids=resolved_ids,
        )
