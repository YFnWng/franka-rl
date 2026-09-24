"""Deterministic command replay installed after Isaac Sim has started."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch


class TargetReplayController:
    """Replace one initialized pose command term's sampler with saved targets."""

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
        if self.document.get("schema_version") != 1:
            raise ValueError(f"Unsupported target-set schema in {self.path}.")

        self._term: Any | None = None
        self._targets: torch.Tensor | None = None
        self._episode: torch.Tensor | None = None
        self._orientation: torch.Tensor | None = None

    def install(self, env: Any, command_name: str) -> None:
        """Install replay on a command term in an initialized environment."""
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
