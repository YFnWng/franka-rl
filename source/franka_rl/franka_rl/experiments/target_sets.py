"""Deterministic target-set generation for paired evaluations."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .suite_config import TargetReplaySettings


def ensure_target_set(
    path: Path,
    *,
    seed: int,
    num_envs: int,
    total_episodes: int,
    settings: TargetReplaySettings,
) -> dict[str, Any]:
    required = math.ceil(total_episodes / num_envs)
    capacity = required + settings.extra_episodes_per_env
    document = _generate_document(seed, num_envs, capacity, settings)
    serialized = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    expected_sha256 = hashlib.sha256(serialized).hexdigest()

    if path.is_file():
        existing = path.read_bytes()
        if hashlib.sha256(existing).hexdigest() != expected_sha256:
            raise ValueError(
                f"Existing target set does not match requested configuration: {path}"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(serialized)
        temporary.replace(path)

    return {
        "path": str(path),
        "sha256": expected_sha256,
        "seed": seed,
        "num_envs": num_envs,
        "episodes_per_env": capacity,
        "required_episodes_per_env": required,
    }


def _generate_document(
    seed: int,
    num_envs: int,
    capacity: int,
    settings: TargetReplaySettings,
) -> dict[str, Any]:
    generator = random.Random(seed)
    targets = []
    for _env_id in range(num_envs):
        environment_targets = []
        for _episode_id in range(capacity):
            environment_targets.append(
                [
                    generator.uniform(*settings.pos_x),
                    generator.uniform(*settings.pos_y),
                    generator.uniform(*settings.pos_z),
                ]
            )
        targets.append(environment_targets)
    return {
        "schema_version": 1,
        "seed": seed,
        "num_envs": num_envs,
        "episodes_per_env": capacity,
        "position_ranges": {
            "x": list(settings.pos_x),
            "y": list(settings.pos_y),
            "z": list(settings.pos_z),
        },
        "settings": asdict(settings),
        "targets": targets,
    }
