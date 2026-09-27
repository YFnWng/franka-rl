"""Configuration tests for deterministic waypoint-path evaluation."""

import math
from pathlib import Path

import pytest

from franka_rl.tasks.manager_based.franka_path_tracking.path_env_cfg import (
    Fr3FrankyIncremental6DCirclePathEnvCfg,
)
from franka_rl.utils.paths import PathCatalog


def test_circle_path_catalog_geometry():
    catalog = PathCatalog.from_yaml()
    path = catalog.get("circle_xy")

    assert path.type == "circle"
    assert len(path.waypoints_m) == 24
    assert path.position_threshold_m == pytest.approx(0.01)
    assert path.waypoint_timeout_s == pytest.approx(1.0)

    center = (0.475, 0.0, 0.35)
    radii = [
        math.hypot(point[0] - center[0], point[1] - center[1])
        for point in path.waypoints_m
    ]
    assert radii == pytest.approx([0.075] * len(radii))
    assert all(point[2] == pytest.approx(center[2]) for point in path.waypoints_m)

    first, second = path.waypoints_m[:2]
    spacing = math.dist(first, second)
    assert spacing > path.position_threshold_m


def test_circle_path_task_is_separate_from_random_point_task(tmp_path, monkeypatch):
    asset = tmp_path / "test.usda"
    asset.write_text("#usda 1.0\n")
    monkeypatch.setenv("FRANKA_RL_FR3_USD", str(asset))

    cfg = Fr3FrankyIncremental6DCirclePathEnvCfg()
    command = cfg.commands.ee_pose

    assert cfg.evaluation_protocol == "waypoint_path"
    assert cfg.path_name == "circle_xy"
    assert len(command.waypoints_m) == 24
    assert command.position_threshold_m == pytest.approx(0.01)
    assert command.resampling_time_range == (1.0, 1.0)
    assert cfg.episode_length_s == pytest.approx(25.0)
    assert cfg.terminations.reached_target is not None
    assert cfg.terminations.waypoint_path_failed is not None
    assert cfg.actions.arm_action.class_type.endswith(
        ":FrankyIncremental6DImpedanceAction"
    )
