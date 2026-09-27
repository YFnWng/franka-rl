"""Configuration tests for deterministic waypoint-path evaluation."""

import math
from pathlib import Path

import pytest

from franka_rl.tasks.manager_based.franka_path_tracking.path_env_cfg import (
    Fr3FrankyIncremental6DCirclePathEnvCfg,
    Fr3FrankyIncremental6DPositionZAxisCirclePathEnvCfg,
)
from franka_rl.tasks.manager_based.franka_rl import mdp
from franka_rl.utils.paths import PathCatalog


def test_circle_path_catalog_geometry():
    catalog = PathCatalog.from_yaml()
    path = catalog.get("circle_xy")

    assert path.type == "circle"
    assert len(path.waypoints_m) == 24
    assert path.position_threshold_m == pytest.approx(0.01)
    assert path.waypoint_timeout_s == pytest.approx(1.0)
    assert path.center_m == pytest.approx((0.475, 0.0, 0.35))
    assert path.orientation_rpy_deg == pytest.approx((0.0, 0.0, 0.0))
    assert path.radius_m == pytest.approx(0.075)
    assert path.waypoint_count == 24
    assert path.target_orientation_xyzw == pytest.approx((0.0, 1.0, 0.0, 0.0))

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


def test_circle_orientation_rotates_the_waypoint_plane(tmp_path):
    catalog_path = tmp_path / "paths.yaml"
    catalog_path.write_text(
        """version: 1
paths:
  rotated:
    type: circle
    center_m: [0.0, 0.0, 0.0]
    orientation_rpy_deg: [90.0, 0.0, 0.0]
    radius_m: 1.0
    waypoint_count: 4
    phase_deg: 0.0
    target_orientation_xyzw: [2.0, 0.0, 0.0, 0.0]
    waypoint_timeout_s: 2.0
    position_threshold_m: 0.02
"""
    )

    path = PathCatalog.from_yaml(catalog_path).get("rotated")

    assert path.waypoints_m[0] == pytest.approx((1.0, 0.0, 0.0), abs=1.0e-12)
    assert path.waypoints_m[1] == pytest.approx((0.0, 0.0, 1.0), abs=1.0e-12)
    assert path.target_orientation_xyzw == pytest.approx((1.0, 0.0, 0.0, 0.0))


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
    assert command.fixed_quaternion_xyzw == pytest.approx((0.0, 1.0, 0.0, 0.0))
    assert cfg.episode_length_s == pytest.approx(25.0)
    assert cfg.terminations.reached_target is not None
    assert cfg.terminations.waypoint_path_failed is not None
    assert cfg.actions.arm_action.class_type.endswith(
        ":FrankyIncremental6DImpedanceAction"
    )

    cfg.configure_path("circle_yz")
    assert cfg.path_name == "circle_yz"
    assert cfg.path_metadata["name"] == "circle_yz"
    assert cfg.commands.ee_pose.waypoints_m[0] == pytest.approx(
        (0.475, 0.0, 0.50), abs=1.0e-12
    )
    assert cfg.commands.ee_pose.waypoints_m[6] == pytest.approx(
        (0.475, -0.15, 0.35), abs=1.0e-12
    )
    assert cfg.commands.ee_pose.fixed_quaternion_xyzw == pytest.approx(
        (0.0, 1.0, 0.0, 0.0)
    )


def test_z_axis_circle_path_preserves_policy_observation_contract(tmp_path, monkeypatch):
    asset = tmp_path / "test.usda"
    asset.write_text("#usda 1.0\n")
    monkeypatch.setenv("FRANKA_RL_FR3_USD", str(asset))

    cfg = Fr3FrankyIncremental6DPositionZAxisCirclePathEnvCfg()
    cfg.configure_path("circle_yz")

    assert cfg.evaluation_protocol == "waypoint_path"
    assert cfg.observations.policy.ee_z_axis_error.func is mdp.ee_z_axis_error_b
    assert cfg.commands.ee_pose.fixed_quaternion_xyzw == pytest.approx(
        (0.0, 1.0, 0.0, 0.0)
    )
    assert cfg.control_contract == "fr3_franky_incremental_6d_position_z_axis_v1"
