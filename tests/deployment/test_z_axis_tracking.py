"""Tests for the position plus tip-z-axis training task."""

import math

import pytest
import torch

from franka_rl.tasks.manager_based.franka_rl import mdp
from franka_rl.tasks.manager_based.franka_z_axis_tracking.z_axis_env_cfg import (
    Fr3FrankyIncremental6DPositionZAxisEnvCfg,
)
from franka_rl.tasks.manager_based.franka_velocity_impedance.rsl_rl_velocity_impedance_ppo_cfg import (
    VelocityImpedancePositionZAxisPPORunnerCfg,
)
from franka_rl.tasks.manager_based.franka_velocity_impedance.velocity_impedance_env_cfg import (
    Fr3FrankyVelocityImpedancePositionZAxisEnvCfg,
)


def test_z_axis_error_is_two_angle_minimal_rotation():
    angle = 0.3
    sine = math.sin(0.5 * angle)
    cosine = math.cos(0.5 * angle)
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    roll_target = torch.tensor([[sine, 0.0, 0.0, cosine]])
    pitch_target = torch.tensor([[0.0, sine, 0.0, cosine]])
    yaw_target = torch.tensor([[0.0, 0.0, sine, cosine]])

    torch.testing.assert_close(
        mdp.z_axis_error_from_quaternions(identity, roll_target),
        torch.tensor([[angle, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        mdp.z_axis_error_from_quaternions(identity, pitch_target),
        torch.tensor([[0.0, angle]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        mdp.z_axis_error_from_quaternions(identity, yaw_target),
        torch.tensor([[0.0, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_z_axis_task_is_separate_and_six_action(tmp_path, monkeypatch):
    asset = tmp_path / "test.usda"
    asset.write_text("#usda 1.0\n")
    monkeypatch.setenv("FRANKA_RL_FR3_USD", str(asset))

    cfg = Fr3FrankyIncremental6DPositionZAxisEnvCfg()

    assert cfg.control_contract == "fr3_franky_incremental_6d_position_z_axis_v1"
    assert cfg.actions.arm_action.class_type.endswith(
        ":FrankyIncremental6DImpedanceAction"
    )
    assert cfg.observations.policy.ee_z_axis_error.func is mdp.ee_z_axis_error_b
    assert cfg.rewards.z_axis_tracking.func is mdp.z_axis_tracking_exp
    assert cfg.rewards.fine_z_axis_tracking.func is mdp.z_axis_tracking_exp
    assert cfg.commands.ee_pose.ranges.roll == pytest.approx(
        (-math.pi / 6.0, math.pi / 6.0)
    )
    assert cfg.commands.ee_pose.ranges.pitch == pytest.approx(
        (5.0 * math.pi / 6.0, 7.0 * math.pi / 6.0)
    )
    assert cfg.commands.ee_pose.ranges.yaw == (0.0, 0.0)


def test_velocity_z_axis_task_preserves_velocity_contract(tmp_path, monkeypatch):
    asset = tmp_path / "test.usda"
    asset.write_text("#usda 1.0\n")
    monkeypatch.setenv("FRANKA_RL_FR3_USD", str(asset))

    cfg = Fr3FrankyVelocityImpedancePositionZAxisEnvCfg()

    assert cfg.control_contract == "fr3_joint_velocity_impedance_position_z_axis_31d_v1"
    assert cfg.required_action_delay_steps == 1
    assert cfg.evaluation_startup_auto_reset_prime is True
    assert cfg.actions.arm_action.class_type.endswith(
        ":FrankyVelocityReference6DImpedanceAction"
    )
    assert cfg.observations.policy.ee_z_axis_error.func is mdp.ee_z_axis_error_b
    assert cfg.rewards.z_axis_tracking.func is mdp.z_axis_tracking_exp
    assert cfg.rewards.z_axis_tracking.weight == pytest.approx(0.5)
    assert cfg.rewards.fine_z_axis_tracking.func is mdp.z_axis_tracking_exp
    assert cfg.rewards.fine_z_axis_tracking.weight == pytest.approx(0.5)
    assert cfg.commands.ee_pose.ranges.roll == pytest.approx(
        (-math.pi / 6.0, math.pi / 6.0)
    )
    assert cfg.commands.ee_pose.ranges.pitch == pytest.approx(
        (5.0 * math.pi / 6.0, 7.0 * math.pi / 6.0)
    )
    assert cfg.commands.ee_pose.ranges.yaw == (0.0, 0.0)

    runner = VelocityImpedancePositionZAxisPPORunnerCfg()
    assert runner.experiment_name == "fr3_velocity_impedance_position_z_axis_reach"
    assert tuple(runner.actor.distribution_cfg.initial_action) == (0.0,) * 6
