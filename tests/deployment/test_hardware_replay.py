"""Pure-Python metric tests for hardware-conditioned replay."""

import json

import pytest
import torch

from franka_rl.utils.hardware_replay import save_simulation_trace


def test_simulation_trace_uses_absolute_acceleration_and_excludes_approach(tmp_path):
    count = 3
    joints = torch.zeros((count, 7))
    velocity = torch.tensor([[0.0] * 7, [-1.0] * 7, [1.0] * 7])
    zeros = torch.zeros_like(joints)
    trace = {
        "joint_position": joints,
        "joint_velocity": velocity,
        "position_reference": joints,
        "target_velocity_reference": zeros,
        "applied_velocity_reference": zeros,
        "normalized_action": torch.zeros((count, 6)),
        "requested_torque": zeros,
        "slew_limited_torque": zeros,
        "filtered_torque": zeros,
        "applied_torque": zeros,
        "end_effector_position_base": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        "waypoint_index": torch.tensor([[0.0], [1.0], [1.0]]),
    }
    circle = {
        "center_m": [0.0, 0.0, 0.0],
        "orientation_rpy_deg": [0.0, 0.0, 0.0],
        "radius_m": 1.0,
    }

    summary = save_simulation_trace(
        trace, tmp_path, physics_dt_s=1.0, circle_path=circle
    )

    acceleration = summary["abs_estimated_acceleration_rad_s2"]
    assert acceleration["mean"] == 1.5
    assert acceleration["median"] == 1.5
    assert summary["circle_error_all_samples_m"]["mean"] == pytest.approx(1.0 / 3.0)
    assert summary["circle_error_excluding_initial_approach_m"]["mean"] == 0.0
    assert json.loads((tmp_path / "simulation_trace_summary.json").read_text()) == summary
    assert (tmp_path / "simulation_trace.csv").is_file()
