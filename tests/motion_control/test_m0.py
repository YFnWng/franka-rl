import copy
from dataclasses import replace
import math
import subprocess
import sys

import numpy as np
import pytest

from franka_motion.config import ControllerProfile, TrajectorySpec, _load, expand_trials, load_motion_suite, validate_suite
from franka_motion.contracts import CartesianReference, JointReference, validate_reference
from franka_motion.time_scaling import QuinticSegment, SmoothPhaseSchedule, validate_time_law
from franka_motion.trajectories import CircleTrajectory, sample_trajectory, validate_trajectory


def test_quintic_boundary_conditions():
    law = QuinticSegment([0., 1.], [2., -1.], 3., start_velocity=[.2, .1],
                         end_velocity=[0., .3], start_acceleration=[0., .1], end_acceleration=[.2, 0.])
    for t, p, v, a in ((0., [0., 1.], [.2, .1], [0., .1]), (3., [2., -1.], [0., .3], [.2, 0.])):
        sample = law.evaluate(t)
        np.testing.assert_allclose(sample.position, p, atol=1e-12)
        np.testing.assert_allclose(sample.velocity, v, atol=1e-12)
        np.testing.assert_allclose(sample.acceleration, a, atol=1e-12)


def test_quintic_analytic_extrema_and_rejection():
    law = QuinticSegment(0., 1., 2.)
    peaks = law.derivative_peaks()
    assert peaks["velocity"][0] == pytest.approx(1.875 / 2)
    assert peaks["acceleration"][0] == pytest.approx(10 / math.sqrt(3) / 4)
    assert peaks["jerk"][0] == pytest.approx(60 / 8)
    validate_time_law(law, max_velocity=1., max_acceleration=2., max_jerk=8.)
    with pytest.raises(ValueError, match="exceeds"):
        validate_time_law(law, max_velocity=.1, max_acceleration=2., max_jerk=8.)


@pytest.mark.parametrize("speed", [.2, .4, 1.])
def test_phase_turns_and_joins(speed):
    law = SmoothPhaseSchedule(speed, 2., 1., .7)
    assert float(law.evaluate(law.duration_s).position) == pytest.approx(.7 + 2 * math.pi)
    for t in (0., law.duration_s):
        sample = law.evaluate(t)
        np.testing.assert_allclose([sample.velocity, sample.acceleration, sample.jerk], 0., atol=1e-12)
    for t in (2., 2. + law.cruise_duration_s):
        left, right = law.evaluate(t - 1e-7), law.evaluate(t + 1e-7)
        for field in ("position", "velocity", "acceleration", "jerk"):
            np.testing.assert_allclose(getattr(left, field), getattr(right, field), atol=2e-6)


@pytest.mark.parametrize("bad_time", [-1., float("nan"), float("inf"), 100.])
def test_invalid_time(bad_time):
    with pytest.raises(ValueError):
        QuinticSegment(0., 1., 1.).evaluate(bad_time)


def test_impossible_schedule():
    with pytest.raises(ValueError, match="too few"):
        SmoothPhaseSchedule(10., 2.)


def test_circle_plane_and_analytic_derivatives():
    path = CircleTrajectory(center_m=[.4, 0., .3], radius_m=.05,
        plane_rpy_deg=[0., 90., 0.], schedule=SmoothPhaseSchedule(.4, 2.))
    for t in (.8, 3., path.duration_s - .8):
        sample = path.evaluate(t)
        assert sample.reference.position_m[0] == pytest.approx(.4)
        h = 1e-4
        before, after = path.evaluate(t - h), path.evaluate(t + h)
        for field, derivative in (("position_m", sample.reference.linear_velocity_m_s),
                ("linear_velocity_m_s", sample.acceleration_m_s2)):
            numerical = (np.array(getattr(after.reference, field)) - getattr(before.reference, field)) / (2 * h)
            np.testing.assert_allclose(numerical, derivative, atol=1e-8)
        numerical = (np.array(after.acceleration_m_s2) - before.acceleration_m_s2) / (2 * h)
        np.testing.assert_allclose(numerical, sample.jerk_m_s3, atol=1e-8)


def test_all_catalog_paths_and_sweep_cells():
    suite = load_motion_suite()
    trials = expand_trials(suite)
    assert len(trials) == 3 and len({t.trial_id for t in trials}) == 3
    assert [t.twist_feedforward for t in trials[:2]] == [False, True]
    limits = suite.workspace.data["reference_limits"]
    for trial in trials:
        path = trial.trajectory.build(trial.angular_speed_rad_s, suite.preview_initial_position_m)
        report = validate_trajectory(path, max_speed_m_s=limits["speed_m_s"],
            max_acceleration_m_s2=limits["acceleration_m_s2"], max_jerk_m_s3=limits["jerk_m_s3"],
            lower_m=suite.workspace.data["cartesian_bounds_m"]["lower"],
            upper_m=suite.workspace.data["cartesian_bounds_m"]["upper"], sample_hz=50.)
        assert report["passed"]
        samples = sample_trajectory(path, sample_hz=10.)
        assert np.isfinite(samples["jerk_m_s3"]).all()
        np.testing.assert_allclose(samples["position_m"][[0, -1]], [suite.preview_initial_position_m] * 2, atol=1e-12)
        for t in (path.approach_duration_s, path.track_start_s, path.track_start_s + path.path.duration_s):
            before, after = path.evaluate(t - 1e-7), path.evaluate(t + 1e-7)
            np.testing.assert_allclose(before.reference.position_m, after.reference.position_m, atol=1e-7)
            np.testing.assert_allclose(before.reference.linear_velocity_m_s, after.reference.linear_velocity_m_s, atol=1e-7)
            np.testing.assert_allclose(before.acceleration_m_s2, after.acceleration_m_s2, atol=1e-7)


def test_action_contracts_and_limits():
    joint = JointReference([0.] * 7, [.1] * 7)
    assert joint.as_action().shape == (14,)
    validate_reference(joint, lower_rad=[-1.] * 7, upper_rad=[1.] * 7, max_velocity_rad_s=[.2] * 7)
    with pytest.raises(ValueError):
        validate_reference(joint, max_velocity_rad_s=[.05] * 7)
    pose = CartesianReference([.4, 0., .4], [0., 0., 0., 1.], [.1, .2, .3])
    np.testing.assert_allclose(pose.as_action()[:7], [.4, 0., .4, 0., 0., 0., 1.])
    np.testing.assert_allclose(pose.as_action(twist_feedforward=False)[7:], 0.)
    assert pose.as_action().shape == (13,)
    with pytest.raises(ValueError):
        CartesianReference([.4, 0., .4], [0., 0., 0., 2.])
    with pytest.raises(ValueError):
        JointReference([float("nan")] * 7, [0.] * 7)


def test_hardware_unapproved_and_mode_mismatch():
    suite = load_motion_suite()
    with pytest.raises(ValueError, match="reviewed"):
        validate_suite(replace(suite, backend="hardware"))
    experiments = copy.deepcopy(suite.experiments)
    experiments[0]["controller"] = "joint_k100_d20"
    with pytest.raises(ValueError, match="Cartesian"):
        validate_suite(replace(suite, experiments=experiments))


@pytest.mark.parametrize("field,value", [("delay_steps", -1), ("reference_hz", 60), ("hardware_approved", True),
    ("units", {}), ("stiffness", [-1.] * 6), ("audit_pending", "missing")])
def test_invalid_controller(field, value):
    data = copy.deepcopy(load_motion_suite().controllers["cartesian_soft"].data)
    data[field] = value
    with pytest.raises(ValueError):
        ControllerProfile("bad", data)


def test_duplicate_unknown_yaml_and_bad_radius(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n")
    with pytest.raises(ValueError, match="Duplicate"):
        _load(path)
    data = copy.deepcopy(load_motion_suite().trajectories["circle_yz_small"].data)
    data["radius_m"] = -.1
    with pytest.raises(ValueError):
        TrajectorySpec("bad", data)
    data["radius_m"] = .1
    data["unknown"] = True
    with pytest.raises(ValueError, match="Invalid fields"):
        TrajectorySpec("bad", data)


def test_import_does_not_launch_simulator():
    import franka_motion
    code = "from franka_motion.config import load_motion_suite; import sys; load_motion_suite(); " \
        "assert not any(x in sys.modules for x in ['isaaclab','isaacsim','torch','franky'])"
    import os
    env = {**os.environ, "PYTHONPATH": str(__import__("pathlib").Path(franka_motion.__file__).parent.parent)}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
