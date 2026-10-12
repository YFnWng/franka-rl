import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

from franka_motion.config import load_motion_suite, expand_trials
from franka_motion.impedance import JointImpedanceController
from franka_motion.impedance import TorqueCommandProcessor
from franka_motion.joint_planning import plan_joint_ptp, validate_joint_trajectory


def torch_controller():
    path = Path(__file__).resolve().parents[2]/"source/franka_rl/franka_rl/tasks/manager_based/franka_motion_control/mdp/controllers.py"
    spec = importlib.util.spec_from_file_location("motion_controller_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BatchedJointImpedanceController


def test_numpy_torch_controller_parity_and_effort_bounds():
    cls = torch_controller()
    batch = cls(2, "cpu", [100.]*7, [20.]*7)
    numpy = [JointImpedanceController([100.]*7, [20.]*7) for _ in range(2)]
    rng = np.random.default_rng(42)
    for _ in range(100):
        inputs = [rng.normal(size=(2, 7)).astype(np.float32) for _ in range(6)]
        batch.compute(*(torch.from_numpy(x) for x in inputs))
        for index in range(2):
            expected = numpy[index].compute(*(x[index] for x in inputs))
            for key in expected:
                np.testing.assert_allclose(batch.components[key][index].numpy(), expected[key], rtol=2e-5, atol=2e-5)
        assert (batch.components["applied"].abs() <= batch.effort).all()


def test_hold_has_immediate_gravity_without_filter_transient():
    controller = JointImpedanceController([100.]*7, [20.]*7)
    q = np.arange(7)*.1
    result = controller.compute(q, [0.]*7, q, [0.]*7, [0.]*7, [1.]*7)
    np.testing.assert_array_equal(result["applied"], 1.)
    np.testing.assert_array_equal(result["requested"], 0.)


def test_shared_previous_output_recurrence_reversal_seed_and_clamp():
    processor = TorqueCommandProcessor(effort_limits_nm=[2.]*7)
    batch = torch_controller()(2, "cpu", [0.]*7, [0.]*7, effort_limits_nm=[2.]*7)
    seed = np.linspace(-.7, .7, 7)
    processor.reset(seed)
    batch.reset(torch.tensor([0]), seed[None, :])
    expected = seed.copy()
    zeros = torch.zeros(2, 7)
    # Sustained saturation, reversal, unsaturated convergence, and gravity clamp.
    for raw in [20.]*10 + [-20.]*10 + [.2]*10:
        requested = np.full(7, raw)
        candidate = expected + np.clip(requested-expected, -1., 1.)
        expected += processor.alpha*(candidate-expected)
        result = processor.compute(requested, np.full(7, 10.))
        batch.compute(zeros, zeros, zeros, zeros,
                      torch.full((2, 7), raw), torch.full((2, 7), 10.))
        np.testing.assert_allclose(result["limited_candidate"], candidate)
        np.testing.assert_allclose(result["filtered"], expected)
        np.testing.assert_allclose(result["applied"], np.clip(expected+10., -2., 2.))
        np.testing.assert_allclose(batch.components["filtered"][0], expected, atol=2e-6)
    processor.reset()
    np.testing.assert_array_equal(processor.filtered, 0.)
    with pytest.raises(ValueError):
        processor.reset([float("nan")]*7)
    with pytest.raises(ValueError):
        batch.reset(torch.tensor([0]), torch.full((1,7), float("nan")))


def test_selective_controller_reset_does_not_affect_other_clone():
    controller = torch_controller()(2, "cpu", [100.]*7, [20.]*7)
    zeros, ones = torch.zeros(2, 7), torch.ones(2, 7)
    controller.compute(zeros, zeros, ones, zeros, zeros, zeros)
    before = {key: value.clone() for key, value in controller.components.items()}
    controller.reset(torch.tensor([0]))
    for key in before:
        assert not controller.components[key][0].any()
        torch.testing.assert_close(controller.components[key][1], before[key][1])


def test_ruckig_ptp_bounds_endpoints_and_sweep():
    suite = load_motion_suite(Path(__file__).resolve().parents[2]/"source/franka_motion/franka_motion/config/suites/joint_smoke.yaml")
    assert len(expand_trials(suite)) == 3
    spec = suite.trajectories["joint_small_ptp"].data
    q0 = suite.workspace.data["initial_postures_rad"]["deployment_home"]
    bounds = suite.workspace.data["joint_bounds_rad"]["commanded"]
    path = plan_joint_ptp(q0, spec["target_position_rad"], lower=bounds["lower"], upper=bounds["upper"],
        max_velocity=spec["max_velocity_rad_s"], max_acceleration=spec["max_acceleration_rad_s2"], max_jerk=spec["max_jerk_rad_s3"])
    assert validate_joint_trajectory(path)["passed"]
    np.testing.assert_allclose(path.evaluate(0.).position_rad, q0, atol=1e-12)
    final = path.evaluate(path.duration_s)
    np.testing.assert_allclose(final.position_rad, spec["target_position_rad"], atol=1e-12)
    np.testing.assert_allclose(final.velocity_rad_s, 0., atol=1e-12)
    with pytest.raises(ValueError):
        path.evaluate(float("nan"))
    with pytest.raises(ValueError):
        plan_joint_ptp(q0, [100.]*7, lower=bounds["lower"], upper=bounds["upper"],
            max_velocity=spec["max_velocity_rad_s"], max_acceleration=spec["max_acceleration_rad_s2"], max_jerk=spec["max_jerk_rad_s3"])


def test_zero_displacement_plan_has_finite_validation():
    path = plan_joint_ptp([0.]*7, [0.]*7, lower=[-1.]*7, upper=[1.]*7,
                          max_velocity=[.1]*7, max_acceleration=[.3]*7, max_jerk=[1.]*7)
    assert validate_joint_trajectory(path)["passed"]
    np.testing.assert_allclose(path.evaluate(0).velocity_rad_s, 0.)
