"""Numerical contract tests for the deployment-oriented impedance controller."""

import importlib.util
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/franka_rl/franka_rl/tasks/manager_based/franka_rl/mdp/impedance_controller.py"
)
SPEC = importlib.util.spec_from_file_location("franka_impedance_controller", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


LOWER = (-2.610666015, -1.652481015, -2.610666015, -2.929015870, -2.588673015, 0.648913185, -2.745750015)
UPPER = (2.610666015, 1.652481015, 2.610666015, -0.264941230, 2.588673015, 4.412542815, 2.745750015)

def test_full_range_position_command_maps_endpoints_and_midpoint():
    lower = torch.tensor(LOWER)
    upper = torch.tensor(UPPER)
    actions = torch.tensor([[-2.0] * 7, [0.0] * 7, [2.0] * 7])

    bounded, mapped = MODULE.full_range_position_command(actions, lower, upper)

    assert torch.equal(bounded[0], torch.full((7,), -1.0))
    assert torch.equal(bounded[1], torch.zeros(7))
    assert torch.equal(bounded[2], torch.ones(7))
    assert torch.allclose(mapped[0], lower)
    assert torch.allclose(mapped[1], 0.5 * (lower + upper))
    assert torch.allclose(mapped[2], upper)


def test_impedance_torque_slew_and_gain_randomization():
    controller = MODULE.FrankyImpedanceController(
        128,
        7,
        "cpu",
        torch.float32,
        dt=0.001,
        nominal_stiffness=100.0,
        gain_alpha_range=(0.5, 2.0),
        position_error_clip=0.5,
        torque_slew_rate=1000.0,
        filter_cutoff_hz=100.0,
        lower=LOWER,
        upper=UPPER,
        limit_activation_distance=0.1,
        limit_stiffness=4.0,
        limit_damping=1.0,
        limit_max_torque=5.0,
    )
    controller.reset(slice(None))
    assert torch.all(controller.gain_alpha >= 0.5)
    assert torch.all(controller.gain_alpha <= 2.0)
    zeros = torch.zeros((128, 7))
    controller.compute(zeros, zeros, torch.full_like(zeros, 0.5), zeros, zeros)
    assert torch.all(torch.abs(controller.previous_limited_torque) <= 1.0 + 1.0e-6)
    previous = controller.previous_limited_torque.clone()
    controller.compute(zeros, zeros, torch.full_like(zeros, -0.5), zeros, zeros)
    assert torch.all(torch.abs(controller.previous_limited_torque - previous) <= 1.0 + 1.0e-6)


def test_incremental_position_command_bounds_step_and_projects_soft_limits():
    lower = torch.tensor(LOWER)
    upper = torch.tensor(UPPER)
    vmax = torch.tensor([0.435] * 4 + [0.522] * 3)
    reference = torch.stack((0.5 * (lower + upper), upper - 0.001))
    actions = torch.stack((torch.ones(7), torch.ones(7) * 2.0))

    bounded, target, increment, projection = MODULE.incremental_position_command(
        actions, reference, vmax, 0.02, lower, upper
    )

    assert torch.equal(bounded[0], torch.ones(7))
    assert torch.equal(bounded[1], torch.ones(7))
    assert torch.allclose(increment[0], vmax * 0.02)
    assert torch.all(target[1] <= upper)
    assert torch.allclose(target[1], upper)
    assert torch.all(projection[1] > 0.0)


def test_zero_increment_holds_any_reference():
    lower = torch.tensor(LOWER)
    upper = torch.tensor(UPPER)
    reference = 0.25 * lower + 0.75 * upper
    _, target, increment, projection = MODULE.incremental_position_command(
        torch.zeros((1, 7)), reference.unsqueeze(0), torch.ones(7), 0.02, lower, upper
    )
    assert torch.allclose(target[0], reference)
    assert torch.count_nonzero(increment) == 0
    assert torch.count_nonzero(projection) == 0


def test_velocity_reference_integrates_at_one_millisecond_and_projects_outward_motion():
    lower = torch.tensor(LOWER[:6])
    upper = torch.tensor(UPPER[:6])
    velocity = torch.tensor([[0.435, -0.435, 0.1, -0.1, 0.522, -0.522]])
    reference = 0.5 * (lower + upper)

    target, applied, projection = MODULE.integrate_velocity_reference(
        velocity, reference.unsqueeze(0), 0.001, lower, upper
    )

    assert torch.allclose(target, reference.unsqueeze(0) + velocity * 0.001)
    assert torch.equal(applied, velocity)
    assert torch.count_nonzero(projection) == 0

    at_bounds = torch.stack((upper, lower))
    outward = torch.stack((torch.ones(6), -torch.ones(6)))
    target, applied, projection = MODULE.integrate_velocity_reference(
        outward, at_bounds, 0.001, lower, upper
    )
    assert torch.allclose(target, at_bounds)
    assert torch.count_nonzero(applied) == 0
    assert torch.all(projection[0] > 0.0)
    assert torch.all(projection[1] < 0.0)

    inward = -outward
    target, applied, projection = MODULE.integrate_velocity_reference(
        inward, at_bounds, 0.001, lower, upper
    )
    assert torch.equal(applied, inward)
    assert torch.count_nonzero(projection) == 0


def test_impedance_controller_tracks_velocity_reference_and_defaults_to_zero():
    kwargs = dict(
        num_envs=1,
        num_joints=7,
        device="cpu",
        dtype=torch.float32,
        dt=0.001,
        nominal_stiffness=100.0,
        gain_alpha_range=None,
        position_error_clip=0.5,
        torque_slew_rate=1.0e9,
        filter_cutoff_hz=1.0e9,
        lower=LOWER,
        upper=UPPER,
        limit_activation_distance=0.0,
        limit_stiffness=0.0,
        limit_damping=0.0,
        limit_max_torque=0.0,
    )
    zeros = torch.zeros((1, 7))
    desired_velocity = torch.full_like(zeros, 0.2)

    velocity_controller = MODULE.FrankyImpedanceController(**kwargs)
    velocity_controller.reset(slice(None))
    velocity_torque = velocity_controller.compute(
        zeros, zeros, zeros, zeros, zeros, velocity_reference=desired_velocity
    )

    position_controller = MODULE.FrankyImpedanceController(**kwargs)
    position_controller.reset(slice(None))
    position_torque = position_controller.compute(zeros, zeros, zeros, zeros, zeros)

    assert torch.allclose(velocity_torque, torch.full_like(zeros, 4.0), atol=1.0e-4)
    assert torch.count_nonzero(position_torque) == 0


def test_impedance_controller_accepts_independent_hardware_damping():
    controller = MODULE.FrankyImpedanceController(
        num_envs=1,
        num_joints=7,
        device="cpu",
        dtype=torch.float32,
        dt=0.001,
        nominal_stiffness=200.0,
        nominal_damping=40.0,
        gain_alpha_range=None,
        position_error_clip=0.5,
        torque_slew_rate=1.0e9,
        filter_cutoff_hz=1.0e9,
        lower=LOWER,
        upper=UPPER,
        limit_activation_distance=0.0,
        limit_stiffness=0.0,
        limit_damping=0.0,
        limit_max_torque=0.0,
    )
    controller.reset(slice(None))
    zeros = torch.zeros((1, 7))
    torque = controller.compute(
        zeros, zeros, zeros, zeros, zeros,
        velocity_reference=torch.full_like(zeros, 0.2),
    )
    assert torch.allclose(torque, torch.full_like(zeros, 8.0), atol=1.0e-4)
