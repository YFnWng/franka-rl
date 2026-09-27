"""Contract tests for the bounded RSL-RL action distribution."""

import pytest
import torch
from franka_rl.utils.rsl_rl_distributions import SquashedGaussianDistribution
from rsl_rl.modules import MLP


def test_squashed_gaussian_is_bounded_and_starts_at_home():
    home = (0.0, -0.475, 0.0, -0.570, 0.0, -0.510, 0.0)
    distribution = SquashedGaussianDistribution(
        7,
        init_std=0.15,
        std_type="log",
        initial_action=home,
    )
    mlp = MLP(3, 7, [8], "elu")
    distribution.init_mlp_weights(mlp)

    latent_mean = mlp(torch.randn(32, 3))
    deterministic_action = distribution.deterministic_output(latent_mean)
    assert torch.allclose(deterministic_action, torch.tensor(home).expand_as(deterministic_action))

    distribution.update(latent_mean)
    sampled_action = distribution.sample()
    assert torch.all(sampled_action > -1.0)
    assert torch.all(sampled_action < 1.0)
    assert torch.isfinite(distribution.log_prob(sampled_action)).all()


def test_squashed_gaussian_export_module_matches_deterministic_output():
    distribution = SquashedGaussianDistribution(2, initial_action=(0.0, 0.0))
    latent = torch.tensor([[-2.0, 2.0]])
    exported = distribution.as_deterministic_output_module()
    assert torch.equal(exported(latent), distribution.deterministic_output(latent))


def test_squashed_gaussian_reports_non_finite_standard_deviation():
    distribution = SquashedGaussianDistribution(2, initial_action=(0.0, 0.0))
    with torch.no_grad():
        distribution.log_std_param[0] = torch.nan

    with pytest.raises(FloatingPointError, match="log_std_param became non-finite"):
        distribution.update(torch.zeros(1, 2))
