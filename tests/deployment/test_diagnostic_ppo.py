"""Numerical diagnostic contract tests for deployment-DR PPO."""

from types import SimpleNamespace

import pytest
import torch

from franka_rl.tasks.manager_based.franka_incremental_impedance.rsl_rl_incremental_impedance_ppo_cfg import (
    Incremental6DDeploymentDRPPORunnerCfg,
)
from franka_rl.utils.diagnostic_ppo import DiagnosticPPO


def test_deployment_dr_uses_fixed_conservative_diagnostic_ppo():
    cfg = Incremental6DDeploymentDRPPORunnerCfg()

    assert cfg.algorithm.class_name.endswith(":DiagnosticPPO")
    assert cfg.algorithm.schedule == "fixed"
    assert cfg.algorithm.learning_rate == 3.0e-4
    assert cfg.save_interval == 5


def test_diagnostic_failure_artifact_is_atomic_and_informative(tmp_path):
    diagnostic = DiagnosticPPO.__new__(DiagnosticPPO)
    diagnostic.diagnostic_dir = tmp_path
    diagnostic.diagnostic_update = 7
    diagnostic.learning_rate = 3.0e-4
    diagnostic.actor = SimpleNamespace(
        distribution=SimpleNamespace(log_std_param=torch.tensor([-1.0, -0.5]))
    )

    path = diagnostic._write_failure(
        "log_ratio_overflow_risk",
        3,
        threshold=80.0,
        tensors={"log_ratio": {"max_abs": 91.0}},
    )

    assert path == tmp_path / "ppo_numerical_failure.json"
    text = path.read_text()
    assert '"stage": "log_ratio_overflow_risk"' in text
    assert '"diagnostic_update": 7' in text
    assert '"log_std_param"' in text
    assert not (tmp_path / "ppo_numerical_failure.json.tmp").exists()


def test_tensor_stats_preserve_non_finite_fraction():
    stats = DiagnosticPPO._tensor_stats(torch.tensor([1.0, float("nan"), float("inf")]))

    assert stats["finite_fraction"] == pytest.approx(1.0 / 3.0)
    assert stats["min"] == 1.0
    assert stats["max"] == 1.0


def test_log_ratio_guard_is_based_on_positive_tail():
    assert DiagnosticPPO.max_positive_log_ratio == 80.0

    harmless_negative_tail = torch.tensor([-91.0, -2.0, 1.0])
    dangerous_positive_tail = torch.tensor([-2.0, 81.0])

    assert not DiagnosticPPO._log_ratio_overflow_risk(
        harmless_negative_tail,
        torch.tensor([-1.0, -1.0, -1.0]),
    )
    assert not DiagnosticPPO._log_ratio_overflow_risk(
        dangerous_positive_tail,
        torch.tensor([-1.0, 1.0]),
    )
    assert DiagnosticPPO._log_ratio_overflow_risk(
        dangerous_positive_tail,
        torch.tensor([1.0, -1.0]),
    )


def test_stable_surrogate_matches_standard_ppo_in_normal_range():
    log_ratio = torch.tensor([-1.0, -0.1, 0.1, 1.0])
    advantages = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])
    ratio = torch.exp(log_ratio)
    standard = torch.maximum(
        -advantages.squeeze() * ratio,
        -advantages.squeeze() * ratio.clamp(0.8, 1.2),
    ).mean()

    stable = DiagnosticPPO._clipped_surrogate_loss(log_ratio, advantages, 0.2)

    assert stable.item() == pytest.approx(standard.item())


def test_stable_surrogate_does_not_exponentiate_clipped_positive_advantage_tail():
    log_ratio = torch.tensor([102.0], requires_grad=True)
    loss = DiagnosticPPO._clipped_surrogate_loss(
        log_ratio,
        torch.tensor([[1.0]]),
        0.2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(-1.2)
    assert torch.isfinite(log_ratio.grad).all()
    assert log_ratio.grad.item() == 0.0
