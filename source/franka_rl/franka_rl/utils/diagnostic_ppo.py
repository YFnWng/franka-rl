"""Fail-fast PPO instrumentation for diagnosing policy-update divergence."""

from __future__ import annotations

import json
import math
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from rsl_rl.algorithms import PPO


class DiagnosticPPO(PPO):
    """PPO with finite checks and compact numerical diagnostics.

    This intentionally supports the plain PPO configuration used by the FR3
    deployment-DR experiment (no RND or symmetry augmentation). It preserves
    the PPO objective but refuses to exponentiate a log likelihood ratio near
    float32 overflow and refuses to apply a non-finite gradient.
    """

    max_positive_log_ratio = 80.0
    action_saturation_threshold = 0.999

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError("DiagnosticPPO currently supports plain PPO without RND or symmetry")
        self.diagnostic_dir: Path | None = None
        self.diagnostic_update = 0

    def set_diagnostic_dir(self, directory: str | Path) -> None:
        self.diagnostic_dir = Path(directory)
        self.diagnostic_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _finite_scalar(value: torch.Tensor) -> float | None:
        detached = value.detach()
        if detached.numel() != 1 or not torch.isfinite(detached).item():
            return None
        return float(detached.item())

    @staticmethod
    def _tensor_stats(value: torch.Tensor) -> dict[str, Any]:
        detached = value.detach().float()
        finite = torch.isfinite(detached)
        stats: dict[str, Any] = {
            "shape": list(detached.shape),
            "finite_fraction": float(finite.float().mean().item()) if detached.numel() else 1.0,
        }
        if finite.any():
            selected = detached[finite]
            stats.update(
                min=float(selected.min().item()),
                max=float(selected.max().item()),
                mean=float(selected.mean().item()),
                max_abs=float(selected.abs().max().item()),
            )
        return stats

    def _distribution_state(self) -> dict[str, Any]:
        distribution = self.actor.distribution
        state: dict[str, Any] = {}
        for name in ("log_std_param", "std_param"):
            value = getattr(distribution, name, None)
            if value is not None:
                state[name] = value.detach().cpu().tolist()
        return state

    @classmethod
    def _log_ratio_overflow_risk(
        cls,
        log_ratio: torch.Tensor,
        advantages: torch.Tensor,
    ) -> bool:
        """Whether PPO must exponentiate a ratio near float32 overflow.

        For positive advantages, PPO clips the upper ratio and a very large
        positive log-ratio is harmless. For negative advantages, the upper
        side is intentionally not clipped and must remain guarded.
        """
        squeezed_advantages = torch.squeeze(advantages.detach())
        dangerous = (squeezed_advantages < 0.0) & (
            log_ratio.detach() >= cls.max_positive_log_ratio
        )
        return bool(dangerous.any().item())

    @staticmethod
    def _clipped_surrogate_loss(
        log_ratio: torch.Tensor,
        advantages: torch.Tensor,
        clip_param: float,
    ) -> torch.Tensor:
        """Evaluate the standard clipped PPO objective without unsafe exp()."""
        squeezed_advantages = torch.squeeze(advantages)
        log_lower = math.log1p(-clip_param)
        log_upper = math.log1p(clip_param)

        # This is algebraically equivalent to max(-A*r, -A*clip(r)):
        # positive A clips only the upper side; negative A clips only the
        # lower side. The remaining upper tail for negative A is guarded
        # before this method is called.
        log_ratio_for_positive_advantage = torch.clamp(log_ratio, max=log_upper)
        log_ratio_for_negative_advantage = torch.clamp(log_ratio, min=log_lower)
        effective_log_ratio = torch.where(
            squeezed_advantages >= 0.0,
            log_ratio_for_positive_advantage,
            log_ratio_for_negative_advantage,
        )
        # Select in log space before exponentiating. Exponentiating both
        # torch.where branches first would still create inf in an unused
        # branch and yield 0 * inf = NaN during autograd.
        effective_ratio = torch.exp(effective_log_ratio)
        return (-squeezed_advantages * effective_ratio).mean()

    def _write_failure(self, stage: str, minibatch: int, **values: Any) -> Path | None:
        payload = {
            "schema_version": 1,
            "stage": stage,
            "diagnostic_update": self.diagnostic_update,
            "minibatch": minibatch,
            "learning_rate": float(self.learning_rate),
            "distribution": self._distribution_state(),
            **values,
        }
        if self.diagnostic_dir is None:
            return None
        path = self.diagnostic_dir / "ppo_numerical_failure.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        return path

    def _fail(self, stage: str, minibatch: int, **values: Any) -> None:
        artifact = self._write_failure(stage, minibatch, **values)
        suffix = f" Diagnostic: {artifact}" if artifact is not None else ""
        raise FloatingPointError(
            f"DiagnosticPPO stopped before applying a corrupt update at {stage} "
            f"(update={self.diagnostic_update}, minibatch={minibatch}).{suffix}"
        )

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        mean_saturation = 0.0
        mean_actor_grad_norm = 0.0
        maximum_abs_log_ratio = 0.0
        maximum_positive_log_ratio = -math.inf
        maximum_abs_latent_mean = 0.0
        update_count = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for minibatch, batch in enumerate(generator):
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1.0e-8
                    )

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            entropy = self.actor.output_entropy
            distribution_params = self.actor.output_distribution_params
            latent_mean = distribution_params[0]
            log_ratio = actions_log_prob - torch.squeeze(batch.old_actions_log_prob)

            stats = {
                "actions": self._tensor_stats(batch.actions),
                "advantages": self._tensor_stats(batch.advantages),
                "old_log_prob": self._tensor_stats(batch.old_actions_log_prob),
                "new_log_prob": self._tensor_stats(actions_log_prob),
                "log_ratio": self._tensor_stats(log_ratio),
                "latent_mean": self._tensor_stats(latent_mean),
                "values": self._tensor_stats(values),
                "returns": self._tensor_stats(batch.returns),
            }
            checked = (actions_log_prob, values, entropy, log_ratio, latent_mean)
            if not all(torch.isfinite(value).all().item() for value in checked):
                self._fail("forward_non_finite", minibatch, tensors=stats)

            detached_log_ratio = log_ratio.detach()
            max_abs_log_ratio = float(detached_log_ratio.abs().max().item())
            max_positive_log_ratio = float(detached_log_ratio.max().item())
            maximum_abs_log_ratio = max(maximum_abs_log_ratio, max_abs_log_ratio)
            maximum_positive_log_ratio = max(maximum_positive_log_ratio, max_positive_log_ratio)
            maximum_abs_latent_mean = max(
                maximum_abs_latent_mean, float(latent_mean.detach().abs().max().item())
            )
            saturation = float(
                (batch.actions.detach().abs() >= self.action_saturation_threshold).float().mean().item()
            )
            mean_saturation += saturation
            # Only the positive tail can overflow exp(log_ratio). A very
            # negative ratio safely underflows toward zero and is handled by
            # PPO clipping, so it remains telemetry rather than an abort.
            if self._log_ratio_overflow_risk(log_ratio, batch.advantages):
                squeezed_advantages = torch.squeeze(batch.advantages.detach())
                dangerous = (squeezed_advantages < 0.0) & (
                    log_ratio.detach() >= self.max_positive_log_ratio
                )
                self._fail(
                    "log_ratio_overflow_risk",
                    minibatch,
                    threshold=self.max_positive_log_ratio,
                    dangerous_sample_count=int(dangerous.sum().item()),
                    dangerous_advantages=self._tensor_stats(squeezed_advantages[dangerous]),
                    dangerous_log_ratios=self._tensor_stats(log_ratio.detach()[dangerous]),
                    action_saturation_fraction=saturation,
                    tensors=stats,
                )

            with torch.inference_mode():
                kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                kl_mean = torch.mean(kl)
            if not torch.isfinite(kl_mean).item():
                self._fail("kl_non_finite", minibatch, tensors=stats, kl=self._tensor_stats(kl))
            mean_kl += float(kl_mean.item())

            surrogate_loss = self._clipped_surrogate_loss(
                log_ratio,
                batch.advantages,
                self.clip_param,
            )

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            if not torch.isfinite(loss).item():
                self._fail(
                    "loss_non_finite",
                    minibatch,
                    tensors=stats,
                    surrogate_loss=self._finite_scalar(surrogate_loss),
                    value_loss=self._finite_scalar(value_loss),
                    entropy=self._finite_scalar(entropy.mean()),
                )

            self.optimizer.zero_grad()
            loss.backward()
            bad_gradients = []
            for model_name, model in (("actor", self.actor), ("critic", self.critic)):
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                        bad_gradients.append(f"{model_name}.{name}")
            if bad_gradients:
                self._fail("gradient_non_finite", minibatch, tensors=stats, bad_gradients=bad_gradients)

            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm, error_if_nonfinite=True
            )
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm, error_if_nonfinite=True
            )
            mean_actor_grad_norm += float(actor_grad_norm.detach().item())
            self.optimizer.step()

            bad_parameters = [
                name
                for name, parameter in chain(self.actor.named_parameters(), self.critic.named_parameters())
                if not torch.isfinite(parameter).all().item()
            ]
            if bad_parameters:
                self._fail("parameter_non_finite_after_step", minibatch, bad_parameters=bad_parameters)

            mean_value_loss += float(value_loss.item())
            mean_surrogate_loss += float(surrogate_loss.item())
            mean_entropy += float(entropy.mean().item())
            update_count += 1

        self.storage.clear()
        self.diagnostic_update += 1
        denominator = max(update_count, 1)
        distribution = self.actor.distribution
        log_std = getattr(distribution, "log_std_param", None)
        return {
            "value": mean_value_loss / denominator,
            "surrogate": mean_surrogate_loss / denominator,
            "entropy": mean_entropy / denominator,
            "diag_kl": mean_kl / denominator,
            "diag_max_abs_log_ratio": maximum_abs_log_ratio,
            "diag_max_positive_log_ratio": maximum_positive_log_ratio,
            "diag_action_saturation_fraction": mean_saturation / denominator,
            "diag_max_abs_latent_mean": maximum_abs_latent_mean,
            "diag_actor_grad_norm": mean_actor_grad_norm / denominator,
            "diag_log_std_min": float(log_std.detach().min().item()) if log_std is not None else math.nan,
            "diag_log_std_max": float(log_std.detach().max().item()) if log_std is not None else math.nan,
        }
