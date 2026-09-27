"""Bounded RSL-RL action distributions used by deployment-oriented policies."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from rsl_rl.modules.distribution import Distribution
from torch import nn
from torch.distributions import Normal


class SquashedGaussianDistribution(Distribution):
    """Diagonal Gaussian transformed through tanh into the open interval (-1, 1)."""

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.01,
        std_type: str = "log",
        initial_action: Sequence[float] | None = None,
        epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__(output_dim)
        if init_std <= 0.0:
            raise ValueError("init_std must be positive")
        if std_type not in ("scalar", "log"):
            raise ValueError("std_type must be 'scalar' or 'log'")

        initial_action = tuple(initial_action or (0.0,) * output_dim)
        if len(initial_action) != output_dim:
            raise ValueError(f"initial_action has {len(initial_action)} values; expected {output_dim}")
        initial_action_tensor = torch.tensor(initial_action, dtype=torch.float32)
        if torch.any(torch.abs(initial_action_tensor) >= 1.0):
            raise ValueError("initial_action values must lie strictly inside (-1, 1)")

        self.std_type = std_type
        self.epsilon = float(epsilon)
        self.register_buffer("initial_action", initial_action_tensor)
        if std_type == "scalar":
            self.std_param = nn.Parameter(init_std * torch.ones(output_dim))
        else:
            self.log_std_param = nn.Parameter(torch.log(init_std * torch.ones(output_dim)))
        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        if self.std_type == "scalar":
            std = torch.clamp(self.std_param, min=self.epsilon).expand_as(mlp_output)
        else:
            std = torch.exp(self.log_std_param).expand_as(mlp_output)
        self._distribution = Normal(mlp_output, std)

    def sample(self) -> torch.Tensor:
        return torch.tanh(self._distribution.sample())  # type: ignore[union-attr]

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        return nn.Tanh()

    @property
    def input_dim(self) -> int:
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        return self._distribution.mean  # type: ignore[union-attr]

    @property
    def std(self) -> torch.Tensor:
        return self._distribution.stddev  # type: ignore[union-attr]

    @property
    def entropy(self) -> torch.Tensor:
        # This latent-Gaussian entropy is a stable exploration proxy. PPO's
        # likelihood ratio below uses the exact transformed density.
        return self._distribution.entropy().sum(dim=-1)  # type: ignore[union-attr]

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        # KL is invariant under the shared bijective tanh transformation.
        return self.mean, self.std

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        bounded = torch.clamp(outputs, -1.0 + self.epsilon, 1.0 - self.epsilon)
        latent = torch.atanh(bounded)
        gaussian_log_prob = self._distribution.log_prob(latent)  # type: ignore[union-attr]
        log_jacobian = torch.log(1.0 - bounded.square() + self.epsilon)
        return (gaussian_log_prob - log_jacobian).sum(dim=-1)

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = Normal(old_mean, old_std)
        new_dist = Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        final_linear = next(module for module in reversed(list(mlp)) if isinstance(module, nn.Linear))
        nn.init.zeros_(final_linear.weight)
        home_latent = torch.atanh(self.initial_action.to(final_linear.bias.device))
        with torch.no_grad():
            final_linear.bias.copy_(home_latent.to(final_linear.bias.dtype))
