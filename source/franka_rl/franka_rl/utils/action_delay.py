"""Policy-step action delay for fixed robustness evaluation scenarios."""

from __future__ import annotations

import gymnasium as gym
import torch


class FixedActionDelayWrapper(gym.Wrapper):
    """Delay batched actions by an exact number of policy steps.

    New episodes start with zero residual actions. Rows that auto-reset are
    cleared immediately after their terminal step, while other environments
    retain their independent histories.
    """

    def __init__(self, env: gym.Env, delay_steps: int):
        if not isinstance(delay_steps, int) or isinstance(delay_steps, bool) or delay_steps < 1:
            raise ValueError("delay_steps must be a positive integer.")
        super().__init__(env)
        self.delay_steps = delay_steps
        self._history: list[torch.Tensor] = []

    def reset(self, **kwargs):
        self._history.clear()
        return self.env.reset(**kwargs)

    def step(self, actions: torch.Tensor):
        if not self._history:
            self._history = [torch.zeros_like(actions) for _ in range(self.delay_steps)]
        elif self._history[0].shape != actions.shape:
            raise ValueError(
                f"Action shape changed from {tuple(self._history[0].shape)} to {tuple(actions.shape)}."
            )

        delayed_actions = self._history.pop(0)
        self._history.append(actions.clone())
        observations, rewards, terminated, truncated, extras = self.env.step(delayed_actions)

        dones = terminated.bool() | truncated.bool()
        if torch.any(dones):
            for buffered_actions in self._history:
                buffered_actions[dones] = 0.0
        return observations, rewards, terminated, truncated, extras
