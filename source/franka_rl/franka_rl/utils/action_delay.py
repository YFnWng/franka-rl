"""Policy-step action delay for fixed robustness evaluation scenarios."""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import torch


ActionProvider = Callable[[], torch.Tensor]


def action_term_delay_fill_provider(env: gym.Env, action_name: str = "arm_action") -> ActionProvider | None:
    """Return a provider for an action term's reset-safe delay fill command.

    Action terms with non-zero neutral commands can expose ``delay_fill_action``.
    Other tasks retain the legacy all-zero fill behavior.
    """

    action_term = env.unwrapped.action_manager.get_term(action_name)
    if not hasattr(action_term, "delay_fill_action"):
        return None
    return lambda: action_term.delay_fill_action


def _fill_actions_like(actions: torch.Tensor, provider: ActionProvider | None) -> torch.Tensor:
    if provider is None:
        return torch.zeros_like(actions)
    fill_actions = provider()
    if fill_actions.shape != actions.shape:
        raise ValueError(
            f"Delay fill action shape {tuple(fill_actions.shape)} does not match "
            f"policy action shape {tuple(actions.shape)}."
        )
    return fill_actions.to(device=actions.device, dtype=actions.dtype).clone()


class FixedActionDelayWrapper(gym.Wrapper):
    """Delay batched actions by an exact number of policy steps.

    New episodes start with the action supplied by ``initial_action_provider``.
    If no provider is supplied, the legacy all-zero action is used. Rows that
    auto-reset are refilled immediately after their terminal step, while other
    environments retain their independent histories.
    """

    def __init__(
        self,
        env: gym.Env,
        delay_steps: int,
        initial_action_provider: ActionProvider | None = None,
    ):
        if not isinstance(delay_steps, int) or isinstance(delay_steps, bool) or delay_steps < 1:
            raise ValueError("delay_steps must be a positive integer.")
        super().__init__(env)
        self.delay_steps = delay_steps
        self.initial_action_provider = initial_action_provider
        self._history: list[torch.Tensor] = []

    def reset(self, **kwargs):
        self._history.clear()
        return self.env.reset(**kwargs)

    def step(self, actions: torch.Tensor):
        if not self._history:
            fill_actions = _fill_actions_like(actions, self.initial_action_provider)
            self._history = [fill_actions.clone() for _ in range(self.delay_steps)]
        elif self._history[0].shape != actions.shape:
            raise ValueError(
                f"Action shape changed from {tuple(self._history[0].shape)} to {tuple(actions.shape)}."
            )

        delayed_actions = self._history.pop(0)
        self._history.append(actions.clone())
        observations, rewards, terminated, truncated, extras = self.env.step(delayed_actions)

        dones = terminated.bool() | truncated.bool()
        if torch.any(dones):
            fill_actions = _fill_actions_like(actions, self.initial_action_provider)
            for buffered_actions in self._history:
                buffered_actions[dones] = fill_actions[dones]
        return observations, rewards, terminated, truncated, extras


class RandomActionDelayWrapper(gym.Wrapper):
    """Sample an independent integer policy-step delay for every episode.

    Delay is sampled uniformly and independently per vectorized environment.
    Histories begin with the action supplied by ``initial_action_provider``
    and are refilled for each row immediately after that environment
    auto-resets. If no provider is supplied, the legacy all-zero action is used.
    """

    def __init__(
        self,
        env: gym.Env,
        min_delay_steps: int,
        max_delay_steps: int,
        initial_action_provider: ActionProvider | None = None,
    ):
        if (
            not isinstance(min_delay_steps, int)
            or isinstance(min_delay_steps, bool)
            or not isinstance(max_delay_steps, int)
            or isinstance(max_delay_steps, bool)
            or min_delay_steps < 0
            or min_delay_steps > max_delay_steps
        ):
            raise ValueError("Action-delay bounds must be ordered nonnegative integers.")
        super().__init__(env)
        self.min_delay_steps = min_delay_steps
        self.max_delay_steps = max_delay_steps
        self.initial_action_provider = initial_action_provider
        self._history: list[torch.Tensor] = []
        self._delays: torch.Tensor | None = None

    def _sample_delays(self, count: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            self.min_delay_steps,
            self.max_delay_steps + 1,
            (count,),
            device=device,
        )

    def reset(self, **kwargs):
        observations, extras = self.env.reset(**kwargs)
        self._history.clear()
        self._delays = None
        return observations, extras

    def step(self, actions: torch.Tensor):
        if self._delays is None:
            self._delays = self._sample_delays(actions.shape[0], actions.device)
            fill_actions = _fill_actions_like(actions, self.initial_action_provider)
            self._history = [fill_actions.clone() for _ in range(self.max_delay_steps)]
        elif self._delays.shape[0] != actions.shape[0]:
            raise ValueError(
                f"Action batch changed from {self._delays.shape[0]} to {actions.shape[0]}."
            )

        candidates = torch.stack([actions, *self._history], dim=1)
        gather_index = self._delays.view(-1, 1, 1).expand(-1, 1, actions.shape[1])
        delayed_actions = candidates.gather(1, gather_index).squeeze(1)
        observations, rewards, terminated, truncated, extras = self.env.step(delayed_actions)

        if self.max_delay_steps:
            self._history = [actions.clone(), *self._history[:-1]]
        dones = terminated.bool() | truncated.bool()
        if torch.any(dones):
            done_count = int(dones.sum())
            self._delays[dones] = self._sample_delays(done_count, actions.device)
            fill_actions = _fill_actions_like(actions, self.initial_action_provider)
            for buffered_actions in self._history:
                buffered_actions[dones] = fill_actions[dones]
        return observations, rewards, terminated, truncated, extras
