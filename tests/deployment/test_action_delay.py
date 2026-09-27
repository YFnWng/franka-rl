"""Regression tests for reset-safe policy action delay."""

import gymnasium as gym
import torch

from franka_rl.utils.action_delay import (
    FixedActionDelayWrapper,
    RandomActionDelayWrapper,
)
from franka_rl.utils.policy_evaluator import prime_vector_env_auto_reset


class _AutoResetVectorEnv(gym.Env):
    def __init__(self):
        self.fill_action = torch.tensor([[0.2, -0.3], [0.4, -0.5]])
        self.received: list[torch.Tensor] = []
        self._step = 0

    def reset(self, **kwargs):
        return torch.zeros((2, 1)), {}

    def step(self, action):
        self.received.append(action.clone())
        self._step += 1
        terminated = torch.tensor([self._step == 1, False])
        if terminated[0]:
            # Model an auto-reset that produces a new randomized initial pose.
            self.fill_action[0] = torch.tensor([-0.7, 0.6])
        return torch.zeros((2, 1)), torch.zeros(2), terminated, torch.zeros(2, dtype=torch.bool), {}


def test_fixed_delay_uses_reset_action_for_initial_and_done_rows():
    base = _AutoResetVectorEnv()
    wrapper = FixedActionDelayWrapper(
        base,
        1,
        initial_action_provider=lambda: base.fill_action,
    )
    first = torch.tensor([[0.8, 0.9], [0.6, 0.7]])
    second = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    wrapper.step(first)
    wrapper.step(second)

    torch.testing.assert_close(
        base.received[0],
        torch.tensor([[0.2, -0.3], [0.4, -0.5]]),
    )
    torch.testing.assert_close(
        base.received[1],
        torch.tensor([[-0.7, 0.6], [0.6, 0.7]]),
    )


def test_random_delay_uses_reset_action_for_delayed_rows():
    base = _AutoResetVectorEnv()
    wrapper = RandomActionDelayWrapper(
        base,
        1,
        1,
        initial_action_provider=lambda: base.fill_action,
    )
    first = torch.tensor([[0.8, 0.9], [0.6, 0.7]])
    second = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    wrapper.step(first)
    wrapper.step(second)

    torch.testing.assert_close(
        base.received[0],
        torch.tensor([[0.2, -0.3], [0.4, -0.5]]),
    )
    torch.testing.assert_close(
        base.received[1],
        torch.tensor([[-0.7, 0.6], [0.6, 0.7]]),
    )


class _AutoResetPrimerEnv:
    def __init__(self):
        self.max_episode_length = 10
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.reset_action = torch.tensor([[0.2, -0.3], [0.4, -0.5]])
        action_term = type("ActionTerm", (), {"delay_fill_action": self.reset_action})()
        self.action_manager = type(
            "ActionManager", (), {"get_term": lambda manager, name: action_term}
        )()
        self.received = None

    def step(self, action):
        assert torch.all(self.episode_length_buf == self.max_episode_length - 1)
        self.received = action.clone()
        self.episode_length_buf.zero_()
        dones = torch.ones(2, dtype=torch.bool)
        return torch.full((2, 1), 7.0), torch.zeros(2), dones, {}


def test_auto_reset_primer_returns_post_reset_observation_and_resets_policy():
    env = _AutoResetPrimerEnv()
    reset_masks = []

    observations = prime_vector_env_auto_reset(
        env, env, lambda dones: reset_masks.append(dones.clone())
    )

    torch.testing.assert_close(env.received, env.reset_action)
    torch.testing.assert_close(observations, torch.full((2, 1), 7.0))
    assert len(reset_masks) == 1
    assert torch.all(reset_masks[0])
