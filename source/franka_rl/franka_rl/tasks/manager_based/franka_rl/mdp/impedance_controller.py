"""GPU-batched reference generation and Franky-like joint impedance control."""

from __future__ import annotations

import math

import torch


def full_range_position_command(
    actions: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip normalized actions and map them through the midpoint to joint bounds."""

    bounded = torch.clamp(actions, -1.0, 1.0)
    center = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    return bounded, center + half_range * bounded


def incremental_position_command(
    actions: torch.Tensor,
    reference: torch.Tensor,
    max_velocity: torch.Tensor,
    policy_dt: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate a bounded reference velocity and project it to soft limits."""

    bounded = torch.clamp(actions, -1.0, 1.0)
    requested = reference + bounded * max_velocity * float(policy_dt)
    projected = torch.maximum(torch.minimum(requested, upper), lower)
    increment = projected - reference
    projection = requested - projected
    return bounded, projected, increment, projection


class FrankyImpedanceController:
    """Explicit torque law matching the validated Franky controller contract."""

    def __init__(
        self,
        num_envs: int,
        num_joints: int,
        device: str,
        dtype: torch.dtype,
        *,
        dt: float,
        nominal_stiffness: float,
        gain_alpha_range: tuple[float, float] | None,
        position_error_clip: float,
        torque_slew_rate: float,
        filter_cutoff_hz: float,
        lower: tuple[float, ...],
        upper: tuple[float, ...],
        limit_activation_distance: float,
        limit_stiffness: float,
        limit_damping: float,
        limit_max_torque: float,
    ) -> None:
        self.dt = float(dt)
        self.nominal_stiffness = float(nominal_stiffness)
        self.gain_alpha_range = gain_alpha_range
        self.position_error_clip = float(position_error_clip)
        self.torque_step = float(torque_slew_rate) * self.dt
        tau = 1.0 / (2.0 * math.pi * float(filter_cutoff_hz))
        self.filter_alpha = self.dt / (self.dt + tau)
        self.lower = torch.tensor(lower, device=device, dtype=dtype)
        self.upper = torch.tensor(upper, device=device, dtype=dtype)
        self.limit_activation_distance = float(limit_activation_distance)
        self.limit_stiffness = float(limit_stiffness)
        self.limit_damping = float(limit_damping)
        self.limit_max_torque = float(limit_max_torque)
        shape = (num_envs, num_joints)
        self.gain_alpha = torch.ones((num_envs, 1), device=device, dtype=dtype)
        self.previous_limited_torque = torch.zeros(shape, device=device, dtype=dtype)
        self.filtered_torque = torch.zeros_like(self.previous_limited_torque)
        self.applied_torque = torch.zeros_like(self.previous_limited_torque)

    def reset(self, env_ids) -> None:
        self.previous_limited_torque[env_ids] = 0.0
        self.filtered_torque[env_ids] = 0.0
        self.applied_torque[env_ids] = 0.0
        if self.gain_alpha_range is None:
            self.gain_alpha[env_ids] = 1.0
        else:
            low, high = self.gain_alpha_range
            if low <= 0.0 or low > high:
                raise ValueError("gain_alpha_range must be ordered and positive")
            sample = torch.empty_like(self.gain_alpha[env_ids]).uniform_(math.log(low), math.log(high))
            self.gain_alpha[env_ids] = torch.exp(sample)

    def compute(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        position_reference: torch.Tensor,
        coriolis: torch.Tensor,
        gravity: torch.Tensor,
    ) -> torch.Tensor:
        stiffness = self.nominal_stiffness * self.gain_alpha
        damping = 2.0 * torch.sqrt(stiffness)
        error = torch.clamp(
            position_reference - position,
            -self.position_error_clip,
            self.position_error_clip,
        )
        limit_torque = self._limit_torque(position, velocity)
        requested = stiffness * error - damping * velocity + coriolis + limit_torque
        limited = self.previous_limited_torque + torch.clamp(
            requested - self.previous_limited_torque,
            -self.torque_step,
            self.torque_step,
        )
        self.previous_limited_torque.copy_(limited)
        self.filtered_torque.add_(self.filter_alpha * (limited - self.filtered_torque))
        self.applied_torque.copy_(self.filtered_torque + gravity)
        return self.applied_torque

    def _limit_torque(self, position: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        lower_activation = self.lower + self.limit_activation_distance
        upper_activation = self.upper - self.limit_activation_distance
        lower = torch.clamp(
            self.limit_stiffness * (lower_activation - position) - self.limit_damping * velocity,
            min=0.0,
            max=self.limit_max_torque,
        )
        upper = torch.clamp(
            self.limit_stiffness * (upper_activation - position) - self.limit_damping * velocity,
            min=-self.limit_max_torque,
            max=0.0,
        )
        return lower + upper
