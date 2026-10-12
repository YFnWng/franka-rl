#!/usr/bin/env python3
"""Numerical checks supporting the M4 Franky/controller source audit."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def torque_processing_check() -> dict[str, object]:
    dt = 0.001
    cutoff_hz = 100.0
    alpha = dt / (dt + 1.0 / (2.0 * math.pi * cutoff_hz))
    raw = 100.0
    max_delta = 1.0

    simulation_limited = 0.0
    simulation_filtered = 0.0
    franky_applied = 0.0
    samples = []
    for tick in range(1, 101):
        # franka_rl TorqueCommandProcessor: independent limiter and filter states.
        simulation_limited += float(np.clip(raw - simulation_limited, -max_delta, max_delta))
        simulation_filtered += alpha * (simulation_limited - simulation_filtered)

        # Current Franky Cartesian controller limits against tau_J_d, then libfranka
        # filters against that same previously applied tau_J_d.
        callback_output = franky_applied + float(np.clip(raw - franky_applied, -max_delta, max_delta))
        franky_applied += alpha * (callback_output - franky_applied)

        if tick in (1, 2, 5, 10, 20, 50, 100):
            samples.append(
                {
                    "tick": tick,
                    "time_ms": tick,
                    "simulation_filtered_non_gravity_nm_before_effort_clip": simulation_filtered,
                    "franky_plus_libfranka_command_nm": franky_applied,
                }
            )

    return {
        "dt_s": dt,
        "cutoff_hz": cutoff_hz,
        "low_pass_alpha": alpha,
        "max_delta_nm_per_tick": max_delta,
        "step_input_nm": raw,
        "samples": samples,
        "interpretation": (
            "The two cascades are not equivalent. With a sustained large step, the current "
            "Franky+libfranka client path can change its outgoing command by alpha*1 Nm per tick, "
            "whereas the simulation limiter state advances by 1 Nm per tick and its filtered "
            "output asymptotically acquires that 1 Nm/tick slope."
        ),
    }


def projector_check() -> dict[str, object]:
    rng = np.random.default_rng(20261011)
    jacobian = rng.normal(size=(6, 7))
    a = rng.normal(size=(7, 7))
    mass = a.T @ a + np.diag(np.linspace(1.0, 2.0, 7))
    identity = np.eye(7)

    # Franky uses I - J^+ J in factored form. M3 static posture uses the
    # algebraically identical I - J.T @ pinv(J.T).
    franky = identity - np.linalg.pinv(jacobian) @ jacobian
    simulation_static = identity - jacobian.T @ np.linalg.pinv(jacobian.T)

    mass_inv = np.linalg.inv(mass)
    simulation_dynamic = (
        identity
        - jacobian.T
        @ np.linalg.pinv(jacobian @ mass_inv @ jacobian.T)
        @ jacobian
        @ mass_inv
    )

    return {
        "seed": 20261011,
        "jacobian_shape": list(jacobian.shape),
        "franky_vs_simulation_static_max_abs": float(np.max(np.abs(franky - simulation_static))),
        "franky_vs_simulation_dynamic_max_abs": float(np.max(np.abs(franky - simulation_dynamic))),
        "franky_vs_simulation_dynamic_frobenius": float(np.linalg.norm(franky - simulation_dynamic)),
        "franky_nullspace_residual_frobenius": float(np.linalg.norm(jacobian @ franky)),
        "dynamic_nullspace_residual_frobenius": float(np.linalg.norm(jacobian @ mass_inv @ simulation_dynamic)),
        "interpretation": (
            "Franky's orthogonal kinematic torque projector matches the M3 static-posture "
            "projector to numerical precision. It is generally different from M4's "
            "mass-weighted dynamically consistent torque projector."
        ),
    }


def main() -> None:
    result = {
        "schema_version": 1,
        "torque_processing": torque_processing_check(),
        "projectors": projector_check(),
    }
    output = Path(__file__).with_name("numerical_results.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
