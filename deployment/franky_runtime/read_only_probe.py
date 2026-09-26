#!/usr/bin/env python3
"""Read one FR3 state through Franky without starting a control loop.

This program exposes no motion, recovery, load, impedance, or Desk operations.
Constructing ``franky.Robot`` in the pinned build does call ``setCollisionBehavior``
with Franky's default 20 Nm joint-torque and 30 N Cartesian-force thresholds.  The
program then reads state and model metadata, writes one JSON record, and exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_FRANKY = "2.0.1.dev58+gf88f0e9b.libfranka.0.21.2"
DEFAULT_HOST = "172.16.0.6"


def _array(value: Any) -> list[Any]:
    return np.asarray(value).tolist()


def _affine_matrix(value: Any) -> list[Any]:
    """Handle both released and development-build Affine matrix APIs."""
    matrix = value.matrix
    return _array(matrix() if callable(matrix) else matrix)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_libfranka() -> dict[str, str]:
    import franky

    site_packages = Path(franky.__file__).resolve().parent.parent
    matches = sorted((site_packages / "franky_control.libs").glob("libfranka*.so*"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one bundled libfranka, found {[str(p) for p in matches]}"
        )
    library = matches[0]
    return {"path": str(library), "sha256": _sha256(library)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--connect",
        action="store_true",
        help="Perform the read-only FCI connection. Without this flag, only describe the probe.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    installed_franky = metadata.version("franky-control")
    if installed_franky != EXPECTED_FRANKY:
        raise RuntimeError(
            f"Franky version mismatch: expected {EXPECTED_FRANKY}, got {installed_franky}"
        )

    environment = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "franky_control": installed_franky,
        "bundled_libfranka": _native_libfranka(),
    }

    if not args.connect:
        print(
            json.dumps(
                {
                    "mode": "offline_description",
                    "host": args.host,
                    "environment": environment,
                    "connection_performed": False,
                    "motion_api_called": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.output is None:
        raise ValueError("--output is required with --connect")

    # Constructing Robot establishes FCI, loads the model, and—inside Franky's
    # constructor—sets its default 20 Nm / 30 N collision thresholds. Only read-only
    # properties are explicitly accessed below. This program never calls move(),
    # recover_from_errors(), stop(), or any setter itself.
    from franky import RealtimeConfig, Robot

    robot = Robot(args.host, realtime_config=RealtimeConfig.Enforce)
    state = robot.state

    record = {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "probe_pid": os.getpid(),
        "host": args.host,
        "environment": environment,
        "connection_performed": True,
        "motion_api_called": False,
        "robot": {
            "mode": state.robot_mode.name,
            "has_errors": bool(robot.has_errors),
            "is_in_control": bool(robot.is_in_control),
            "robot_time_s": state.time.to_sec(),
            "control_command_success_rate": state.control_command_success_rate,
            "q_rad": _array(state.q),
            "dq_rad_s": _array(state.dq),
            "q_desired_rad": _array(state.q_d),
            "dq_desired_rad_s": _array(state.dq_d),
            "tau_joint_nm": _array(state.tau_J),
            "tau_external_filtered_nm": _array(state.tau_ext_hat_filtered),
            "current_errors": repr(state.current_errors),
            "last_motion_errors": repr(state.last_motion_errors),
            "F_T_EE": _affine_matrix(state.F_T_EE),
            "EE_T_K": _affine_matrix(state.EE_T_K),
            "load": {
                "mass_kg": state.m_load,
                "center_of_mass_flange_m": _array(state.F_x_Cload),
                "inertia_kg_m2": _array(state.I_load),
            },
            "limits_max": {
                "joint_velocity_rad_s": _array(robot.joint_velocity_limit.max),
                "joint_acceleration_rad_s2": _array(robot.joint_acceleration_limit.max),
                "joint_jerk_rad_s3": _array(robot.joint_jerk_limit.max),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
