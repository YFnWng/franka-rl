#!/usr/bin/env python3
"""Run subprocess-isolated J4/J6 velocity-reference step diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


DEFAULT_DATA_ROOT = Path("/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data")
REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_SCRIPT = REPO_ROOT / "scripts" / "rsl_rl" / "diagnose_velocity_impedance.py"
EXPECTED_KEYS = {
    "version",
    "name",
    "task",
    "joints",
    "normalized_amplitudes",
    "ramp_down_durations_s",
    "duration_s",
    "step_start_s",
    "step_end_s",
    "target_position_base_m",
    "seed",
    "device",
}


def _load_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    document = yaml.safe_load(raw)
    if not isinstance(document, dict) or set(document) != EXPECTED_KEYS:
        raise ValueError(f"{path} must contain exactly these keys: {sorted(EXPECTED_KEYS)}")
    if document["version"] != 1:
        raise ValueError("Only step-sweep config version 1 is supported")
    if not isinstance(document["name"], str) or not document["name"]:
        raise ValueError("name must be a non-empty string")
    joints = document["joints"]
    if not isinstance(joints, list) or not joints or len(set(joints)) != len(joints):
        raise ValueError("joints must be a non-empty list without duplicates")
    if any(not isinstance(joint, int) or isinstance(joint, bool) or not 1 <= joint <= 6 for joint in joints):
        raise ValueError("Every swept joint must be an integer in [1, 6]")
    amplitudes = document["normalized_amplitudes"]
    if not isinstance(amplitudes, list) or not amplitudes:
        raise ValueError("normalized_amplitudes must be a non-empty list")
    amplitudes = [float(value) for value in amplitudes]
    if len(set(amplitudes)) != len(amplitudes) or any(not 0.0 < value <= 1.0 for value in amplitudes):
        raise ValueError("normalized_amplitudes must be unique values in (0, 1]")
    document["normalized_amplitudes"] = amplitudes
    ramp_down_durations = document["ramp_down_durations_s"]
    if not isinstance(ramp_down_durations, list) or not ramp_down_durations:
        raise ValueError("ramp_down_durations_s must be a non-empty list")
    ramp_down_durations = [float(value) for value in ramp_down_durations]
    if len(set(ramp_down_durations)) != len(ramp_down_durations) or any(
        value < 0.0 for value in ramp_down_durations
    ):
        raise ValueError("ramp_down_durations_s must be unique nonnegative values")
    document["ramp_down_durations_s"] = ramp_down_durations
    duration = float(document["duration_s"])
    start = float(document["step_start_s"])
    end = float(document["step_end_s"])
    if not 0.0 <= start < end <= duration:
        raise ValueError("Require 0 <= step_start_s < step_end_s <= duration_s")
    if any(end + value > duration for value in ramp_down_durations):
        raise ValueError("Every ramp-down profile must finish by duration_s")
    target = document["target_position_base_m"]
    if not isinstance(target, list) or len(target) != 3:
        raise ValueError("target_position_base_m must contain x, y, z")
    document["target_position_base_m"] = [float(value) for value in target]
    return document, hashlib.sha256(raw).hexdigest()


def _validate_output(output: Path, data_root: Path) -> None:
    if not data_root.is_dir() or not os.access(data_root, os.W_OK):
        raise RuntimeError(f"FRANKA_RL_DATA_ROOT is missing or not writable: {data_root}")
    if shutil.disk_usage(data_root).free < 100 * 1024 * 1024:
        raise RuntimeError(f"FRANKA_RL_DATA_ROOT has less than 100 MiB free: {data_root}")
    try:
        output.relative_to(data_root)
    except ValueError as error:
        raise RuntimeError(f"Sweep output must be under FRANKA_RL_DATA_ROOT ({data_root}): {output}") from error


def _job_name(joint: int, amplitude: float, ramp_down_duration: float) -> str:
    return (
        f"joint_{joint}/amplitude_{amplitude:.3f}/ramp_down_{ramp_down_duration:.3f}"
        .replace(".", "p")
    )


def _classification(row: dict) -> str:
    if row["model_limit_hit"] or row["velocity_amplification"] >= 3.0 or row["slew_active_fraction"] >= 0.5:
        return "unstable"
    if row["velocity_amplification"] >= 2.0 or row["slew_active_fraction"] >= 0.1:
        return "marginal"
    return "stable"


def _compile_job(
    config: dict,
    joint: int,
    amplitude: float,
    ramp_down_duration: float,
    job_dir: Path,
) -> dict:
    summary = json.loads((job_dir / "summary.json").read_text())
    metrics = summary["per_joint"][f"joint_{joint}"]
    commanded_velocity = metrics["peak_abs_target_velocity_reference_rad_s"]
    peak_velocity = metrics["peak_abs_measured_velocity_rad_s"]
    row = {
        "joint": joint,
        "normalized_amplitude": amplitude,
        "release_profile": "abrupt" if ramp_down_duration == 0.0 else "linear_ramp",
        "ramp_down_duration_s": ramp_down_duration,
        "commanded_velocity_rad_s": commanded_velocity,
        "peak_measured_velocity_rad_s": peak_velocity,
        "velocity_amplification": peak_velocity / max(commanded_velocity, 1.0e-12),
        "qualification_ratio": metrics["peak_qualification_ratio"],
        "fraction_ticks_above_qualification_limit": metrics["fraction_ticks_above_qualification_limit"],
        "model_limit_ratio": metrics["peak_model_limit_ratio"],
        "model_limit_hit": metrics["reached_99pct_model_velocity_limit"],
        "peak_tracking_error_rad": metrics["peak_abs_position_tracking_error_rad"],
        "peak_measured_acceleration_rad_s2": metrics["peak_abs_measured_acceleration_rad_s2"],
        "measured_velocity_sign_reversals": metrics[
            "measured_velocity_sign_reversals_above_0p01_rad_s"
        ],
        "slew_active_fraction": metrics["fraction_ticks_slew_limiter_active"],
        "requested_filtered_opposite_fraction": metrics[
            "fraction_meaningful_ticks_requested_filtered_torque_opposite"
        ],
        "requested_applied_total_opposite_fraction": metrics[
            "fraction_meaningful_ticks_requested_applied_total_torque_opposite"
        ],
        "peak_requested_torque_nm": metrics["peak_abs_requested_torque_nm"],
        "peak_applied_torque_nm": metrics["peak_abs_applied_torque_nm"],
        "artifact_dir": str(job_dir),
    }
    row["classification"] = _classification(row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve(strict=True)
    config, config_sha256 = _load_config(config_path)
    data_root = Path(os.environ.get("FRANKA_RL_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output = data_root / "diagnostic_sweeps" / f"{stamp}_{config['name']}"
    else:
        output = args.output_dir.expanduser().resolve()
    _validate_output(output, data_root)
    output.mkdir(parents=True, exist_ok=False)

    plan = [
        {
            "joint": joint,
            "normalized_amplitude": amplitude,
            "ramp_down_duration_s": ramp_down_duration,
            "job": _job_name(joint, amplitude, ramp_down_duration),
        }
        for joint in config["joints"]
        for amplitude in config["normalized_amplitudes"]
        for ramp_down_duration in config["ramp_down_durations_s"]
    ]
    manifest = {
        "schema_version": 1,
        "name": config["name"],
        "source_config": str(config_path),
        "source_config_sha256": config_sha256,
        "created_at": datetime.now().astimezone().isoformat(),
        "config": config,
        "classification_heuristic": {
            "unstable": "model limit hit OR velocity amplification >= 3 OR slew active fraction >= 0.5",
            "marginal": "velocity amplification >= 2 OR slew active fraction >= 0.1",
            "stable": "none of the preceding diagnostic thresholds",
            "note": (
                "Opposite-torque fraction is reported but does not classify isolated steps because "
                "normal start/stop transients can reverse corrective torque. Classification is diagnostic, "
                "not a hardware safety certification."
            ),
        },
        "jobs": plan,
        "dry_run": args.dry_run,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.dry_run:
        print(f"Sweep plan written to: {output / 'manifest.json'}")
        return 0

    rows: list[dict] = []
    failures: list[dict] = []
    for index, job in enumerate(plan, start=1):
        joint = job["joint"]
        amplitude = job["normalized_amplitude"]
        ramp_down_duration = job["ramp_down_duration_s"]
        job_dir = output / "jobs" / job["job"]
        job_dir.parent.mkdir(parents=True, exist_ok=True)
        log_path = job_dir.parent / f"{job_dir.name}.stdout.log"
        command = [
            "direnv",
            "exec",
            str(REPO_ROOT),
            sys.executable,
            "-u",
            str(DIAGNOSTIC_SCRIPT),
            "--task",
            config["task"],
            "--case",
            "step",
            "--duration",
            str(config["duration_s"]),
            "--step-start",
            str(config["step_start_s"]),
            "--step-end",
            str(config["step_end_s"]),
            "--joint",
            str(joint),
            "--amplitude",
            str(amplitude),
            "--ramp-down-duration",
            str(ramp_down_duration),
            "--target",
            *(str(value) for value in config["target_position_base_m"]),
            "--seed",
            str(config["seed"]),
            "--device",
            config["device"],
            "--output-dir",
            str(job_dir),
            "--viz",
            "none",
        ]
        print(
            f"[{index}/{len(plan)}] joint {joint}, normalized amplitude {amplitude:.3f}, "
            f"ramp down {ramp_down_duration:.3f} s",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0 or not (job_dir / "summary.json").is_file():
            failure = {**job, "returncode": result.returncode, "stdout": str(log_path)}
            failures.append(failure)
            print(f"  FAILED ({result.returncode}); see {log_path}")
            if args.fail_fast:
                break
            continue
        row = _compile_job(config, joint, amplitude, ramp_down_duration, job_dir)
        rows.append(row)
        print(
            f"  {row['classification']}: peak={row['peak_measured_velocity_rad_s']:.3f} rad/s, "
            f"amplification={row['velocity_amplification']:.2f}, slew={row['slew_active_fraction']:.1%}"
        )

    compiled = output / "compiled"
    compiled.mkdir(exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "joint", "normalized_amplitude", "classification", "artifact_dir"
    ]
    with (compiled / "sweep_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result_document = {
        "schema_version": 1,
        "completed_jobs": len(rows),
        "expected_jobs": len(plan),
        "failures": failures,
        "rows": rows,
    }
    (compiled / "sweep_summary.json").write_text(json.dumps(result_document, indent=2) + "\n")
    print(f"Compiled sweep: {compiled}")
    return 0 if not failures and len(rows) == len(plan) else 1


if __name__ == "__main__":
    raise SystemExit(main())
