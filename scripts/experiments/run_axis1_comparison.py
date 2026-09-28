#!/usr/bin/env python3
"""Evaluate a completed Axis-1 three-seed checkpoint manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from franka_rl.experiments import EvaluationSuiteConfig, EvaluationSuiteCoordinator

DEFAULT_DATA_ROOT = Path("/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data")
MIN_FREE_BYTES = 100 * 1024**2
EVALUATION_SEED = 20260927
EVALUATION_SCENARIO = "action_delay_1"
SUMMARY_METRICS = (
    "success_rate",
    "unsafe_failure_rate",
    "mean_final_position_error_m",
    "mean_min_position_error_m",
    "mean_position_error_m",
    "mean_time_to_success_s",
    "mean_waypoints_reached",
    "total_waypoint_timeouts",
    "velocity_envelope_exceedance_rate",
    "mean_peak_joint_velocity_ratio",
    "mean_command_difference_norm_rad",
    "mean_peak_command_difference_norm_rad",
    "mean_peak_applied_torque_norm_nm",
    "mean_tracking_error_norm_rad",
    "mean_reference_projection_norm_rad",
    "mean_action_clipping_fraction",
)


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.expanduser().resolve().read_text())
    if manifest.get("version") != 1 or not isinstance(manifest.get("policies"), dict):
        raise ValueError("Unsupported or malformed Axis-1 checkpoint manifest")
    target_seeds = manifest.get("target_seeds")
    if not isinstance(target_seeds, list) or not target_seeds:
        raise ValueError("Checkpoint manifest lacks target_seeds")
    for group, policy in manifest["policies"].items():
        for key in ("point_task", "circle_task", "checkpoints"):
            if key not in policy:
                raise ValueError(f"Policy {group} lacks {key}")
        for seed in target_seeds:
            entry = policy["checkpoints"].get(str(seed))
            if not entry or not entry.get("complete"):
                raise ValueError(f"Policy {group} seed {seed} is incomplete")
            checkpoint = Path(entry["path"])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
    return manifest


def _suite_document(manifest: dict[str, Any], mode: str) -> tuple[dict[str, Any], dict[str, tuple[str, int]]]:
    policies = {}
    identities = {}
    task_key = "point_task" if mode == "point" else "circle_task"
    for group, policy in manifest["policies"].items():
        for seed in manifest["target_seeds"]:
            name = f"{group}_seed_{seed}"
            policies[name] = {
                "checkpoint": policy["checkpoints"][str(seed)]["path"],
                "task": policy[task_key],
                "description": f"{group} training seed {seed}",
            }
            identities[name] = (group, int(seed))
    baseline = "position_increment_seed_42"
    if baseline not in policies:
        baseline = next(iter(policies))
    evaluation = {
        "task": next(iter(policies.values()))["task"],
        "num_envs": 64,
        "episodes_per_job": 512,
        "success_threshold": 0.03,
        "success_steps": 5,
        "device": str(manifest.get("device", "cuda:0")),
        "deterministic": True,
        "visualizer": "none",
    }
    if mode == "point":
        evaluation["target_replay"] = {
            "enabled": True,
            "extra_episodes_per_env": 32,
            "position_ranges": {"x": [0.35, 0.60], "y": [-0.20, 0.20], "z": [0.20, 0.50]},
        }
    else:
        evaluation.update(
            {"num_envs": 16, "episodes_per_job": 16, "success_threshold": 0.01, "success_steps": 1, "path": "circle_yz"}
        )
    return (
        {
            "version": 1,
            "name": f"axis1_policy_output_{mode}_three_seed",
            "baseline_policy": baseline,
            "policies": policies,
            "scenarios": [EVALUATION_SCENARIO],
            "seeds": [EVALUATION_SEED],
            "evaluation": evaluation,
        },
        identities,
    )


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _compile_groups(suite_dir: Path, identities: dict[str, tuple[str, int]]) -> None:
    rows = []
    for policy_name, (group, training_seed) in identities.items():
        summary_path = (
            suite_dir / "jobs" / policy_name / EVALUATION_SCENARIO / f"seed_{EVALUATION_SEED}" / "summary.json"
        )
        summary = json.loads(summary_path.read_text())
        row = {
            "output_type": group,
            "training_seed": training_seed,
            "evaluation_seed": EVALUATION_SEED,
            "checkpoint_policy": policy_name,
            "episodes_recorded": summary.get("episodes_recorded"),
            "summary_path": str(summary_path),
        }
        for metric in SUMMARY_METRICS:
            row[metric] = summary.get(metric)
        rows.append(row)
    rows.sort(key=lambda r: (r["output_type"], r["training_seed"]))
    compiled = suite_dir / "compiled"
    compiled.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (compiled / "axis1_checkpoint_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    grouped = []
    for group in sorted({r["output_type"] for r in rows}):
        members = [r for r in rows if r["output_type"] == group]
        result = {
            "output_type": group,
            "num_training_seeds": len(members),
            "training_seeds": ";".join(str(r["training_seed"]) for r in members),
            "episodes_recorded": sum(int(r["episodes_recorded"]) for r in members),
        }
        for metric in SUMMARY_METRICS:
            values = [v for r in members if (v := _number(r.get(metric))) is not None]
            result[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
        grouped.append(result)
    group_fields = list(grouped[0])
    with (compiled / "axis1_output_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=group_fields)
        w.writeheader()
        w.writerows(grouped)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("point", "circle", "both"), default="both")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    manifest = _load_manifest(args.training_manifest)
    data_root = Path(os.environ.get("FRANKA_RL_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()
    if not data_root.is_dir() or not os.access(data_root, os.W_OK):
        raise RuntimeError(f"FRANKA_RL_DATA_ROOT unavailable: {data_root}")
    if shutil.disk_usage(data_root).free < MIN_FREE_BYTES:
        raise RuntimeError("FRANKA_RL_DATA_ROOT has less than 100 MiB free")
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output = data_root / "evaluation_suites" / f"{stamp}_axis1_policy_output_three_seed"
    else:
        output = args.output_dir.expanduser().resolve()
    try:
        output.relative_to(data_root)
    except ValueError as e:
        raise RuntimeError(f"Output must be below {data_root}") from e
    output.mkdir(parents=True, exist_ok=True)
    modes = ("point", "circle") if args.mode == "both" else (args.mode,)
    index = {"version": 1, "training_manifest": str(args.training_manifest.expanduser().resolve()), "modes": {}}
    success = True
    for mode in modes:
        document, identities = _suite_document(manifest, mode)
        suite_yaml = output / f"{mode}_suite.yaml"
        suite_yaml.write_text(yaml.safe_dump(document, sort_keys=False))
        suite_dir = output / mode
        config = EvaluationSuiteConfig.from_yaml(suite_yaml)
        coordinator = EvaluationSuiteCoordinator(
            config, output_dir=suite_dir, resume=not args.no_resume, fail_fast=args.fail_fast
        )
        mode_ok = coordinator.run()
        success &= mode_ok
        if mode_ok:
            _compile_groups(suite_dir, identities)
        index["modes"][mode] = {"suite": str(suite_yaml), "output": str(suite_dir), "complete": mode_ok}
        if not mode_ok and args.fail_fast:
            break
    (output / "axis1_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Axis-1 comparison index: {output / 'axis1_index.json'}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
