#!/usr/bin/env python3
"""Evaluate matched nominal and curriculum-DR incremental policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from franka_rl.experiments import EvaluationSuiteConfig, EvaluationSuiteCoordinator


DEFAULT_DATA_ROOT = Path("/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data")
MIN_FREE_BYTES = 100 * 1024**2
DEFAULT_EVALUATION_SEED = 20260929
TASK = "Franka-FR3v2-FrankyImpedance-Incremental6DReach-v0"
SCENARIOS = (
    "nominal",
    "gain_alpha_050",
    "gain_alpha_200",
    "action_delay_1",
    "action_delay_2",
    "payload_050",
    "payload_100_z005",
    "deployment_combined_boundary",
    "fr3_incremental_deployment_dr_v1",
)
SUMMARY_METRICS = (
    "success_rate",
    "unsafe_failure_rate",
    "mean_final_position_error_m",
    "mean_min_position_error_m",
    "mean_position_error_m",
    "mean_time_to_success_s",
    "velocity_envelope_exceedance_rate",
    "mean_peak_joint_velocity_ratio",
    "mean_command_difference_norm_rad",
    "mean_peak_command_difference_norm_rad",
    "mean_reference_acceleration_ratio",
    "mean_peak_reference_acceleration_ratio",
    "worst_reference_acceleration_ratio",
    "mean_peak_applied_torque_norm_nm",
    "mean_tracking_error_norm_rad",
    "mean_reference_projection_norm_rad",
    "mean_action_clipping_fraction",
)


def load_curriculum_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1 or not isinstance(manifest.get("seeds"), dict):
        raise ValueError("Unsupported or malformed Axis-3 curriculum manifest")
    target_seeds = manifest.get("target_seeds")
    if not isinstance(target_seeds, list) or not target_seeds:
        raise ValueError("Curriculum manifest lacks target_seeds")
    for seed in target_seeds:
        seed_entry = manifest["seeds"].get(str(seed))
        if not isinstance(seed_entry, dict):
            raise ValueError(f"Curriculum manifest lacks seed {seed}")
        nominal = Path(seed_entry["source_checkpoint"]).expanduser().resolve()
        full = seed_entry.get("stages", {}).get("full")
        if not isinstance(full, dict) or not full.get("complete"):
            raise ValueError(f"Curriculum full stage for seed {seed} is incomplete")
        dr = Path(full["checkpoint"]).expanduser().resolve()
        for label, checkpoint in (("nominal", nominal), ("curriculum DR", dr)):
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing {label} checkpoint for seed {seed}: {checkpoint}")
        failure = Path(full["run_dir"]) / "ppo_numerical_failure.json"
        if failure.is_file():
            raise ValueError(f"Selected curriculum run for seed {seed} contains {failure}")
    return manifest


def write_scenario_catalog(path: Path) -> None:
    raw = files("franka_rl").joinpath("config/scenarios.yaml").read_text()
    document = yaml.safe_load(raw)
    document["scenarios"].update(
        {
            "gain_alpha_050": {
                "type": "specified",
                "description": "Franky impedance stiffness and damping scaled to 50 percent.",
                "control": {"impedance_gain_alpha": 0.50},
            },
            "gain_alpha_200": {
                "type": "specified",
                "description": "Franky impedance stiffness and damping scaled to 200 percent.",
                "control": {"impedance_gain_alpha": 2.00},
            },
            "deployment_combined_boundary": {
                "type": "specified",
                "description": "Low gain, one-step delay, and offset 1 kg payload.",
                "physics": {
                    "payload_mass_kg": 1.0,
                    "payload_com_offset_m": [0.03, 0.03, 0.05],
                },
                "control": {
                    "action_delay_steps": 1,
                    "impedance_gain_alpha": 0.50,
                },
            },
        }
    )
    path.write_text(yaml.safe_dump(document, sort_keys=False))


def suite_document(
    manifest: dict[str, Any],
    scenario_file: Path,
    *,
    num_envs: int,
    episodes_per_job: int,
    evaluation_seed: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, tuple[str, int]]]:
    policies: dict[str, dict[str, str]] = {}
    identities: dict[str, tuple[str, int]] = {}
    for seed_value in manifest["target_seeds"]:
        seed = int(seed_value)
        entry = manifest["seeds"][str(seed)]
        for group, checkpoint in (
            ("nominal", entry["source_checkpoint"]),
            ("curriculum_dr", entry["stages"]["full"]["checkpoint"]),
        ):
            name = f"{group}_seed_{seed}"
            policies[name] = {
                "checkpoint": str(checkpoint),
                "task": TASK,
                "description": f"{group} position-increment policy, training seed {seed}.",
            }
            identities[name] = (group, seed)
    return (
        {
            "version": 1,
            "name": "axis3_curriculum_nominal_vs_dr",
            "baseline_policy": "nominal_seed_42" if "nominal_seed_42" in policies else next(iter(policies)),
            "policies": policies,
            "scenario_file": str(scenario_file),
            "scenarios": list(SCENARIOS),
            "seeds": [evaluation_seed],
            "evaluation": {
                "task": TASK,
                "num_envs": num_envs,
                "episodes_per_job": episodes_per_job,
                "success_threshold": 0.03,
                "success_steps": 5,
                "device": device,
                "deterministic": True,
                "visualizer": "none",
                "target_replay": {
                    "enabled": True,
                    "extra_episodes_per_env": 32,
                    "position_ranges": {
                        "x": [0.35, 0.60],
                        "y": [-0.20, 0.20],
                        "z": [0.20, 0.50],
                    },
                },
            },
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


def compile_training_groups(
    suite_dir: Path,
    identities: dict[str, tuple[str, int]],
    evaluation_seed: int,
) -> None:
    rows: list[dict[str, Any]] = []
    for policy_name, (group, training_seed) in identities.items():
        for scenario in SCENARIOS:
            summary_path = (
                suite_dir / "jobs" / policy_name / scenario / f"seed_{evaluation_seed}" / "summary.json"
            )
            summary = json.loads(summary_path.read_text())
            row: dict[str, Any] = {
                "policy_group": group,
                "training_seed": training_seed,
                "scenario": scenario,
                "evaluation_seed": evaluation_seed,
                "checkpoint_policy": policy_name,
                "episodes_recorded": summary.get("episodes_recorded"),
                "summary_path": str(summary_path),
            }
            for metric in SUMMARY_METRICS:
                row[metric] = summary.get(metric)
            rows.append(row)
    rows.sort(key=lambda row: (row["scenario"], row["policy_group"], row["training_seed"]))
    compiled = suite_dir / "compiled"
    compiled.mkdir(parents=True, exist_ok=True)
    with (compiled / "axis3_checkpoint_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for group in ("nominal", "curriculum_dr"):
            members = [row for row in rows if row["scenario"] == scenario and row["policy_group"] == group]
            result: dict[str, Any] = {
                "scenario": scenario,
                "policy_group": group,
                "num_training_seeds": len(members),
                "training_seeds": ";".join(str(row["training_seed"]) for row in members),
                "episodes_recorded": sum(int(row["episodes_recorded"]) for row in members),
            }
            for metric in SUMMARY_METRICS:
                values = [value for row in members if (value := _number(row.get(metric))) is not None]
                result[f"{metric}_mean"] = statistics.fmean(values) if values else ""
                result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
            grouped.append(result)
    with (compiled / "axis3_group_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes-per-job", type=int, default=256)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.episodes_per_job <= 0:
        raise ValueError("num-envs and episodes-per-job must be positive")

    manifest = load_curriculum_manifest(args.training_manifest)
    data_root = Path(os.environ.get("FRANKA_RL_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()
    if not data_root.is_dir() or not os.access(data_root, os.W_OK):
        raise RuntimeError(f"FRANKA_RL_DATA_ROOT unavailable: {data_root}")
    if shutil.disk_usage(data_root).free < MIN_FREE_BYTES:
        raise RuntimeError("FRANKA_RL_DATA_ROOT has less than 100 MiB free")
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output = data_root / "evaluation_suites" / f"{stamp}_axis3_curriculum_dr"
    else:
        output = args.output_dir.expanduser().resolve()
    try:
        output.relative_to(data_root)
    except ValueError as error:
        raise RuntimeError(f"Output must be below {data_root}") from error
    output.mkdir(parents=True, exist_ok=True)

    scenario_file = output / "scenarios.yaml"
    write_scenario_catalog(scenario_file)
    document, identities = suite_document(
        manifest,
        scenario_file,
        num_envs=args.num_envs,
        episodes_per_job=args.episodes_per_job,
        evaluation_seed=args.evaluation_seed,
        device=args.device,
    )
    suite_file = output / "suite.yaml"
    suite_file.write_text(yaml.safe_dump(document, sort_keys=False))
    suite_dir = output / "results"
    coordinator = EvaluationSuiteCoordinator(
        EvaluationSuiteConfig.from_yaml(suite_file),
        output_dir=suite_dir,
        resume=not args.no_resume,
        fail_fast=args.fail_fast,
    )
    success = coordinator.run()
    if success:
        compile_training_groups(suite_dir, identities, args.evaluation_seed)
    index = {
        "version": 1,
        "training_manifest": str(args.training_manifest.expanduser().resolve()),
        "suite": str(suite_file),
        "results": str(suite_dir),
        "complete": success,
    }
    index_path = output / "axis3_index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"Axis-3 comparison index: {index_path}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
