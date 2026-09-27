#!/usr/bin/env python3
"""Expand an immutable path grid into ordinary evaluation suites."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from franka_rl.experiments.coordinator import DEFAULT_DATA_ROOT, EvaluationSuiteCoordinator
from franka_rl.experiments.suite_config import EvaluationSuiteConfig
from franka_rl.utils.paths import PathCatalog


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(source: Path, value: str) -> Path:
    candidate = Path(os.path.expandvars(value)).expanduser()
    if "$" in str(candidate):
        raise ValueError(f"Unresolved environment variable in {value!r}")
    return (source.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the FR3 YZ path calibration grid.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and materialize suite YAMLs without Isaac.")
    args = parser.parse_args()

    source = args.config.expanduser().resolve(strict=True)
    raw = source.read_bytes()
    document = yaml.safe_load(raw)
    required = {
        "version", "name", "path_catalog", "paths", "scenario_file", "scenarios",
        "seeds", "policies", "evaluation", "baseline_policy",
    }
    if not isinstance(document, dict) or set(document) != required or document["version"] != 1:
        raise ValueError(f"{source} must be a version-1 calibration file with keys {sorted(required)}")

    path_catalog_path = _resolve(source, document["path_catalog"])
    scenario_file = _resolve(source, document["scenario_file"])
    catalog = PathCatalog.from_yaml(path_catalog_path)
    path_names = document["paths"]
    if not isinstance(path_names, list) or not path_names or len(set(path_names)) != len(path_names):
        raise ValueError("paths must be a non-empty list without duplicates")
    for name in path_names:
        catalog.get(name)

    data_root = Path(os.environ.get("FRANKA_RL_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output = data_root / "evaluation_suites" / f"{stamp}_{document['name']}"
    else:
        output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = output / "generated_suite_configs"
    configs.mkdir(exist_ok=True)

    results = []
    all_passed = True
    for path_name in path_names:
        suite_document = {
            "version": 1,
            "name": f"{document['name']}__{path_name}",
            "policies": document["policies"],
            "scenarios": document["scenarios"],
            "seeds": document["seeds"],
            "evaluation": {
                **document["evaluation"],
                "path": path_name,
                "path_file": str(path_catalog_path),
            },
            "baseline_policy": document["baseline_policy"],
            "scenario_file": str(scenario_file),
        }
        suite_path = configs / f"{path_name}.yaml"
        suite_path.write_text(yaml.safe_dump(suite_document, sort_keys=False))
        path_output = output / "paths" / path_name
        parsed_suite = EvaluationSuiteConfig.from_yaml(suite_path)
        if args.dry_run:
            passed = True
        else:
            passed = EvaluationSuiteCoordinator(
                parsed_suite,
                output_dir=path_output,
                resume=not args.no_resume,
                fail_fast=args.fail_fast,
            ).run()
        results.append({"path": path_name, "passed": passed, "output_dir": str(path_output)})
        all_passed = all_passed and passed
        if args.fail_fast and not passed:
            break

    index = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "path_catalog": str(path_catalog_path),
        "path_catalog_sha256": catalog.sha256,
        "scenario_file": str(scenario_file),
        "scenario_file_sha256": _sha256(scenario_file),
        "hardware_action_delay_steps": 1,
        "action_delay_evidence": (
            "deployment/hardware_control_audit/2026-09-27-shadow-qualification/qualification.json"
        ),
        "results": results,
        "passed": all_passed and len(results) == len(path_names),
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        combined_rows: list[dict[str, str]] = []
        fieldnames: list[str] = []
        for result in results:
            summary_path = Path(result["output_dir"]) / "compiled" / "scenario_summary.csv"
            if not summary_path.is_file():
                continue
            with summary_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    combined = {"path": str(result["path"]), **row}
                    combined_rows.append(combined)
                    for field in combined:
                        if field not in fieldnames:
                            fieldnames.append(field)
        comparison_path = output / "path_scenario_summary.csv"
        with comparison_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)
        index["path_scenario_summary"] = str(comparison_path)
        index["selection"] = {
            "status": "pending_review",
            "reason": "Select a policy only after reviewing every scenario and all safety margins.",
        }
    (output / "calibration_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Calibration index: {output / 'calibration_index.json'}")
    return 0 if index["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
