#!/usr/bin/env python3
"""Train a seeded policy matrix, reusing declared checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DATA_ROOT = Path("/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data")
MIN_FREE_BYTES = 2 * 1024**3
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class PolicyTrainingSpec:
    name: str
    description: str
    train_task: str
    scenario: str
    point_task: str
    circle_task: str
    experiment_name: str
    run_name_prefix: str
    existing_checkpoints: dict[int, Path]


@dataclass(frozen=True)
class TrainingMatrix:
    name: str
    scenario: str
    target_seeds: tuple[int, ...]
    num_envs: int
    max_iterations: int
    device: str
    policies: tuple[PolicyTrainingSpec, ...]
    source: Path
    source_sha256: str


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ValueError(f"Unresolved environment variable in {value!r}")
        return expanded
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _safe(value: Any, context: str) -> str:
    if not isinstance(value, str) or SAFE_NAME.fullmatch(value) is None:
        raise ValueError(f"Invalid {context}: {value!r}")
    return value


def load_matrix(path: Path) -> TrainingMatrix:
    source = path.expanduser().resolve()
    raw = source.read_bytes()
    doc = _expand(yaml.safe_load(raw) or {})
    if doc.get("version") != 1:
        raise ValueError("Training matrix version must be 1")
    allowed = {"version", "name", "scenario", "target_seeds", "num_envs", "max_iterations", "device", "policies"}
    unknown = set(doc) - allowed
    if unknown:
        raise ValueError(f"Unknown training-matrix keys: {sorted(unknown)}")
    seeds = doc.get("target_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(not isinstance(v, int) or isinstance(v, bool) for v in seeds)
    ):
        raise ValueError("target_seeds must contain at least two integer seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("target_seeds must be unique")
    raw_policies = doc.get("policies")
    if not isinstance(raw_policies, dict) or not raw_policies:
        raise ValueError("policies must be a non-empty mapping")
    policies = []
    fields = {
        "description",
        "train_task",
        "scenario",
        "point_task",
        "circle_task",
        "experiment_name",
        "run_name_prefix",
        "existing_checkpoints",
    }
    for name, value in raw_policies.items():
        name = _safe(name, "policy name")
        if not isinstance(value, dict):
            raise TypeError(f"Policy {name} must be a mapping")
        extra = set(value) - fields
        if extra:
            raise ValueError(f"Unknown keys for policy {name}: {sorted(extra)}")
        checkpoints = {}
        raw_checkpoints = value.get("existing_checkpoints", {})
        if not isinstance(raw_checkpoints, dict):
            raise TypeError(f"{name}.existing_checkpoints must be a mapping")
        for seed, checkpoint in raw_checkpoints.items():
            seed = int(seed)
            if seed not in seeds:
                raise ValueError(f"Existing seed {seed} for {name} is not in target_seeds")
            checkpoints[seed] = Path(str(checkpoint)).expanduser().resolve()
        policies.append(
            PolicyTrainingSpec(
                name=name,
                description=str(value.get("description", "")),
                train_task=str(value["train_task"]),
                scenario=_safe(value.get("scenario", doc.get("scenario")), "policy scenario"),
                point_task=str(value["point_task"]),
                circle_task=str(value["circle_task"]),
                experiment_name=_safe(value["experiment_name"], "experiment name"),
                run_name_prefix=_safe(value["run_name_prefix"], "run-name prefix"),
                existing_checkpoints=checkpoints,
            )
        )
    for field in ("num_envs", "max_iterations"):
        if not isinstance(doc.get(field), int) or isinstance(doc[field], bool) or doc[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    return TrainingMatrix(
        name=_safe(doc.get("name"), "matrix name"),
        scenario=_safe(doc.get("scenario"), "scenario"),
        target_seeds=tuple(seeds),
        num_envs=doc["num_envs"],
        max_iterations=doc["max_iterations"],
        device=str(doc["device"]),
        policies=tuple(policies),
        source=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


class SeedTrainingCoordinator:
    def __init__(
        self, matrix: TrainingMatrix, output_dir: Path | None, dry_run: bool, fail_fast: bool, selected: set[str] | None
    ):
        self.matrix = matrix
        self.dry_run = dry_run
        self.fail_fast = fail_fast
        self.selected = selected
        self.repo_root = Path(__file__).resolve().parents[2]
        self.train_script = self.repo_root / "scripts" / "rsl_rl" / "train.py"
        self.data_root = Path(os.environ.get("FRANKA_RL_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()
        self._validate_root()
        if output_dir is None:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = self.data_root / "training_matrices" / f"{stamp}_{matrix.name}"
        self.output_dir = output_dir.expanduser().resolve()
        self._under_root(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "checkpoint_manifest.json"
        self.manifest = self._initial_manifest()
        if self.manifest_path.is_file():
            previous = json.loads(self.manifest_path.read_text())
            if previous.get("source_sha256") != matrix.source_sha256:
                raise RuntimeError("Existing output directory belongs to a different training matrix")
            self.manifest = previous

    def _initial_manifest(self) -> dict[str, Any]:
        policies = {}
        for policy in self.matrix.policies:
            checkpoints = {
                str(seed): {"path": str(path), "source": "existing", "complete": path.is_file()}
                for seed, path in policy.existing_checkpoints.items()
            }
            policies[policy.name] = {
                "description": policy.description,
                "train_task": policy.train_task,
                "scenario": policy.scenario,
                "point_task": policy.point_task,
                "circle_task": policy.circle_task,
                "experiment_name": policy.experiment_name,
                "checkpoints": checkpoints,
            }
        return {
            "version": 1,
            "name": self.matrix.name,
            "source": str(self.matrix.source),
            "source_sha256": self.matrix.source_sha256,
            "target_seeds": list(self.matrix.target_seeds),
            "num_envs": self.matrix.num_envs,
            "max_iterations": self.matrix.max_iterations,
            "device": self.matrix.device,
            "scenario": self.matrix.scenario,
            "policies": policies,
            "jobs": [],
        }

    def run(self) -> bool:
        failures = []
        jobs = []
        for policy in self.matrix.policies:
            if self.selected and policy.name not in self.selected:
                continue
            for seed in self.matrix.target_seeds:
                entry = self.manifest["policies"][policy.name]["checkpoints"].get(str(seed))
                if entry and entry.get("complete") and Path(entry["path"]).is_file():
                    print(f"[reuse] {policy.name} seed {seed}: {entry['path']}")
                    continue
                jobs.append((policy, seed))
        print(f"Training jobs required: {len(jobs)}")
        for index, (policy, seed) in enumerate(jobs, 1):
            print(f"[{index}/{len(jobs)}] {policy.name} seed {seed}")
            command = self._command(policy, seed)
            if self.dry_run:
                print("  " + " ".join(command))
                continue
            ok, error = self._run_job(policy, seed, command)
            if not ok:
                failures.append({"policy": policy.name, "seed": seed, "error": error})
                print(f"  FAILED: {error}")
                if self.fail_fast:
                    break
        self.manifest["failures"] = failures
        self.manifest["updated_at"] = datetime.now().astimezone().isoformat()
        self._write_manifest()
        print(f"Checkpoint manifest: {self.manifest_path}")
        return not failures and (self.dry_run or self._all_selected_complete())

    def _command(self, policy: PolicyTrainingSpec, seed: int) -> list[str]:
        return [
            sys.executable,
            "-u",
            str(self.train_script),
            "--task",
            policy.train_task,
            "--scenario",
            policy.scenario,
            "--num_envs",
            str(self.matrix.num_envs),
            "--max_iterations",
            str(self.matrix.max_iterations),
            "--seed",
            str(seed),
            "--device",
            self.matrix.device,
            "--run_name",
            f"{policy.run_name_prefix}_seed_{seed}",
            "--headless",
        ]

    def _run_job(self, policy: PolicyTrainingSpec, seed: int, command: list[str]) -> tuple[bool, str | None]:
        job_dir = self.output_dir / "jobs" / policy.name / f"seed_{seed}"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        run_name = f"{policy.run_name_prefix}_seed_{seed}"
        root = self.data_root / "runs" / "logs" / "rsl_rl" / policy.experiment_name
        root.mkdir(parents=True, exist_ok=True)
        before = {p.resolve() for p in root.glob(f"*_{run_name}") if p.is_dir()}
        runs_dir = self.data_root / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        with (job_dir / "stdout.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=runs_dir,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            return False, f"trainer returned {result.returncode}; see {job_dir / 'stdout.log'}"
        candidates = [p.resolve() for p in root.glob(f"*_{run_name}") if p.is_dir() and p.resolve() not in before]
        if not candidates:
            candidates = [p.resolve() for p in root.glob(f"*_{run_name}") if p.is_dir()]
        if not candidates:
            return False, f"no run directory found below {root}"
        run_dir = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        checkpoint = run_dir / f"model_{self.matrix.max_iterations - 1}.pt"
        if not checkpoint.is_file():
            return False, f"expected final checkpoint not found: {checkpoint}"
        entry = {
            "path": str(checkpoint),
            "source": "trained",
            "complete": True,
            "run_dir": str(run_dir),
            "seed": seed,
            "command": command,
        }
        self.manifest["policies"][policy.name]["checkpoints"][str(seed)] = entry
        self.manifest["jobs"].append(
            {
                "policy": policy.name,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "completed_at": datetime.now().astimezone().isoformat(),
            }
        )
        self._write_manifest()
        print(f"  checkpoint: {checkpoint}")
        return True, None

    def _all_selected_complete(self) -> bool:
        names = self.selected or {p.name for p in self.matrix.policies}
        return all(
            self.manifest["policies"][name]["checkpoints"].get(str(seed), {}).get("complete")
            and Path(self.manifest["policies"][name]["checkpoints"][str(seed)]["path"]).is_file()
            for name in names
            for seed in self.matrix.target_seeds
        )

    def _write_manifest(self):
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2) + "\n")
        tmp.replace(self.manifest_path)

    def _validate_root(self):
        if not self.data_root.is_dir() or not os.access(self.data_root, os.W_OK):
            raise RuntimeError(f"FRANKA_RL_DATA_ROOT unavailable or not writable: {self.data_root}")
        if shutil.disk_usage(self.data_root).free < MIN_FREE_BYTES:
            raise RuntimeError("FRANKA_RL_DATA_ROOT has less than 2 GiB free")
        if not self.train_script.is_file():
            raise FileNotFoundError(self.train_script)

    def _under_root(self, path: Path):
        try:
            path.relative_to(self.data_root)
        except ValueError as e:
            raise RuntimeError(f"Output must be under {self.data_root}: {path}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--policy", action="append", dest="policies", help="Run only this policy group; repeatable")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    matrix = load_matrix(args.config)
    selected = set(args.policies) if args.policies else None
    known = {p.name for p in matrix.policies}
    if selected and not selected <= known:
        parser.error(f"unknown policies: {sorted(selected - known)}")
    runner = SeedTrainingCoordinator(matrix, args.output_dir, args.dry_run, args.fail_fast, selected)
    return 0 if runner.run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
