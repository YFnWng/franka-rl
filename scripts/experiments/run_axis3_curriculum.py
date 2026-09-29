#!/usr/bin/env python3
"""Run a matched three-seed nominal-to-DR curriculum."""

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
MODEL_NAME = re.compile(r"model_(\d+)\.pt")


@dataclass(frozen=True)
class Stage:
    name: str
    task: str
    scenario: str
    iterations: int
    learning_rate: float
    schedule: str


@dataclass(frozen=True)
class Curriculum:
    name: str
    seeds: tuple[int, ...]
    num_envs: int
    device: str
    source_manifest: Path
    source_policy: str
    experiment_name: str
    stages: tuple[Stage, ...]
    source: Path
    source_sha256: str


def checkpoint_iteration(path: Path) -> int:
    """Return the iteration encoded by an RSL-RL checkpoint filename."""
    match = MODEL_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Checkpoint must be named model_<iteration>.pt: {path}")
    return int(match.group(1))


def expected_final_iteration(source: Path, iterations: int) -> int:
    """Compute the final iteration produced by RSL-RL's inclusive loop."""
    return checkpoint_iteration(source) + iterations - 1


def stage_completion_error(
    run_dir: Path,
    checkpoint: Path,
    source: Path,
    iterations: int,
) -> str | None:
    """Return why a stage is incomplete, or ``None`` when it is valid."""
    failure = run_dir / "ppo_numerical_failure.json"
    if failure.is_file():
        return f"numerical failure artifact exists: {failure}"
    expected = expected_final_iteration(source, iterations)
    actual = checkpoint_iteration(checkpoint)
    if actual < expected:
        return f"last checkpoint is model_{actual}.pt; expected at least model_{expected}.pt"
    return None


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


def load_curriculum(path: Path) -> Curriculum:
    source = path.expanduser().resolve()
    raw = source.read_bytes()
    doc = _expand(yaml.safe_load(raw) or {})
    expected = {
        "version", "name", "target_seeds", "num_envs", "device",
        "source_manifest", "source_policy", "experiment_name", "stages",
    }
    if doc.get("version") != 1:
        raise ValueError("Curriculum version must be 1")
    unknown = set(doc) - expected
    if unknown:
        raise ValueError(f"Unknown curriculum keys: {sorted(unknown)}")
    seeds = doc.get("target_seeds")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
        raise TypeError("target_seeds must be a non-empty integer list")
    if len(set(seeds)) != len(seeds):
        raise ValueError("target_seeds must be unique")
    if type(doc.get("num_envs")) is not int or doc["num_envs"] <= 0:
        raise ValueError("num_envs must be a positive integer")
    raw_stages = doc.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("stages must be a non-empty list")
    stages = []
    known_stage_keys = {"name", "task", "scenario", "iterations", "learning_rate", "schedule"}
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, dict):
            raise TypeError("Each stage must be a mapping")
        extra = set(raw_stage) - known_stage_keys
        if extra:
            raise ValueError(f"Unknown stage keys: {sorted(extra)}")
        iterations = raw_stage.get("iterations")
        learning_rate = raw_stage.get("learning_rate")
        schedule = raw_stage.get("schedule")
        if type(iterations) is not int or iterations <= 0:
            raise ValueError("Stage iterations must be positive integers")
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("Stage learning_rate must be positive")
        if schedule not in {"fixed", "adaptive"}:
            raise ValueError("Stage schedule must be fixed or adaptive")
        stages.append(
            Stage(
                name=_safe(raw_stage.get("name"), "stage name"),
                task=str(raw_stage["task"]),
                scenario=_safe(raw_stage["scenario"], "scenario"),
                iterations=iterations,
                learning_rate=float(learning_rate),
                schedule=schedule,
            )
        )
    names = [stage.name for stage in stages]
    if len(names) != len(set(names)):
        raise ValueError("Stage names must be unique")
    return Curriculum(
        name=_safe(doc.get("name"), "curriculum name"),
        seeds=tuple(seeds),
        num_envs=doc["num_envs"],
        device=str(doc["device"]),
        source_manifest=Path(str(doc["source_manifest"])).expanduser().resolve(),
        source_policy=_safe(doc["source_policy"], "source policy"),
        experiment_name=_safe(doc["experiment_name"], "experiment name"),
        stages=tuple(stages),
        source=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


class CurriculumRunner:
    def __init__(self, curriculum: Curriculum, output_dir: Path | None, dry_run: bool):
        self.cfg = curriculum
        self.dry_run = dry_run
        self.repo_root = Path(__file__).resolve().parents[2]
        self.train_script = self.repo_root / "scripts" / "rsl_rl" / "train.py"
        self.data_root = Path(os.environ.get("FRANKA_RL_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()
        self._validate_data_root()
        if output_dir is None:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = self.data_root / "training_matrices" / f"{stamp}_{curriculum.name}"
        self.output_dir = output_dir.expanduser().resolve()
        self._require_under_data_root(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "curriculum_manifest.json"
        self.sources = self._load_sources()
        self.manifest = self._initial_manifest()
        if self.manifest_path.is_file():
            previous = json.loads(self.manifest_path.read_text())
            if previous.get("source_sha256") != self.cfg.source_sha256:
                raise RuntimeError("Output directory belongs to a different curriculum config")
            self.manifest = previous

    def _validate_data_root(self) -> None:
        if not self.data_root.is_dir() or not os.access(self.data_root, os.W_OK):
            raise RuntimeError(f"FRANKA_RL_DATA_ROOT unavailable or not writable: {self.data_root}")
        if shutil.disk_usage(self.data_root).free < MIN_FREE_BYTES:
            raise RuntimeError("FRANKA_RL_DATA_ROOT has less than 2 GiB free")
        if not self.train_script.is_file():
            raise FileNotFoundError(self.train_script)

    def _require_under_data_root(self, path: Path) -> None:
        try:
            path.relative_to(self.data_root)
        except ValueError as exc:
            raise RuntimeError(f"Output must be under {self.data_root}: {path}") from exc

    def _load_sources(self) -> dict[int, Path]:
        if not self.cfg.source_manifest.is_file():
            raise FileNotFoundError(self.cfg.source_manifest)
        doc = json.loads(self.cfg.source_manifest.read_text())
        entries = doc.get("policies", {}).get(self.cfg.source_policy, {}).get("checkpoints", {})
        sources = {}
        for seed in self.cfg.seeds:
            entry = entries.get(str(seed))
            if not entry or not entry.get("complete"):
                raise ValueError(f"Missing complete nominal checkpoint for seed {seed}")
            checkpoint = Path(entry["path"]).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            sources[seed] = checkpoint
        return sources

    def _initial_manifest(self) -> dict[str, Any]:
        return {
            "version": 1,
            "name": self.cfg.name,
            "source": str(self.cfg.source),
            "source_sha256": self.cfg.source_sha256,
            "source_manifest": str(self.cfg.source_manifest),
            "source_policy": self.cfg.source_policy,
            "target_seeds": list(self.cfg.seeds),
            "num_envs": self.cfg.num_envs,
            "device": self.cfg.device,
            "experiment_name": self.cfg.experiment_name,
            "stages": [
                {
                    "name": stage.name,
                    "task": stage.task,
                    "scenario": stage.scenario,
                    "iterations": stage.iterations,
                    "learning_rate": stage.learning_rate,
                    "schedule": stage.schedule,
                }
                for stage in self.cfg.stages
            ],
            "seeds": {
                str(seed): {
                    "source_checkpoint": str(self.sources[seed]),
                    "stages": {},
                }
                for seed in self.cfg.seeds
            },
            "jobs": [],
        }

    def run(self) -> bool:
        if not self.dry_run:
            # A loaded manifest may have been written by an older coordinator
            # that incorrectly marked an interrupted matrix as complete.
            self.manifest.pop("completed_at", None)
        for seed in self.cfg.seeds:
            checkpoint = self.sources[seed]
            for stage in self.cfg.stages:
                existing = self.manifest["seeds"][str(seed)]["stages"].get(stage.name)
                if existing and existing.get("complete"):
                    existing_checkpoint = Path(existing["checkpoint"])
                    existing_run_dir = Path(existing["run_dir"])
                    source_matches = Path(existing.get("source_checkpoint", "")) == checkpoint
                    if existing_checkpoint.is_file() and source_matches:
                        error = stage_completion_error(
                            existing_run_dir,
                            existing_checkpoint,
                            checkpoint,
                            stage.iterations,
                        )
                        if error is None:
                            checkpoint = existing_checkpoint
                            print(f"[reuse] seed {seed} / {stage.name}: {checkpoint}")
                            continue
                        existing["complete"] = False
                        existing["validation_error"] = error
                        if not self.dry_run:
                            self._write_manifest()
                        print(f"[rerun] seed {seed} / {stage.name}: {error}")
                print(f"[seed {seed}] stage {stage.name}: resume {checkpoint}")
                command = self._command(seed, stage, checkpoint)
                if self.dry_run:
                    print("  " + " ".join(command))
                    checkpoint = Path(f"<dry-run-{seed}-{stage.name}>")
                    continue
                checkpoint = self._run_stage(seed, stage, checkpoint, command)
        if not self.dry_run:
            self.manifest["completed_at"] = datetime.now().astimezone().isoformat()
            self._write_manifest()
        print(f"Curriculum manifest: {self.manifest_path}")
        return True

    def _command(self, seed: int, stage: Stage, checkpoint: Path) -> list[str]:
        return [
            sys.executable,
            "-u",
            str(self.train_script),
            "--task", stage.task,
            "--scenario", stage.scenario,
            "--num_envs", str(self.cfg.num_envs),
            "--max_iterations", str(stage.iterations),
            "--seed", str(seed),
            "--device", self.cfg.device,
            "--experiment_name", self.cfg.experiment_name,
            "--run_name", f"axis3_curriculum_seed_{seed}_{stage.name}",
            "--learning-rate", str(stage.learning_rate),
            "--schedule", stage.schedule,
            "--resume",
            "--resume-path", str(checkpoint),
            "--reset-optimizer",
            "--headless",
        ]

    def _run_stage(self, seed: int, stage: Stage, source: Path, command: list[str]) -> Path:
        job_dir = self.output_dir / "jobs" / f"seed_{seed}" / stage.name
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        log_root = self.data_root / "runs" / "logs" / "rsl_rl" / self.cfg.experiment_name
        log_root.mkdir(parents=True, exist_ok=True)
        suffix = f"_axis3_curriculum_seed_{seed}_{stage.name}"
        before = {path.resolve() for path in log_root.glob(f"*{suffix}") if path.is_dir()}
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
            raise RuntimeError(
                f"Seed {seed} stage {stage.name} returned {result.returncode}; "
                f"see {job_dir / 'stdout.log'}"
            )
        candidates = [
            path.resolve()
            for path in log_root.glob(f"*{suffix}")
            if path.is_dir() and path.resolve() not in before
        ]
        if not candidates:
            raise RuntimeError(f"No new run directory found below {log_root} for {suffix}")
        run_dir = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        checkpoints = []
        for path in run_dir.glob("model_*.pt"):
            match = MODEL_NAME.fullmatch(path.name)
            if match:
                checkpoints.append((int(match.group(1)), path.resolve()))
        if not checkpoints:
            raise RuntimeError(f"No numbered checkpoint found in {run_dir}")
        final_iteration, checkpoint = max(checkpoints)
        completion_error = stage_completion_error(
            run_dir,
            checkpoint,
            source,
            stage.iterations,
        )
        entry = {
            "complete": completion_error is None,
            "checkpoint": str(checkpoint),
            "checkpoint_iteration": final_iteration,
            "expected_final_iteration": expected_final_iteration(source, stage.iterations),
            "source_checkpoint": str(source),
            "run_dir": str(run_dir),
            "command": command,
        }
        if completion_error is not None:
            entry["validation_error"] = completion_error
        self.manifest["seeds"][str(seed)]["stages"][stage.name] = entry
        job = {
            "seed": seed,
            "stage": stage.name,
            "checkpoint": str(checkpoint),
            "complete": completion_error is None,
            "finished_at": datetime.now().astimezone().isoformat(),
        }
        if completion_error is not None:
            job["validation_error"] = completion_error
        self.manifest["jobs"].append(job)
        self._write_manifest()
        if completion_error is not None:
            raise RuntimeError(
                f"Seed {seed} stage {stage.name} exited without a valid completed run: "
                f"{completion_error}; see {job_dir / 'stdout.log'}"
            )
        print(f"  checkpoint: {checkpoint}")
        return checkpoint

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2) + "\n")
        temporary.replace(self.manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    runner = CurriculumRunner(load_curriculum(args.config), args.output_dir, args.dry_run)
    return 0 if runner.run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
