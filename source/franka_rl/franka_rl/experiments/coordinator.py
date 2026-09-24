"""Subprocess-based coordinator for reproducible Isaac Lab evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from .aggregator import CompletedJob, compile_results
from .artifact_reader import read_artifacts
from .suite_config import EvaluationSuiteConfig, PolicySpec
from .target_sets import ensure_target_set


DEFAULT_DATA_ROOT = Path("/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data")
MIN_FREE_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class EvaluationJob:
    policy: PolicySpec
    checkpoint_sha256: str
    scenario: str
    seed: int
    job_id: str
    output_dir: Path
    target_set_path: Path | None
    target_set_sha256: str | None


class EvaluationSuiteCoordinator:
    def __init__(
        self,
        config: EvaluationSuiteConfig,
        *,
        output_dir: Path | None = None,
        resume: bool = True,
        fail_fast: bool = False,
    ):
        self.config = config
        self.resume = resume
        self.fail_fast = fail_fast
        self.data_root = Path(os.environ.get("FRANKA_RL_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()
        self._validate_data_root()
        self.repo_root = Path(__file__).resolve().parents[4]
        self.evaluate_script = self.repo_root / "scripts" / "rsl_rl" / "evaluate.py"
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = self.data_root / "evaluation_suites" / f"{timestamp}_{config.name}"
        self.output_dir = output_dir.expanduser().resolve()
        self._require_under_data_root(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_log = self.output_dir / "jobs.jsonl"
        self._available_scenarios, self._scenario_catalog_sha256 = self._load_scenario_catalog()
        self._checkpoint_hashes = self._validate_inputs()
        self._target_sets = self._prepare_target_sets()

    def run(self) -> bool:
        jobs = self._make_jobs()
        self._write_suite_manifest(jobs)
        completed: list[CompletedJob] = []
        failures: list[dict[str, Any]] = []

        for index, job in enumerate(jobs, start=1):
            print(
                f"[{index}/{len(jobs)}] {job.policy.name} / {job.scenario} / seed {job.seed}"
            )
            artifacts = self._read_if_valid(job) if self.resume else None
            if artifacts is not None:
                print(f"  Reusing completed job {job.job_id}")
                self._append_status("reused", {"job_id": job.job_id})
                completed.append(self._completed_job(job, artifacts))
                continue

            success, error = self._run_job(job)
            if success:
                artifacts = self._read_if_valid(job)
                if artifacts is None:
                    success = False
                    error = "Evaluator exited successfully but artifacts failed validation."
                else:
                    completed.append(self._completed_job(job, artifacts))
                    self._append_status("complete", {"job_id": job.job_id})
            if not success:
                failure = {
                    "job_id": job.job_id,
                    "policy": job.policy.name,
                    "scenario": job.scenario,
                    "seed": job.seed,
                    "error": error,
                }
                failures.append(failure)
                self._append_status("failed", failure)
                print(f"  FAILED: {error}")
                if self.fail_fast:
                    break

        compile_results(
            completed,
            self.output_dir / "compiled",
            baseline_policy=self.config.baseline_policy,
        )
        self._write_json_atomic(self.output_dir / "failures.json", failures)
        print(f"Compiled results: {self.output_dir / 'compiled'}")
        return not failures and len(completed) == len(jobs)

    def _make_jobs(self) -> list[EvaluationJob]:
        jobs: list[EvaluationJob] = []
        for policy in self.config.policies:
            checkpoint_sha256 = self._checkpoint_hashes[policy.name]
            for scenario in self.config.scenarios:
                for seed in self.config.seeds:
                    descriptor = {
                        "suite_sha256": self.config.source_sha256,
                        "policy": policy.name,
                        "checkpoint_sha256": checkpoint_sha256,
                        "scenario": scenario,
                        "scenario_catalog_sha256": self._scenario_catalog_sha256,
                        "target_set_sha256": (
                            self._target_sets[seed]["sha256"] if seed in self._target_sets else None
                        ),
                        "seed": seed,
                        "evaluation": asdict(self.config.evaluation),
                    }
                    digest = hashlib.sha256(
                        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                    job_id = f"{policy.name}__{scenario}__seed_{seed}__{digest[:12]}"
                    output_dir = self.output_dir / "jobs" / policy.name / scenario / f"seed_{seed}"
                    target_set = self._target_sets.get(seed)
                    jobs.append(
                        EvaluationJob(
                            policy,
                            checkpoint_sha256,
                            scenario,
                            seed,
                            job_id,
                            output_dir,
                            Path(target_set["path"]) if target_set else None,
                            str(target_set["sha256"]) if target_set else None,
                        )
                    )
        return jobs

    def _run_job(self, job: EvaluationJob) -> tuple[bool, str | None]:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        # A rerun is not complete until the evaluator writes a fresh marker.
        (job.output_dir / "completed.json").unlink(missing_ok=True)
        (job.output_dir / "completed.json.tmp").unlink(missing_ok=True)
        command = self._command(job)
        self._write_json_atomic(
            job.output_dir / "job.json",
            {
                "job_id": job.job_id,
                "policy": job.policy.name,
                "scenario": job.scenario,
                "seed": job.seed,
                "checkpoint": str(job.policy.checkpoint),
                "checkpoint_sha256": job.checkpoint_sha256,
                "target_set": (
                    {"path": str(job.target_set_path), "sha256": job.target_set_sha256}
                    if job.target_set_path is not None
                    else None
                ),
                "command": command,
            },
        )
        self._append_status("started", {"job_id": job.job_id})
        log_path = job.output_dir / "stdout.log"
        runs_dir = self.data_root / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    command,
                    cwd=runs_dir,
                    env=os.environ.copy(),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
        except OSError as error:
            return False, f"Could not launch evaluator: {error}"
        if result.returncode != 0:
            return False, f"Evaluator returned {result.returncode}; see {log_path}."
        return True, None

    def _command(self, job: EvaluationJob) -> list[str]:
        evaluation = self.config.evaluation
        command = [
            sys.executable,
            "-u",
            str(self.evaluate_script),
            "--task",
            evaluation.task,
            "--checkpoint",
            str(job.policy.checkpoint),
            "--scenario",
            job.scenario,
            "--num_envs",
            str(evaluation.num_envs),
            "--num_episodes",
            str(evaluation.episodes_per_job),
            "--success_threshold",
            str(evaluation.success_threshold),
            "--success_steps",
            str(evaluation.success_steps),
            "--seed",
            str(job.seed),
            "--device",
            evaluation.device,
            "--viz",
            evaluation.visualizer,
            "--output-dir",
            str(job.output_dir),
            "--job-id",
            job.job_id,
        ]
        if evaluation.deterministic:
            command.append("--deterministic")
        if self.config.scenario_file is not None:
            command.extend(("--scenario-file", str(self.config.scenario_file)))
        if job.target_set_path is not None:
            command.extend(("--target-set", str(job.target_set_path)))
        return command

    def _read_if_valid(self, job: EvaluationJob):
        try:
            return read_artifacts(
                job.output_dir,
                expected_job_id=job.job_id,
                expected_episodes=self.config.evaluation.episodes_per_job,
                expected_checkpoint_sha256=job.checkpoint_sha256,
                expected_target_set_sha256=job.target_set_sha256,
            )
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _completed_job(self, job: EvaluationJob, artifacts) -> CompletedJob:
        return CompletedJob(
            policy=job.policy.name,
            scenario=job.scenario,
            seed=job.seed,
            job_id=job.job_id,
            checkpoint_sha256=job.checkpoint_sha256,
            artifacts=artifacts,
        )

    def _validate_inputs(self) -> dict[str, str]:
        if not self.evaluate_script.is_file():
            raise FileNotFoundError(f"Evaluator script not found: {self.evaluate_script}")
        unknown = set(self.config.scenarios) - self._available_scenarios
        if unknown:
            raise ValueError(f"Unknown suite scenarios: {', '.join(sorted(unknown))}.")
        hashes: dict[str, str] = {}
        for policy in self.config.policies:
            if not policy.checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint not found for {policy.name}: {policy.checkpoint}")
            hashes[policy.name] = _sha256(policy.checkpoint)
        return hashes

    def _load_scenario_catalog(self) -> tuple[set[str], str]:
        if self.config.scenario_file is None:
            raw = files("franka_rl").joinpath("config/scenarios.yaml").read_bytes()
        else:
            if not self.config.scenario_file.is_file():
                raise FileNotFoundError(f"Scenario file not found: {self.config.scenario_file}")
            raw = self.config.scenario_file.read_bytes()
        document = yaml.safe_load(raw) or {}
        scenarios = document.get("scenarios")
        if not isinstance(scenarios, dict):
            raise TypeError("Scenario catalog must contain a scenarios mapping.")
        return set(scenarios), hashlib.sha256(raw).hexdigest()

    def _write_suite_manifest(self, jobs: list[EvaluationJob]) -> None:
        manifest = {
            "created_at": datetime.now().astimezone().isoformat(),
            "config": self.config.to_dict(),
            "checkpoint_sha256": self._checkpoint_hashes,
            "scenario_catalog_sha256": self._scenario_catalog_sha256,
            "target_sets": self._target_sets,
            "jobs": [job.job_id for job in jobs],
        }
        self._write_json_atomic(self.output_dir / "suite_manifest.json", manifest)

    def _append_status(self, status: str, values: dict[str, Any]) -> None:
        record = {
            "time": datetime.now().astimezone().isoformat(),
            "status": status,
            **values,
        }
        with self.jobs_log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
            file.flush()

    @staticmethod
    def _write_json_atomic(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2)
            file.write("\n")
        temporary.replace(path)

    def _validate_data_root(self) -> None:
        if not self.data_root.is_dir():
            raise RuntimeError(f"Franka RL data root is unavailable: {self.data_root}")
        if not os.access(self.data_root, os.W_OK):
            raise RuntimeError(f"Franka RL data root is not writable: {self.data_root}")
        free = shutil.disk_usage(self.data_root).free
        if free < MIN_FREE_BYTES:
            raise RuntimeError(f"Franka RL data root has only {free / 1024**2:.1f} MiB free.")

    def _require_under_data_root(self, path: Path) -> None:
        try:
            path.relative_to(self.data_root)
        except ValueError as error:
            raise RuntimeError(f"Suite output must be under {self.data_root}: {path}") from error

    def _prepare_target_sets(self) -> dict[int, dict[str, Any]]:
        settings = self.config.evaluation.target_replay
        if settings is None:
            return {}
        result: dict[int, dict[str, Any]] = {}
        for seed in self.config.seeds:
            path = self.output_dir / "target_sets" / f"seed_{seed}.json"
            result[seed] = ensure_target_set(
                path,
                seed=seed,
                num_envs=self.config.evaluation.num_envs,
                total_episodes=self.config.evaluation.episodes_per_job,
                settings=settings,
            )
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
