from __future__ import annotations

import csv
import hashlib
import importlib.metadata as importlib_metadata
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("franka-rl", "isaaclab", "isaaclab-rl", "isaacsim", "rsl-rl-lib", "gymnasium", "torch"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_metadata() -> dict[str, str | bool | None]:
    repo_root = next((parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()), None)
    if repo_root is None:
        return {"root": None, "revision": None, "dirty": None}

    try:
        revision = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"root": str(repo_root), "revision": None, "dirty": None}

    return {"root": str(repo_root), "revision": revision, "dirty": dirty}


@dataclass(frozen=True)
class EvaluationConfig:
    num_episodes: int
    success_threshold: float
    success_steps: int
    seed: int
    output_dir: Path

    scenario: dict[str, Any] = field(default_factory=dict)
    record_domain_parameters: bool = True
    domain_parameter_schema: dict[str, Any] = field(default_factory=dict)
    task_name: str = ""
    command_name: str = "ee_pose"
    real_time: bool = False
    deterministic: bool = False

    def __post_init__(self):
        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be positive.")
        if self.success_threshold <= 0.0:
            raise ValueError("success_threshold must be positive.")
        if self.success_steps <= 0:
            raise ValueError("success_steps must be positive.")


@dataclass
class EvaluationResults:
    env_ids: torch.Tensor
    episode_ids: torch.Tensor
    target_positions: torch.Tensor

    successes: torch.Tensor
    timeouts: torch.Tensor
    unsafe_failures: torch.Tensor
    joint_position_failures: torch.Tensor
    joint_velocity_failures: torch.Tensor
    non_finite_failures: torch.Tensor
    other_failures: torch.Tensor

    episode_steps: torch.Tensor
    time_to_success: torch.Tensor
    domain_parameters: dict[str, torch.Tensor]

    metadata: dict[str, Any]

    @property
    def num_episodes(self) -> int:
        return int(self.successes.numel())

    def summary(self) -> dict[str, Any]:
        count = self.num_episodes

        success_count = int(self.successes.sum())
        timeout_count = int(self.timeouts.sum())
        unsafe_count = int(self.unsafe_failures.sum())
        other_count = int(self.other_failures.sum())

        successful_times = self.time_to_success[
            self.successes & torch.isfinite(self.time_to_success)
        ]

        summary: dict[str, Any] = {
            "episodes_recorded": count,
            "successes": success_count,
            "success_rate": success_count / count if count else None,
            "timeouts": timeout_count,
            "timeout_rate": timeout_count / count if count else None,
            "unsafe_failures": unsafe_count,
            "unsafe_failure_rate": unsafe_count / count if count else None,
            "other_failures": other_count,
            "other_failure_rate": other_count / count if count else None,
            "joint_position_failures": int(
                self.joint_position_failures.sum()
            ),
            "joint_velocity_failures": int(
                self.joint_velocity_failures.sum()
            ),
            "non_finite_failures": int(self.non_finite_failures.sum()),
            "mean_episode_steps": (
                float(self.episode_steps.float().mean()) if count else None
            ),
        }

        if successful_times.numel() > 0:
            summary.update(
                {
                    "mean_time_to_success_s": float(
                        successful_times.mean()
                    ),
                    "median_time_to_success_s": float(
                        successful_times.median()
                    ),
                    "p90_time_to_success_s": float(
                        torch.quantile(successful_times, 0.90)
                    ),
                }
            )
        else:
            summary.update(
                {
                    "mean_time_to_success_s": None,
                    "median_time_to_success_s": None,
                    "p90_time_to_success_s": None,
                }
            )

        domain_summary: dict[str, Any] = {}
        schema = self.metadata.get("domain_parameter_schema", {})
        for name, values in self.domain_parameters.items():
            per_joint: dict[str, Any] = {}
            joint_names = schema.get(name, {}).get("joint_names", [])
            for joint_index, joint_name in enumerate(joint_names):
                joint_values = values[:, joint_index].float()
                per_joint[joint_name] = {
                    "min": float(joint_values.min()),
                    "mean": float(joint_values.mean()),
                    "max": float(joint_values.max()),
                }
            domain_summary[name] = {
                "unit": schema.get(name, {}).get("unit"),
                "min": float(values.float().min()),
                "mean": float(values.float().mean()),
                "max": float(values.float().max()),
                "per_joint": per_joint,
            }
        summary["domain_parameters"] = domain_summary

        return summary

    def print_summary(self) -> None:
        summary = self.summary()
        count = summary["episodes_recorded"]

        def percentage(value: int) -> str:
            return f"{value / count:.2%}" if count else "N/A"

        print("\nSuccess evaluation report")
        print(f"Checkpoint: {self.metadata['checkpoint']}")
        print(f"Seed: {self.metadata['seed']}")
        print(
            f"Scenario: {self.metadata['scenario']['name']} "
            f"({self.metadata['scenario']['type']})"
        )
        print(f"Environments: {self.metadata['num_envs']}")
        print(f"Episodes recorded: {count}")
        print(
            "Success criterion: "
            f"error < {self.metadata['success_threshold_m']:.4f} m "
            f"for {self.metadata['success_steps']} consecutive policy steps"
        )

        print(
            f"Sustained successes: {summary['successes']} "
            f"({percentage(summary['successes'])})"
        )
        print(
            f"Timeouts: {summary['timeouts']} "
            f"({percentage(summary['timeouts'])})"
        )
        print(
            f"Unsafe failures: {summary['unsafe_failures']} "
            f"({percentage(summary['unsafe_failures'])})"
        )
        print(
            "  Joint-position failures: "
            f"{summary['joint_position_failures']}"
        )
        print(
            "  Joint-velocity failures: "
            f"{summary['joint_velocity_failures']}"
        )
        print(
            f"  Non-finite failures: {summary['non_finite_failures']}"
        )

        if summary["other_failures"]:
            print(f"Other failures: {summary['other_failures']}")

        if summary["mean_episode_steps"] is not None:
            print(
                "Mean episode length: "
                f"{summary['mean_episode_steps']:.2f} policy steps"
            )

        if summary["mean_time_to_success_s"] is not None:
            print(
                "Mean time to success: "
                f"{summary['mean_time_to_success_s']:.3f} s"
            )
            print(
                "Median time to success: "
                f"{summary['median_time_to_success_s']:.3f} s"
            )
            print(
                "90th-percentile time to success: "
                f"{summary['p90_time_to_success_s']:.3f} s"
            )
        else:
            print("Time to success: N/A")

        if summary["domain_parameters"]:
            print("Episode-start domain parameters:")
            for name, statistics in summary["domain_parameters"].items():
                unit = statistics["unit"] or "unitless"
                print(
                    f"  {name}: min={statistics['min']:.4g}, "
                    f"mean={statistics['mean']:.4g}, "
                    f"max={statistics['max']:.4g} {unit}"
                )

    def save(self, output_dir: Path | None = None) -> None:
        destination = (
            Path(output_dir)
            if output_dir is not None
            else Path(self.metadata["output_dir"])
        )
        destination.mkdir(parents=True, exist_ok=True)

        summary_path = destination / "summary.json"
        manifest_path = destination / "manifest.json"
        episodes_path = destination / "episodes.csv"

        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(self.summary(), file, indent=2)

        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(self.metadata, file, indent=2)

        with episodes_path.open(
            "w", encoding="utf-8", newline=""
        ) as file:
            fieldnames = [
                "env_id",
                "episode_id",
                "target_x",
                "target_y",
                "target_z",
                "success",
                "timeout",
                "unsafe_failure",
                "joint_position_failure",
                "joint_velocity_failure",
                "non_finite_failure",
                "other_failure",
                "episode_steps",
                "time_to_success_s",
            ]
            domain_columns: list[tuple[str, str, int]] = []
            schema = self.metadata.get("domain_parameter_schema", {})
            for parameter_name, values in self.domain_parameters.items():
                joint_names = schema.get(parameter_name, {}).get("joint_names", [])
                if values.ndim != 2 or values.shape[1] != len(joint_names):
                    raise ValueError(
                        f"Domain parameter {parameter_name!r} does not match its schema."
                    )
                for joint_index, joint_name in enumerate(joint_names):
                    column_name = f"{parameter_name}__{joint_name}"
                    fieldnames.append(column_name)
                    domain_columns.append((column_name, parameter_name, joint_index))
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for index in range(self.num_episodes):
                success_time = float(self.time_to_success[index])

                row = {
                        "env_id": int(self.env_ids[index]),
                        "episode_id": int(self.episode_ids[index]),
                        "target_x": float(
                            self.target_positions[index, 0]
                        ),
                        "target_y": float(
                            self.target_positions[index, 1]
                        ),
                        "target_z": float(
                            self.target_positions[index, 2]
                        ),
                        "success": bool(self.successes[index]),
                        "timeout": bool(self.timeouts[index]),
                        "unsafe_failure": bool(
                            self.unsafe_failures[index]
                        ),
                        "joint_position_failure": bool(
                            self.joint_position_failures[index]
                        ),
                        "joint_velocity_failure": bool(
                            self.joint_velocity_failures[index]
                        ),
                        "non_finite_failure": bool(
                            self.non_finite_failures[index]
                        ),
                        "other_failure": bool(
                            self.other_failures[index]
                        ),
                        "episode_steps": int(
                            self.episode_steps[index]
                        ),
                        "time_to_success_s": (
                            success_time
                            if torch.isfinite(
                                self.time_to_success[index]
                            )
                            else ""
                        ),
                    }
                for column_name, parameter_name, joint_index in domain_columns:
                    row[column_name] = float(
                        self.domain_parameters[parameter_name][index, joint_index]
                    )
                writer.writerow(row)

        print(f"Evaluation artifacts written to: {destination}")


class PolicyEvaluator:
    """Run finite, balanced evaluation over vectorized environments."""

    def __init__(
        self,
        env,
        policy: Callable[[Any], torch.Tensor],
        config: EvaluationConfig,
        checkpoint_path: str | Path,
        reset_policy: Callable[[torch.Tensor], None] | None = None,
        domain_parameter_reader: Callable[[Any], dict[str, torch.Tensor]] | None = None,
    ):
        self.env = env
        self.policy = policy
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.reset_policy = reset_policy
        self.domain_parameter_reader = domain_parameter_reader

        if config.record_domain_parameters and domain_parameter_reader is None:
            raise ValueError(
                "record_domain_parameters=True requires a domain_parameter_reader."
            )
        if not config.record_domain_parameters and domain_parameter_reader is not None:
            raise ValueError(
                "A domain_parameter_reader was provided while recording is disabled."
            )

        self.base_env = env.unwrapped
        self.num_envs = self.base_env.num_envs
        self.device = self.base_env.device
        self.step_dt = self.base_env.step_dt

    def run(self) -> EvaluationResults:
        quotas = self._make_balanced_quotas()
        capacity = int(quotas.max())

        shape = (self.num_envs, capacity)

        completed = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        current_steps = torch.zeros_like(completed)

        recorded = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        successes = torch.zeros_like(recorded)
        timeouts = torch.zeros_like(recorded)
        unsafe_failures = torch.zeros_like(recorded)
        joint_position_failures = torch.zeros_like(recorded)
        joint_velocity_failures = torch.zeros_like(recorded)
        non_finite_failures = torch.zeros_like(recorded)
        other_failures = torch.zeros_like(recorded)

        episode_steps = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        time_to_success = torch.full(
            shape, float("nan"), device=self.device
        )
        target_positions = torch.zeros(
            (*shape, 3), dtype=torch.float32, device=self.device
        )

        # Start from an explicit reset. This is required for reset-mode domain
        # randomizers to affect the first recorded episode as well as later ones.
        obs, _ = self.env.reset()
        if self.reset_policy is not None:
            self.reset_policy(
                torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            )

        # Save the target before stepping. On termination, Isaac Lab may
        # already have reset the environment and sampled its next target.
        current_targets = self._get_target_positions().clone()
        if self.domain_parameter_reader is None:
            current_domain_parameters: dict[str, torch.Tensor] = {}
        else:
            current_domain_parameters = self.domain_parameter_reader(self.base_env)

        domain_parameters: dict[str, torch.Tensor] = {}
        for name, values in current_domain_parameters.items():
            if values.shape[0] != self.num_envs:
                raise ValueError(
                    f"Domain parameter {name!r} has {values.shape[0]} rows; "
                    f"expected {self.num_envs}."
                )
            domain_parameters[name] = torch.zeros(
                (*shape, *values.shape[1:]),
                dtype=values.dtype,
                device=self.device,
            )

        interrupted = False

        try:
            while not torch.all(completed >= quotas):
                if not self._visualizer_is_running():
                    interrupted = True
                    break

                start_time = time.time()

                with torch.inference_mode():
                    active = completed < quotas
                    current_steps[active] += 1

                    actions = self.policy(obs)
                    obs, _, dones, _ = self.env.step(actions)
                    dones = dones.bool()

                    terms = self.base_env.termination_manager

                    success_term = terms.get_term("reached_target").bool()
                    timeout_term = terms.get_term("time_out").bool()
                    joint_position = terms.get_term(
                        "joint_position_limit"
                    ).bool()
                    joint_velocity = terms.get_term(
                        "joint_velocity_limit"
                    ).bool()
                    non_finite = terms.get_term(
                        "non_finite_state"
                    ).bool()

                    unsafe = (
                        joint_position | joint_velocity | non_finite
                    )

                    # Make the top-level outcomes mutually exclusive.
                    valid_success = success_term & ~unsafe
                    valid_timeout = (
                        timeout_term & ~valid_success & ~unsafe
                    )
                    unknown_failure = (
                        dones
                        & ~valid_success
                        & ~valid_timeout
                        & ~unsafe
                    )

                    record_finished = dones & active
                    env_ids = torch.where(record_finished)[0]
                    episode_ids = completed[env_ids]

                    recorded[env_ids, episode_ids] = True
                    episode_steps[env_ids, episode_ids] = (
                        current_steps[env_ids]
                    )
                    target_positions[env_ids, episode_ids] = (
                        current_targets[env_ids]
                    )
                    for name, values in current_domain_parameters.items():
                        domain_parameters[name][env_ids, episode_ids] = values[env_ids]

                    successes[env_ids, episode_ids] = (
                        valid_success[env_ids]
                    )
                    timeouts[env_ids, episode_ids] = (
                        valid_timeout[env_ids]
                    )
                    unsafe_failures[env_ids, episode_ids] = unsafe[
                        env_ids
                    ]
                    joint_position_failures[
                        env_ids, episode_ids
                    ] = joint_position[env_ids]
                    joint_velocity_failures[
                        env_ids, episode_ids
                    ] = joint_velocity[env_ids]
                    non_finite_failures[
                        env_ids, episode_ids
                    ] = non_finite[env_ids]
                    other_failures[
                        env_ids, episode_ids
                    ] = unknown_failure[env_ids]

                    successful_env_ids = env_ids[
                        valid_success[env_ids]
                    ]
                    successful_episode_ids = episode_ids[
                        valid_success[env_ids]
                    ]
                    time_to_success[
                        successful_env_ids,
                        successful_episode_ids,
                    ] = (
                        current_steps[successful_env_ids]
                        * self.step_dt
                    )

                    completed[env_ids] += 1
                    current_steps[dones] = 0

                    # env.step() has reset done environments, so this now
                    # reads each environment's next command.
                    next_targets = self._get_target_positions()
                    current_targets[dones] = next_targets[dones]

                    if self.domain_parameter_reader is not None and torch.any(dones):
                        next_parameters = self.domain_parameter_reader(self.base_env)
                        if next_parameters.keys() != current_domain_parameters.keys():
                            raise RuntimeError(
                                "Domain parameter reader changed its keys during evaluation."
                            )
                        for name, values in next_parameters.items():
                            if values.shape != current_domain_parameters[name].shape:
                                raise RuntimeError(
                                    f"Domain parameter {name!r} changed shape during evaluation."
                                )
                            current_domain_parameters[name][dones] = values[dones]

                    if self.reset_policy is not None:
                        self.reset_policy(dones)

                if self.config.real_time:
                    sleep_time = self.step_dt - (
                        time.time() - start_time
                    )
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            interrupted = True

        return self._build_results(
            recorded=recorded,
            target_positions=target_positions,
            successes=successes,
            timeouts=timeouts,
            unsafe_failures=unsafe_failures,
            joint_position_failures=joint_position_failures,
            joint_velocity_failures=joint_velocity_failures,
            non_finite_failures=non_finite_failures,
            other_failures=other_failures,
            episode_steps=episode_steps,
            time_to_success=time_to_success,
            domain_parameters=domain_parameters,
            interrupted=interrupted,
        )

    def _make_balanced_quotas(self) -> torch.Tensor:
        base = self.config.num_episodes // self.num_envs
        remainder = self.config.num_episodes % self.num_envs

        quotas = torch.full(
            (self.num_envs,),
            base,
            dtype=torch.long,
            device=self.device,
        )

        if remainder:
            # Seeded rotation avoids always assigning extra episodes to the
            # same low-numbered environments.
            start = self.config.seed % self.num_envs
            extra_ids = (
                torch.arange(remainder, device=self.device) + start
            ) % self.num_envs
            quotas[extra_ids] += 1

        return quotas

    def _get_target_positions(self) -> torch.Tensor:
        command = self.base_env.command_manager.get_command(
            self.config.command_name
        )
        return command[:, :3]

    def _visualizer_is_running(self) -> bool:
        visualizers = self.base_env.sim.visualizers
        if not visualizers:
            return True

        return any(
            visualizer.is_running() and not visualizer.is_closed
            for visualizer in visualizers
        )

    def _build_results(
        self,
        *,
        recorded: torch.Tensor,
        target_positions: torch.Tensor,
        successes: torch.Tensor,
        timeouts: torch.Tensor,
        unsafe_failures: torch.Tensor,
        joint_position_failures: torch.Tensor,
        joint_velocity_failures: torch.Tensor,
        non_finite_failures: torch.Tensor,
        other_failures: torch.Tensor,
        episode_steps: torch.Tensor,
        time_to_success: torch.Tensor,
        domain_parameters: dict[str, torch.Tensor],
        interrupted: bool,
    ) -> EvaluationResults:
        env_grid = torch.arange(
            self.num_envs, device=self.device
        ).unsqueeze(1).expand_as(recorded)

        episode_grid = torch.arange(
            recorded.shape[1], device=self.device
        ).unsqueeze(0).expand_as(recorded)

        checkpoint_path = self.checkpoint_path.resolve()

        # Transfer only once, after simulation has finished.
        return EvaluationResults(
            env_ids=env_grid[recorded].cpu(),
            episode_ids=episode_grid[recorded].cpu(),
            target_positions=target_positions[recorded].cpu(),
            successes=successes[recorded].cpu(),
            timeouts=timeouts[recorded].cpu(),
            unsafe_failures=unsafe_failures[recorded].cpu(),
            joint_position_failures=(
                joint_position_failures[recorded].cpu()
            ),
            joint_velocity_failures=(
                joint_velocity_failures[recorded].cpu()
            ),
            non_finite_failures=(
                non_finite_failures[recorded].cpu()
            ),
            other_failures=other_failures[recorded].cpu(),
            episode_steps=episode_steps[recorded].cpu(),
            time_to_success=time_to_success[recorded].cpu(),
            domain_parameters={
                name: values[recorded].cpu()
                for name, values in domain_parameters.items()
            },
            metadata={
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "seed": self.config.seed,
                "task": self.config.task_name,
                "scenario": self.config.scenario,
                "domain_parameter_recording": (
                    "episode_start" if self.config.record_domain_parameters else "none"
                ),
                "domain_parameter_schema": self.config.domain_parameter_schema,
                "num_envs": self.num_envs,
                "requested_episodes": self.config.num_episodes,
                "success_threshold_m": (
                    self.config.success_threshold
                ),
                "success_steps": self.config.success_steps,
                "command_name": self.config.command_name,
                "step_dt_s": self.step_dt,
                "device": str(self.device),
                "deterministic": self.config.deterministic,
                "interrupted": interrupted,
                "output_dir": str(self.config.output_dir),
                "git": _git_metadata(),
                "package_versions": _package_versions(),
            },
        )
