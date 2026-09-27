from __future__ import annotations

import csv
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
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

    job_id: str | None = None
    target_set: dict[str, Any] | None = None
    scenario: dict[str, Any] = field(default_factory=dict)
    record_domain_parameters: bool = True
    domain_parameter_schema: dict[str, Any] = field(default_factory=dict)
    initial_joint_names: tuple[str, ...] = ()
    task_name: str = ""
    command_name: str = "ee_pose"
    evaluation_protocol: str = "random_point"
    path: dict[str, Any] | None = None
    z_axis_threshold_rad: float | None = None
    real_time: bool = False
    deterministic: bool = False

    def __post_init__(self):
        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be positive.")
        if self.success_threshold <= 0.0:
            raise ValueError("success_threshold must be positive.")
        if self.success_steps <= 0:
            raise ValueError("success_steps must be positive.")
        if self.z_axis_threshold_rad is not None and self.z_axis_threshold_rad <= 0.0:
            raise ValueError("z_axis_threshold_rad must be positive when provided.")


@dataclass
class EvaluationResults:
    env_ids: torch.Tensor
    episode_ids: torch.Tensor
    target_positions: torch.Tensor
    initial_joint_positions: torch.Tensor
    initial_joint_velocities: torch.Tensor

    successes: torch.Tensor
    timeouts: torch.Tensor
    unsafe_failures: torch.Tensor
    joint_position_failures: torch.Tensor
    joint_velocity_failures: torch.Tensor
    non_finite_failures: torch.Tensor
    other_failures: torch.Tensor

    episode_steps: torch.Tensor
    waypoints_reached: torch.Tensor
    waypoints_timed_out: torch.Tensor
    waypoint_outcomes: torch.Tensor
    waypoint_elapsed_steps: torch.Tensor
    waypoint_final_error_m: torch.Tensor
    waypoint_min_error_m: torch.Tensor
    time_to_success: torch.Tensor
    final_position_error_m: torch.Tensor
    min_position_error_m: torch.Tensor
    integrated_position_error_m_s: torch.Tensor
    mean_position_error_m: torch.Tensor
    threshold_overshoot_m: torch.Tensor
    entered_success_region: torch.Tensor
    final_z_axis_error_rad: torch.Tensor
    min_z_axis_error_rad: torch.Tensor
    integrated_z_axis_error_rad_s: torch.Tensor
    mean_z_axis_error_rad: torch.Tensor
    entered_z_axis_region: torch.Tensor
    mean_action_magnitude: torch.Tensor
    peak_action_magnitude: torch.Tensor
    min_joint_limit_margin_rad: torch.Tensor
    velocity_envelope_exceeded: torch.Tensor
    peak_joint_velocity_ratio: torch.Tensor
    mean_command_difference_norm_rad: torch.Tensor
    peak_command_difference_norm_rad: torch.Tensor
    action_clipping_fraction: torch.Tensor
    peak_action_clipping: torch.Tensor
    peak_applied_torque_norm_nm: torch.Tensor
    domain_parameters: dict[str, torch.Tensor]

    metadata: dict[str, Any]

    @property
    def num_episodes(self) -> int:
        return int(self.successes.numel())

    def summary(self) -> dict[str, Any]:
        count = self.num_episodes

        def finite_mean(values: torch.Tensor) -> float | None:
            finite = values[torch.isfinite(values)].float()
            return float(finite.mean()) if finite.numel() else None

        success_count = int(self.successes.sum())
        timeout_count = int(self.timeouts.sum())
        unsafe_count = int(self.unsafe_failures.sum())
        other_count = int(self.other_failures.sum())

        successful_times = self.time_to_success[self.successes & torch.isfinite(self.time_to_success)]

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
            "joint_position_failures": int(self.joint_position_failures.sum()),
            "joint_velocity_failures": int(self.joint_velocity_failures.sum()),
            "non_finite_failures": int(self.non_finite_failures.sum()),
            "mean_episode_steps": (float(self.episode_steps.float().mean()) if count else None),
            "mean_waypoints_reached": (
                float(self.waypoints_reached.float().mean())
                if count and torch.all(self.waypoints_reached >= 0)
                else None
            ),
            "total_waypoint_timeouts": (
                int(self.waypoints_timed_out.sum())
                if count and torch.all(self.waypoints_timed_out >= 0)
                else None
            ),
            "mean_final_position_error_m": (float(self.final_position_error_m.float().mean()) if count else None),
            "mean_min_position_error_m": (float(self.min_position_error_m.float().mean()) if count else None),
            "mean_integrated_position_error_m_s": (
                float(self.integrated_position_error_m_s.float().mean()) if count else None
            ),
            "mean_position_error_m": (float(self.mean_position_error_m.float().mean()) if count else None),
            "mean_threshold_overshoot_m": (float(self.threshold_overshoot_m.float().mean()) if count else None),
            "threshold_entry_rate": (float(self.entered_success_region.float().mean()) if count else None),
            "mean_final_z_axis_error_rad": finite_mean(self.final_z_axis_error_rad),
            "mean_min_z_axis_error_rad": finite_mean(self.min_z_axis_error_rad),
            "mean_integrated_z_axis_error_rad_s": finite_mean(self.integrated_z_axis_error_rad_s),
            "mean_z_axis_error_rad": finite_mean(self.mean_z_axis_error_rad),
            "z_axis_threshold_entry_rate": (
                float(self.entered_z_axis_region.float().mean())
                if count and self.metadata.get("z_axis_threshold_rad") is not None
                else None
            ),
            "mean_action_magnitude": (float(self.mean_action_magnitude.float().mean()) if count else None),
            "mean_peak_action_magnitude": (float(self.peak_action_magnitude.float().mean()) if count else None),
            "mean_min_joint_limit_margin_rad": (
                float(self.min_joint_limit_margin_rad.float().mean()) if count else None
            ),
            "worst_joint_limit_margin_rad": (float(self.min_joint_limit_margin_rad.float().min()) if count else None),
            "velocity_envelope_exceeded_episodes": int(self.velocity_envelope_exceeded.sum()),
            "velocity_envelope_exceedance_rate": (
                float(self.velocity_envelope_exceeded.float().mean()) if count else None
            ),
            "mean_peak_joint_velocity_ratio": (float(self.peak_joint_velocity_ratio.float().mean()) if count else None),
            "worst_peak_joint_velocity_ratio": (float(self.peak_joint_velocity_ratio.float().max()) if count else None),
            "mean_command_difference_norm_rad": (
                float(self.mean_command_difference_norm_rad.float().mean()) if count else None
            ),
            "mean_peak_command_difference_norm_rad": (
                float(self.peak_command_difference_norm_rad.float().mean()) if count else None
            ),
            "worst_command_difference_norm_rad": (
                float(self.peak_command_difference_norm_rad.float().max()) if count else None
            ),
            "mean_action_clipping_fraction": (float(self.action_clipping_fraction.float().mean()) if count else None),
            "worst_action_clipping": (float(self.peak_action_clipping.float().max()) if count else None),
            "mean_peak_applied_torque_norm_nm": (
                float(self.peak_applied_torque_norm_nm.float().mean()) if count else None
            ),
            "worst_applied_torque_norm_nm": (float(self.peak_applied_torque_norm_nm.float().max()) if count else None),
        }

        if successful_times.numel() > 0:
            summary.update(
                {
                    "mean_time_to_success_s": float(successful_times.mean()),
                    "median_time_to_success_s": float(successful_times.median()),
                    "p90_time_to_success_s": float(torch.quantile(successful_times, 0.90)),
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
            component_names = schema.get(name, {}).get("component_names", schema.get(name, {}).get("joint_names", []))
            for component_index, component_name in enumerate(component_names):
                component_values = values[:, component_index].float()
                per_joint[component_name] = {
                    "min": float(component_values.min()),
                    "mean": float(component_values.mean()),
                    "max": float(component_values.max()),
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
        print(f"Scenario: {self.metadata['scenario']['name']} ({self.metadata['scenario']['type']})")
        print(f"Environments: {self.metadata['num_envs']}")
        print(f"Episodes recorded: {count}")
        if self.metadata["evaluation_protocol"] == "waypoint_path":
            path = self.metadata["path"]
            print(
                "Success criterion: all "
                f"{len(path['waypoints_m'])} waypoints reached within "
                f"{path['position_threshold_m']:.4f} m before each "
                f"{path['waypoint_timeout_s']:.3f} s timeout"
            )
        elif self.metadata["evaluation_protocol"] == "position_z_axis":
            print(
                "Success criterion: position error < "
                f"{self.metadata['success_threshold_m']:.4f} m and tip-z error < "
                f"{math.degrees(self.metadata['z_axis_threshold_rad']):.2f} deg "
                f"for {self.metadata['success_steps']} consecutive policy steps"
            )
        else:
            print(
                "Success criterion: "
                f"error < {self.metadata['success_threshold_m']:.4f} m "
                f"for {self.metadata['success_steps']} consecutive policy steps"
            )

        if self.metadata["evaluation_protocol"] == "waypoint_path":
            print(f"Complete path successes: {summary['successes']} ({percentage(summary['successes'])})")
            print(
                "Path failures (waypoint or episode timeout): "
                f"{summary['timeouts']} ({percentage(summary['timeouts'])})"
            )
        else:
            print(f"Sustained successes: {summary['successes']} ({percentage(summary['successes'])})")
            print(f"Timeouts: {summary['timeouts']} ({percentage(summary['timeouts'])})")
        print(f"Unsafe failures: {summary['unsafe_failures']} ({percentage(summary['unsafe_failures'])})")
        print(f"  Joint-position failures: {summary['joint_position_failures']}")
        print(f"  Joint-velocity termination failures: {summary['joint_velocity_failures']}")
        print(f"  Non-finite failures: {summary['non_finite_failures']}")

        if summary["other_failures"]:
            print(f"Other failures: {summary['other_failures']}")

        if summary["mean_episode_steps"] is not None:
            print(f"Mean episode length: {summary['mean_episode_steps']:.2f} policy steps")
        if summary["mean_waypoints_reached"] is not None:
            print(f"Mean waypoints reached: {summary['mean_waypoints_reached']:.2f}")
            print(f"Total waypoint timeouts: {summary['total_waypoint_timeouts']}")

        if summary["mean_position_error_m"] is not None:
            print(f"Mean trajectory position error: {summary['mean_position_error_m']:.4f} m")
            print(f"Mean final position error: {summary['mean_final_position_error_m']:.4f} m")
            print(f"Mean integrated position error: {summary['mean_integrated_position_error_m_s']:.4f} m*s")
            print(f"Mean policy-action magnitude: {summary['mean_action_magnitude']:.4f}")
            print(f"Worst soft joint-limit margin: {summary['worst_joint_limit_margin_rad']:.4f} rad")
            print(
                "20%-velocity envelope exceeded: "
                f"{summary['velocity_envelope_exceeded_episodes']} "
                f"({summary['velocity_envelope_exceedance_rate']:.2%})"
            )
            print(f"Worst measured/allowed velocity ratio: {summary['worst_peak_joint_velocity_ratio']:.3f}")
            print(
                "Mean / worst peak command difference: "
                f"{summary['mean_peak_command_difference_norm_rad']:.4f} / "
                f"{summary['worst_command_difference_norm_rad']:.4f} rad"
            )
            print(f"Mean action-clipping fraction: {summary['mean_action_clipping_fraction']:.2%}")
            print(f"Worst applied torque-vector norm: {summary['worst_applied_torque_norm_nm']:.3f} Nm")

        if summary["mean_z_axis_error_rad"] is not None:
            print(
                "Mean trajectory / final / minimum tip-z error: "
                f"{math.degrees(summary['mean_z_axis_error_rad']):.2f} / "
                f"{math.degrees(summary['mean_final_z_axis_error_rad']):.2f} / "
                f"{math.degrees(summary['mean_min_z_axis_error_rad']):.2f} deg"
            )
            print(
                "Mean integrated tip-z error: "
                f"{summary['mean_integrated_z_axis_error_rad_s']:.4f} rad*s"
            )
            print(
                "Tip-z threshold entered: "
                f"{summary['z_axis_threshold_entry_rate']:.2%}"
            )

        if summary["mean_time_to_success_s"] is not None:
            print(f"Mean time to success: {summary['mean_time_to_success_s']:.3f} s")
            print(f"Median time to success: {summary['median_time_to_success_s']:.3f} s")
            print(f"90th-percentile time to success: {summary['p90_time_to_success_s']:.3f} s")
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
        destination = Path(output_dir) if output_dir is not None else Path(self.metadata["output_dir"])
        destination.mkdir(parents=True, exist_ok=True)

        summary_path = destination / "summary.json"
        manifest_path = destination / "manifest.json"
        episodes_path = destination / "episodes.csv"
        waypoints_path = destination / "waypoints.csv"

        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(self.summary(), file, indent=2)

        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(self.metadata, file, indent=2)

        with episodes_path.open("w", encoding="utf-8", newline="") as file:
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
                "waypoints_reached",
                "waypoints_timed_out",
                "time_to_success_s",
                "final_position_error_m",
                "min_position_error_m",
                "integrated_position_error_m_s",
                "mean_position_error_m",
                "threshold_overshoot_m",
                "entered_success_region",
                "final_z_axis_error_rad",
                "min_z_axis_error_rad",
                "integrated_z_axis_error_rad_s",
                "mean_z_axis_error_rad",
                "entered_z_axis_region",
                "mean_action_magnitude",
                "peak_action_magnitude",
                "min_joint_limit_margin_rad",
                "velocity_envelope_exceeded",
                "peak_joint_velocity_ratio",
                "mean_command_difference_norm_rad",
                "peak_command_difference_norm_rad",
                "action_clipping_fraction",
                "peak_action_clipping",
                "peak_applied_torque_norm_nm",
            ]
            initial_state_columns: list[tuple[str, str, int]] = []
            for joint_index, joint_name in enumerate(self.metadata.get("initial_joint_names", [])):
                position_column = f"initial_joint_position__{joint_name}"
                velocity_column = f"initial_joint_velocity__{joint_name}"
                fieldnames.extend((position_column, velocity_column))
                initial_state_columns.extend(
                    (
                        (position_column, "position", joint_index),
                        (velocity_column, "velocity", joint_index),
                    )
                )
            domain_columns: list[tuple[str, str, int]] = []
            schema = self.metadata.get("domain_parameter_schema", {})
            for parameter_name, values in self.domain_parameters.items():
                component_names = schema.get(parameter_name, {}).get(
                    "component_names", schema.get(parameter_name, {}).get("joint_names", [])
                )
                if values.ndim != 2 or values.shape[1] != len(component_names):
                    raise ValueError(f"Domain parameter {parameter_name!r} does not match its schema.")
                for component_index, component_name in enumerate(component_names):
                    column_name = f"{parameter_name}__{component_name}"
                    fieldnames.append(column_name)
                    domain_columns.append((column_name, parameter_name, component_index))
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for index in range(self.num_episodes):
                success_time = float(self.time_to_success[index])

                row = {
                    "env_id": int(self.env_ids[index]),
                    "episode_id": int(self.episode_ids[index]),
                    "target_x": float(self.target_positions[index, 0]),
                    "target_y": float(self.target_positions[index, 1]),
                    "target_z": float(self.target_positions[index, 2]),
                    "success": bool(self.successes[index]),
                    "timeout": bool(self.timeouts[index]),
                    "unsafe_failure": bool(self.unsafe_failures[index]),
                    "joint_position_failure": bool(self.joint_position_failures[index]),
                    "joint_velocity_failure": bool(self.joint_velocity_failures[index]),
                    "non_finite_failure": bool(self.non_finite_failures[index]),
                    "other_failure": bool(self.other_failures[index]),
                    "episode_steps": int(self.episode_steps[index]),
                    "waypoints_reached": (
                        int(self.waypoints_reached[index]) if self.waypoints_reached[index] >= 0 else ""
                    ),
                    "waypoints_timed_out": (
                        int(self.waypoints_timed_out[index]) if self.waypoints_timed_out[index] >= 0 else ""
                    ),
                    "time_to_success_s": (success_time if torch.isfinite(self.time_to_success[index]) else ""),
                    "final_position_error_m": float(self.final_position_error_m[index]),
                    "min_position_error_m": float(self.min_position_error_m[index]),
                    "integrated_position_error_m_s": float(self.integrated_position_error_m_s[index]),
                    "mean_position_error_m": float(self.mean_position_error_m[index]),
                    "threshold_overshoot_m": float(self.threshold_overshoot_m[index]),
                    "entered_success_region": bool(self.entered_success_region[index]),
                    "final_z_axis_error_rad": (
                        float(self.final_z_axis_error_rad[index])
                        if torch.isfinite(self.final_z_axis_error_rad[index])
                        else ""
                    ),
                    "min_z_axis_error_rad": (
                        float(self.min_z_axis_error_rad[index])
                        if torch.isfinite(self.min_z_axis_error_rad[index])
                        else ""
                    ),
                    "integrated_z_axis_error_rad_s": (
                        float(self.integrated_z_axis_error_rad_s[index])
                        if torch.isfinite(self.integrated_z_axis_error_rad_s[index])
                        else ""
                    ),
                    "mean_z_axis_error_rad": (
                        float(self.mean_z_axis_error_rad[index])
                        if torch.isfinite(self.mean_z_axis_error_rad[index])
                        else ""
                    ),
                    "entered_z_axis_region": (
                        bool(self.entered_z_axis_region[index])
                        if self.metadata.get("z_axis_threshold_rad") is not None
                        else ""
                    ),
                    "mean_action_magnitude": float(self.mean_action_magnitude[index]),
                    "peak_action_magnitude": float(self.peak_action_magnitude[index]),
                    "min_joint_limit_margin_rad": float(self.min_joint_limit_margin_rad[index]),
                    "velocity_envelope_exceeded": bool(self.velocity_envelope_exceeded[index]),
                    "peak_joint_velocity_ratio": float(self.peak_joint_velocity_ratio[index]),
                    "mean_command_difference_norm_rad": float(self.mean_command_difference_norm_rad[index]),
                    "peak_command_difference_norm_rad": float(self.peak_command_difference_norm_rad[index]),
                    "action_clipping_fraction": float(self.action_clipping_fraction[index]),
                    "peak_action_clipping": float(self.peak_action_clipping[index]),
                    "peak_applied_torque_norm_nm": float(self.peak_applied_torque_norm_nm[index]),
                }
                for column_name, parameter_name, joint_index in domain_columns:
                    row[column_name] = float(self.domain_parameters[parameter_name][index, joint_index])
                for column_name, state_name, joint_index in initial_state_columns:
                    values = self.initial_joint_positions if state_name == "position" else self.initial_joint_velocities
                    row[column_name] = float(values[index, joint_index])
                writer.writerow(row)

        if self.waypoint_outcomes.shape[1] > 0:
            path_waypoints = self.metadata["path"]["waypoints_m"]
            outcome_names = {0: "unresolved", 1: "reached", 2: "timed_out"}
            with waypoints_path.open("w", encoding="utf-8", newline="") as file:
                fieldnames = [
                    "env_id",
                    "episode_id",
                    "waypoint_index",
                    "target_x",
                    "target_y",
                    "target_z",
                    "outcome",
                    "elapsed_steps",
                    "elapsed_s",
                    "final_error_m",
                    "min_error_m",
                ]
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for episode_index in range(self.num_episodes):
                    for waypoint_index, target in enumerate(path_waypoints):
                        outcome = int(self.waypoint_outcomes[episode_index, waypoint_index])
                        elapsed_steps = int(
                            self.waypoint_elapsed_steps[episode_index, waypoint_index]
                        )
                        final_error = self.waypoint_final_error_m[
                            episode_index, waypoint_index
                        ]
                        min_error = self.waypoint_min_error_m[
                            episode_index, waypoint_index
                        ]
                        writer.writerow(
                            {
                                "env_id": int(self.env_ids[episode_index]),
                                "episode_id": int(self.episode_ids[episode_index]),
                                "waypoint_index": waypoint_index,
                                "target_x": float(target[0]),
                                "target_y": float(target[1]),
                                "target_z": float(target[2]),
                                "outcome": outcome_names[outcome],
                                "elapsed_steps": elapsed_steps,
                                "elapsed_s": elapsed_steps * self.metadata["step_dt_s"],
                                "final_error_m": (
                                    float(final_error) if torch.isfinite(final_error) else ""
                                ),
                                "min_error_m": (
                                    float(min_error) if torch.isfinite(min_error) else ""
                                ),
                            }
                        )

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
        trajectory_state_reader: Callable[[Any], dict[str, torch.Tensor]] | None = None,
        initial_state_reader: Callable[[Any], dict[str, torch.Tensor]] | None = None,
        configure_episode_quotas: Callable[[torch.Tensor], None] | None = None,
    ):
        self.env = env
        self.policy = policy
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.reset_policy = reset_policy
        self.domain_parameter_reader = domain_parameter_reader
        self.trajectory_state_reader = trajectory_state_reader
        self.initial_state_reader = initial_state_reader
        self.configure_episode_quotas = configure_episode_quotas

        if config.record_domain_parameters and domain_parameter_reader is None:
            raise ValueError("record_domain_parameters=True requires a domain_parameter_reader.")
        if not config.record_domain_parameters and domain_parameter_reader is not None:
            raise ValueError("A domain_parameter_reader was provided while recording is disabled.")
        if trajectory_state_reader is None:
            raise ValueError("A trajectory_state_reader is required for continuous evaluation metrics.")
        if initial_state_reader is None:
            raise ValueError("An initial_state_reader is required for paired evaluation.")
        if not config.initial_joint_names:
            raise ValueError("initial_joint_names must be provided.")

        self.base_env = env.unwrapped
        self.num_envs = self.base_env.num_envs
        self.device = self.base_env.device
        self.step_dt = self.base_env.step_dt

    def run(self) -> EvaluationResults:
        quotas = self._make_balanced_quotas()
        if self.configure_episode_quotas is not None:
            self.configure_episode_quotas(quotas.clone())
        capacity = int(quotas.max())

        shape = (self.num_envs, capacity)

        completed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        current_steps = torch.zeros_like(completed)

        recorded = torch.zeros(shape, dtype=torch.bool, device=self.device)
        successes = torch.zeros_like(recorded)
        timeouts = torch.zeros_like(recorded)
        unsafe_failures = torch.zeros_like(recorded)
        joint_position_failures = torch.zeros_like(recorded)
        joint_velocity_failures = torch.zeros_like(recorded)
        non_finite_failures = torch.zeros_like(recorded)
        other_failures = torch.zeros_like(recorded)

        episode_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        waypoints_reached = torch.full(shape, -1, dtype=torch.long, device=self.device)
        waypoints_timed_out = torch.full(shape, -1, dtype=torch.long, device=self.device)
        path_command = (
            self.base_env.command_manager.get_term(self.config.command_name)
            if self.config.evaluation_protocol == "waypoint_path"
            else None
        )
        waypoint_count = path_command.waypoint_count if path_command is not None else 0
        waypoint_shape = (*shape, waypoint_count)
        waypoint_outcomes = torch.zeros(
            waypoint_shape, dtype=torch.int8, device=self.device
        )
        waypoint_elapsed_steps = torch.zeros(
            waypoint_shape, dtype=torch.long, device=self.device
        )
        waypoint_final_error_m = torch.full(
            waypoint_shape, float("nan"), device=self.device
        )
        waypoint_min_error_m = torch.full_like(
            waypoint_final_error_m, float("nan")
        )
        time_to_success = torch.full(shape, float("nan"), device=self.device)
        final_position_error_m = torch.full(shape, float("nan"), device=self.device)
        min_position_error_m = torch.full_like(final_position_error_m, float("nan"))
        integrated_position_error_m_s = torch.full_like(final_position_error_m, float("nan"))
        mean_position_error_m = torch.full_like(final_position_error_m, float("nan"))
        threshold_overshoot_m = torch.full_like(final_position_error_m, float("nan"))
        entered_success_region = torch.zeros_like(recorded)
        final_z_axis_error_rad = torch.full_like(final_position_error_m, float("nan"))
        min_z_axis_error_rad = torch.full_like(final_position_error_m, float("nan"))
        integrated_z_axis_error_rad_s = torch.full_like(final_position_error_m, float("nan"))
        mean_z_axis_error_rad = torch.full_like(final_position_error_m, float("nan"))
        entered_z_axis_region = torch.zeros_like(recorded)
        mean_action_magnitude = torch.full_like(final_position_error_m, float("nan"))
        peak_action_magnitude = torch.full_like(final_position_error_m, float("nan"))
        min_joint_limit_margin_rad = torch.full_like(final_position_error_m, float("nan"))
        velocity_envelope_exceeded = torch.zeros_like(recorded)
        peak_joint_velocity_ratio = torch.full_like(final_position_error_m, float("nan"))
        mean_command_difference_norm_rad = torch.full_like(final_position_error_m, float("nan"))
        peak_command_difference_norm_rad = torch.full_like(final_position_error_m, float("nan"))
        action_clipping_fraction = torch.full_like(final_position_error_m, float("nan"))
        peak_action_clipping = torch.full_like(final_position_error_m, float("nan"))
        peak_applied_torque_norm_nm = torch.full_like(final_position_error_m, float("nan"))
        target_positions = torch.zeros((*shape, 3), dtype=torch.float32, device=self.device)
        joint_count = len(self.config.initial_joint_names)
        initial_joint_positions = torch.zeros((*shape, joint_count), dtype=torch.float32, device=self.device)
        initial_joint_velocities = torch.zeros_like(initial_joint_positions)

        current_final_error = torch.full((self.num_envs,), float("nan"), device=self.device)
        current_min_error = torch.full((self.num_envs,), float("inf"), device=self.device)
        current_integrated_error = torch.zeros(self.num_envs, device=self.device)
        current_overshoot = torch.zeros(self.num_envs, device=self.device)
        current_entered_region = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        current_final_z_axis_error = torch.full((self.num_envs,), float("nan"), device=self.device)
        current_min_z_axis_error = torch.full((self.num_envs,), float("inf"), device=self.device)
        current_integrated_z_axis_error = torch.zeros(self.num_envs, device=self.device)
        current_entered_z_axis_region = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        current_action_sum = torch.zeros(self.num_envs, device=self.device)
        current_peak_action = torch.zeros(self.num_envs, device=self.device)
        current_min_joint_margin = torch.full((self.num_envs,), float("inf"), device=self.device)
        current_velocity_envelope_exceeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        current_peak_joint_velocity_ratio = torch.zeros(self.num_envs, device=self.device)
        current_command_difference_sum = torch.zeros(self.num_envs, device=self.device)
        current_peak_command_difference = torch.zeros(self.num_envs, device=self.device)
        current_clipping_steps = torch.zeros(self.num_envs, device=self.device)
        current_peak_action_clipping = torch.zeros(self.num_envs, device=self.device)
        current_peak_applied_torque = torch.zeros(self.num_envs, device=self.device)

        # Start from an explicit reset. This is required for reset-mode domain
        # randomizers to affect the first recorded episode as well as later ones.
        obs, _ = self.env.reset()
        if self.reset_policy is not None:
            self.reset_policy(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))

        # Save the target before stepping. On termination, Isaac Lab may
        # already have reset the environment and sampled its next target.
        current_targets = self._get_target_positions().clone()
        current_initial_state = self._read_initial_state()
        if self.domain_parameter_reader is None:
            current_domain_parameters: dict[str, torch.Tensor] = {}
        else:
            current_domain_parameters = self.domain_parameter_reader(self.base_env)

        domain_parameters: dict[str, torch.Tensor] = {}
        for name, values in current_domain_parameters.items():
            if values.shape[0] != self.num_envs:
                raise ValueError(f"Domain parameter {name!r} has {values.shape[0]} rows; expected {self.num_envs}.")
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
                    action_magnitude = torch.linalg.vector_norm(actions.reshape(self.num_envs, -1), dim=1)
                    current_action_sum[active] += action_magnitude[active]
                    current_peak_action[active] = torch.maximum(current_peak_action[active], action_magnitude[active])
                    obs, _, dones, _ = self.env.step(actions)
                    dones = dones.bool()

                    trajectory_state = self.trajectory_state_reader(self.base_env)
                    required_state_keys = {
                        "position_error_m",
                        "joint_limit_margin_rad",
                    }
                    if not required_state_keys.issubset(trajectory_state):
                        raise RuntimeError(
                            "Trajectory state reader is missing required keys "
                            f"{sorted(required_state_keys - trajectory_state.keys())}."
                        )
                    position_error = trajectory_state["position_error_m"]
                    joint_margin = trajectory_state["joint_limit_margin_rad"]
                    expected_shape = (self.num_envs,)
                    if any(values.shape != expected_shape for values in trajectory_state.values()):
                        raise RuntimeError(f"Trajectory state tensors must have shape {expected_shape}.")
                    joint_velocity_ratio = trajectory_state.get(
                        "joint_velocity_ratio", torch.full_like(position_error, float("nan"))
                    )
                    command_difference = trajectory_state.get(
                        "command_difference_norm_rad", torch.full_like(position_error, float("nan"))
                    )
                    action_clipping = trajectory_state.get(
                        "action_clipping", torch.full_like(position_error, float("nan"))
                    )
                    applied_torque = trajectory_state.get(
                        "applied_torque_norm_nm", torch.full_like(position_error, float("nan"))
                    )
                    z_axis_error = trajectory_state.get(
                        "z_axis_error_rad", torch.full_like(position_error, float("nan"))
                    )
                    if self.config.z_axis_threshold_rad is not None and not torch.all(
                        torch.isfinite(z_axis_error[active])
                    ):
                        raise RuntimeError(
                            "Z-axis evaluation requires finite z_axis_error_rad values from the trajectory reader."
                        )

                    current_final_error[active] = position_error[active]
                    current_min_error[active] = torch.minimum(current_min_error[active], position_error[active])
                    current_integrated_error[active] += position_error[active] * self.step_dt
                    current_overshoot[active] = torch.where(
                        current_entered_region[active],
                        torch.maximum(
                            current_overshoot[active],
                            torch.clamp(
                                position_error[active] - self.config.success_threshold,
                                min=0.0,
                            ),
                        ),
                        current_overshoot[active],
                    )
                    current_entered_region[active] |= position_error[active] < self.config.success_threshold
                    if self.config.z_axis_threshold_rad is not None:
                        current_final_z_axis_error[active] = z_axis_error[active]
                        current_min_z_axis_error[active] = torch.minimum(
                            current_min_z_axis_error[active], z_axis_error[active]
                        )
                        current_integrated_z_axis_error[active] += z_axis_error[active] * self.step_dt
                        current_entered_z_axis_region[active] |= (
                            z_axis_error[active] < self.config.z_axis_threshold_rad
                        )
                    current_min_joint_margin[active] = torch.minimum(
                        current_min_joint_margin[active],
                        joint_margin[active],
                    )
                    current_velocity_envelope_exceeded[active] |= joint_velocity_ratio[active] > 1.0
                    current_peak_joint_velocity_ratio[active] = torch.maximum(
                        current_peak_joint_velocity_ratio[active],
                        torch.nan_to_num(joint_velocity_ratio[active]),
                    )
                    current_command_difference_sum[active] += torch.nan_to_num(command_difference[active])
                    current_peak_command_difference[active] = torch.maximum(
                        current_peak_command_difference[active],
                        torch.nan_to_num(command_difference[active]),
                    )
                    current_clipping_steps[active] += torch.nan_to_num(action_clipping[active]) > 0.0
                    current_peak_action_clipping[active] = torch.maximum(
                        current_peak_action_clipping[active],
                        torch.nan_to_num(action_clipping[active]),
                    )
                    current_peak_applied_torque[active] = torch.maximum(
                        current_peak_applied_torque[active],
                        torch.nan_to_num(applied_torque[active]),
                    )

                    terms = self.base_env.termination_manager

                    success_term = terms.get_term("reached_target").bool()
                    timeout_term = terms.get_term("time_out").bool()
                    if "waypoint_path_failed" in terms.active_terms:
                        timeout_term |= terms.get_term("waypoint_path_failed").bool()
                    joint_position = terms.get_term("joint_position_limit").bool()
                    joint_velocity = (
                        terms.get_term("joint_velocity_limit").bool()
                        if "joint_velocity_limit" in terms.active_terms
                        else torch.zeros_like(dones)
                    )
                    non_finite = terms.get_term("non_finite_state").bool()

                    unsafe = joint_position | joint_velocity | non_finite

                    # Make the top-level outcomes mutually exclusive.
                    valid_success = success_term & ~unsafe
                    valid_timeout = timeout_term & ~valid_success & ~unsafe
                    unknown_failure = dones & ~valid_success & ~valid_timeout & ~unsafe

                    record_finished = dones & active
                    env_ids = torch.where(record_finished)[0]
                    episode_ids = completed[env_ids]

                    recorded[env_ids, episode_ids] = True
                    episode_steps[env_ids, episode_ids] = current_steps[env_ids]
                    if path_command is not None:
                        waypoints_reached[env_ids, episode_ids] = (
                            path_command.last_episode_waypoints_reached[env_ids]
                        )
                        waypoints_timed_out[env_ids, episode_ids] = (
                            path_command.last_episode_waypoints_timed_out[env_ids]
                        )
                        waypoint_outcomes[env_ids, episode_ids] = (
                            path_command.last_episode_outcomes[env_ids]
                        )
                        waypoint_elapsed_steps[env_ids, episode_ids] = (
                            path_command.last_episode_elapsed_steps[env_ids]
                        )
                        waypoint_final_error_m[env_ids, episode_ids] = (
                            path_command.last_episode_final_error_m[env_ids]
                        )
                        waypoint_min_error_m[env_ids, episode_ids] = (
                            path_command.last_episode_min_error_m[env_ids]
                        )
                    target_positions[env_ids, episode_ids] = current_targets[env_ids]
                    initial_joint_positions[env_ids, episode_ids] = current_initial_state["joint_position"][env_ids]
                    initial_joint_velocities[env_ids, episode_ids] = current_initial_state["joint_velocity"][env_ids]
                    for name, values in current_domain_parameters.items():
                        domain_parameters[name][env_ids, episode_ids] = values[env_ids]

                    successes[env_ids, episode_ids] = valid_success[env_ids]
                    timeouts[env_ids, episode_ids] = valid_timeout[env_ids]
                    unsafe_failures[env_ids, episode_ids] = unsafe[env_ids]
                    joint_position_failures[env_ids, episode_ids] = joint_position[env_ids]
                    joint_velocity_failures[env_ids, episode_ids] = joint_velocity[env_ids]
                    non_finite_failures[env_ids, episode_ids] = non_finite[env_ids]
                    other_failures[env_ids, episode_ids] = unknown_failure[env_ids]

                    successful_env_ids = env_ids[valid_success[env_ids]]
                    successful_episode_ids = episode_ids[valid_success[env_ids]]
                    time_to_success[
                        successful_env_ids,
                        successful_episode_ids,
                    ] = current_steps[successful_env_ids] * self.step_dt

                    final_position_error_m[env_ids, episode_ids] = current_final_error[env_ids]
                    min_position_error_m[env_ids, episode_ids] = current_min_error[env_ids]
                    integrated_position_error_m_s[env_ids, episode_ids] = current_integrated_error[env_ids]
                    mean_position_error_m[env_ids, episode_ids] = current_integrated_error[env_ids] / (
                        current_steps[env_ids] * self.step_dt
                    )
                    threshold_overshoot_m[env_ids, episode_ids] = current_overshoot[env_ids]
                    entered_success_region[env_ids, episode_ids] = current_entered_region[env_ids]
                    if self.config.z_axis_threshold_rad is not None:
                        final_z_axis_error_rad[env_ids, episode_ids] = current_final_z_axis_error[env_ids]
                        min_z_axis_error_rad[env_ids, episode_ids] = current_min_z_axis_error[env_ids]
                        integrated_z_axis_error_rad_s[env_ids, episode_ids] = (
                            current_integrated_z_axis_error[env_ids]
                        )
                        mean_z_axis_error_rad[env_ids, episode_ids] = (
                            current_integrated_z_axis_error[env_ids]
                            / (current_steps[env_ids] * self.step_dt)
                        )
                        entered_z_axis_region[env_ids, episode_ids] = current_entered_z_axis_region[env_ids]
                    mean_action_magnitude[env_ids, episode_ids] = current_action_sum[env_ids] / current_steps[env_ids]
                    peak_action_magnitude[env_ids, episode_ids] = current_peak_action[env_ids]
                    min_joint_limit_margin_rad[env_ids, episode_ids] = current_min_joint_margin[env_ids]
                    velocity_envelope_exceeded[env_ids, episode_ids] = current_velocity_envelope_exceeded[env_ids]
                    peak_joint_velocity_ratio[env_ids, episode_ids] = current_peak_joint_velocity_ratio[env_ids]
                    mean_command_difference_norm_rad[env_ids, episode_ids] = (
                        current_command_difference_sum[env_ids] / current_steps[env_ids]
                    )
                    peak_command_difference_norm_rad[env_ids, episode_ids] = current_peak_command_difference[env_ids]
                    action_clipping_fraction[env_ids, episode_ids] = (
                        current_clipping_steps[env_ids] / current_steps[env_ids]
                    )
                    peak_action_clipping[env_ids, episode_ids] = current_peak_action_clipping[env_ids]
                    peak_applied_torque_norm_nm[env_ids, episode_ids] = current_peak_applied_torque[env_ids]

                    completed[env_ids] += 1
                    current_steps[dones] = 0
                    current_final_error[dones] = float("nan")
                    current_min_error[dones] = float("inf")
                    current_integrated_error[dones] = 0.0
                    current_overshoot[dones] = 0.0
                    current_entered_region[dones] = False
                    current_final_z_axis_error[dones] = float("nan")
                    current_min_z_axis_error[dones] = float("inf")
                    current_integrated_z_axis_error[dones] = 0.0
                    current_entered_z_axis_region[dones] = False
                    current_action_sum[dones] = 0.0
                    current_peak_action[dones] = 0.0
                    current_min_joint_margin[dones] = float("inf")
                    current_velocity_envelope_exceeded[dones] = False
                    current_peak_joint_velocity_ratio[dones] = 0.0
                    current_command_difference_sum[dones] = 0.0
                    current_peak_command_difference[dones] = 0.0
                    current_clipping_steps[dones] = 0.0
                    current_peak_action_clipping[dones] = 0.0
                    current_peak_applied_torque[dones] = 0.0

                    # env.step() has reset done environments, so this now
                    # reads each environment's next command.
                    next_targets = self._get_target_positions()
                    current_targets[dones] = next_targets[dones]
                    next_initial_state = self._read_initial_state()
                    for name, values in next_initial_state.items():
                        current_initial_state[name][dones] = values[dones]

                    if self.domain_parameter_reader is not None and torch.any(dones):
                        next_parameters = self.domain_parameter_reader(self.base_env)
                        if next_parameters.keys() != current_domain_parameters.keys():
                            raise RuntimeError("Domain parameter reader changed its keys during evaluation.")
                        for name, values in next_parameters.items():
                            if values.shape != current_domain_parameters[name].shape:
                                raise RuntimeError(f"Domain parameter {name!r} changed shape during evaluation.")
                            current_domain_parameters[name][dones] = values[dones]

                    if self.reset_policy is not None:
                        self.reset_policy(dones)

                if self.config.real_time:
                    sleep_time = self.step_dt - (time.time() - start_time)
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            interrupted = True

        return self._build_results(
            recorded=recorded,
            target_positions=target_positions,
            initial_joint_positions=initial_joint_positions,
            initial_joint_velocities=initial_joint_velocities,
            successes=successes,
            timeouts=timeouts,
            unsafe_failures=unsafe_failures,
            joint_position_failures=joint_position_failures,
            joint_velocity_failures=joint_velocity_failures,
            non_finite_failures=non_finite_failures,
            other_failures=other_failures,
            episode_steps=episode_steps,
            waypoints_reached=waypoints_reached,
            waypoints_timed_out=waypoints_timed_out,
            waypoint_outcomes=waypoint_outcomes,
            waypoint_elapsed_steps=waypoint_elapsed_steps,
            waypoint_final_error_m=waypoint_final_error_m,
            waypoint_min_error_m=waypoint_min_error_m,
            time_to_success=time_to_success,
            final_position_error_m=final_position_error_m,
            min_position_error_m=min_position_error_m,
            integrated_position_error_m_s=integrated_position_error_m_s,
            mean_position_error_m=mean_position_error_m,
            threshold_overshoot_m=threshold_overshoot_m,
            entered_success_region=entered_success_region,
            final_z_axis_error_rad=final_z_axis_error_rad,
            min_z_axis_error_rad=min_z_axis_error_rad,
            integrated_z_axis_error_rad_s=integrated_z_axis_error_rad_s,
            mean_z_axis_error_rad=mean_z_axis_error_rad,
            entered_z_axis_region=entered_z_axis_region,
            mean_action_magnitude=mean_action_magnitude,
            peak_action_magnitude=peak_action_magnitude,
            min_joint_limit_margin_rad=min_joint_limit_margin_rad,
            velocity_envelope_exceeded=velocity_envelope_exceeded,
            peak_joint_velocity_ratio=peak_joint_velocity_ratio,
            mean_command_difference_norm_rad=mean_command_difference_norm_rad,
            peak_command_difference_norm_rad=peak_command_difference_norm_rad,
            action_clipping_fraction=action_clipping_fraction,
            peak_action_clipping=peak_action_clipping,
            peak_applied_torque_norm_nm=peak_applied_torque_norm_nm,
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
            extra_ids = (torch.arange(remainder, device=self.device) + start) % self.num_envs
            quotas[extra_ids] += 1

        return quotas

    def _get_target_positions(self) -> torch.Tensor:
        command = self.base_env.command_manager.get_command(self.config.command_name)
        return command[:, :3]

    def _read_initial_state(self) -> dict[str, torch.Tensor]:
        state = self.initial_state_reader(self.base_env)
        expected = {"joint_position", "joint_velocity"}
        if state.keys() != expected:
            raise ValueError(f"Initial-state reader returned {set(state)}, expected {expected}.")
        expected_shape = (self.num_envs, len(self.config.initial_joint_names))
        for name, values in state.items():
            if tuple(values.shape) != expected_shape:
                raise ValueError(f"Initial-state {name!r} has shape {tuple(values.shape)}; expected {expected_shape}.")
        return {name: values.clone() for name, values in state.items()}

    def _visualizer_is_running(self) -> bool:
        visualizers = self.base_env.sim.visualizers
        if not visualizers:
            return True

        return any(visualizer.is_running() and not visualizer.is_closed for visualizer in visualizers)

    def _build_results(
        self,
        *,
        recorded: torch.Tensor,
        target_positions: torch.Tensor,
        initial_joint_positions: torch.Tensor,
        initial_joint_velocities: torch.Tensor,
        successes: torch.Tensor,
        timeouts: torch.Tensor,
        unsafe_failures: torch.Tensor,
        joint_position_failures: torch.Tensor,
        joint_velocity_failures: torch.Tensor,
        non_finite_failures: torch.Tensor,
        other_failures: torch.Tensor,
        episode_steps: torch.Tensor,
        waypoints_reached: torch.Tensor,
        waypoints_timed_out: torch.Tensor,
        waypoint_outcomes: torch.Tensor,
        waypoint_elapsed_steps: torch.Tensor,
        waypoint_final_error_m: torch.Tensor,
        waypoint_min_error_m: torch.Tensor,
        time_to_success: torch.Tensor,
        final_position_error_m: torch.Tensor,
        min_position_error_m: torch.Tensor,
        integrated_position_error_m_s: torch.Tensor,
        mean_position_error_m: torch.Tensor,
        threshold_overshoot_m: torch.Tensor,
        entered_success_region: torch.Tensor,
        final_z_axis_error_rad: torch.Tensor,
        min_z_axis_error_rad: torch.Tensor,
        integrated_z_axis_error_rad_s: torch.Tensor,
        mean_z_axis_error_rad: torch.Tensor,
        entered_z_axis_region: torch.Tensor,
        mean_action_magnitude: torch.Tensor,
        peak_action_magnitude: torch.Tensor,
        min_joint_limit_margin_rad: torch.Tensor,
        velocity_envelope_exceeded: torch.Tensor,
        peak_joint_velocity_ratio: torch.Tensor,
        mean_command_difference_norm_rad: torch.Tensor,
        peak_command_difference_norm_rad: torch.Tensor,
        action_clipping_fraction: torch.Tensor,
        peak_action_clipping: torch.Tensor,
        peak_applied_torque_norm_nm: torch.Tensor,
        domain_parameters: dict[str, torch.Tensor],
        interrupted: bool,
    ) -> EvaluationResults:
        env_grid = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand_as(recorded)

        episode_grid = torch.arange(recorded.shape[1], device=self.device).unsqueeze(0).expand_as(recorded)

        checkpoint_path = self.checkpoint_path.resolve()

        # Transfer only once, after simulation has finished.
        return EvaluationResults(
            env_ids=env_grid[recorded].cpu(),
            episode_ids=episode_grid[recorded].cpu(),
            target_positions=target_positions[recorded].cpu(),
            initial_joint_positions=initial_joint_positions[recorded].cpu(),
            initial_joint_velocities=initial_joint_velocities[recorded].cpu(),
            successes=successes[recorded].cpu(),
            timeouts=timeouts[recorded].cpu(),
            unsafe_failures=unsafe_failures[recorded].cpu(),
            joint_position_failures=(joint_position_failures[recorded].cpu()),
            joint_velocity_failures=(joint_velocity_failures[recorded].cpu()),
            non_finite_failures=(non_finite_failures[recorded].cpu()),
            other_failures=other_failures[recorded].cpu(),
            episode_steps=episode_steps[recorded].cpu(),
            waypoints_reached=waypoints_reached[recorded].cpu(),
            waypoints_timed_out=waypoints_timed_out[recorded].cpu(),
            waypoint_outcomes=waypoint_outcomes[recorded].cpu(),
            waypoint_elapsed_steps=waypoint_elapsed_steps[recorded].cpu(),
            waypoint_final_error_m=waypoint_final_error_m[recorded].cpu(),
            waypoint_min_error_m=waypoint_min_error_m[recorded].cpu(),
            time_to_success=time_to_success[recorded].cpu(),
            final_position_error_m=(final_position_error_m[recorded].cpu()),
            min_position_error_m=min_position_error_m[recorded].cpu(),
            integrated_position_error_m_s=(integrated_position_error_m_s[recorded].cpu()),
            mean_position_error_m=mean_position_error_m[recorded].cpu(),
            threshold_overshoot_m=threshold_overshoot_m[recorded].cpu(),
            entered_success_region=entered_success_region[recorded].cpu(),
            final_z_axis_error_rad=final_z_axis_error_rad[recorded].cpu(),
            min_z_axis_error_rad=min_z_axis_error_rad[recorded].cpu(),
            integrated_z_axis_error_rad_s=integrated_z_axis_error_rad_s[recorded].cpu(),
            mean_z_axis_error_rad=mean_z_axis_error_rad[recorded].cpu(),
            entered_z_axis_region=entered_z_axis_region[recorded].cpu(),
            mean_action_magnitude=mean_action_magnitude[recorded].cpu(),
            peak_action_magnitude=peak_action_magnitude[recorded].cpu(),
            min_joint_limit_margin_rad=(min_joint_limit_margin_rad[recorded].cpu()),
            velocity_envelope_exceeded=(velocity_envelope_exceeded[recorded].cpu()),
            peak_joint_velocity_ratio=(peak_joint_velocity_ratio[recorded].cpu()),
            mean_command_difference_norm_rad=(mean_command_difference_norm_rad[recorded].cpu()),
            peak_command_difference_norm_rad=(peak_command_difference_norm_rad[recorded].cpu()),
            action_clipping_fraction=(action_clipping_fraction[recorded].cpu()),
            peak_action_clipping=peak_action_clipping[recorded].cpu(),
            peak_applied_torque_norm_nm=(peak_applied_torque_norm_nm[recorded].cpu()),
            domain_parameters={name: values[recorded].cpu() for name, values in domain_parameters.items()},
            metadata={
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "job_id": self.config.job_id,
                "target_set": self.config.target_set,
                "initial_state_recording": "episode_start",
                "initial_joint_names": list(self.config.initial_joint_names),
                "seed": self.config.seed,
                "task": self.config.task_name,
                "scenario": self.config.scenario,
                "domain_parameter_recording": ("episode_start" if self.config.record_domain_parameters else "none"),
                "domain_parameter_schema": self.config.domain_parameter_schema,
                "num_envs": self.num_envs,
                "requested_episodes": self.config.num_episodes,
                "success_threshold_m": (self.config.success_threshold),
                "z_axis_threshold_rad": self.config.z_axis_threshold_rad,
                "success_steps": self.config.success_steps,
                "command_name": self.config.command_name,
                "evaluation_protocol": self.config.evaluation_protocol,
                "path": self.config.path,
                "step_dt_s": self.step_dt,
                "trajectory_metric_sampling": (
                    "post-physics pre-reset manager term; integrated error uses policy step_dt"
                ),
                "action_metric": "L2 norm of the policy command before fixed-delay processing",
                "joint_limit_margin": "minimum distance to any soft joint-position limit",
                "velocity_ratio": "1 kHz peak measured speed divided by the 20-percent hardware envelope",
                "command_difference": "L2 norm of consecutive physical held joint-position commands",
                "action_clipping": "maximum amount outside normalized [-1, 1] before clipping",
                "applied_torque": "L2 norm of the 1 kHz peak absolute per-joint torque vector",
                "threshold_overshoot": ("maximum distance outside the success radius after first entering it"),
                "device": str(self.device),
                "deterministic": self.config.deterministic,
                "interrupted": interrupted,
                "output_dir": str(self.config.output_dir),
                "git": _git_metadata(),
                "package_versions": _package_versions(),
            },
        )
