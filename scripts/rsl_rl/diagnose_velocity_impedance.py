"""Diagnose the 50 Hz velocity policy and 1 kHz Franky impedance loop."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import cli_args  # isort: skip

import franka_rl.tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from franka_rl.utils.action_delay import FixedActionDelayWrapper, action_term_delay_fill_provider


DEFAULT_DATA_ROOT = Path("/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data")
MIN_FREE_BYTES = 20 * 1024 * 1024
JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 8))


def _validate_output_root(root: Path) -> None:
    if not root.is_dir() or not os.access(root, os.W_OK):
        raise RuntimeError(f"Diagnostic data root is missing or not writable: {root}")
    if shutil.disk_usage(root).free < MIN_FREE_BYTES:
        raise RuntimeError(f"Diagnostic data root has less than {MIN_FREE_BYTES / 1024**2:.0f} MiB free: {root}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_trace_csv(path: Path, trace: dict[str, torch.Tensor], physics_dt: float, decimation: int) -> None:
    vector_fields = tuple(trace)
    header = ["tick", "time_s", "policy_step"]
    for field in vector_fields:
        width = trace[field].shape[1]
        suffixes = JOINT_NAMES[:width] if width == 7 else tuple(f"action_{index}" for index in range(1, width + 1))
        header.extend(f"{field}_{suffix}" for suffix in suffixes)
    arrays = {name: value.numpy() for name, value in trace.items()}
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for tick in range(next(iter(trace.values())).shape[0]):
            row: list[float | int] = [tick, tick * physics_dt, tick // decimation]
            for field in vector_fields:
                row.extend(float(value) for value in arrays[field][tick])
            writer.writerow(row)


def _command_reversals(values: torch.Tensor, epsilon: float = 1.0e-5) -> list[int]:
    results: list[int] = []
    for joint_values in values.T:
        nonzero = joint_values[torch.abs(joint_values) > epsilon]
        if nonzero.numel() < 2:
            results.append(0)
        else:
            results.append(int(((nonzero[1:] * nonzero[:-1]) < 0.0).sum().item()))
    return results


def _summarize(
    trace: dict[str, torch.Tensor],
    physics_dt: float,
    decimation: int,
    qualification_limits: torch.Tensor,
    model_limits: torch.Tensor,
) -> dict:
    measured_velocity = trace["joint_velocity"]
    target_velocity = trace["target_velocity_reference"]
    applied_velocity = trace["applied_velocity_reference"]
    tracking_error = trace["position_reference"] - trace["joint_position"]
    peak_velocity = measured_velocity.abs().amax(dim=0)
    qualification_ratio = peak_velocity / qualification_limits
    model_ratio = peak_velocity / model_limits
    if measured_velocity.shape[0] > 1:
        acceleration = torch.diff(measured_velocity, dim=0) / physics_dt
        peak_acceleration = acceleration.abs().amax(dim=0)
        rms_acceleration = torch.sqrt(torch.mean(acceleration.square(), dim=0))
    else:
        peak_acceleration = torch.zeros(7)
        rms_acceleration = torch.zeros(7)

    policy_target = target_velocity[::decimation]
    policy_action = trace["normalized_action"][::decimation]
    per_joint: dict[str, dict[str, float | int | bool]] = {}
    reversals = _command_reversals(policy_target)
    for index, name in enumerate(JOINT_NAMES):
        measured_nonzero = measured_velocity[:, index][measured_velocity[:, index].abs() > 1.0e-2]
        measured_reversals = (
            int(((measured_nonzero[1:] * measured_nonzero[:-1]) < 0.0).sum().item())
            if measured_nonzero.numel() >= 2
            else 0
        )
        requested_torque = trace["requested_torque"][:, index]
        filtered_torque = trace["filtered_torque"][:, index]
        applied_total_torque = trace["applied_torque"][:, index]
        meaningful_filtered_torque = (requested_torque.abs() > 1.0) & (filtered_torque.abs() > 1.0)
        requested_filtered_opposite_fraction = (
            float(
                (requested_torque[meaningful_filtered_torque] * filtered_torque[meaningful_filtered_torque] < 0.0)
                .float()
                .mean()
            )
            if torch.any(meaningful_filtered_torque)
            else 0.0
        )
        meaningful_total_torque = (requested_torque.abs() > 1.0) & (applied_total_torque.abs() > 1.0)
        requested_applied_total_opposite_fraction = (
            float(
                (
                    requested_torque[meaningful_total_torque]
                    * applied_total_torque[meaningful_total_torque]
                    < 0.0
                )
                .float()
                .mean()
            )
            if torch.any(meaningful_total_torque)
            else 0.0
        )
        action_difference = (
            torch.diff(policy_action[:, index])
            if index < policy_action.shape[1]
            else policy_action.new_empty(0)
        )
        per_joint[name] = {
            "peak_abs_measured_velocity_rad_s": float(peak_velocity[index]),
            "qualification_velocity_limit_rad_s": float(qualification_limits[index]),
            "peak_qualification_ratio": float(qualification_ratio[index]),
            "fraction_ticks_above_qualification_limit": float(
                (measured_velocity[:, index].abs() > qualification_limits[index]).float().mean()
            ),
            "model_velocity_limit_rad_s": float(model_limits[index]),
            "peak_model_limit_ratio": float(model_ratio[index]),
            "reached_99pct_model_velocity_limit": bool(model_ratio[index] >= 0.99),
            "peak_abs_target_velocity_reference_rad_s": float(target_velocity[:, index].abs().max()),
            "peak_abs_applied_velocity_reference_rad_s": float(applied_velocity[:, index].abs().max()),
            "policy_command_sign_reversals": reversals[index],
            "measured_velocity_sign_reversals_above_0p01_rad_s": measured_reversals,
            "policy_action_difference_rms": (
                float(torch.sqrt(torch.mean(action_difference.square()))) if action_difference.numel() else 0.0
            ),
            "policy_action_difference_max": (
                float(action_difference.abs().max()) if action_difference.numel() else 0.0
            ),
            "peak_abs_position_tracking_error_rad": float(tracking_error[:, index].abs().max()),
            "rms_position_tracking_error_rad": float(torch.sqrt(torch.mean(tracking_error[:, index].square()))),
            "peak_abs_measured_acceleration_rad_s2": float(peak_acceleration[index]),
            "rms_measured_acceleration_rad_s2": float(rms_acceleration[index]),
            "peak_abs_requested_torque_nm": float(trace["requested_torque"][:, index].abs().max()),
            "peak_abs_slew_limited_torque_nm": float(trace["slew_limited_torque"][:, index].abs().max()),
            "peak_abs_filtered_torque_nm": float(trace["filtered_torque"][:, index].abs().max()),
            "peak_abs_applied_torque_nm": float(trace["applied_torque"][:, index].abs().max()),
            "fraction_ticks_slew_limiter_active": float(
                (
                    (trace["requested_torque"][:, index] - trace["slew_limited_torque"][:, index]).abs()
                    > 1.0e-3
                )
                .float()
                .mean()
            ),
            "fraction_meaningful_ticks_requested_filtered_torque_opposite": (
                requested_filtered_opposite_fraction
            ),
            "fraction_meaningful_ticks_requested_applied_total_torque_opposite": (
                requested_applied_total_opposite_fraction
            ),
        }
    worst_qualification_joint = int(torch.argmax(qualification_ratio).item())
    worst_model_joint = int(torch.argmax(model_ratio).item())
    return {
        "sample_count": int(measured_velocity.shape[0]),
        "physics_dt_s": physics_dt,
        "policy_dt_s": physics_dt * decimation,
        "worst_qualification_velocity": {
            "joint": JOINT_NAMES[worst_qualification_joint],
            "ratio": float(qualification_ratio[worst_qualification_joint]),
        },
        "worst_model_velocity": {
            "joint": JOINT_NAMES[worst_model_joint],
            "ratio": float(model_ratio[worst_model_joint]),
            "reached_99pct_limit": bool(model_ratio[worst_model_joint] >= 0.99),
        },
        "per_joint": per_joint,
    }


def _write_plot(path: Path, trace: dict[str, torch.Tensor], physics_dt: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is unavailable; skipping trace.png")
        return
    time_s = torch.arange(trace["joint_position"].shape[0]).numpy() * physics_dt
    colors = plt.cm.tab10.colors
    figure, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for joint in range(7):
        color = colors[joint]
        axes[0].plot(time_s, trace["joint_velocity"][:, joint], color=color, label=f"J{joint + 1}")
        axes[0].plot(time_s, trace["target_velocity_reference"][:, joint], color=color, alpha=0.35, linestyle="--")
        axes[1].plot(time_s, trace["position_reference"][:, joint] - trace["joint_position"][:, joint], color=color)
        axes[2].plot(time_s, trace["requested_torque"][:, joint], color=color, alpha=0.45)
        axes[2].plot(time_s, trace["applied_torque"][:, joint], color=color)
    for joint in range(6):
        axes[3].plot(time_s, trace["normalized_action"][:, joint], color=colors[joint], label=f"J{joint + 1}")
    axes[0].set_ylabel("velocity [rad/s]\nsolid measured, dashed ref")
    axes[1].set_ylabel("q_ref - q [rad]")
    axes[2].set_ylabel("torque [Nm]\nfaint requested, solid applied")
    axes[3].set_ylabel("normalized action")
    axes[3].set_xlabel("time [s]")
    axes[0].legend(ncol=7, fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Franka-FR3v2-FrankyVelocityImpedance-Reach-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--case", choices=("zero", "step", "policy"), required=True)
parser.add_argument("--duration", type=float, default=3.0, help="Trace duration in seconds.")
parser.add_argument("--step-start", type=float, default=0.5)
parser.add_argument("--step-end", type=float, default=1.0)
parser.add_argument(
    "--ramp-down-duration",
    type=float,
    default=0.0,
    help="For the step case, linearly release the command over this many seconds after --step-end.",
)
parser.add_argument("--joint", type=int, default=6, help="One-based commanded joint for the step case (1--6).")
parser.add_argument("--amplitude", type=float, default=0.2, help="Normalized velocity action for the step case.")
parser.add_argument("--target", type=float, nargs=3, default=(0.475, 0.0, 0.5), metavar=("X", "Y", "Z"))
parser.add_argument("--output-dir", type=Path, default=None)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--external_callback", default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

remaining_args_env_registration = None
if args_cli.external_callback:
    remaining_args_env_registration = string_to_callable(args_cli.external_callback, separator=".")()
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    if args_cli.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if not 1 <= args_cli.joint <= 6:
        raise ValueError("--joint must be between 1 and 6")
    if not -1.0 <= args_cli.amplitude <= 1.0:
        raise ValueError("--amplitude must be within [-1, 1]")
    if args_cli.case == "step":
        if not 0.0 <= args_cli.step_start < args_cli.step_end <= args_cli.duration:
            raise ValueError("Require 0 <= --step-start < --step-end <= --duration")
        if args_cli.ramp_down_duration < 0.0 or args_cli.step_end + args_cli.ramp_down_duration > args_cli.duration:
            raise ValueError("Require a nonnegative ramp-down duration ending no later than --duration")
    if args_cli.case == "policy" and not args_cli.checkpoint:
        raise ValueError("--checkpoint is required for --case policy")

    data_root = Path(os.environ.get("FRANKA_RL_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()
    _validate_output_root(data_root)
    if args_cli.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        output_dir = data_root / "diagnostics" / f"{timestamp}_{args_cli.case}"
    else:
        output_dir = args_cli.output_dir.expanduser().resolve()
        output_dir.relative_to(data_root)
    output_dir.mkdir(parents=True, exist_ok=False)

    completed = False
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli), installed_version)
        env_cfg.scene.num_envs = 1
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), args_cli.duration + 1.0)
        ranges = env_cfg.commands.ee_pose.ranges
        ranges.pos_x = (args_cli.target[0], args_cli.target[0])
        ranges.pos_y = (args_cli.target[1], args_cli.target[1])
        ranges.pos_z = (args_cli.target[2], args_cli.target[2])

        env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            action_term = env.unwrapped.action_manager.get_term("arm_action")
            if not hasattr(action_term, "start_diagnostic_trace"):
                raise TypeError("Task does not use FrankyVelocityReference6DImpedanceAction")
            robot = env.unwrapped.scene["robot"]
            joint_ids, runtime_joint_names = robot.find_joints([f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)
            model_limits = robot.data.joint_vel_limits.torch[0, joint_ids].detach().cpu()
            qualification_limits = torch.tensor(env_cfg.actions.arm_action.max_measured_velocity)
            physics_dt = float(env.unwrapped.cfg.sim.dt)
            policy_dt = float(env.unwrapped.step_dt)
            decimation = round(policy_dt / physics_dt)
            policy_steps = math.ceil(args_cli.duration / policy_dt)
            trace_ticks = policy_steps * decimation

            env.reset()
            action_term.start_diagnostic_trace(trace_ticks, env_id=0)
            delay_steps = int(getattr(env.unwrapped.cfg, "required_action_delay_steps", 0))
            if delay_steps != 1:
                raise RuntimeError(f"Expected the velocity task's one-step delay contract, got {delay_steps}")
            env = FixedActionDelayWrapper(
                env,
                delay_steps,
                initial_action_provider=action_term_delay_fill_provider(env),
            )

            resume_path: Path | None = None
            policy = None
            reset_policy = None
            if args_cli.case == "policy":
                resume_path = Path(retrieve_file_path(args_cli.checkpoint)).resolve()
                env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                if agent_cfg.class_name == "OnPolicyRunner":
                    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
                elif agent_cfg.class_name == "DistillationRunner":
                    runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
                else:
                    raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
                configure_seed(env_cfg.seed, True)
                runner.load(str(resume_path))
                policy = runner.get_inference_policy(device=env.unwrapped.device)
                reset_policy = policy.reset if version.parse(installed_version) >= version.parse("4.0.0") else None
                observations = env.get_observations()

            for step in range(policy_steps):
                with torch.inference_mode():
                    if args_cli.case == "policy":
                        actions = policy(observations)
                        observations, _, dones, _ = env.step(actions)
                        if reset_policy is not None:
                            reset_policy(dones)
                    else:
                        actions = torch.zeros((1, 6), device=env.unwrapped.device)
                        time_s = step * policy_dt
                        if args_cli.case == "step":
                            if args_cli.step_start <= time_s < args_cli.step_end:
                                actions[:, args_cli.joint - 1] = args_cli.amplitude
                            elif (
                                args_cli.ramp_down_duration > 0.0
                                and args_cli.step_end <= time_s < args_cli.step_end + args_cli.ramp_down_duration
                            ):
                                ramp_fraction = 1.0 - (
                                    (time_s - args_cli.step_end) / args_cli.ramp_down_duration
                                )
                                actions[:, args_cli.joint - 1] = args_cli.amplitude * ramp_fraction
                        _, _, terminated, truncated, _ = env.step(actions)
                        dones = terminated | truncated
                    if bool(torch.any(dones)):
                        raise RuntimeError(f"Environment terminated at policy step {step}; trace would cross an auto-reset")

            action_term.stop_diagnostic_trace()
            trace = action_term.diagnostic_trace()
            summary = _summarize(trace, physics_dt, decimation, qualification_limits, model_limits)
            summary.update(
                {
                    "schema_version": 2,
                    "case": args_cli.case,
                    "task": args_cli.task,
                    "seed": args_cli.seed,
                    "target_position_base_m": list(args_cli.target),
                    "joint_names": list(runtime_joint_names),
                    "action_delay_steps": delay_steps,
                    "checkpoint": None if resume_path is None else str(resume_path),
                    "checkpoint_sha256": None if resume_path is None else _sha256(resume_path),
                    "step": {
                        "joint": args_cli.joint,
                        "normalized_amplitude": args_cli.amplitude,
                        "start_s": args_cli.step_start,
                        "end_s": args_cli.step_end,
                        "ramp_down_duration_s": args_cli.ramp_down_duration,
                    }
                    if args_cli.case == "step"
                    else None,
                }
            )
            _write_trace_csv(output_dir / "trace_1khz.csv", trace, physics_dt, decimation)
            _write_plot(output_dir / "trace.png", trace, physics_dt)
            with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2)
                stream.write("\n")
            print(json.dumps(summary["worst_qualification_velocity"], indent=2))
            print(json.dumps(summary["worst_model_velocity"], indent=2))
            print(f"Diagnostic artifacts written to: {output_dir}")
            completed = True
        finally:
            env.close()
    if not completed:
        raise RuntimeError("Velocity diagnostic failed; see the preceding simulator traceback")


if __name__ == "__main__":
    main()
