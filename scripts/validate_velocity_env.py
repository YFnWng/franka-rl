"""Validate the FR3 joint-velocity task before PPO training."""

import argparse
import contextlib
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

parser = argparse.ArgumentParser(description="Validate the FR3 velocity task.")
parser.add_argument(
    "--task", default="Franka-FR3v2-JointVelocity-Reach-v0"
)
parser.add_argument("--scenario", default="nominal")
parser.add_argument(
    "--scenario-file",
    type=str,
    default=None,
    help="Optional scenario YAML; defaults to the packaged catalog.",
)
parser.add_argument("--num_envs", type=int, default=16)
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import franka_rl.tasks  # noqa: E402, F401
from franka_rl.utils.scenarios import ScenarioCatalog, ScenarioModifier  # noqa: E402


def main() -> None:
    torch.manual_seed(42)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.ee_pose.debug_vis = False
    scenario_catalog = ScenarioCatalog.from_yaml(args_cli.scenario_file)
    scenario = scenario_catalog.get(args_cli.scenario)
    ScenarioModifier(scenario, scenario_catalog).apply(env_cfg)

    with launch_simulation(env_cfg, args_cli):
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env = gym.make(args_cli.task, cfg=env_cfg)
        base_env = env.unwrapped
        try:
            obs, _ = env.reset(seed=42)
            assert obs["policy"].shape == (args_cli.num_envs, 24)
            assert env.action_space.shape == (args_cli.num_envs, 7)

            robot = base_env.scene["robot"]
            arm_ids, _ = robot.find_joints(
                [f"panda_joint{i}" for i in range(1, 8)], preserve_order=True
            )
            q_start = robot.data.joint_pos.torch[:, arm_ids].clone()
            zeros = torch.zeros(env.action_space.shape, device=base_env.device)
            terminations = 0

            # Zero velocity must hold under gravity rather than falling toward
            # a stale position target.
            for _ in range(60):
                obs, _, terminated, truncated, _ = env.step(zeros)
                terminations += int((terminated | truncated).sum().item())
            q_after_hold = robot.data.joint_pos.torch[:, arm_ids].clone()
            max_hold_drift = float((q_after_hold - q_start).abs().max().item())

            # A small positive joint-1 command must move in the correct
            # direction and remain within the governed physical target.
            pulse = torch.zeros_like(zeros)
            pulse[:, 0] = 0.25
            q_before_pulse = q_after_hold.clone()
            for _ in range(30):
                obs, _, terminated, truncated, _ = env.step(pulse)
                terminations += int((terminated | truncated).sum().item())
            q_after_pulse = robot.data.joint_pos.torch[:, arm_ids].clone()
            median_joint1_motion = float(
                torch.median(q_after_pulse[:, 0] - q_before_pulse[:, 0]).item()
            )

            # Exercise clipping and rate limiting without reaching timeout.
            for _ in range(30):
                random_actions = 0.25 * (
                    2.0 * torch.rand_like(zeros) - 1.0
                )
                obs, rewards, terminated, truncated, _ = env.step(random_actions)
                terminations += int((terminated | truncated).sum().item())
                assert torch.isfinite(obs["policy"]).all()
                assert torch.isfinite(rewards).all()

            # Explicitly verify the normalized action boundary. One step is
            # enough to inspect the accepted action without approaching a
            # joint limit.
            over_range = torch.zeros_like(zeros)
            over_range[:, 0] = 2.0
            obs, rewards, terminated, truncated, _ = env.step(over_range)
            terminations += int((terminated | truncated).sum().item())

            action_term = base_env.action_manager.get_term("arm_action")
            max_applied_velocity = float(action_term.applied_actions.abs().max().item())
            max_accepted_action = float(action_term.accepted_actions.abs().max().item())
            target_scale_min = float(action_term.velocity_target_scale.min().item())
            target_scale_max = float(action_term.velocity_target_scale.max().item())
            acceleration_scale_min = float(action_term.acceleration_scale.min().item())
            acceleration_scale_max = float(action_term.acceleration_scale.max().item())

            assert terminations == 0, f"Observed {terminations} unexpected resets"
            assert max_hold_drift < 0.03, (
                f"Zero-action drift {max_hold_drift:.5f} rad exceeds 0.03 rad"
            )
            assert median_joint1_motion > 0.02, (
                "Positive joint-1 velocity did not produce positive motion: "
                f"{median_joint1_motion:.5f} rad"
            )
            target_range = env_cfg.actions.arm_action.velocity_target_scale_range
            acceleration_range = env_cfg.actions.arm_action.acceleration_scale_range
            max_target_scale = target_range[1] if target_range is not None else 1.0
            assert max_applied_velocity <= 0.30001 * max_target_scale
            assert max_accepted_action <= 1.00001
            assert max_accepted_action >= 0.99999
            if target_range is not None:
                assert target_range[0] <= target_scale_min <= target_scale_max <= target_range[1]
                assert target_scale_max - target_scale_min > 0.01
            if acceleration_range is not None:
                assert (
                    acceleration_range[0]
                    <= acceleration_scale_min
                    <= acceleration_scale_max
                    <= acceleration_range[1]
                )
                assert acceleration_scale_max - acceleration_scale_min > 0.01

            print("Velocity environment validation report")
            print(f"Task: {args_cli.task}")
            print(f"Scenario: {args_cli.scenario}")
            print(f"Environments: {args_cli.num_envs}")
            print(f"Zero-action max joint drift: {max_hold_drift:.6f} rad")
            print(
                "Median joint-1 motion under +0.25 action: "
                f"{median_joint1_motion:.6f} rad"
            )
            print(f"Maximum applied target speed: {max_applied_velocity:.6f} rad/s")
            print(f"Maximum accepted normalized action: {max_accepted_action:.6f}")
            print(
                "Velocity target scale range realized: "
                f"[{target_scale_min:.4f}, {target_scale_max:.4f}]"
            )
            print(
                "Acceleration scale range realized: "
                f"[{acceleration_scale_min:.4f}, {acceleration_scale_max:.4f}]"
            )
            print("Unexpected resets: 0")
            print("[PASS] FR3 governed joint-velocity task")
        finally:
            env.close()


if __name__ == "__main__":
    main()
