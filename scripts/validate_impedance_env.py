"""Finite smoke test for the deployment-faithful FR3 impedance task."""

import argparse
import contextlib
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Validate the FR3 Franky impedance task.")
parser.add_argument("--task", default="Franka-FR3v2-FrankyImpedance-Reach-v0")
parser.add_argument("--scenario", default="nominal")
parser.add_argument("--scenario-file", default=None)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=100)
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]

import franka_rl.tasks  # noqa: E402, F401
from franka_rl.utils.scenarios import ScenarioCatalog, ScenarioModifier  # noqa: E402


def main() -> None:
    torch.manual_seed(42)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.ee_pose.debug_vis = False
    catalog = ScenarioCatalog.from_yaml(args_cli.scenario_file)
    ScenarioModifier(catalog.get(args_cli.scenario), catalog).apply(env_cfg)

    with launch_simulation(env_cfg, args_cli):
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            observations, _ = env.reset(seed=42)
            action_term = env.unwrapped.action_manager.get_term("arm_action")
            action_dim = action_term.action_dim
            assert env.action_space.shape == (args_cli.num_envs, action_dim)
            expected_observations = (
                17 + 2 * action_dim
                if hasattr(action_term, "normalized_reference_position")
                else 24
            )
            if getattr(env_cfg.observations.policy, "ee_z_axis_error", None) is not None:
                expected_observations += 2
            assert observations["policy"].shape == (args_cli.num_envs, expected_observations)
            max_command_difference = 0.0
            max_joint7_command_difference = 0.0
            max_action_clipping = 0.0
            max_torque = 0.0
            max_measured_speed = 0.0
            max_contact_force = 0.0
            reset_count = 0
            termination_counts = {
                name: 0 for name in env.unwrapped.termination_manager.active_terms
            }

            hold_action = action_term.delay_fill_action.clone()
            for step in range(args_cli.steps):
                actions = hold_action.clone()
                if step >= 10:
                    actions[:, 0] += 0.01
                observations, rewards, terminated, truncated, _ = env.step(actions)
                assert torch.isfinite(observations["policy"]).all()
                assert torch.isfinite(rewards).all()
                assert torch.isfinite(action_term.applied_torque).all()
                max_command_difference = max(
                    max_command_difference, float(action_term.command_difference.abs().max())
                )
                if action_dim == 6:
                    max_joint7_command_difference = max(
                        max_joint7_command_difference,
                        float(action_term.command_difference[:, 6].abs().max()),
                    )
                max_action_clipping = max(
                    max_action_clipping, float(action_term.action_clipping.max())
                )
                max_torque = max(max_torque, float(action_term.applied_torque.abs().max()))
                max_measured_speed = max(
                    max_measured_speed,
                    float(env.unwrapped.scene["robot"].data.joint_vel.torch[:, :7].abs().max()),
                )
                contact_history = env.unwrapped.scene.sensors["arm_contacts"].data.net_forces_w_history.torch
                max_contact_force = max(
                    max_contact_force,
                    float(torch.linalg.norm(contact_history, dim=-1).max()),
                )
                reset_count += int((terminated | truncated).sum())
                for name in termination_counts:
                    termination_counts[name] += int(
                        env.unwrapped.termination_manager.get_term(name).sum()
                    )

            assert max_action_clipping == 0.0
            if hasattr(action_term, "max_reference_velocity"):
                max_increment = float((action_term.max_reference_velocity * env.unwrapped.step_dt).max())
                assert max_command_difference <= max_increment + 1.0e-6
            assert not action_term.faulted.any()
            if action_dim == 6:
                assert max_joint7_command_difference == 0.0, (
                    "Six-action task changed the held joint-7 reference"
                )

            print("Franky impedance environment validation report")
            print(f"Task: {args_cli.task}")
            print(f"Scenario: {args_cli.scenario}")
            print(f"Environments: {args_cli.num_envs}")
            print(f"Policy action dimension: {action_dim}")
            print(f"Maximum command difference: {max_command_difference:.6f} rad")
            if action_dim == 6:
                print(f"Maximum joint-7 command difference: {max_joint7_command_difference:.6f} rad")
            print(f"Maximum normalized action clipping: {max_action_clipping:.6f}")
            print(f"Maximum measured speed: {max_measured_speed:.6f} rad/s")
            print(f"Peak 1 kHz speed by joint: {action_term.peak_measured_speed.amax(dim=0).tolist()}")
            print(f"Maximum applied torque: {max_torque:.6f} N*m")
            print(f"Maximum moving-link contact force: {max_contact_force:.6f} N")
            print(f"Episode resets observed: {reset_count}")
            for name, count in termination_counts.items():
                print(f"  {name}: {count}")
            unsafe_resets = sum(
                count for name, count in termination_counts.items() if name != "time_out"
            )
            assert unsafe_resets == 0, (
                f"Observed {unsafe_resets} unsafe resets; controller validation failed"
            )
            print("[PASS] FR3 Franky impedance task")
        finally:
            env.close()


if __name__ == "__main__":
    main()
