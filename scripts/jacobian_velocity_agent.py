"""Damped least-squares reachability baseline for the velocity-action task."""

import argparse
import contextlib
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

parser = argparse.ArgumentParser(description="FR3 Jacobian velocity baseline.")
parser.add_argument(
    "--task", default="Franka-FR3v2-JointVelocity-Reach-v0"
)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=300)
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import franka_rl.tasks  # noqa: E402, F401


def main() -> None:
    torch.manual_seed(42)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = max(20.0, args_cli.num_steps / 30.0 + 1.0)
    env_cfg.commands.ee_pose.debug_vis = False

    with launch_simulation(env_cfg, args_cli):
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env = gym.make(args_cli.task, cfg=env_cfg)
        base_env = env.unwrapped
        try:
            env.reset(seed=42)
            robot = base_env.scene["robot"]
            body_name = env_cfg.commands.ee_pose.body_name
            robot_cfg = SceneEntityCfg(
                "robot",
                joint_names=[f"panda_joint{i}" for i in range(1, 8)],
                body_names=[body_name],
            )
            robot_cfg.resolve(base_env.scene)
            ee_body_id = robot_cfg.body_ids[0]
            ee_jacobian_id = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
            arm_joint_ids = robot_cfg.joint_ids
            if isinstance(arm_joint_ids, slice):
                arm_joint_ids = list(range(robot.num_joints))[arm_joint_ids]
            jacobian_joint_ids = [
                joint_id + robot.num_base_dofs for joint_id in arm_joint_ids
            ]

            controller = DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="position",
                    use_relative_mode=False,
                    ik_method="dls",
                    ik_params={"lambda_val": 0.1},
                ),
                num_envs=base_env.num_envs,
                device=base_env.device,
            )

            best_error = torch.full(
                (base_env.num_envs,), float("inf"), device=base_env.device
            )
            ever_below = torch.zeros(
                base_env.num_envs, dtype=torch.bool, device=base_env.device
            )
            failure_resets = 0

            for _ in range(args_cli.num_steps):
                target_pos_b = base_env.command_manager.get_command("ee_pose")[:, :3]
                ee_pose_w = robot.data.body_pose_w.torch[:, ee_body_id]
                root_pose_w = robot.data.root_pose_w.torch
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, :3],
                    root_pose_w[:, 3:7],
                    ee_pose_w[:, :3],
                    ee_pose_w[:, 3:7],
                )
                controller.set_command(target_pos_b, ee_quat=ee_quat_b)
                jacobian = robot.data.body_link_jacobian_w.torch[
                    :, ee_jacobian_id, :, jacobian_joint_ids
                ]
                joint_pos = robot.data.joint_pos.torch[:, arm_joint_ids]
                joint_pos_desired = controller.compute(
                    ee_pos_b, ee_quat_b, jacobian, joint_pos
                )

                # Convert the DLS position correction to a bounded velocity
                # request. The environment applies the same acceleration and
                # joint-limit governor used by PPO.
                joint_velocity = torch.clamp(
                    (joint_pos_desired - joint_pos) / base_env.step_dt,
                    -0.30,
                    0.30,
                )
                actions = joint_velocity / 0.30
                _, _, terminated, truncated, _ = env.step(actions)
                failure_resets += int(terminated.sum().item())

                error = torch.linalg.vector_norm(target_pos_b - ee_pos_b, dim=1)
                best_error = torch.minimum(best_error, error)
                ever_below |= error < 0.03

            target_pos_b = base_env.command_manager.get_command("ee_pose")[:, :3]
            ee_pose_w = robot.data.body_pose_w.torch[:, ee_body_id]
            root_pose_w = robot.data.root_pose_w.torch
            ee_pos_b, _ = subtract_frame_transforms(
                root_pose_w[:, :3],
                root_pose_w[:, 3:7],
                ee_pose_w[:, :3],
                ee_pose_w[:, 3:7],
            )
            final_error = torch.linalg.vector_norm(target_pos_b - ee_pos_b, dim=1)

            print("Jacobian velocity baseline report")
            print(f"Completed steps: {args_cli.num_steps}")
            print(f"Final mean error: {final_error.mean().item():.4f} m")
            print(f"Final median error: {final_error.median().item():.4f} m")
            print(f"Best mean error: {best_error.mean().item():.4f} m")
            print(f"Ever below 3 cm: {ever_below.float().mean().item():.2%}")
            print(f"Failure resets: {failure_resets}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
