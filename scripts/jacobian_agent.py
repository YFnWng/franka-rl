# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

import argparse
import contextlib
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

import isaaclab.envs.mdp as mdp

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import scale_transform, subtract_frame_transforms

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Jacobian DLS agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--num_steps",
    type=int,
    default=150,
    help="Number of control steps to run.",
)
# append AppLauncher cli args
add_launcher_args(parser)
# simple agents should open Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import franka_rl.tasks  # noqa: F401


def main():
    """Jacobian DLS agent with Isaac Lab environment."""

    torch.manual_seed(42)

    # parse configuration via Hydra (supports preset selection, e.g. env.sim.physics=newton_mjwarp)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.terminations.reached_target = None
    env_cfg.terminations.time_out = None

    # Use Isaac Lab's high-PD settings for the Jacobian reachability baseline.
    env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True

    env_cfg.scene.robot.actuators["panda_shoulder"].stiffness = 400.0
    env_cfg.scene.robot.actuators["panda_shoulder"].damping = 80.0

    env_cfg.scene.robot.actuators["panda_forearm"].stiffness = 400.0
    env_cfg.scene.robot.actuators["panda_forearm"].damping = 80.0

    with launch_simulation(env_cfg, args_cli):
        # override with CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        env_cfg.actions.arm_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            rescale_to_limits=True,
        )

        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg)

        # print info (this is vectorized environment)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        # reset environment
        env.reset()

        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        robot_cfg = SceneEntityCfg(
            "robot",
            joint_names=["panda_joint.*"],
            body_names=["panda_hand"],
        )
        robot_cfg.resolve(base_env.scene)

        ee_body_id = robot_cfg.body_ids[0]
        ee_jacobian_id = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

        jacobian_joint_ids = [
            joint_id + robot.num_base_dofs for joint_id in robot_cfg.joint_ids
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

        soft_joint_limits = robot.data.soft_joint_pos_limits.torch[
            :, robot_cfg.joint_ids, :
        ]

        joint_lower = soft_joint_limits[:, :, 0] + 0.05
        joint_upper = soft_joint_limits[:, :, 1] - 0.05

        # simulate environment
        # keep running while any visualizer is open, otherwise fall back to MAX_STEPS
        sim = env.unwrapped.sim
        completed_steps = 0
        saturated_values = torch.zeros((), device=base_env.device)
        total_action_values = 0
        failure_resets = torch.zeros((), device=base_env.device)
        timeout_resets = torch.zeros((), device=base_env.device)

        termination_counts = {
            name: torch.zeros((), device=base_env.device)
            for name in base_env.termination_manager.active_terms
        }

        ever_failed = torch.zeros(
            base_env.num_envs,
            dtype=torch.bool,
            device=base_env.device,
        )

        joint_limit_clamped_values = torch.zeros(
            (),
            device=base_env.device,
        )
        total_joint_targets = 0

        best_error = torch.full(
            (base_env.num_envs,),
            float("inf"),
            device=base_env.device,
        )
        for step in range(args_cli.num_steps):
            if sim.visualizers:
                # visualizer mode: run until the visualizer window is closed
                if not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break
            # run everything in inference mode
            with torch.inference_mode():
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

                joint_pos = robot.data.joint_pos.torch[:, robot_cfg.joint_ids]

                unlimited_joint_pos_desired = controller.compute(
                    ee_pos_b,
                    ee_quat_b,
                    jacobian,
                    joint_pos,
                )

                joint_delta = torch.clamp(
                    unlimited_joint_pos_desired - joint_pos,
                    -0.05,
                    0.05,
                )

                unclamped_joint_pos_desired = joint_pos + joint_delta

                joint_pos_desired = torch.maximum(
                    torch.minimum(unclamped_joint_pos_desired, joint_upper),
                    joint_lower,
                )

                joint_limit_clamped_values += (
                    (joint_pos_desired - unclamped_joint_pos_desired).abs() > 1.0e-6
                ).sum()

                total_joint_targets += joint_pos_desired.numel()

                default_joint_pos = robot.data.default_joint_pos.torch[
                    :, robot_cfg.joint_ids
                ]

                physical_lower = soft_joint_limits[:, :, 0]
                physical_upper = soft_joint_limits[:, :, 1]

                unclipped_actions = scale_transform(
                    joint_pos_desired,
                    physical_lower,
                    physical_upper,
                )

                saturated_values += (unclipped_actions.abs() > 1.0).sum()
                total_action_values += unclipped_actions.numel()

                actions = torch.clamp(unclipped_actions, -1.0, 1.0)

                # saturated_values += (actions.abs() > 0.99).sum()
                # total_action_values += actions.numel()

                # apply actions
                _, _, terminated, truncated, _ = env.step(actions)

                ever_failed |= terminated

                for name in base_env.termination_manager.active_terms:
                    termination_counts[name] += (
                        base_env.termination_manager.get_term(name).sum()
                        )

                failure_resets += terminated.sum()
                timeout_resets += truncated.sum()
                completed_steps += 1

                best_error = torch.minimum(
                    best_error, 
                    torch.linalg.vector_norm(
                        target_pos_b - ee_pos_b,
                        dim=1,
                    )
                    )

        # report statistics
        with torch.inference_mode():
            final_target_pos_b = base_env.command_manager.get_command("ee_pose")[:, :3]

            final_ee_pose_w = robot.data.body_pose_w.torch[:, ee_body_id]
            final_root_pose_w = robot.data.root_pose_w.torch

            final_ee_pos_b, _ = subtract_frame_transforms(
                final_root_pose_w[:, :3],
                final_root_pose_w[:, 3:7],
                final_ee_pose_w[:, :3],
                final_ee_pose_w[:, 3:7],
            )

            position_error = torch.linalg.vector_norm(
                final_target_pos_b - final_ee_pos_b,
                dim=1,
            )

            print("\nJacobian baseline report")
            print(f"Completed steps: {completed_steps}")
            print(f"Mean error: {position_error.mean().item():.4f} m")
            print(f"Median error: {position_error.median().item():.4f} m")
            print(
                "Fraction below 3 cm: "
                f"{(position_error < 0.03).float().mean().item():.2%}"
            )
            print(
                "Action saturation: "
                f"{(saturated_values / total_action_values).item():.2%}"
            )
            print(f"Failure resets: {failure_resets.item():.0f}")
            print(f"Timeout resets: {timeout_resets.item():.0f}")
            print("Termination counts:")
            for name, count in termination_counts.items():
                print(f"  {name}: {count.item():.0f}")

            print(f"Environments that ever failed: {ever_failed.sum().item()}")

            valid = ~ever_failed
            valid_error = position_error[valid]

            print(f"Valid environments: {valid.sum().item()}")

            if valid.any():
                print(f"Valid mean error: {valid_error.mean().item():.4f} m")
                print(f"Valid median error: {valid_error.median().item():.4f} m")
                print(
                    "Valid fraction below 3 cm: "
                    f"{(valid_error < 0.03).float().mean().item():.2%}"
                )
            print(
                "Joint-limit clamp rate: "
                f"{(joint_limit_clamped_values / total_joint_targets).item():.2%}"
            )

            print(f"Best mean error: {best_error.mean().item():.4f} m")
            print(f"Best median error: {best_error.median().item():.4f} m")
            print(
                "Ever below 3 cm: "
                f"{(best_error < 0.03).float().mean().item():.2%}"
            )

        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
