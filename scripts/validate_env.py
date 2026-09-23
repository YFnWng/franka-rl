"""Validate the Franka reaching environment contract."""

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

parser = argparse.ArgumentParser(
    description="Validate the Franka RL environment."
)
parser.add_argument("--task", type=str, required=True)
add_launcher_args(parser)

args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

# Import after parsing so the custom task is registered.
import franka_rl.tasks  # noqa: F401


def main() -> None:
    torch.manual_seed(42)

    env_cfg, _ = resolve_task_config(args_cli.task, "")

    # Keep validation small and disable command markers.
    env_cfg.scene.num_envs = 4
    env_cfg.commands.ee_pose.debug_vis = False

    with launch_simulation(env_cfg, args_cli):
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        env = gym.make(args_cli.task, cfg=env_cfg)
        base_env = env.unwrapped

        try:
            observations, _ = env.reset(seed=42)
            policy_obs = observations["policy"]

            #
            # 1. Shape contract
            #
            assert policy_obs.shape == (4, 24), (
                f"Expected observation shape (4, 24), got {policy_obs.shape}"
            )
            assert env.action_space.shape == (4, 7), (
                f"Expected action shape (4, 7), got {env.action_space.shape}"
            )

            #
            # 2. Device and finite-value contract
            #
            assert policy_obs.device == torch.device(base_env.device)
            assert torch.isfinite(policy_obs).all(), (
                "Initial observation contains NaN or infinity"
            )

            #
            # 3. Observation layout
            #
            joint_pos = policy_obs[:, 0:7]
            joint_vel = policy_obs[:, 7:14]
            position_error = policy_obs[:, 14:17]
            previous_action = policy_obs[:, 17:24]

            assert torch.isfinite(joint_pos).all()
            assert torch.isfinite(joint_vel).all()
            assert torch.isfinite(position_error).all()
            assert torch.isfinite(previous_action).all()

            # A world-frame clone-origin leak would create errors of several
            # metres because environments are spaced 2.5 m apart.
            assert position_error.abs().max() < 2.0, (
                "Position error may contain cloned-environment world offsets"
            )

            # Action history should be zero immediately after reset.
            torch.testing.assert_close(
                previous_action,
                torch.zeros_like(previous_action),
            )

            #
            # 4. Reward consistency
            #
            position_reward_cfg = base_env.reward_manager.get_term_cfg(
                "position_tracking"
            )
            raw_position_reward = position_reward_cfg.func(
                base_env,
                **position_reward_cfg.params,
            )

            sigma = position_reward_cfg.params["sigma"]
            expected_position_reward = torch.exp(
                -torch.sum(torch.square(position_error), dim=1) / sigma
            )

            torch.testing.assert_close(
                raw_position_reward,
                expected_position_reward,
            )

            assert raw_position_reward.shape == (4,)
            assert (raw_position_reward > 0.0).all()
            assert (raw_position_reward <= 1.0).all()

            #
            # 5. One environment step
            #
            actions = torch.zeros(
                env.action_space.shape,
                device=base_env.device,
            )

            (
                next_observations,
                rewards,
                terminated,
                truncated,
                _,
            ) = env.step(actions)

            assert next_observations["policy"].shape == (4, 24)
            assert rewards.shape == (4,)
            assert terminated.shape == (4,)
            assert truncated.shape == (4,)

            assert terminated.dtype == torch.bool
            assert truncated.dtype == torch.bool

            assert torch.isfinite(next_observations["policy"]).all()
            assert torch.isfinite(rewards).all()

            #
            # 6. Selective-reset isolation
            #
            commands_before = (
                base_env.command_manager.get_command("ee_pose").clone()
            )
            joint_positions_before = (
                base_env.scene["robot"].data.joint_pos.torch.clone()
            )

            reset_ids = torch.tensor(
                [0],
                dtype=torch.int32,
                device=base_env.device,
            )
            reset_observations, _ = base_env.reset(env_ids=reset_ids)

            commands_after = (
                base_env.command_manager.get_command("ee_pose").clone()
            )
            joint_positions_after = (
                base_env.scene["robot"].data.joint_pos.torch.clone()
            )

            # Environment zero should receive a new target.
            assert not torch.allclose(
                commands_before[0, :3],
                commands_after[0, :3],
            ), "Reset environment did not receive a new target"

            # Other environments must retain their targets.
            torch.testing.assert_close(
                commands_after[1:],
                commands_before[1:],
            )

            # Other robots must not be reset.
            torch.testing.assert_close(
                joint_positions_after[1:],
                joint_positions_before[1:],
            )

            assert reset_observations["policy"].shape == (4, 24)
            assert torch.isfinite(reset_observations["policy"]).all()

            print("[PASS] Observation shape: (4, 24)")
            print("[PASS] Action shape: (4, 7)")
            print("[PASS] Observation device and finite values")
            print("[PASS] Base-frame error has no clone-origin leakage")
            print("[PASS] Position reward matches observation error")
            print("[PASS] Step outputs have valid shapes and values")
            print("[PASS] Selective reset changes only the requested environment")

        finally:
            env.close()


if __name__ == "__main__":
    main()