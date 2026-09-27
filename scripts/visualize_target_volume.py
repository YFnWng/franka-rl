"""Show the configured Cartesian target volume around one robot."""

import argparse
import contextlib
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli


parser = argparse.ArgumentParser(description="Visualize a task's complete Cartesian target sampling volume.")
parser.add_argument("--task", default="Franka-FR3v2-FrankyImpedance-Reach-v0")
parser.add_argument("--num_samples", type=int, default=400)
parser.add_argument("--seed", type=int, default=42)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.utils.math import transform_points  # noqa: E402

import franka_rl.tasks  # noqa: E402, F401


def _position_bounds(command_cfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract XYZ lower and upper bounds from a uniform pose command config."""

    ranges = command_cfg.ranges
    lower = torch.tensor(
        [ranges.pos_x[0], ranges.pos_y[0], ranges.pos_z[0]],
        device=device,
        dtype=torch.float32,
    )
    upper = torch.tensor(
        [ranges.pos_x[1], ranges.pos_y[1], ranges.pos_z[1]],
        device=device,
        dtype=torch.float32,
    )
    if torch.any(upper <= lower):
        raise ValueError(f"Target position ranges must have positive width: lower={lower}, upper={upper}")
    return lower, upper


def _create_markers(size: tuple[float, float, float]) -> tuple[VisualizationMarkers, VisualizationMarkers]:
    """Create a translucent bounds marker and a target-sample marker group."""

    volume = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/TargetVolume/Bounds",
            markers={
                "bounds": sim_utils.CuboidCfg(
                    size=size,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.1, 0.45, 1.0),
                        opacity=0.12,
                    ),
                )
            },
        )
    )
    samples = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/TargetVolume/Samples",
            markers={
                "sample": sim_utils.SphereCfg(
                    radius=0.006,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.25, 0.05),
                        emissive_color=(0.2, 0.02, 0.0),
                    ),
                )
            },
        )
    )
    return volume, samples


def main() -> None:
    if args_cli.num_samples <= 0:
        raise ValueError("--num_samples must be positive")

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = 1
    env_cfg.commands.ee_pose.debug_vis = True

    with launch_simulation(env_cfg, args_cli):
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            env.reset(seed=args_cli.seed)
            base_env = env.unwrapped
            robot = base_env.scene["robot"]
            device = base_env.device

            lower_b, upper_b = _position_bounds(env_cfg.commands.ee_pose, device)
            center_b = 0.5 * (lower_b + upper_b)
            size = upper_b - lower_b

            generator = torch.Generator(device=device)
            generator.manual_seed(args_cli.seed)
            samples_b = lower_b + size * torch.rand(
                (args_cli.num_samples, 3),
                generator=generator,
                device=device,
            )

            root_pose_w = robot.data.root_pose_w.torch[0]
            center_w = transform_points(
                center_b.unsqueeze(0),
                pos=root_pose_w[:3],
                quat=root_pose_w[3:7],
            )
            samples_w = transform_points(
                samples_b,
                pos=root_pose_w[:3],
                quat=root_pose_w[3:7],
            )

            volume_marker, sample_markers = _create_markers(tuple(float(value) for value in size))
            volume_marker.visualize(
                translations=center_w,
                orientations=root_pose_w[3:7].unsqueeze(0),
            )
            sample_markers.visualize(translations=samples_w)

            eye_b = torch.tensor([1.25, 1.05, 0.95], device=device)
            look_at_b = torch.tensor([0.45, 0.0, 0.30], device=device)
            eye_w = transform_points(eye_b.unsqueeze(0), root_pose_w[:3], root_pose_w[3:7])[0]
            look_at_w = transform_points(look_at_b.unsqueeze(0), root_pose_w[:3], root_pose_w[3:7])[0]
            base_env.sim.set_camera_view(eye_w.tolist(), look_at_w.tolist())

            print("Target-volume visualization")
            print(f"Task: {args_cli.task}")
            print(f"End-effector body: {env_cfg.commands.ee_pose.body_name}")
            print(f"Base-frame lower bounds: {lower_b.tolist()} m")
            print(f"Base-frame upper bounds: {upper_b.tolist()} m")
            print(f"Samples: {args_cli.num_samples} (seed {args_cli.seed})")

            actions = torch.zeros(env.action_space.shape, device=device)
            while any(
                visualizer.is_running() and not visualizer.is_closed
                for visualizer in base_env.sim.visualizers
            ):
                with torch.inference_mode():
                    env.step(actions)
        finally:
            env.close()


if __name__ == "__main__":
    main()
