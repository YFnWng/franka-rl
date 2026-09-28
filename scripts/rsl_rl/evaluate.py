# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL checkpoint and write reproducible result artifacts."""

import argparse
import contextlib
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
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

import franka_rl.tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from franka_rl.tasks.manager_based.franka_rl import mdp
from franka_rl.utils.policy_evaluator import (
    EvaluationConfig,
    PolicyEvaluator,
)
from franka_rl.utils.scenarios import ScenarioCatalog, ScenarioModifier

DEFAULT_DATA_ROOT = Path("/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data")
MIN_FREE_BYTES = 100 * 1024 * 1024


def validate_data_root(data_root: Path) -> None:
    """Fail rather than silently writing artifacts outside the mounted data volume."""
    if not data_root.exists():
        raise RuntimeError(
            f"Franka RL data root does not exist: {data_root}. "
            "Mount the data volume or set FRANKA_RL_DATA_ROOT to an existing directory."
        )
    if not data_root.is_dir():
        raise RuntimeError(f"Franka RL data root is not a directory: {data_root}")
    if not os.access(data_root, os.W_OK):
        raise RuntimeError(f"Franka RL data root is not writable: {data_root}")

    free_bytes = shutil.disk_usage(data_root).free
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(
            f"Franka RL data root has only {free_bytes / 1024**2:.1f} MiB free; "
            f"at least {MIN_FREE_BYTES / 1024**2:.0f} MiB is required."
        )


# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--num_episodes", type=int, default=5120, help="Number of episodes to evaluate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
parser.add_argument("--success_threshold", type=float, default=0.03)
parser.add_argument("--success_steps", type=int, default=5)
parser.add_argument(
    "--z-axis-threshold-deg",
    type=float,
    default=5.0,
    help="Tip-z angular success threshold in degrees for z-axis tracking tasks.",
)
parser.add_argument(
    "--path",
    default=None,
    help="Named path from paths.yaml. Only valid for waypoint-path evaluation tasks.",
)
parser.add_argument(
    "--path-file",
    type=Path,
    default=None,
    help="Immutable path catalog YAML. Requires --path; defaults to the packaged catalog.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Write artifacts to this exact directory instead of a timestamped directory.",
)
parser.add_argument(
    "--job-id",
    default=None,
    help="Stable coordinator job identifier stored in the evaluation artifacts.",
)
parser.add_argument(
    "--target-set",
    type=Path,
    default=None,
    help="Optional deterministic target-set JSON for paired evaluation.",
)
parser.add_argument(
    "--hardware-session",
    type=Path,
    default=None,
    help=(
        "Packaged hardware session directory for exact controller, path, "
        "checkpoint, and initial-state replay. Requires one environment and episode."
    ),
)
parser.add_argument(
    "--scenario",
    default="nominal",
    help="Named scenario from the scenario YAML catalog.",
)
parser.add_argument(
    "--scenario-file",
    type=Path,
    default=None,
    help="Optional scenario YAML file. Defaults to the packaged catalog.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True


# Call an external callback if requested. This gives opportunity to external code to register the environments
# The function is expected to return a list of arguments that were not consumed by the callback.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# clear out sys.argv for Hydra
# The remaining arguments are the arguments that were not consumed by both this scripts
# argparser and (optionally) the external callback function. Both sides of this
# intersection are pre-fold (the callback reads the user's original sys.argv), so
# preset tokens like ``physics=NAME`` compare correctly here. Fold runs after.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# Check for installed RSL-RL version
installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate an RSL-RL agent."""
    scenario_catalog = ScenarioCatalog.from_yaml(args_cli.scenario_file)
    scenario = scenario_catalog.get(args_cli.scenario)
    scenario_modifier = ScenarioModifier(scenario, scenario_catalog)
    if args_cli.path_file is not None and args_cli.path is None:
        raise ValueError("--path-file requires --path")
    if args_cli.path is not None:
        configure_path = getattr(env_cfg, "configure_path", None)
        if configure_path is None:
            raise ValueError("--path is only valid for a waypoint-path evaluation task")
        configure_path(args_cli.path, args_cli.path_file)
    evaluation_protocol = getattr(env_cfg, "evaluation_protocol", "random_point")
    evaluates_z_axis = (
        getattr(env_cfg.observations.policy, "ee_z_axis_error", None) is not None
    )
    if evaluation_protocol == "random_point" and evaluates_z_axis:
        evaluation_protocol = "position_z_axis"
    z_axis_threshold_rad = (
        math.radians(args_cli.z_axis_threshold_deg) if evaluates_z_axis else None
    )
    path_metadata = getattr(env_cfg, "path_metadata", None)
    evaluation_threshold = (
        path_metadata["position_threshold_m"]
        if evaluation_protocol == "waypoint_path"
        else args_cli.success_threshold
    )
    if evaluation_protocol == "waypoint_path" and args_cli.target_set is not None:
        raise ValueError("--target-set cannot be combined with a waypoint-path task")

    if args_cli.target_set is None:
        target_set_metadata = None
    else:
        target_set_path = args_cli.target_set.expanduser().resolve()
        if not target_set_path.is_file():
            raise FileNotFoundError(f"Target set not found: {target_set_path}")
        target_set_sha256 = hashlib.sha256(target_set_path.read_bytes()).hexdigest()
        target_set_metadata = {
            "path": str(target_set_path),
            "sha256": target_set_sha256,
        }

    data_root = Path(os.environ.get("FRANKA_RL_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()
    validate_data_root(data_root)

    hardware_replay = None
    hardware_replay_metadata = None
    if args_cli.hardware_session is not None:
        from franka_rl.utils.hardware_replay import HardwareSessionReplay

        hardware_replay = HardwareSessionReplay.from_directory(
            args_cli.hardware_session, data_root=data_root
        )
        hardware_replay.validate_task(args_cli.task)
        hardware_replay.validate_path_metadata(path_metadata)
        if args_cli.target_set is not None:
            raise ValueError("--hardware-session cannot be combined with --target-set")
        if args_cli.num_envs != 1 or args_cli.num_episodes != 1:
            raise ValueError("Hardware replay requires --num_envs 1 --num_episodes 1")
        if scenario.control.action_delay_steps != 1:
            raise ValueError(
                "Hardware replay requires a scenario with exactly one policy-step delay"
            )
        if Path(args_cli.checkpoint).expanduser().resolve() != hardware_replay.checkpoint:
            raise ValueError(
                "--checkpoint does not match the checkpoint recorded by the hardware bundle"
            )
        controller_metadata = hardware_replay.apply_controller_config(
            env_cfg.actions.arm_action
        )
        hardware_replay_metadata = {
            **hardware_replay.artifact_metadata(),
            "simulator_controller": controller_metadata,
        }

    if args_cli.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        output_dir = data_root / "evaluations" / f"{timestamp}_{args_cli.scenario}"
    else:
        output_dir = args_cli.output_dir.expanduser().resolve()
        try:
            output_dir.relative_to(data_root)
        except ValueError as error:
            raise RuntimeError(
                f"Evaluation output directory must be under FRANKA_RL_DATA_ROOT ({data_root}): {output_dir}"
            ) from error

    evaluation_completed = False
    with launch_simulation(env_cfg, args_cli):
        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # Random-point evaluation terminates on sustained proximity. Path tasks
        # already define strict whole-path success and waypoint-timeout outcomes.
        if evaluation_protocol == "position_z_axis":
            env_cfg.terminations.reached_target = TerminationTermCfg(
                func=mdp.SustainedPositionZAxisSuccess,
                time_out=False,
                params={
                    "command_name": "ee_pose",
                    "asset_cfg": SceneEntityCfg(
                        "robot", body_names=[env_cfg.commands.ee_pose.body_name]
                    ),
                    "distance_threshold": args_cli.success_threshold,
                    "z_axis_threshold_rad": z_axis_threshold_rad,
                    "required_steps": args_cli.success_steps,
                },
            )
        elif evaluation_protocol != "waypoint_path":
            env_cfg.terminations.reached_target = TerminationTermCfg(
                func=mdp.SustainedPositionSuccess,
                time_out=False,
                params={
                    "command_name": "ee_pose",
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=[env_cfg.commands.ee_pose.body_name],
                    ),
                    "distance_threshold": args_cli.success_threshold,
                    "required_steps": args_cli.success_steps,
                },
            )
        action_cfg = env_cfg.actions.arm_action
        max_joint_velocity = getattr(action_cfg, "max_measured_velocity", None)
        env_cfg.terminations.evaluation_state_metrics = TerminationTermCfg(
            func=mdp.EvaluationStateMetrics,
            time_out=False,
            params={
                "command_name": "ee_pose",
                "hand_asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[env_cfg.commands.ee_pose.body_name],
                ),
                "joint_asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["panda_joint.*"],
                ),
                "max_joint_velocity": max_joint_velocity,
                "action_name": "arm_action",
                "measure_z_axis": evaluates_z_axis,
            },
        )

        scenario_metadata = scenario_modifier.apply(env_cfg)
        if getattr(env_cfg, "robot_model", "panda") == "fr3v2_bare_flange":
            asset_path = Path(env_cfg.scene.robot.spawn.usd_path)
            asset_manifest = asset_path.parent / "manifest.json"
            scenario_metadata["robot_model"] = {
                "name": env_cfg.robot_model,
                "usd_sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
                "asset_manifest": json.loads(asset_manifest.read_text()),
                "control_contract": getattr(env_cfg, "control_contract", "legacy_fr3_implicit_pd"),
                "controller_assumption": getattr(
                    env_cfg,
                    "controller_profile",
                    "implicit_pd_80_4_armature_0.001_60hz_physics_30hz_policy",
                ),
                "payload_reference": "fr3_flange origin",
            }
        if hardware_replay_metadata is not None:
            scenario_metadata["hardware_replay"] = hardware_replay_metadata
        print("[INFO] Robustness scenario:")
        print_dict(scenario_metadata, nesting=4)

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        evaluation_state_term = env.unwrapped.termination_manager.get_term_cfg("evaluation_state_metrics").func

        def trajectory_state_reader(_env):
            return {
                "position_error_m": evaluation_state_term.position_error_m,
                "z_axis_error_rad": evaluation_state_term.z_axis_error_rad,
                "joint_limit_margin_rad": evaluation_state_term.joint_limit_margin_rad,
                "joint_velocity_ratio": evaluation_state_term.joint_velocity_ratio,
                "command_difference_norm_rad": evaluation_state_term.command_difference_norm_rad,
                "reference_acceleration_ratio": evaluation_state_term.reference_acceleration_ratio,
                "action_clipping": evaluation_state_term.action_clipping,
                "applied_torque_norm_nm": evaluation_state_term.applied_torque_norm_nm,
                "tracking_error_norm_rad": evaluation_state_term.tracking_error_norm_rad,
                "reference_projection_norm_rad": evaluation_state_term.reference_projection_norm_rad,
            }

        # Install deterministic replay only after SimulationApp and the
        # environment exist. An eager custom Isaac command import initializes
        # USD/pxr too early and can make native Kit startup crash.
        replay_controller = None
        if hardware_replay is not None:
            from franka_rl.utils.hardware_replay import HardwareInitialStateController

            hardware_initial_state = HardwareInitialStateController(
                hardware_replay.initial_joint_position,
                hardware_replay.initial_joint_velocity,
            )
            hardware_initial_state.install(env.unwrapped, reset_event_name="reset_arm")
        if args_cli.target_set is not None:
            from franka_rl.utils.target_replay import TargetReplayController

            replay_controller = TargetReplayController(args_cli.target_set, target_set_sha256)
            replay_controller.install(
                env.unwrapped,
                command_name="ee_pose",
                reset_event_name="reset_arm",
            )
            target_set_metadata.update(
                {
                    "schema_version": replay_controller.schema_version,
                    "replays_initial_joint_state": (replay_controller.replays_initial_joint_state),
                }
            )

        robot = env.unwrapped.scene["robot"]
        initial_joint_ids, initial_joint_names = robot.find_joints("panda_joint.*")
        trace_action_term = None
        if hardware_replay is not None:
            trace_action_term = env.unwrapped.action_manager.get_term("arm_action")
            trace_ticks = math.ceil(env_cfg.episode_length_s / env_cfg.sim.dt) + 2000
            trace_action_term.start_diagnostic_trace(trace_ticks, env_id=0)

        def initial_state_reader(_env):
            return {
                "joint_position": robot.data.joint_pos.torch[:, initial_joint_ids],
                "joint_velocity": robot.data.joint_vel.torch[:, initial_joint_ids],
            }

        action_delay_steps = scenario.control.action_delay_steps
        delay_source = "scenario"
        if action_delay_steps is None:
            action_delay_steps = int(getattr(env.unwrapped.cfg, "required_action_delay_steps", 0))
            delay_source = "task_contract"
        if action_delay_steps:
            from franka_rl.utils.action_delay import (
                FixedActionDelayWrapper,
                action_term_delay_fill_provider,
            )

            delay_fill_provider = action_term_delay_fill_provider(env)
            env = FixedActionDelayWrapper(
                env,
                action_delay_steps,
                initial_action_provider=delay_fill_provider,
            )
            scenario_metadata["runtime_control"] = {
                "action_delay_steps": action_delay_steps,
                "source": delay_source,
                "initial_action": (
                    "action_term_reset_command" if delay_fill_provider is not None else "zero"
                ),
                "reset_behavior": "refill_done_environment_history",
            }
        elif scenario.control.action_delay_range is not None:
            from franka_rl.utils.action_delay import (
                RandomActionDelayWrapper,
                action_term_delay_fill_provider,
            )

            min_delay, max_delay = scenario.control.action_delay_range
            delay_fill_provider = action_term_delay_fill_provider(env)
            env = RandomActionDelayWrapper(
                env,
                min_delay,
                max_delay,
                initial_action_provider=delay_fill_provider,
            )
            scenario_metadata["runtime_control"] = {
                "action_delay_range": [min_delay, max_delay],
                "sampling": "uniform_integer_per_environment_per_episode",
                "initial_action": (
                    "action_term_reset_command" if delay_fill_provider is not None else "zero"
                ),
                "reset_behavior": "refill_done_environment_history",
            }

        if scenario_modifier.records_episode_parameters:
            # Resolve runtime joint names now so the artifact schema exactly
            # matches the tensors sampled at each episode start.
            scenario_modifier.capture(env.unwrapped)
            domain_parameter_schema = scenario_modifier.parameter_schema
            domain_parameter_reader = scenario_modifier.capture
            scenario_metadata["runtime_validation"] = scenario_modifier.validate_runtime(env.unwrapped)
        else:
            domain_parameter_schema = {}
            domain_parameter_reader = None

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": str(output_dir / "videos"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording evaluation video.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        # configure_seed must be called after runner construction so that PyTorch deterministic settings
        # do not interfere with the runner's internal initialization.
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        if version.parse(installed_version) >= version.parse("4.0.0"):
            reset_policy = policy.reset
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic
            reset_policy = policy_nn.reset

        # The primer must not advance deterministic replay cursors. Paired
        # target/state evaluations retain their existing explicit-reset path.
        startup_auto_reset_prime = bool(
            getattr(env.unwrapped.cfg, "evaluation_startup_auto_reset_prime", False)
        )
        if startup_auto_reset_prime and (
            args_cli.target_set is not None or hardware_replay is not None
        ):
            startup_auto_reset_prime = False
            initialization_reason = (
                "disabled_for_hardware_replay"
                if hardware_replay is not None
                else "disabled_for_paired_target_replay"
            )
        else:
            initialization_reason = "task_contract"
        scenario_metadata["runtime_initialization"] = {
            "mode": "unrecorded_timeout_auto_reset",
            "enabled": startup_auto_reset_prime,
            "source": initialization_reason,
        }

        # initialize evaluator
        evaluation_cfg = EvaluationConfig(
            job_id=args_cli.job_id,
            target_set=target_set_metadata,
            hardware_replay=hardware_replay_metadata,
            scenario=scenario_metadata,
            record_domain_parameters=scenario_modifier.records_episode_parameters,
            domain_parameter_schema=domain_parameter_schema,
            initial_joint_names=tuple(initial_joint_names),
            num_episodes=args_cli.num_episodes,
            success_threshold=evaluation_threshold,
            success_steps=args_cli.success_steps,
            z_axis_threshold_rad=z_axis_threshold_rad,
            seed=env_cfg.seed,
            output_dir=output_dir,
            task_name=args_cli.task,
            evaluation_protocol=evaluation_protocol,
            path=path_metadata,
            real_time=args_cli.real_time,
            deterministic=args_cli.deterministic,
            startup_auto_reset_prime=startup_auto_reset_prime,
        )

        evaluator = PolicyEvaluator(
            env=env,
            policy=policy,
            config=evaluation_cfg,
            checkpoint_path=resume_path,
            reset_policy=reset_policy,
            domain_parameter_reader=domain_parameter_reader,
            trajectory_state_reader=trajectory_state_reader,
            initial_state_reader=initial_state_reader,
            configure_episode_quotas=(replay_controller.set_episode_quotas if replay_controller is not None else None),
        )

        # simulate environment
        try:
            results = evaluator.run()
            if results.num_episodes != args_cli.num_episodes:
                raise RuntimeError("Evaluation stopped before recording all requested episodes.")
            results.print_summary()
            results.save()
            if hardware_replay is not None:
                from franka_rl.utils.hardware_replay import save_simulation_trace

                trace_action_term.stop_diagnostic_trace()
                save_simulation_trace(
                    trace_action_term.diagnostic_trace(),
                    output_dir,
                    physics_dt_s=float(env_cfg.sim.dt),
                    circle_path=hardware_replay.path,
                )
            completion = {
                "status": "complete",
                "job_id": args_cli.job_id,
                "episodes_recorded": results.num_episodes,
                "completed_at": datetime.now().astimezone().isoformat(),
            }
            completion_tmp = output_dir / "completed.json.tmp"
            completion_path = output_dir / "completed.json"
            with completion_tmp.open("w", encoding="utf-8") as file:
                json.dump(completion, file, indent=2)
                file.write("\n")
            completion_tmp.replace(completion_path)
        finally:
            env.close()
        evaluation_completed = True

    # launch_simulation can print and suppress exceptions from its body.
    if not evaluation_completed:
        raise RuntimeError("Evaluation failed; see the preceding simulator traceback.")


if __name__ == "__main__":
    main()
