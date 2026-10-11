"""Run finite M1 non-RL joint trials; all target/controller choices come from YAML."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys

import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
from franka_motion.config import load_motion_suite, expand_trials

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Franka-FR3v2-MotionControl-Joint-v0")
parser.add_argument("--suite", type=Path, default=Path(__file__).resolve().parents[2]/"source/franka_motion/franka_motion/config/suites/joint_smoke.yaml")
parser.add_argument("--trial", help="Expanded trial ID; omit to execute the finite YAML suite serially")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--verify-reset-parity", action="store_true", help="After trials, test rejection/fault capture/selective reset on cloned simulation only")
parser.add_argument("--output-dir", type=Path, help="New output directory; default is on FRANKA_RL_DATA_ROOT")
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]
import franka_rl.tasks
from franka_rl.experiments.motion_runner import IsaacMotionRunner, verify_reset_parity


def main():
    suite = load_motion_suite(args.suite)
    if args.verify_reset_parity and args.num_envs < 2:
        raise ValueError("--verify-reset-parity requires at least two environments")
    if suite.backend != "simulation" or not 1 <= args.num_envs <= 16:
        raise ValueError("M1 simulation runner supports simulation suites and 1..16 environments")
    trials = [t for t in expand_trials(suite) if args.trial is None or t.trial_id == args.trial]
    if not trials or any(t.trajectory.data["type"] not in ("joint_hold", "joint_ptp") for t in trials):
        raise ValueError("Select a valid M1 joint trial/suite")
    root = Path(os.environ.get("FRANKA_RL_DATA_ROOT", "/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data"))
    if args.output_dir is None:
        if not root.is_dir() or not any(os.path.ismount(p) for p in (root, *root.parents) if str(p) != "/"):
            raise RuntimeError("Data volume not mounted; refusing fallback")
        output = root/"motion_control"/(datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")+"_"+suite.name)
    else:
        output = args.output_dir.expanduser().resolve()
        if not (output.is_relative_to(root.resolve()) or output.is_relative_to(Path("/tmp"))):
            raise ValueError("Generated motion artifacts must stay on the data volume (or /tmp for bounded verification)")
    parent = next(p for p in (output.parent, *output.parents) if p.exists())
    if not os.access(parent, os.W_OK) or shutil.disk_usage(parent).free < 128*1024**2:
        raise RuntimeError("Output filesystem unavailable or less than 128 MiB free")
    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}")
    env_cfg, _ = resolve_task_config(args.task, "")
    from franka_rl.tasks.manager_based.franka_motion_control.motion_env_cfg import make_fr3_robot_cfg
    env_cfg.scene.robot = make_fr3_robot_cfg(suite.workspace.data)
    bounds = suite.workspace.data["joint_bounds_rad"]
    env_cfg.actions.joint_reference.lower, env_cfg.actions.joint_reference.upper = (tuple(bounds["commanded"][key]) for key in ("lower", "upper"))
    env_cfg.actions.joint_reference.abort_lower, env_cfg.actions.joint_reference.abort_upper = (tuple(bounds["measured_abort"][key]) for key in ("lower", "upper"))
    env_cfg.scene.num_envs = args.num_envs
    # One profile per process. Different gains/rates must use separate invocations.
    profile = trials[0].controller
    if any(t.controller != profile for t in trials) or profile.data["delay_steps"] != 0:
        raise ValueError("M1 process requires a single controller profile and zero publication delay")
    if profile.data["physics_hz"] != 1000:
        raise ValueError("M1 servo requires 1000 Hz physics")
    env_cfg.decimation = round(profile.data["physics_hz"]/profile.data["reference_hz"])
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.actions.joint_reference.stiffness = tuple(profile.data["stiffness"])
    env_cfg.actions.joint_reference.damping = tuple(profile.data["damping"])
    servo = profile.data["joint_servo"]
    if any(value > limit for value, limit in zip(servo["effort_limits_nm"], (87.,)*4+(12.,)*3)):
        raise ValueError("Controller effort bounds exceed live model limits")
    for cfg_key, yaml_key in (("position_error_clip", "position_error_clip_rad"), ("slew_rate_nm_s", "slew_rate_nm_s"),
        ("cutoff_hz", "cutoff_hz"), ("effort_limits_nm", "effort_limits_nm"), ("max_measured_velocity", "max_measured_velocity_rad_s")):
        value = servo[yaml_key]
        setattr(env_cfg.actions.joint_reference, cfg_key, tuple(value) if isinstance(value, list) else value)
    # Allow each trial's configured derivative limit; runner does not synthesize a clamp.
    env_cfg.actions.joint_reference.max_reference_velocity = tuple(
        max(t.trajectory.data["max_velocity_rad_s"][j]*t.speed_scale for t in trials) for j in range(7))
    if args.device is not None:
        env_cfg.sim.device = args.device
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            output.mkdir(parents=True, exist_ok=False)
            runner = IsaacMotionRunner(env, suite)
            results = []
            for trial in trials:
                result = runner.run_trial(trial, output/trial.trial_id)
                results.append(result)
                print(f"{trial.trial_id}: {result['status']}; acceptance={result['acceptance_passed']}; hold={result['hold_max_error_rad']} rad", flush=True)
                if result["status"] == "fault":
                    break
            if args.verify_reset_parity and all(r["status"] == "complete" for r in results):
                (output/"reset_parity.json").write_text(json.dumps(verify_reset_parity(env), indent=2)+"\n")
            (output/"index.json").write_text(json.dumps({"suite": suite.name,
                "results": results, "all_accepted": len(results) == len(trials) and all(r["acceptance_passed"] for r in results)}, indent=2)+"\n")
            print(f"Motion artifacts: {output}", flush=True)
            return 0 if len(results) == len(trials) and all(r["acceptance_passed"] for r in results) else 1
        finally:
            env.close()
    # launch_simulation logs/suppresses some initialization exceptions. Never
    # turn that fall-through into SystemExit(None), which would report success.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
