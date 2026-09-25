"""Small actual-Isaac governor integration check; no policy training or hardware."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import franka_rl.tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

original_arguments = sys.argv[1:]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--num_envs", type=int, default=2)
add_launcher_args(parser)
args, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining


def main():
    if args.num_envs < 2 or args.steps < 6:
        raise ValueError("Use at least two environments and six policy steps.")
    root = Path(os.environ["FRANKA_RL_DATA_ROOT"]).resolve()
    output = args.output.resolve()
    output.relative_to(root)
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    cfg, _ = resolve_task_config("Franka-FR3v2-Governed-Reach-v0", "")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 123
    cfg.commands.ee_pose.debug_vis = False
    asset = Path(cfg.scene.robot.spawn.usd_path)
    manifest = json.loads((asset.parent / "manifest.json").read_text())
    if manifest.get("source_robot") != "fr3v2.1":
        raise ValueError("Smoke test requires the corrected live fr3v2.1 asset.")
    completed = False
    with launch_simulation(cfg, args):
        env = gym.make("Franka-FR3v2-Governed-Reach-v0", cfg=cfg).unwrapped
        try:
            env.reset()
            term = env.action_manager.get_term("arm_action")
            robot = env.scene["robot"]
            config = json.loads(Path(term.cfg.governor_config).read_text())["core"]
            peaks = dict(reference_velocity=0., reference_acceleration=0., reference_jerk=0., measured_velocity=0.)
            samples = 0
            last = {}
            original_apply = term.apply_actions

            def observe_apply():
                nonlocal samples
                original_apply()
                if (term._physics_tick - 1) % 3:
                    return
                samples += 1
                for i, value in enumerate(term._governor.outputs):
                    assert value.command_valid
                    q, v, a = map(np.asarray, (value.q, value.dq, value.ddq))
                    assert np.isfinite(q).all()
                    assert np.all(q >= np.asarray(config["lower"]) + config["margin"] - 1e-9)
                    assert np.all(q <= np.asarray(config["upper"]) - config["margin"] + 1e-9)
                    assert np.all(abs(v) <= np.asarray(config["max_velocity"]) + 1e-8)
                    assert np.all(abs(a) <= np.asarray(config["max_acceleration"]) + 1e-8)
                    session = int(term._governor.sessions[i])
                    if i in last and last[i][0] == session:
                        jerk = (a - last[i][1]) / .001
                        assert np.all(abs(jerk) <= np.asarray(config["max_jerk"]) + 1e-6)
                        peaks["reference_jerk"] = max(peaks["reference_jerk"], float(abs(jerk).max()))
                    last[i] = session, a.copy()
                    peaks["reference_velocity"] = max(peaks["reference_velocity"], float(abs(v).max()))
                    peaks["reference_acceleration"] = max(peaks["reference_acceleration"], float(abs(a).max()))
                peaks["measured_velocity"] = max(peaks["measured_velocity"], float(robot.data.joint_vel.torch.abs().max()))

            term.apply_actions = observe_apply
            actions = torch.zeros((args.num_envs, 7), device=env.device)
            actions[:, 0] = 0.05  # small 0.025 rad policy target offset
            started = time.perf_counter()
            partial_reset_verified = False
            for step in range(args.steps):
                if step == args.steps // 2:
                    before = term._governor.sessions.copy()
                    env._reset_idx(torch.tensor([0], device=env.device))
                _, _, terminated, truncated, _ = env.step(actions)
                assert not (terminated | truncated).any(), "Unexpected termination during smoke rollout"
                if step == args.steps // 2:
                    assert term._governor.sessions[0] == before[0] + 1
                    np.testing.assert_array_equal(term._governor.sessions[1:], before[1:])
                    partial_reset_verified = True
            elapsed = time.perf_counter() - started
            # Explicit simulated stop: verify governor_fault reaches the environment.
            term._governor.batch.stop(0)
            _, _, terminated, _, _ = env.step(actions)
            assert bool(terminated[0])
            assert bool(env.termination_manager.get_term("governor_fault")[0])
            result = dict(passed=True, scope="synthetic governor integration, not hardware equivalence",
                          num_envs=args.num_envs, policy_steps=args.steps, policy_dt=env.step_dt,
                          physics_dt=env.physics_dt, observed_governor_ticks=samples,
                          wall_seconds=elapsed, env_steps_per_second=args.steps * args.num_envs / elapsed,
                          partial_reset_verified=partial_reset_verified, stop_termination_verified=True,
                          peaks=peaks, governor_config_sha256=term.config_sha256,
                          asset_manifest=manifest, usd_sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
        except Exception as error:
            # Preserve failure evidence as well as successful runs. Never turn a
            # failed plant/governor compatibility check into a passing smoke test.
            diagnostics = []
            if "term" in locals():
                for i, value in enumerate(term._governor.outputs):
                    diagnostics.append(dict(env_id=i, status=str(value.status), reason=str(value.reason),
                                            command_valid=value.command_valid, reference_q=list(value.q),
                                            reference_dq=list(value.dq),
                                            measured_q=robot.data.joint_pos.torch[i].cpu().tolist(),
                                            measured_dq=robot.data.joint_vel.torch[i].cpu().tolist()))
            output.write_text(json.dumps(dict(passed=False, error=str(error),
                                             physics_tick=getattr(locals().get("term"), "_physics_tick", None),
                                             diagnostics=diagnostics, asset_manifest=manifest), indent=2) + "\n")
            raise
        finally:
            env.close()
        completed = True
    if not completed:
        raise RuntimeError("Governor smoke test failed; see preceding simulator traceback.")


if __name__ == "__main__":
    if os.environ.get("FRANKA_GOVERNOR_SMOKE_CHILD") == "1":
        main()
    else:
        # Kit shutdown can terminate its process with code zero even while an
        # exception is unwinding. Verify the fresh report outside that process.
        if args.output.exists():
            raise FileExistsError(args.output)
        child_env = os.environ.copy()
        child_env["FRANKA_GOVERNOR_SMOKE_CHILD"] = "1"
        # Preserve all launcher/preset flags, whose spelling varies by version.
        # sys.argv was rewritten for Hydra above, so save original arguments.
        command = [sys.executable, str(Path(__file__).resolve()), *original_arguments]
        child = subprocess.run(command, env=child_env, check=False)
        passed = False
        if args.output.is_file():
            passed = json.loads(args.output.read_text()).get("passed") is True
        if child.returncode or not passed:
            raise SystemExit(1)
