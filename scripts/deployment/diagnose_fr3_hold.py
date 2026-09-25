"""Paired stationary-reference gravity ablation, without policy/governor/reset stepping.

Two separate Isaac processes use identical seeded resets and hold their initial
measured positions. Physics and implicit PD remain identical to the governed task.
This diagnostic deliberately bypasses governor termination to observe the plant.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def child(args):
    import gymnasium as gym
    import torch
    import franka_rl.tasks  # noqa: F401
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

    launcher_parser = argparse.ArgumentParser()
    add_launcher_args(launcher_parser)
    launch_args = launcher_parser.parse_args(["--device", args.device, "--viz", "none"])

    cfg, _ = resolve_task_config("Franka-FR3v2-Reach-v0", "")
    cfg.scene.num_envs = 2
    cfg.seed = 123
    cfg.sim.dt = 1 / 3000
    cfg.decimation = 100
    cfg.sim.render_interval = 100
    cfg.commands.ee_pose.debug_vis = False
    cfg.scene.robot.spawn.rigid_props.disable_gravity = False
    cfg.sim.gravity = (0., 0., -9.81 if args.case == "gravity_on" else 0.)
    asset = Path(cfg.scene.robot.spawn.usd_path)
    manifest = json.loads((asset.parent / "manifest.json").read_text())
    if manifest.get("source_robot") != "fr3v2.1":
        raise ValueError("Select the corrected live FR3v2.1 asset via FRANKA_RL_FR3_USD")
    # Reuse this installation's standard launcher flags.
    with launch_simulation(cfg, launch_args):
        env = gym.make("Franka-FR3v2-Reach-v0", cfg=cfg).unwrapped
        try:
            env.reset()
            robot = env.scene["robot"]
            q_ref = robot.data.joint_pos.torch.clone()
            dq_initial = robot.data.joint_vel.torch.clone()
            robot.set_joint_position_target_index(target=q_ref)
            robot.set_joint_velocity_target_index(target=torch.zeros_like(q_ref))
            view = robot.data._root_view
            steps = round(args.duration / env.physics_dt)
            trace = {key: [] for key in ("q", "dq", "position_error", "pd_estimate",
                                         "isaac_applied_torque", "gravity_compensation",
                                         "physx_actuation_forces")}
            stiffness = robot.data.joint_stiffness.torch.clone()
            damping = robot.data.joint_damping.torch.clone()
            for step in range(steps + 1):
                q = robot.data.joint_pos.torch
                dq = robot.data.joint_vel.torch
                values = dict(q=q, dq=dq, position_error=q_ref-q,
                              pd_estimate=stiffness*(q_ref-q)-damping*dq,
                              isaac_applied_torque=robot.data.applied_torque.torch,
                              gravity_compensation=robot.data.gravity_compensation_forces.torch)
                for key, value in values.items():
                    trace[key].append(value.detach().cpu().numpy().copy())
                trace["physx_actuation_forces"].append(view.get_dof_actuation_forces().numpy().copy())
                if step < steps:
                    robot.set_joint_position_target_index(target=q_ref)
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(env.physics_dt)
            arrays = {k: np.stack(v) for k, v in trace.items()}
            assert all(np.isfinite(v).all() for v in arrays.values())
            time_s = np.arange(steps+1)*env.physics_dt
            arrays.update(time_s=time_s, q_reference=q_ref.cpu().numpy(), dq_initial=dq_initial.cpu().numpy(),
                          stiffness=stiffness.cpu().numpy(), damping=damping.cpu().numpy())
            np.savez_compressed(args.output_dir / f"{args.case}.npz", **arrays)
            crossing = np.argwhere(abs(arrays["dq"]) > .3)
            first = crossing[0] if len(crossing) else None
            summary = dict(completed=True, case=args.case, seed=123, num_envs=2,
                           physics_dt=env.physics_dt, steps=steps,
                           joint_names=robot.joint_names,
                           initial_q=arrays["q_reference"].tolist(), initial_dq=arrays["dq_initial"].tolist(),
                           peak_speed_rad_s=np.max(abs(arrays["dq"]), axis=(0,1)).tolist(),
                           final_position_error_rad=arrays["position_error"][-1].tolist(),
                           initial_gravity_compensation_nm=arrays["gravity_compensation"][0].tolist(),
                           first_speed_crossing=None if first is None else dict(
                               time_s=float(time_s[first[0]]), env_id=int(first[1]),
                               joint=robot.joint_names[first[2]], velocity=float(arrays["dq"][tuple(first)])),
                           usd_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                           source_urdf_sha256=manifest["urdf_sha256"],
                           script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           torque_notes="pd_estimate and isaac_applied_torque are implicit-actuator estimates, not measured drive torque. physx_actuation_forces is raw API output, not assumed total drive/constraint torque. gravity_compensation is g(q), not an applied feedforward term.",
                           isolation="No env.step, policy action, governor, automatic resets or termination. Exact initial measured q is held. Gravity vector is the only between-case configuration change.")
            (args.output_dir / f"{args.case}.json").write_text(json.dumps(summary, indent=2)+"\n")
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=.2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case", choices=["gravity_on", "gravity_off"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]  # task resolution invokes Hydra
    if not 0 < args.duration <= 2:
        raise ValueError("Diagnostic duration must be in (0, 2] seconds")
    args.output_dir = args.output_dir.resolve()
    root = Path(os.environ["FRANKA_RL_DATA_ROOT"]).resolve()
    args.output_dir.relative_to(root)
    if args.case:
        child(args)
        return
    import shutil
    if not root.is_dir() or not os.access(root, os.W_OK) or shutil.disk_usage(root).free < 100*1024**2:
        raise RuntimeError("Data volume unavailable, unwritable or insufficient space")
    args.output_dir.mkdir(parents=False, exist_ok=False)
    summaries = {}
    for case in ("gravity_on", "gravity_off"):
        command = [sys.executable, str(Path(__file__).resolve()), "--output-dir", str(args.output_dir),
                   "--case", case, "--duration", str(args.duration), "--device", args.device]
        with (args.output_dir / f"{case}.log").open("w") as log:
            result = subprocess.run(command, cwd=args.output_dir, stdout=log, stderr=subprocess.STDOUT, check=False)
        path = args.output_dir / f"{case}.json"
        if result.returncode or not path.is_file():
            raise RuntimeError(f"{case} failed; inspect {args.output_dir / (case + '.log')}")
        summaries[case] = json.loads(path.read_text())
    for key in ("initial_q", "initial_dq"):
        np.testing.assert_array_equal(summaries["gravity_on"][key], summaries["gravity_off"][key])
    report = dict(initial_states_match=True, cases=summaries)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
