"""Reconstruct training physics and export raw initialized PhysX tensors."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deployment"))
import franka_rl.tasks  # noqa: F401
from franka_policy_bundle.bundle import git_state, sha256
from franka_policy_bundle.contract import TrainingLoader
from franka_rl.utils.scenarios import ScenarioCatalog, ScenarioModifier

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenario", choices=["nominal", "franka_dr_train_v2"], required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--task", default="Template-Franka-Rl-v0")
add_launcher_args(parser)
args, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")


def array(value):
    if hasattr(value, "numpy") and not isinstance(value, torch.Tensor):
        return value.numpy().copy()
    if hasattr(value, "torch"):
        value = value.torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def snapshot(env, label):
    robot = env.scene["robot"]
    view = robot.data._root_view
    methods = {
        "mass_kg": "get_masses",
        "com_pose_body_xyz_xyzw": "get_coms",
        "inertia_body_at_com_column_major_kg_m2": "get_inertias",
        "position_limits_rad": "get_dof_limits",
        "velocity_limits_rad_s": "get_dof_max_velocities",
        "effort_limits_Nm": "get_dof_max_forces",
        "stiffness_Nm_rad": "get_dof_stiffnesses",
        "damping_Nm_s_rad": "get_dof_dampings",
        "armature_kg_m2": "get_dof_armatures",
        "friction_backend_properties": "get_dof_friction_properties",
        "q_rad": "get_dof_positions",
        "dq_rad_s": "get_dof_velocities",
    }
    result = dict(
        label=label,
        environment_indices=list(range(env.num_envs)),
        body_names=robot.body_names,
        joint_names=robot.joint_names,
        tensors={},
    )
    for name, method in methods.items():
        result["tensors"][name] = array(getattr(view, method)()).tolist()
    inertia = np.asarray(result["tensors"]["inertia_body_at_com_column_major_kg_m2"])
    matrices = inertia.reshape(*inertia.shape[:-1], 3, 3).swapaxes(-1, -2)
    result["tensors"]["inertia_body_at_com_matrix_kg_m2"] = matrices.tolist()
    result["actuators"] = {}
    for name, actuator in robot.actuators.items():
        values = {}
        for field in ["stiffness", "damping", "effort_limit", "velocity_limit", "armature", "friction"]:
            if hasattr(actuator, field):
                values[field] = array(getattr(actuator, field)).tolist()
        result["actuators"][name] = values
    return result


def usd_evidence(env, url, audit):
    import omni.client
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = env.sim.stage
    result = dict(
        root_url=url,
        meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
        kilograms_per_unit=UsdPhysics.GetStageKilogramsPerUnit(stage),
        up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        layers=[],
        authored_bodies=[],
        joints=[],
        variants={},
    )
    prefix = url.rsplit("/", 1)[0] + "/"
    for layer in stage.GetUsedLayers():
        identifier = layer.realPath or layer.identifier
        record = dict(identifier=identifier, anonymous=layer.anonymous)
        if layer.anonymous:
            record["composed_text_sha256"] = hashlib.sha256(layer.ExportToString().encode()).hexdigest()
        else:
            try:
                if Path(identifier).is_file():
                    record["sha256"] = sha256(identifier)
                else:
                    status, _, content = omni.client.read_file(identifier)
                    if status != omni.client.Result.OK:
                        raise RuntimeError(str(status))
                    record["sha256"] = hashlib.sha256(memoryview(content)).hexdigest()
                key = (
                    identifier.split("/FrankaPanda/", 1)[-1]
                    if "/FrankaPanda/" in identifier
                    else identifier.removeprefix(prefix)
                )
                record["hardware_audit_key"] = key if key in audit["asset_sha256"] else None
                if key in audit["asset_sha256"]:
                    record["matches_hardware_audit"] = record["sha256"] == audit["asset_sha256"][key]
            except Exception as exc:
                record["hash_error"] = str(exc)
        result["layers"].append(record)
    root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        selections = {n: prim.GetVariantSet(n).GetVariantSelection() for n in prim.GetVariantSets().GetNames()}
        if selections:
            result["variants"][str(prim.GetPath())] = selections
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(prim)
            inertia = mass.GetDiagonalInertiaAttr().Get()
            com = mass.GetCenterOfMassAttr().Get()
            axes = mass.GetPrincipalAxesAttr().Get()
            result["authored_bodies"].append(
                dict(
                    path=str(prim.GetPath()),
                    name=prim.GetName(),
                    mass_kg=mass.GetMassAttr().Get(),
                    diagonal_inertia_kg_m2=list(inertia),
                    com_m=list(com),
                    principal_axes_xyzw=[*axes.GetImaginary(), axes.GetReal()],
                    principal_axes_authored=mass.GetPrincipalAxesAttr().HasAuthoredValueOpinion(),
                )
            )
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            result["joints"].append(
                dict(
                    name=prim.GetName(),
                    path=str(prim.GetPath()),
                    body0=[str(p) for p in joint.GetBody0Rel().GetTargets()],
                    body1=[str(p) for p in joint.GetBody1Rel().GetTargets()],
                )
            )
    return result


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return sum((differences(a.get(k), b.get(k), f"{path}/{k}") for k in sorted(a.keys() | b.keys())), [])
    return [] if a == b else [dict(path=path, saved=a, reconstructed=b)]


def main():
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < 256 * 1024**2:
        raise RuntimeError("Output parent must exist with at least 256 MiB free")
    output.mkdir()
    shutil.copyfile(__file__, output / "extract_training_physics.py")
    evidence = REPO / "deployment/model_audit/2026-09-24"
    saved_path = evidence / ("nominal_env.yaml" if args.scenario == "nominal" else "dr_env.yaml")
    saved = yaml.load(saved_path.read_text(), Loader=TrainingLoader)
    catalog = ScenarioCatalog.from_yaml()
    cfg, _ = resolve_task_config(args.task, "")
    modifier = ScenarioModifier(catalog.get(args.scenario), catalog)
    scenario = modifier.apply(cfg)
    # Compare configuration before changing instance count, seed or rendering.
    serialized = yaml.dump(cfg.to_dict())
    (output / "reconstructed_env.yaml").write_text(serialized)
    reconstructed = yaml.load(serialized, Loader=TrainingLoader)
    write(output / "config_differences.json", differences(saved, reconstructed))
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.commands.ee_pose.debug_vis = False
    cfg.sim.device = args.device or "cuda:0"
    versions = {}
    for name in ["isaacsim", "isaaclab", "isaaclab-physx", "torch", "rsl-rl-lib", "numpy"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    report = dict(
        schema_version=1,
        scenario=scenario,
        seed=args.seed,
        num_envs=args.num_envs,
        versions=versions,
        python=sys.version,
        invocation=sys.orig_argv,
        source=git_state(REPO),
        isaaclab_source=git_state(REPO.parent / "IsaacLab"),
        saved_config_sha256=sha256(saved_path),
        reconstruction_overrides=dict(
            num_envs=args.num_envs, seed=args.seed, command_debug_vis=False, device=cfg.sim.device
        ),
        caveats=[
            "Current runtime reconstruction, not a historical training-time dump.",
            "Reduced environment count changes seeded DR sample sequence versus training.",
            "Raw PhysX inertia is column-major, at COM, expressed in body-prim frame; not world frame.",
            "COM orientation is principal-axes frame relative to body, quaternion xyzw.",
        ],
        snapshots=[],
    )
    checkpoint_name = (
        "2026-09-23_01-15-04_no_success_termination/model_299.pt"
        if args.scenario == "nominal"
        else "2026-09-24_00-48-20_dr_v2/model_999.pt"
    )
    checkpoint = (
        Path(os.environ.get("FRANKA_RL_DATA_ROOT", "/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data"))
        / "runs/logs/rsl_rl/franka_reach"
        / checkpoint_name
    )
    report["checkpoint"] = dict(path=str(checkpoint), sha256=sha256(checkpoint))
    report["training_source_revision"] = "d0bfa66331a4813bcff0a0b1d24762618603d012"
    report["task_source_diff_from_training"] = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO),
            "diff",
            report["training_source_revision"],
            "--",
            "source/franka_rl/franka_rl/tasks",
            "source/franka_rl/franka_rl/utils/scenarios.py",
        ],
        text=True,
    )
    report["saved_config_seed"] = saved.get("seed")
    report["saved_config_num_envs"] = saved["scene"]["num_envs"]
    report["runtime_scope"] = "Physics initialization and resets only. Policy and action-delay wrapper not executed."
    shutil.copyfile(saved_path, output / "saved_training_env.yaml")
    write(output / "runtime_model.json", report)
    with launch_simulation(cfg, args):
        effective_cfg = yaml.dump(cfg.to_dict())
        (output / "launched_env.yaml").write_text(effective_cfg)
        write(
            output / "launched_config_differences.json",
            differences(saved, yaml.load(effective_cfg, Loader=TrainingLoader)),
        )
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        report["physics_extensions"] = {
            name: manager.get_enabled_extension_id(name) for name in ["omni.physx", "omni.physics.tensors", "omni.usd"]
        }

        env = gym.make(args.task, cfg=cfg).unwrapped
        try:
            runtime_cfg = yaml.dump(env.cfg.to_dict())
            (output / "runtime_env.yaml").write_text(runtime_cfg)
            write(
                output / "runtime_config_differences.json",
                differences(saved, yaml.load(runtime_cfg, Loader=TrainingLoader)),
            )
            physics_cfg = env.sim.cfg.physics
            report["resolved_physics_configuration"] = yaml.load(
                yaml.dump(physics_cfg.to_dict()), Loader=TrainingLoader
            )
            report["physics_configuration_differences"] = differences(
                saved["sim"]["physics"], report["resolved_physics_configuration"]
            )
            report["snapshots"].append(snapshot(env, "initialized_after_physics_and_startup_before_reset"))
            report["usd"] = usd_evidence(
                env, cfg.scene.robot.spawn.usd_path, json.loads((evidence / "report.json").read_text())
            )
            write(output / "runtime_model.json", report)
            for i in range(2):
                env.reset(seed=args.seed if i == 0 else None)
                report["snapshots"].append(snapshot(env, f"after_reset_{i + 1}"))
                write(output / "runtime_model.json", report)
            report["complete"] = True
            write(output / "runtime_model.json", report)
            print(f"EXTRACTION_COMPLETE: {output}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
