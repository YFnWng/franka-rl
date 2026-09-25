"""Check FR3 runtime mass properties, flange FK, and reset-time payload composition."""

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import franka_rl.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
from franka_rl.utils.scenarios import ScenarioCatalog, ScenarioModifier
from scipy.spatial.transform import Rotation

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

REPO = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--scenario", default="fr3_payload_dr")
add_launcher_args(parser)
args, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining


def main():
    cfg, _ = resolve_task_config("Franka-FR3v2-Reach-v0", "")
    cfg.scene.num_envs = 4
    cfg.seed = 123
    cfg.commands.ee_pose.debug_vis = False
    catalog = ScenarioCatalog.from_yaml(REPO / "source/franka_rl/franka_rl/config/fr3_scenarios.yaml")
    modifier = ScenarioModifier(catalog.get(args.scenario), catalog)
    modifier.apply(cfg)
    asset = Path(cfg.scene.robot.spawn.usd_path)
    reference = json.loads((asset.parent / "manifest.json").read_text())
    completed = False
    with launch_simulation(cfg, args):
        env = gym.make("Franka-FR3v2-Reach-v0", cfg=cfg).unwrapped
        try:
            robot = env.scene["robot"]
            assert robot.joint_names == [f"panda_joint{i}" for i in range(1, 8)]
            assert robot.num_bodies == 8
            view = robot.data._root_view
            masses = view.get_masses().numpy().copy()
            coms = view.get_coms().numpy().copy()
            inertias = view.get_inertias().numpy().reshape(4, 8, 3, 3).copy()
            rows = {r["body"]: r for r in reference["bodies"]}
            for i, name in enumerate(robot.body_names):
                r = rows[name]
                np.testing.assert_allclose(masses[:, i], r["mass_kg"], rtol=1e-5)
                np.testing.assert_allclose(coms[0, i, :3], r["com_body_m"], atol=1e-7)
                np.testing.assert_allclose(inertias[0, i], r["inertia_body_kg_m2"], atol=1e-7)
            source = asset.parent / "source.urdf"
            if not source.exists():
                source = REPO / "deployment/model_audit/2026-09-24/fr3v2_no_hand_fake.urdf"
            assert hashlib.sha256(source.read_bytes()).hexdigest() == reference["urdf_sha256"]
            urdf = ET.parse(source).getroot()
            prefix = reference.get("source_joint_prefix", "fr3v2_")
            # Independently verify manifest mass properties against source URDF.
            for i, name in enumerate([f"fr3_link{j}" for j in range(7)] + ["fr3_flange"]):
                inertial = urdf.find(f"link[@name='{prefix}link{i}']/inertial")
                np.testing.assert_allclose(rows[name]["mass_kg"], float(inertial.find("mass").get("value")))
                com = np.fromstring(inertial.find("origin").get("xyz"), sep=" ")
                if i == 7:
                    com -= [0, 0, 0.107]
                np.testing.assert_allclose(rows[name]["com_body_m"], com, atol=1e-9)
                v = inertial.find("inertia").attrib
                tensor = np.array([[float(v[k]) for k in row] for row in
                                   [("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz")]])
                rotation = Rotation.from_euler("xyz", np.fromstring(inertial.find("origin").get("rpy"), sep=" ")).as_matrix()
                np.testing.assert_allclose(rows[name]["inertia_body_kg_m2"], rotation @ tensor @ rotation.T, atol=1e-9)
            pose = np.eye(4)
            q = robot.data.joint_pos.torch[0].cpu().numpy()
            expected_limits = []
            expected_vel = []
            for i in range(1, 8):
                joint = urdf.find(f"joint[@name='{prefix}joint{i}']")
                origin = joint.find("origin")
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
                transform[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
                turn = np.eye(4)
                turn[:3, :3] = Rotation.from_rotvec([0, 0, q[i - 1]]).as_matrix()
                pose = pose @ transform @ turn
                limits = joint.find("limit")
                expected_limits.append([float(limits.get("lower")), float(limits.get("upper"))])
                expected_vel.append(float(limits.get("velocity")))
            expected = pose[:3, 3] + pose[:3, :3] @ np.array([0, 0, 0.107])
            flange = robot.body_names.index("fr3_flange")
            observed = (robot.data.body_pos_w.torch[0, flange] - robot.data.root_pos_w.torch[0]).cpu().numpy()
            np.testing.assert_allclose(observed, expected, atol=1e-6)
            np.testing.assert_allclose(view.get_dof_limits().numpy()[0], expected_limits, atol=1e-6)
            np.testing.assert_allclose(view.get_dof_max_velocities().numpy()[0], expected_vel, rtol=1e-5)
            samples = []
            for reset in range(2):
                env.reset()
                m = view.get_masses().numpy().copy()
                c = view.get_coms().numpy().copy()
                inertia = view.get_inertias().numpy().reshape(4, 8, 3, 3).copy()
                payload = m[:, flange] - masses[:, flange]
                pos = (m[:, flange, None] * c[:, flange, :3] - masses[:, flange, None] * coms[:, flange, :3]) / payload[
                    :, None
                ]
                assert np.all(payload >= 0) and np.all(payload <= 1.000001)
                np.testing.assert_allclose(pos[:, :2], 0, atol=1e-6)
                assert np.all(pos[:, 2] >= -1e-6) and np.all(pos[:, 2] <= 0.050001)
                for i in range(4):

                    def pa(mass, delta):
                        return mass * ((delta @ delta) * np.eye(3) - np.outer(delta, delta))

                    combined = (
                        inertias[i, flange]
                        + pa(masses[i, flange], coms[i, flange, :3] - c[i, flange, :3])
                        + pa(payload[i], pos[i] - c[i, flange, :3])
                    )
                    np.testing.assert_allclose(inertia[i, flange], combined, atol=1e-7)
                samples.append(dict(payload_kg=payload.tolist(), payload_position_flange_m=pos.tolist()))
            result = dict(
                passed=True,
                body_names=robot.body_names,
                joint_names=robot.joint_names,
                max_fk_error_m=float(np.max(np.abs(observed - expected))),
                reset_samples=samples,
                asset_manifest=reference,
            )
            if args.output.exists():
                raise FileExistsError(args.output)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("FR3_MODEL_VALIDATION_PASSED", flush=True)
        finally:
            env.close()
        completed = True
    if not completed:
        raise RuntimeError("FR3 validation failed; see simulator traceback.")


if __name__ == "__main__":
    main()
