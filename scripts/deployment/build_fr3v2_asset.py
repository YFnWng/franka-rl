"""Build a bare-flange FR3v2 USD from audited URDF inertials and pinned collision meshes.

The final physical link is re-expressed at the flange origin (link7 + 0.107 m Z).
This removes massless fixed bodies without adding a fictitious gripper/payload.
Joint names remain panda_joint1..7 as policy-interface aliases.
"""

import argparse
import hashlib
import json
import os
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REVISION = "1ccde30d5a30a710f335c9f6545528447a04bc7c"
REPO = Path(__file__).resolve().parents[2]


def xyz(text):
    return np.fromstring(text or "0 0 0", sep=" ")


def quat(matrix):
    from pxr import Gf

    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("FRANKA_RL_DATA_ROOT", "/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data"))
        / "generated_assets/fr3v2_bare_flange_v1",
    )
    args = parser.parse_args()
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.output_dir.parent.is_dir():
        raise FileNotFoundError("Create the generated_assets directory on the mounted data volume first")
    out = args.output_dir
    out.mkdir()
    urdf = REPO / "deployment/model_audit/2026-09-24/fr3v2_no_hand_fake.urdf"
    robot = ET.parse(urdf).getroot()
    meshes = out / "meshes"
    meshes.mkdir()
    hashes = {}
    for i in range(8):
        relative = f"meshes/robot_arms/fr3v2/collision/link{i}.stl"
        url = f"https://raw.githubusercontent.com/frankarobotics/franka_description/{REVISION}/{relative}"
        raw = urllib.request.urlopen(
            url, timeout=60, context=ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")
        ).read()
        (meshes / f"link{i}.stl").write_bytes(raw)
        hashes[relative] = hashlib.sha256(raw).hexdigest()
    stage = Usd.Stage.CreateNew(str(out / "fr3v2.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    ar = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    ar.CreateEnabledSelfCollisionsAttr(True)
    poses = [np.eye(4)]
    joints = [robot.find(f"joint[@name='fr3v2_joint{i}']") for i in range(1, 8)]
    for joint in joints:
        origin = joint.find("origin")
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", xyz(origin.get("rpy"))).as_matrix()
        pose[:3, 3] = xyz(origin.get("xyz"))
        poses.append(poses[-1] @ pose)
    shift = np.array([0.0, 0.0, 0.107])
    names = [f"fr3_link{i}" for i in range(7)] + ["fr3_flange"]
    reference = []
    for i, name in enumerate(names):
        link = robot.find(f"link[@name='fr3v2_link{i}']")
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value"))
        origin = inertial.find("origin")
        rotation = Rotation.from_euler("xyz", xyz(origin.get("rpy"))).as_matrix()
        v = inertial.find("inertia").attrib
        inertia = np.array(
            [
                [float(v[k]) for k in row]
                for row in [("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz")]
            ]
        )
        inertia = rotation @ inertia @ rotation.T
        com = xyz(origin.get("xyz")) - (shift if i == 7 else 0)
        values, axes = np.linalg.eigh(inertia)
        assert np.all(values > 0)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        body = UsdGeom.Xform.Define(stage, f"/Robot/{name}")
        position = poses[i][:3, 3] + (poses[i][:3, :3] @ shift if i == 7 else 0)
        body.AddTranslateOp().Set(Gf.Vec3d(*position))
        body.AddOrientOp().Set(quat(poses[i][:3, :3]))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        m = UsdPhysics.MassAPI.Apply(body.GetPrim())
        m.CreateMassAttr(mass)
        m.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
        m.CreatePrincipalAxesAttr(quat(axes))
        mesh = trimesh.load(str(meshes / f"link{i}.stl"), force="mesh")
        collision_origin = link.find("collision/origin")
        vertices = np.asarray(mesh.vertices)
        if collision_origin is not None:
            rot = Rotation.from_euler("xyz", xyz(collision_origin.get("rpy"))).as_matrix()
            vertices = vertices @ rot.T + xyz(collision_origin.get("xyz"))
        if i == 7:
            vertices = vertices - shift
        usdmesh = UsdGeom.Mesh.Define(stage, f"/Robot/{name}/collision")
        usdmesh.CreatePointsAttr(vertices.astype(np.float32).tolist())
        usdmesh.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
        usdmesh.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces).ravel().tolist())
        usdmesh.CreateSubdivisionSchemeAttr("none")
        usdmesh.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.78, 0.82)])
        UsdPhysics.CollisionAPI.Apply(usdmesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(usdmesh.GetPrim()).CreateApproximationAttr("convexHull")
        reference.append(dict(body=name, mass_kg=mass, com_body_m=com.tolist(), inertia_body_kg_m2=inertia.tolist()))
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Robot/fixed_base")
    fixed.CreateBody1Rel().SetTargets(["/Robot/fr3_link0"])
    for i, joint in enumerate(joints, 1):
        origin = joint.find("origin")
        usdjoint = UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/panda_joint{i}")
        usdjoint.CreateBody0Rel().SetTargets([f"/Robot/{names[i - 1]}"])
        usdjoint.CreateBody1Rel().SetTargets([f"/Robot/{names[i]}"])
        usdjoint.CreateAxisAttr("Z")
        assert np.allclose(xyz(joint.find("axis").get("xyz")), [0, 0, 1])
        usdjoint.CreateLocalPos0Attr(Gf.Vec3f(*xyz(origin.get("xyz"))))
        usdjoint.CreateLocalRot0Attr(quat(Rotation.from_euler("xyz", xyz(origin.get("rpy"))).as_matrix()))
        usdjoint.CreateLocalPos1Attr(Gf.Vec3f(*(-shift if i == 7 else np.zeros(3))))
        usdjoint.CreateLocalRot1Attr(Gf.Quatf(1))
        limits = joint.find("limit").attrib
        usdjoint.CreateLowerLimitAttr(float(np.degrees(float(limits["lower"]))))
        usdjoint.CreateUpperLimitAttr(float(np.degrees(float(limits["upper"]))))
        drive = UsdPhysics.DriveAPI.Apply(usdjoint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateMaxForceAttr(float(limits["effort"]))
        PhysxSchema.PhysxJointAPI.Apply(usdjoint.GetPrim()).CreateMaxJointVelocityAttr(
            float(np.degrees(float(limits["velocity"])))
        )
    stage.GetRootLayer().Save()
    manifest = dict(
        schema_version=1,
        model="fr3v2_bare_flange",
        description_revision=REVISION,
        urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        mesh_sha256=hashes,
        usd_sha256=hashlib.sha256((out / "fr3v2.usda").read_bytes()).hexdigest(),
        bodies=reference,
        joint_names=[f"panda_joint{i}" for i in range(1, 8)],
        flange_offset_from_link7_m=shift.tolist(),
        limitations=[
            "Collision geometry uses convex hulls of pinned URDF collision meshes; visuals reuse those meshes.",
            "Controller gains, friction, armature and governor are simulation assumptions, not hardware identification.",
        ],
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out / "fr3v2.usda")
    launcher.app.close()


if __name__ == "__main__":
    main()
