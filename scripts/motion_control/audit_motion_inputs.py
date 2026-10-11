#!/usr/bin/env python3
"""Read-only M0 model/source audit. Archived evidence is not a fresh simulator run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from franka_motion.config import load_motion_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("FRANKA_RL_DATA_ROOT",
        "/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data")))
    parser.add_argument("--output", type=Path, help="Optional new JSON evidence file; refuses overwrite")
    args = parser.parse_args()
    suite = load_motion_suite(args.suite)
    model = suite.workspace.data["model"]
    asset = args.data_root / model["asset_relative_to_data_root"]
    urdf = args.repo / model["archived_urdf"]
    checks, errors = {}, []
    for name, path in (("usd", asset), ("urdf", urdf)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        checks[name] = {"path": str(path), "sha256": digest, "matches": digest == model[f"{name}_sha256"]}
        if not checks[name]["matches"]:
            errors.append(f"Missing or hash-mismatched {name}: {path}")
    if urdf.is_file():
        joints = [joint for joint in ET.parse(urdf).getroot().findall("joint") if joint.attrib["type"] == "revolute"]
        actual = {key: [float(joint.find("limit").attrib[key]) for joint in joints] for key in ("lower", "upper")}
        checks["urdf_joint_bounds"] = {"joint_names": [j.attrib["name"] for j in joints], "bounds": actual,
            "matches": len(joints) == 7 and all(np.allclose(actual[key], suite.workspace.data["joint_bounds_rad"]["model"][key],
                rtol=0., atol=1e-10) for key in actual)}
        if not checks["urdf_joint_bounds"]["matches"]:
            errors.append("YAML joint limits differ from archived robot URDF")
    revision = subprocess.run(["git", "-C", str(args.repo.parent / "IsaacLab"), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False)
    checks["isaaclab_revision"] = {"actual": revision.stdout.strip(), "expected": model["isaaclab_revision"],
        "matches": revision.returncode == 0 and revision.stdout.strip() == model["isaaclab_revision"]}
    if not checks["isaaclab_revision"]["matches"]:
        errors.append("Isaac Lab revision differs from M0 model provenance pin")
    archive = asset.parent / "runtime_validation.json"
    archived = json.loads(archive.read_text()) if archive.is_file() else None
    checks["archived_runtime_validation"] = {"path": str(archive), "present": archived is not None,
        "passed": archived.get("passed") if archived else None, "fresh_runtime_test": False}
    if archived is None or archived.get("passed") is not True:
        errors.append("Archived runtime model validation missing or unsuccessful")
    versions = {}
    for package in ("numpy", "PyYAML", "isaacsim", "isaaclab", "franky-control", "franka-motion"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {"schema_version": 1, "local_model_checks_passed": not errors, "checks": checks,
        "versions_on_simulation_workstation": versions, "errors": errors,
        "hardware_cartesian_controller_audit": "pending; no current pinned Franky source/RT installation probe available",
        "hardware_execution_authorized": False, "suite_file_hashes": suite.file_hashes}
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.expanduser().open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
