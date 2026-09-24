"""Validate a transferred handoff without importing or activating hardware."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def contained(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes handoff: {relative}")
    return path


def preflight(root, expected_hash=None):
    root = Path(root).resolve(strict=True)
    manifest_path = root / "handoff_manifest.json"
    if expected_hash and digest(manifest_path) != expected_hash:
        raise ValueError("Handoff manifest trust anchor mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported handoff schema")
    for name, sha in manifest["files"].items():
        if digest(contained(root, name)) != sha:
            raise ValueError(f"Hash mismatch: {name}")
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p != manifest_path
    }
    if actual != set(manifest["files"]):
        raise ValueError("Handoff inventory differs from manifest")
    # Only import packaged code after the complete inventory has been checked.
    sys.path.insert(0, str(root / "source/deployment"))
    import yaml
    from franka_policy_bundle.bundle import verify
    from franka_policy_bundle.suite import resolve_suite

    reports = {}
    contracts = []
    for name, item in manifest["bundles"].items():
        bundle = contained(root, item["path"])
        reports[name] = verify(bundle, item["manifest_sha256"])
        contracts.append(yaml.safe_load((bundle / "policy_contract.yaml").read_text()))
    if any(c != contracts[0] for c in contracts):
        raise ValueError("Policy contracts differ; paired comparison is invalid")
    plans = {}
    for name in manifest["suites"]:
        path = contained(root, name)
        suite = yaml.safe_load(path.read_text())
        for key in ("policy_bundle", "target_set"):
            target = (path.parent / suite[key]["path"]).resolve(strict=True)
            if not target.is_relative_to(root):
                raise ValueError("Suite references data outside the handoff")
        plans[name] = resolve_suite(path)
        if suite["mode"] != "shadow":
            raise ValueError("Initial handoff only permits shadow suites")
    schedules = [plan["trials"] for plan in plans.values()]
    if not schedules or any(s != schedules[0] for s in schedules):
        raise ValueError("Paired target schedules differ")
    request = yaml.safe_load((root / "run_request.yaml").read_text())
    if request["execute_hardware"] is not False or request["suite"] not in plans:
        raise ValueError("Invalid initial run request")
    comparison = yaml.safe_load((root / "comparison.yaml").read_text())
    if comparison["suites"] != manifest["suites"] or comparison["execute_hardware"] is not False:
        raise ValueError("Invalid comparison definition")
    return dict(
        passed=True,
        manifest_sha256=digest(manifest_path),
        bundles=reports,
        suites={name: plan["trials"] for name, plan in plans.items()},
        hardware_executed=False,
        recorded_observation_parity=False,
        cpp_parity=False,
        source=manifest["source"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    print(json.dumps(preflight(args.directory, args.manifest_sha256), indent=2))
