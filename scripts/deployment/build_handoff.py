"""Build a portable, checksummed handoff from immutable policy bundles."""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deployment"))
from franka_policy_bundle.bundle import git_state, sha256, verify, write_json
from preflight_handoff import preflight


def build(config_path, data_root):
    config = yaml.safe_load(Path(config_path).read_text())
    data_root = Path(data_root).resolve(strict=True)
    if not data_root.is_dir() or not os.access(data_root, os.W_OK):
        raise ValueError("Data root must exist and be writable")
    if str(data_root).startswith("/media/") and not any(
        os.path.ismount(p) for p in [data_root, *data_root.parents] if str(p) not in {"/", "/media"}
    ):
        raise ValueError("Data volume is not mounted")
    name = config["handoff_id"]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Invalid handoff ID")
    parent = data_root / "deployment_handoffs"
    parent.mkdir(exist_ok=True)
    output = parent / name
    archive = parent / f"{name}.tar.gz"
    if output.exists() or archive.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if shutil.disk_usage(parent).free < 512 * 1024**2:
        raise ValueError("Less than 512 MiB free on data volume")
    root = Path(tempfile.mkdtemp(prefix=f".{name}.staging-", dir=parent))
    # Preserve staging on failure for inspection.
    print(f"Staging: {root}", flush=True)
    bundles = {}
    for label, item in config["bundles"].items():
        src = data_root / "deployment_bundles" / item["directory"]
        report = verify(src, item["manifest_sha256"])
        dst = root / "bundles" / label
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        bundles[label] = dict(path=f"bundles/{label}", manifest_sha256=report["manifest_sha256"])
    # Capture tracked files plus nonignored source additions; no Git mutation.
    files = (
        subprocess.check_output(
            ["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
        )
        .decode()
        .split("\0")
    )
    for relative in sorted(set(filter(None, files))):
        src = REPO / relative
        if not src.exists():
            continue
        if src.is_symlink() or src.stat().st_size > 10 * 1024**2:
            raise ValueError(f"Unexpected source snapshot entry: {relative}")
        dst = root / "source" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    subprocess.run(
        ["git", "-C", str(REPO), "archive", "--format=tar", f"--output={root / 'source_base_commit.tar'}", "HEAD"],
        check=True,
    )
    shutil.copyfile(REPO / "scripts/deployment/preflight_handoff.py", root / "preflight.py")
    shutil.copyfile(REPO / "deployment/requirements.txt", root / "requirements.txt")
    for name in ["REAL_ROBOT_DEPLOYMENT_PLAN.md", "deployment/HANDOFF_AGENT.md"]:
        shutil.copyfile(REPO / name, root / ("AGENTS.md" if name.endswith("HANDOFF_AGENT.md") else Path(name).name))
    (root / "target_sets").mkdir()
    target = root / "target_sets/shadow_smoke.yaml"
    shutil.copyfile(REPO / "deployment/target_sets/shadow_smoke.yaml", target)
    template = yaml.safe_load((REPO / "deployment/suites/shadow_smoke.template.yaml").read_text())
    (root / "suites").mkdir()
    suites = []
    for label, bundle in bundles.items():
        suite = dict(template)
        suite["name"] = f"{label}_shadow_smoke"
        suite["policy_bundle"] = dict(path=f"../{bundle['path']}", sha256=bundle["manifest_sha256"])
        suite["target_set"] = dict(path="../target_sets/shadow_smoke.yaml", sha256=sha256(target))
        suite["artifacts"] = dict(root="../hardware_runs")
        relative = f"suites/{label}_shadow_smoke.yaml"
        (root / relative).write_text(yaml.safe_dump(suite, sort_keys=False))
        suites.append(relative)
    (root / "comparison.yaml").write_text(
        yaml.safe_dump(
            dict(
                schema_version=1,
                name="nominal_vs_dr_shadow",
                suites=suites,
                execute_hardware=False,
                require_identical_target_order=True,
            )
        )
    )
    (root / "run_request.yaml").write_text(
        yaml.safe_dump(dict(schema_version=1, suite=suites[0], execute_hardware=False))
    )
    (root / "README.md").write_text((REPO / "deployment/HANDOFF_README.md").read_text())
    manifest = dict(
        schema_version=1,
        handoff_id=config["handoff_id"],
        created_utc=datetime.now(timezone.utc).isoformat(),
        source=git_state(REPO),
        source_snapshot="source",
        base_revision_archive="source_base_commit.tar",
        bundles=bundles,
        suites=suites,
        files={str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()},
    )
    write_json(root / "handoff_manifest.json", manifest)
    report = preflight(root, sha256(root / "handoff_manifest.json"))
    root.rename(output)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(
            output, arcname=output.name, filter=lambda info: None if "__pycache__" in Path(info.name).parts else info
        )
    write_json(parent / f"{output.name}.preflight.json", report)
    (parent / f"{output.name}.sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n"
        f"{sha256(output / 'handoff_manifest.json')}  {output.name}/handoff_manifest.json\n"
    )
    print(f"Handoff: {output}\nArchive: {archive}\nManifest SHA-256: {report['manifest_sha256']}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "deployment/handoff.yaml"))
    parser.add_argument(
        "--data-root",
        default=os.environ.get("FRANKA_RL_DATA_ROOT", "/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data"),
    )
    args = parser.parse_args()
    build(args.config, args.data_root)
