#!/usr/bin/env python3
"""Build a portable governor release; never imports Isaac/ROS or contacts a robot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / 'deployment/reference_governor'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == REPO or REPO in out.parents:
        raise ValueError('release artifacts must be outside the source repository')
    out.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, '-I', '-m', 'build', '--no-isolation', '--outdir', str(out), str(PACKAGE)], check=True)
    integration = out / 'repo_integration'
    for relative in [
        'source/franka_rl/franka_rl/tasks/manager_based/franka_rl/fr3_governed_env_cfg.py',
        'source/franka_rl/franka_rl/tasks/manager_based/franka_rl/__init__.py',
        'deployment/REFERENCE_GOVERNOR_PLAN.md',
    ]:
        dst = integration / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/relative, dst)
    (integration/'README.md').write_text(
        'Prefer updating the matching franka-rl repo revision/source snapshot. These files\n'
        'show the opt-in task integration for an existing FR3 task checkout. Review/merge\n'
        'the registration entry; do not blindly replace a workstation __init__.py with\n'
        'local changes. The wheel contains the core, Python API, configs and Isaac action\n'
        'term. The task config also needs the existing FR3 base task and corrected USD.\n')
    sources = {}
    for p in sorted(PACKAGE.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:
            sources[str(p.relative_to(REPO))] = sha(p)
    sources[str(Path(__file__).resolve().relative_to(REPO))] = sha(Path(__file__).resolve())
    manifest = {
        'schema_version': 1, 'package': 'franka-reference-governor', 'version': '0.1.0',
        'git_revision': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'git_status': subprocess.check_output(['git', '-C', str(REPO), 'status', '--short'], text=True),
        'python': sys.version, 'source_sha256': sources,
        'artifact_sha256': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        'isaac_sim_target': '6.0.1', 'isaac_lab_target': 'v3.0.0-beta2.patch1',
        'isaac_runtime_tested': False, 'hardware_accepted': False,
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(
        f'{sha(p)}  {p.relative_to(out)}\n' for p in sorted(out.rglob('*'))
        if p.is_file() and p.name != 'SHA256SUMS'))
    print(f'Release: {out}\nManifest SHA-256: {sha(out/"manifest.json")}')


if __name__ == '__main__':
    main()
