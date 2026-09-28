#!/usr/bin/env python3
"""Train the reduced Axis-2 reward study, optionally reusing the Axis-1 control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_axis1_seed_training import SeedTrainingCoordinator, load_matrix


def _import_axis1_control(matrix, manifest_path: Path) -> None:
    """Reuse the three position-increment checkpoints as the full-reward control."""
    manifest = json.loads(manifest_path.expanduser().resolve().read_text())
    source = manifest.get("policies", {}).get("position_increment", {})
    checkpoints = source.get("checkpoints", {})
    destination = next((policy for policy in matrix.policies if policy.name == "full_reward"), None)
    if destination is None:
        raise ValueError("Axis-2 matrix has no full_reward policy")
    for seed in matrix.target_seeds:
        entry = checkpoints.get(str(seed))
        if not entry or not entry.get("complete"):
            raise ValueError(f"Axis-1 position_increment seed {seed} is incomplete")
        checkpoint = Path(entry["path"]).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        destination.existing_checkpoints[seed] = checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--axis1-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--policy", action="append", dest="policies", help="Run only this reward variant")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    matrix = load_matrix(args.config)
    if args.axis1_manifest is not None:
        _import_axis1_control(matrix, args.axis1_manifest)
    selected = set(args.policies) if args.policies else None
    known = {policy.name for policy in matrix.policies}
    if selected and not selected <= known:
        parser.error(f"unknown policies: {sorted(selected - known)}")
    runner = SeedTrainingCoordinator(matrix, args.output_dir, args.dry_run, args.fail_fast, selected)
    return 0 if runner.run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
