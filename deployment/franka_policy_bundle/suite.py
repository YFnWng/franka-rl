"""Validate and expand local YAML suites; never activate hardware."""

import argparse
import json
import random
from pathlib import Path

import jsonschema
import numpy as np
import yaml

from .bundle import sha256, verify
from .contract import require


def resolve_suite(path):
    path = Path(path).resolve(strict=True)
    schemas = Path(__file__).parents[1] / "schemas"
    suite = yaml.safe_load(path.read_text())
    jsonschema.validate(suite, json.loads((schemas / "hardware_suite.schema.json").read_text()))
    bundle = (path.parent / suite["policy_bundle"]["path"]).resolve(strict=True)
    targets_path = (path.parent / suite["target_set"]["path"]).resolve(strict=True)
    verify(bundle, suite["policy_bundle"]["sha256"])
    require(sha256(targets_path) == suite["target_set"]["sha256"], "Target file hash mismatch")
    targets = yaml.safe_load(targets_path.read_text())
    jsonschema.validate(targets, json.loads((schemas / "target_set.schema.json").read_text()))
    c = yaml.safe_load((bundle / "policy_contract.yaml").read_text())
    ids = [t["id"] for t in targets["targets"]]
    require(len(set(ids)) == len(ids), "Duplicate target IDs")
    bounds = np.asarray([c["target_workspace_m"][a] for a in "xyz"])
    for t in targets["targets"]:
        xyz = np.asarray(t["position_m"])
        require(
            np.isfinite(xyz).all() and ((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1])).all(),
            f"Target outside policy workspace: {t['id']}",
        )
    e = suite["execution"]
    require(
        suite["mode"] != "shadow" or not e["move_to_start_before_each_trial"],
        "Shadow suites cannot request start-pose motion",
    )
    require(
        np.isfinite(e["trial_timeout_s"])
        and np.isfinite(e["inter_trial_hold_s"])
        and np.isfinite(suite["success"]["position_threshold_m"]),
        "Nonfinite suite settings",
    )
    require(
        e["trial_timeout_s"] >= c["policy_period_s"] * suite["success"]["consecutive_policy_steps"],
        "Timeout shorter than success dwell",
    )
    trials = [
        dict(repetition=r, target_id=t["id"], position_m=t["position_m"])
        for r in range(e["repetitions"])
        for t in targets["targets"]
    ]
    if e["order"] == "seeded_shuffle":
        random.Random(e["seed"]).shuffle(trials)
    return dict(
        schema_version=1,
        suite=suite,
        suite_sha256=sha256(path),
        target_set=targets,
        trials=trials,
        hardware_authorized=False,
    )


def main():
    p = argparse.ArgumentParser(description="Validate a local hardware suite without moving hardware")
    p.add_argument("suite")
    a = p.parse_args()
    print(json.dumps(resolve_suite(a.suite), indent=2, allow_nan=False))
