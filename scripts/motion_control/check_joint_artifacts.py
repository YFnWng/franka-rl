"""Offline M1 finalization/hash, clone/repeat and optional single/batched parity checks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def validate_trial(directory):
    directory = Path(directory)
    summary_path = directory/"summary.json"
    summary = json.loads(summary_path.read_text())
    marker = json.loads((directory/"FINALIZED.json").read_text())
    if marker["summary_sha256"] != hashlib.sha256(summary_path.read_bytes()).hexdigest():
        raise ValueError("Summary hash mismatch")
    if summary["trace_sha256"] != hashlib.sha256((directory/"trace.npz").read_bytes()).hexdigest():
        raise ValueError("Trace hash mismatch")
    if marker["status"] != summary["status"] or not summary["acceptance_passed"]:
        raise ValueError("Trial failed acceptance or finalization status mismatch")
    with np.load(directory/"trace.npz", allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    count = summary["physics_samples"]
    if trace["q"].shape != (count, summary["num_envs"], 7) or trace["dq"].shape != trace["q"].shape:
        raise ValueError("Telemetry shape mismatch")
    for key in ("q", "dq", "q_ref", "dq_ref", "applied", "terminal_q", "terminal_dq"):
        if not np.isfinite(trace[key]).all():
            raise ValueError(f"Nonfinite accepted telemetry: {key}")
    np.testing.assert_allclose(trace["time_s"], np.arange(count)*.001, rtol=0, atol=1e-12)
    if trace["fault_code"].any() or np.any(trace["first_fault_tick"] != -1):
        raise ValueError("Accepted trace contains a fault")
    clone_error = max(float(np.max(np.abs(trace[key]-trace[key][:, :1]))) for key in ("q", "dq"))
    if clone_error > 1e-6:
        raise ValueError(f"Clone parity exceeds 1e-6: {clone_error}")
    return summary, trace, clone_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--compare-suite", type=Path, help="Matched single/batched suite with identical trial IDs")
    args = parser.parse_args()
    index = json.loads((args.suite_dir/"index.json").read_text())
    if not index["all_accepted"]:
        raise ValueError("Suite did not pass acceptance")
    results, traces = [], {}
    for row in index["results"]:
        name = row["trial_id"]
        summary, trace, clone_error = validate_trial(args.suite_dir/name)
        result = {"trial_id": name, "passed": True, "clone_max_difference": clone_error}
        # IDs keep cell identity separate from repeat identity.
        cell = name.rsplit("__repeat_", 1)[0]
        if cell in traces:
            repeat_error = max(float(np.max(np.abs(trace[key]-traces[cell][key]))) for key in ("q", "dq"))
            if repeat_error > 1e-6:
                raise ValueError(f"Repeat parity failed: {repeat_error}")
            result["repeat_max_difference"] = repeat_error
        traces[cell] = trace
        if args.compare_suite:
            _, other, _ = validate_trial(args.compare_suite/name)
            parity = max(float(np.max(np.abs(trace[key][:, 0]-other[key][:, 0]))) for key in ("q", "dq"))
            if parity > 1e-6:
                raise ValueError(f"Single/batched parity failed: {parity}")
            result["single_batched_max_difference"] = parity
        results.append(result)
    reset = args.suite_dir/"reset_parity.json"
    print(json.dumps({"passed": True, "results": results,
        "reset_parity": json.loads(reset.read_text()) if reset.exists() else "not requested"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
