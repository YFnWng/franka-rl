#!/usr/bin/env python3
"""Verified ONNX worker using newline-delimited local stdio IPC."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

def emit(value):
    sys.stdout.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    root = Path(args.bundle).resolve(strict=True)
    sys.path.insert(0, str(root))
    from franka_policy_bundle.bundle import session, verify
    report = verify(root, args.manifest_sha256)
    model = session(root / "policy.onnx")
    import numpy as np
    emit({"type": "ready", "manifest_sha256": report["manifest_sha256"], "provider": report["provider"]})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            observation = np.asarray(request["observation"], dtype=np.float32).reshape(1, 24)
            if not np.isfinite(observation).all(): raise ValueError("nonfinite observation")
            action = model.run(["action"], {"observation": observation})[0].reshape(7)
            if not np.isfinite(action).all(): raise ValueError("nonfinite action")
            emit({"type": "result", "request_id": request["request_id"],
                  "trial_id": request["trial_id"],
                  "observation_sequence": request["observation_sequence"],
                  "observation_monotonic_ns": request["observation_monotonic_ns"],
                  "completed_monotonic_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                  "action": [float(v) for v in action]})
        except Exception as exc:
            emit({"type": "error", "error": str(exc)})
            return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
