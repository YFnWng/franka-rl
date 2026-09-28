#!/usr/bin/env python3
"""Validate and summarize a completed no-motion Franky shadow run."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("latency series is empty, negative, or non-finite")
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
    }


def summarize(run_directory: str | Path) -> dict[str, Any]:
    run = Path(run_directory)
    final = json.loads((run / "final_metadata.json").read_text())
    if final.get("execution_context") != "shadow":
        raise ValueError("artifact is not a shadow run")
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    observations = [event for event in events
                    if event.get("event") == "policy_observation_submitted"]
    updates = [event for event in events if event.get("event") in
               {"shadow_reference_updated", "shadow_velocity_updated"}]
    if not observations or not updates:
        raise ValueError("shadow run has no observation or consumed-result events")

    for event in observations:
        values = event.get("observation", [])
        position = event.get("ee_feedback_position_base_m", [])
        orientation = event.get("ee_feedback_orientation_base_xyzw", [])
        if len(values) != 29 or not all(math.isfinite(float(value)) for value in values):
            raise ValueError("shadow observation is not a finite 29-vector")
        if (len(position) != 3 or len(orientation) != 4 or
                not all(math.isfinite(float(value)) for value in position + orientation) or
                event.get("flange_position_base_m") != position or
                not event.get("ee_feedback_source") or
                event.get("ee_feedback_base_frame") != "fr3_link0" or
                event.get("ee_feedback_frame") != "fr3_flange"):
            raise ValueError("shadow EE feedback is invalid or ambiguous")
    for event in updates:
        action = event.get("raw_action", [])
        target = event.get("target", event.get("reference", []))
        if (len(action) != 6 or len(target) != 7 or
                not all(math.isfinite(float(value)) for value in action + target) or
                any(abs(float(value)) > 1.0 + 1e-6 for value in action)):
            raise ValueError("shadow result has an invalid action or reference")

    first = observations[0]
    q0 = [float(value) for value in first["measured_q_rad"]]
    reference0 = [float(value) for value in first["current_reference_rad"]]
    previous0 = [float(value) for value in first["previous_action"]]
    if len(q0) != 7 or len(reference0) != 7 or len(previous0) != 6:
        raise ValueError("shadow reset vectors have invalid dimensions")

    observed_sequences = {int(event["observation_sequence"]) for event in observations}
    consumed_sequences = {int(event["observation_sequence"]) for event in updates}
    if not consumed_sequences.issubset(observed_sequences):
        raise ValueError("a consumed result has no recorded source observation")

    inference_ms = [float(event["inference_latency_ns"]) / 1e6 for event in updates]
    state_to_reference_ms = [
        float(event["state_to_reference_consumption_ns"]) / 1e6 for event in updates
    ]
    delay_steps = [float(event["effective_delay_policy_steps"]) for event in updates]
    joint7 = reference0[6]
    ready = next((event for event in events if event.get("event") == "policy_worker_ready"), {})
    return {
        "schema_version": 1,
        "session_id": final["session_id"],
        "terminal_state": final["terminal_state"],
        "terminal_reason": final["terminal_reason"],
        "manifest_sha256": ready.get("manifest_sha256"),
        "provider": ready.get("provider"),
        "observation_count": len(observations),
        "consumed_result_count": len(updates),
        "discarded_stale_result_count": sum(
            event.get("event") == "policy_result_discarded" for event in events
        ),
        "observation_size": 29,
        "action_size": 6,
        "ee_feedback_sources": sorted({
            str(event["ee_feedback_source"]) for event in observations
        }),
        "ee_feedback_base_frame": "fr3_link0",
        "ee_feedback_frame": "fr3_flange",
        "max_encoder_fk_reported_position_error_m": max(
            float(event["encoder_fk_reported_position_error_m"]) for event in observations
        ),
        "max_encoder_fk_reported_orientation_error_rad": max(
            float(event["encoder_fk_reported_orientation_error_rad"]) for event in observations
        ),
        "encoder_fk_compute_ms": _stats([
            float(event["encoder_fk_compute_ns"]) / 1e6 for event in observations
        ]),
        "max_abs_action": max(abs(float(value))
                              for event in updates for value in event["raw_action"]),
        "initial_reference_max_abs_error_rad": max(
            abs(reference0[index] - q0[index]) for index in range(7)
        ),
        "initial_previous_action_max_abs": max(abs(value) for value in previous0),
        "held_joint7_max_drift_rad": max(
            abs(float(event["target"][6]) - joint7) for event in updates
        ),
        "inference_latency_ms": _stats(inference_ms),
        "state_to_reference_consumption_ms": _stats(state_to_reference_ms),
        "effective_delay_policy_steps": _stats(delay_steps),
        "dropped_samples": final["dropped_samples"],
        "smooth_stop_ok": final["smooth_stop_ok"],
        "runtime_source_sha256": final["runtime_source_sha256"],
        "config_sha256": final["config_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = summarize(args.run_directory)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with Path(args.output).open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
