#!/usr/bin/env python3
"""Analyze one response-identification CSV without modifying it."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty trace")
    required = {
        "session_id", "rt_sequence", "state_sequence", "observation_sequence", "robot_time_s", "host_state_received_ns",
        "period_ns", "dropped_samples", "command_valid",
    }
    for joint in range(1, 8):
        required |= {f"q_ref_j{joint}", f"dq_ref_j{joint}", f"q_j{joint}", f"dq_j{joint}"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return rows


def vector(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def robust_clock_fit(robot_s, host_ns):
    keep = np.ones(robot_s.size, dtype=bool)
    for _ in range(4):
        robot_center = np.mean(robot_s[keep])
        host_center = np.mean(host_ns[keep])
        centered_robot = robot_s[keep] - robot_center
        denominator = np.dot(centered_robot, centered_robot)
        if denominator == 0:
            raise ValueError("robot clock did not advance")
        slope = np.dot(centered_robot, host_ns[keep] - host_center) / denominator
        offset = host_center - slope * robot_center
        residual = host_ns - (slope * robot_s + offset)
        median = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - median))
        if mad == 0:
            break
        keep = np.abs(residual - median) <= 6.0 * 1.4826 * mad
    return slope, offset, residual, keep


def lag_and_errors(reference, measured, dt_s, max_lag_s=0.1):
    x = reference - np.mean(reference)
    y = measured - np.mean(measured)
    max_lag = min(int(max_lag_s / dt_s), len(x) // 4)
    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.asarray([
        np.dot(x[max(0, -lag):min(len(x), len(x) - lag)],
               y[max(0, lag):min(len(y), len(y) + lag)])
        for lag in lags
    ])
    lag = int(lags[int(np.argmax(scores))])
    error = measured - reference
    return lag * dt_s, float(np.sqrt(np.mean(error ** 2))), float(np.max(np.abs(error)))


def sine_fit(signal, time_s, frequency_hz):
    omega = 2.0 * np.pi * frequency_hz
    design = np.column_stack((np.sin(omega * time_s), np.cos(omega * time_s), np.ones_like(time_s)))
    sin_c, cos_c, offset = np.linalg.lstsq(design, signal, rcond=None)[0]
    return float(np.hypot(sin_c, cos_c)), float(np.arctan2(cos_c, sin_c)), float(offset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--frequency-hz", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    rows = read_csv(args.trace)
    sessions = {row["session_id"] for row in rows}
    if len(sessions) != 1:
        raise ValueError("analyze one session per file")
    if any(int(row["dropped_samples"]) != 0 for row in rows):
        raise ValueError("trace has dropped RT samples")
    if any(int(row["command_valid"]) != 1 for row in rows):
        raise ValueError("trace contains invalid commands; inspect as faulted trial")

    robot_s = vector(rows, "robot_time_s")
    host_ns = vector(rows, "host_state_received_ns")
    slope, offset, residual, keep = robust_clock_fit(robot_s, host_ns)
    dt_s = float(np.median(vector(rows, "period_ns"))) * 1e-9
    result = {
        "schema_version": 1,
        "trace": str(args.trace.resolve()),
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "sample_count": len(rows),
        "session_id": next(iter(sessions)),
        "median_period_s": dt_s,
        "clock_fit": {
            "host_ns_per_robot_s": float(slope),
            "offset_ns": float(offset),
            "inliers": int(np.count_nonzero(keep)),
            "residual_rms_ns": float(np.sqrt(np.mean(residual[keep] ** 2))),
            "residual_max_abs_ns": float(np.max(np.abs(residual[keep]))),
        },
        "joints": {},
        "identifiability": "effective response only; firmware gains are not recovered",
    }
    relative_time = robot_s - robot_s[0]
    for joint in range(1, 8):
        q_ref = vector(rows, f"q_ref_j{joint}")
        q = vector(rows, f"q_j{joint}")
        lag_s, rms_rad, peak_rad = lag_and_errors(q_ref, q, dt_s)
        metrics = {"tracking_lag_s": lag_s, "position_rms_rad": rms_rad,
                   "position_peak_abs_rad": peak_rad}
        if args.frequency_hz:
            in_amp, in_phase, _ = sine_fit(q_ref, relative_time, args.frequency_hz)
            out_amp, out_phase, _ = sine_fit(q, relative_time, args.frequency_hz)
            metrics.update({
                "frequency_hz": args.frequency_hz,
                "gain": out_amp / in_amp if in_amp > 1e-9 else None,
                "phase_lag_rad": float(np.angle(np.exp(1j * (in_phase - out_phase)))),
            })
        result["joints"][str(joint)] = metrics

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
