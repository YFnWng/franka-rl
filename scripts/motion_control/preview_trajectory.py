#!/usr/bin/env python3
"""Offline M0 preview; does not launch Kit, solve IK, or authorize hardware motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from franka_motion.config import expand_trials, load_motion_suite
from franka_motion.trajectories import sample_trajectory, validate_trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, help="Defaults to the packaged simulation_smoke.yaml")
    parser.add_argument("--trial", help="Expanded trial ID; defaults to the first trial")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for small offline artifacts")
    parser.add_argument("--sample-hz", type=float, default=200.)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    suite = load_motion_suite(args.suite)
    trials = expand_trials(suite)
    matches = [t for t in trials if args.trial is None or t.trial_id == args.trial]
    if not matches:
        parser.error(f"Unknown trial. Available: {[t.trial_id for t in trials]}")
    trial = matches[0]
    trajectory = trial.trajectory.build(trial.angular_speed_rad_s, suite.preview_initial_position_m)
    workspace = suite.workspace.data
    limits = workspace["reference_limits"]
    validation = validate_trajectory(trajectory, sample_hz=args.sample_hz,
        max_speed_m_s=limits["speed_m_s"], max_acceleration_m_s2=limits["acceleration_m_s2"],
        max_jerk_m_s3=limits["jerk_m_s3"], lower_m=workspace["cartesian_bounds_m"]["lower"],
        upper_m=workspace["cartesian_bounds_m"]["upper"])
    samples = sample_trajectory(trajectory, sample_hz=args.sample_hz)
    samples["commanded_linear_velocity_m_s"] = (samples["linear_velocity_m_s"].copy()
        if trial.twist_feedforward else np.zeros_like(samples["linear_velocity_m_s"]))
    # Import before writing so missing optional plotting packages do not leave incomplete output.
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "reference.npz", **samples)
    metadata = {"schema_version": 1, "suite": suite.name, "trial_id": trial.trial_id,
        "scope": "offline reference preview; no robot FK, IK, collision or dynamics validation",
        "duration_s": trajectory.duration_s, "twist_feedforward": trial.twist_feedforward,
        "frames": workspace["frames"], "controller": trial.controller.data,
        "trajectory": trial.trajectory.data, "angular_speed_rad_s": trial.angular_speed_rad_s,
        "preview_initial_position_m": suite.preview_initial_position_m,
        "file_hashes": suite.file_hashes, "validation": validation}
    if not args.no_plot:
        fig = plt.figure(figsize=(11, 6))
        ax = fig.add_subplot(121, projection="3d")
        p = samples["position_m"]
        ax.plot(*p.T)
        ax.scatter(*p[0], label="preview start (not home FK)")
        ax.set(xlabel="base x [m]", ylabel="base y [m]", zlabel="base z [m]", title=trial.trial_id)
        ax.set_box_aspect(np.ptp(p, axis=0).clip(min=.05))
        ax.legend(fontsize=8)
        for index, (field, units) in enumerate((("linear_velocity_m_s", "m/s"),
                ("acceleration_m_s2", "m/s²"), ("jerk_m_s3", "m/s³"))):
            axis = fig.add_subplot(3, 2, 2 + 2 * index)
            axis.plot(samples["time_s"], np.linalg.norm(samples[field], axis=1))
            axis.set(ylabel=units, xlabel="time [s]")
        fig.tight_layout()
        fig.savefig(output / "reference.png", dpi=140)
        plt.close(fig)
    (output / "validation.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Validated {trial.trial_id}: {trajectory.duration_s:.3f} s; {validation['sample_count']} samples")
    print(f"Artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
