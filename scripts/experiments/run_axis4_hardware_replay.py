#!/usr/bin/env python3
"""Replay the four packaged hardware sessions in Isaac Lab and compile parity deltas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from franka_rl.utils.hardware_replay import HardwareSessionReplay, rotation_matrix_xyz

DEFAULT_DATA_ROOT = Path("/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data")
PACKAGE_NAME = "hardware_tracking_2026-09-27_complete"
EVALUATION_SEED = 20260927
MIN_FREE_BYTES = 100 * 1024**2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty sample set")
    ordered = sorted(values)

    def quantile(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)

    return {
        "mean": sum(ordered) / len(ordered),
        "median": quantile(0.5),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "maximum": ordered[-1],
    }


def _circle_error(point: tuple[float, float, float], path: dict[str, Any]) -> float:
    center = tuple(float(value) for value in path["center_m"])
    rpy = tuple(math.radians(float(value)) for value in path["orientation_rpy_deg"])
    rotation = rotation_matrix_xyz(*rpy)
    normal = tuple(rotation[row][2] for row in range(3))
    offset = tuple(point[index] - center[index] for index in range(3))
    axial = sum(offset[index] * normal[index] for index in range(3))
    radial_sq = sum((offset[index] - axial * normal[index]) ** 2 for index in range(3))
    return math.sqrt(axial * axial + (math.sqrt(radial_sq) - float(path["radius_m"])) ** 2)


def _hardware_trace_summary(replay: HardwareSessionReplay) -> dict[str, Any]:
    q_error: list[float] = []
    speed: list[float] = []
    torque: list[float] = []
    circle_all: list[float] = []
    circle_traversal: list[float] = []
    velocity_rows: list[tuple[int, tuple[float, ...]]] = []
    with (replay.session_dir / "samples.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("command_valid", "1").strip().lower() not in ("1", "true"):
                continue
            try:
                q = tuple(float(row[f"q_j{i}"]) for i in range(1, 8))
                dq = tuple(float(row[f"dq_j{i}"]) for i in range(1, 8))
                q_ref = tuple(float(row[f"q_ref_j{i}"]) for i in range(1, 8))
                tau = tuple(float(row[f"tau_command_j{i}"]) for i in range(1, 8))
                point = tuple(float(row[f"ee_feedback_{axis}_m"]) for axis in ("x", "y", "z"))
                timestamp = int(row["host_update_entry_ns"])
                trial = int(row["trial_id"])
            except (KeyError, TypeError, ValueError):
                continue
            values = (*q, *dq, *q_ref, *tau, *point)
            if not all(math.isfinite(value) for value in values):
                continue
            q_error.extend(abs(q[i] - q_ref[i]) for i in range(7))
            speed.extend(abs(value) for value in dq)
            torque.extend(abs(value) for value in tau)
            error = _circle_error(point, replay.path)
            circle_all.append(error)
            if trial > 1:
                circle_traversal.append(error)
            velocity_rows.append((timestamp, dq))
    acceleration: list[float] = []
    for (previous_time, previous), (current_time, current) in zip(velocity_rows, velocity_rows[1:]):
        dt = (current_time - previous_time) * 1.0e-9
        if dt <= 0.0:
            continue
        acceleration.extend(abs((current[i] - previous[i]) / dt) for i in range(7))
    return {
        "schema_version": 1,
        "sample_count": len(velocity_rows),
        "circle_error_excluding_initial_approach_m": _stats(circle_traversal),
        "circle_error_all_samples_m": _stats(circle_all),
        "abs_tracking_error_rad": _stats(q_error),
        "abs_joint_velocity_rad_s": _stats(speed),
        "abs_estimated_acceleration_rad_s2": _stats(acceleration),
        "abs_commanded_torque_nm": _stats(torque),
    }


def _completed_job_valid(job_dir: Path, replay: HardwareSessionReplay) -> bool:
    try:
        completion = json.loads((job_dir / "completed.json").read_text())
        summary = json.loads((job_dir / "summary.json").read_text())
        artifact_manifest = json.loads((job_dir / "manifest.json").read_text())
        trace = json.loads((job_dir / "simulation_trace_summary.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    hardware = artifact_manifest.get("hardware_replay")
    return (
        completion.get("status") == "complete"
        and summary.get("episodes_recorded") == 1
        and hardware is not None
        and int(hardware.get("session_id", -1)) == replay.session_id
        and hardware.get("checkpoint_sha256") == replay.checkpoint_sha256
        and trace.get("sample_count", 0) > 1
    )


def _run_job(
    replay: HardwareSessionReplay,
    *,
    output: Path,
    evaluate_script: Path,
    device: str,
    visualizer: str,
    resume: bool,
) -> tuple[bool, Path]:
    job_dir = output / "jobs" / f"session_{replay.session_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    path_catalog = replay.write_path_catalog(job_dir / "hardware_path.yaml")
    if resume and _completed_job_valid(job_dir, replay):
        print(f"  Reusing completed session {replay.session_id}")
        return True, job_dir
    for marker in ("completed.json", "completed.json.tmp"):
        (job_dir / marker).unlink(missing_ok=True)
    command = [
        sys.executable, "-u", str(evaluate_script),
        "--task", replay.task,
        "--checkpoint", str(replay.checkpoint),
        "--scenario", "action_delay_1",
        "--num_envs", "1",
        "--num_episodes", "1",
        "--success_threshold", str(replay.path["position_threshold_m"]),
        "--success_steps", "1",
        "--path", replay.path_name,
        "--path-file", str(path_catalog),
        "--hardware-session", str(replay.session_dir),
        "--seed", str(EVALUATION_SEED),
        "--device", device,
        "--viz", visualizer,
        "--deterministic",
        "--output-dir", str(job_dir),
        "--job-id", f"axis4_hardware_session_{replay.session_id}",
    ]
    job = {
        "schema_version": 1,
        "session_id": replay.session_id,
        "hardware_session": str(replay.session_dir),
        "task": replay.task,
        "checkpoint": str(replay.checkpoint),
        "checkpoint_sha256": replay.checkpoint_sha256,
        "bundle_manifest_sha256": replay.bundle_manifest_sha256,
        "path_catalog": str(path_catalog),
        "path_catalog_sha256": _sha256(path_catalog),
        "command": command,
    }
    (job_dir / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    runs_dir = Path(os.environ["FRANKA_RL_DATA_ROOT"]) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    with (job_dir / "stdout.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=runs_dir, env=os.environ.copy(), stdout=log,
            stderr=subprocess.STDOUT, text=True, check=False,
        )
    return result.returncode == 0 and _completed_job_valid(job_dir, replay), job_dir


def _compile(replays: list[HardwareSessionReplay], output: Path) -> None:
    rows: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for replay in replays:
        job_dir = output / "jobs" / f"session_{replay.session_id}"
        evaluation = json.loads((job_dir / "summary.json").read_text())
        simulation = json.loads((job_dir / "simulation_trace_summary.json").read_text())
        hardware = _hardware_trace_summary(replay)
        path_summary = replay.hardware_summary
        sim_reached = int(evaluation["mean_waypoints_reached"])
        sim_timed_out = int(evaluation["total_waypoint_timeouts"])
        hw_circle = hardware["circle_error_excluding_initial_approach_m"]
        sim_circle = simulation["circle_error_excluding_initial_approach_m"]
        hw_acceleration = hardware["abs_estimated_acceleration_rad_s2"]
        sim_acceleration = simulation["abs_estimated_acceleration_rad_s2"]
        hw_speed = hardware["abs_joint_velocity_rad_s"]
        sim_speed = simulation["abs_joint_velocity_rad_s"]
        hw_tracking = hardware["abs_tracking_error_rad"]
        sim_tracking = simulation["abs_tracking_error_rad"]
        hw_torque = hardware["abs_commanded_torque_nm"]
        sim_torque = simulation["abs_commanded_torque_nm"]
        stiffness = float(replay.controller["stiffness_nm_rad"][0])
        damping = float(replay.controller["damping_nms_rad"][0])
        sim_fault = int(evaluation["unsafe_failures"]) > 0
        hardware_fault = replay.final_metadata["terminal_state"] != "complete"
        row = {
            "session_id": replay.session_id,
            "route": replay.controller_mode,
            "stiffness_nm_rad": stiffness,
            "damping_nms_rad": damping,
            "hardware_waypoints_reached": int(path_summary["reached"]),
            "simulation_waypoints_reached": sim_reached,
            "delta_waypoints_reached": sim_reached - int(path_summary["reached"]),
            "hardware_waypoints_timed_out": int(path_summary["timed_out"]),
            "simulation_waypoints_timed_out": sim_timed_out,
            "hardware_circle_error_mean_m": hw_circle["mean"],
            "simulation_circle_error_mean_m": sim_circle["mean"],
            "delta_circle_error_mean_m": sim_circle["mean"] - hw_circle["mean"],
            "hardware_circle_error_p95_m": hw_circle["p95"],
            "simulation_circle_error_p95_m": sim_circle["p95"],
            "delta_circle_error_p95_m": sim_circle["p95"] - hw_circle["p95"],
            "hardware_p99_abs_acceleration_rad_s2": hw_acceleration["p99"],
            "simulation_p99_abs_acceleration_rad_s2": sim_acceleration["p99"],
            "delta_p99_abs_acceleration_rad_s2": sim_acceleration["p99"] - hw_acceleration["p99"],
            "hardware_peak_joint_speed_rad_s": hw_speed["maximum"],
            "simulation_peak_joint_speed_rad_s": sim_speed["maximum"],
            "delta_peak_joint_speed_rad_s": sim_speed["maximum"] - hw_speed["maximum"],
            "hardware_p99_tracking_error_rad": hw_tracking["p99"],
            "simulation_p99_tracking_error_rad": sim_tracking["p99"],
            "delta_p99_tracking_error_rad": sim_tracking["p99"] - hw_tracking["p99"],
            "hardware_peak_commanded_torque_nm": hw_torque["maximum"],
            "simulation_peak_commanded_torque_nm": sim_torque["maximum"],
            "delta_peak_commanded_torque_nm": sim_torque["maximum"] - hw_torque["maximum"],
            "hardware_fault": hardware_fault,
            "simulation_fault": sim_fault,
            "fault_mismatch": hardware_fault != sim_fault,
            "job_dir": str(job_dir),
        }
        rows.append(row)
        documents.append({
            "session_id": replay.session_id,
            "hardware": hardware,
            "simulation": simulation,
            "evaluation_summary": evaluation,
            "deltas": row,
        })
    compiled = output / "compiled"
    compiled.mkdir(parents=True, exist_ok=True)
    with (compiled / "paired_replay.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (compiled / "paired_replay.json").write_text(
        json.dumps({"schema_version": 1, "sessions": documents}, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-package", type=Path)
    parser.add_argument("--session-id", type=int, action="append", dest="session_ids")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--viz", default="none")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    data_root = Path(os.environ.get("FRANKA_RL_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()
    if not data_root.is_dir() or not os.access(data_root, os.W_OK):
        raise RuntimeError(f"FRANKA_RL_DATA_ROOT unavailable: {data_root}")
    if shutil.disk_usage(data_root).free < MIN_FREE_BYTES:
        raise RuntimeError("FRANKA_RL_DATA_ROOT has less than 100 MiB free")
    os.environ["FRANKA_RL_DATA_ROOT"] = str(data_root)
    package = (args.hardware_package or data_root / PACKAGE_NAME).expanduser().resolve()
    manifest = json.loads((package / "MANIFEST.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported hardware package manifest")
    requested = set(args.session_ids or ())
    entries = [entry for entry in manifest["sessions"] if not requested or int(entry["session_id"]) in requested]
    if requested - {int(entry["session_id"]) for entry in entries}:
        raise ValueError(f"Unknown session IDs: {sorted(requested - {int(e['session_id']) for e in entries})}")
    replays = [
        HardwareSessionReplay.from_directory(package / entry["relative_path"], data_root=data_root)
        for entry in entries
    ]
    repo_root = Path(__file__).resolve().parents[2]
    evaluate_script = repo_root / "scripts" / "rsl_rl" / "evaluate.py"
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output = data_root / "evaluation_suites" / f"{stamp}_axis4_hardware_paired_replay"
    else:
        output = args.output_dir.expanduser().resolve()
    try:
        output.relative_to(data_root)
    except ValueError as error:
        raise RuntimeError(f"Output must be under FRANKA_RL_DATA_ROOT ({data_root})") from error
    output.mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": 1,
        "hardware_package": str(package),
        "hardware_manifest_sha256": _sha256(package / "MANIFEST.json"),
        "effective_policy_delay_steps": 1,
        "evaluation_seed": EVALUATION_SEED,
        "sessions": [],
    }
    failures = []
    for number, replay in enumerate(replays, start=1):
        print(f"[{number}/{len(replays)}] hardware session {replay.session_id} / {replay.controller_mode}")
        success, job_dir = _run_job(
            replay, output=output, evaluate_script=evaluate_script,
            device=args.device, visualizer=args.viz, resume=not args.no_resume,
        )
        index["sessions"].append({
            "session_id": replay.session_id,
            "job_dir": str(job_dir),
            "complete": success,
        })
        if not success:
            failures.append(replay.session_id)
            print(f"  FAILED: see {job_dir / 'stdout.log'}")
            if args.fail_fast:
                break
    if not failures and len(index["sessions"]) == len(replays):
        _compile(replays, output)
    index["failed_session_ids"] = failures
    (output / "axis4_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Axis-4 replay index: {output / 'axis4_index.json'}")
    if not failures:
        print(f"Compiled paired replay: {output / 'compiled'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
