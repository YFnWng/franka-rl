"""Validation and loading of evaluator artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationArtifacts:
    output_dir: Path
    summary: dict[str, Any]
    manifest: dict[str, Any]
    completion: dict[str, Any]


def read_artifacts(
    output_dir: Path,
    *,
    expected_job_id: str,
    expected_episodes: int,
    expected_checkpoint_sha256: str,
    expected_target_set_sha256: str | None = None,
) -> EvaluationArtifacts:
    summary = _read_json(output_dir / "summary.json")
    manifest = _read_json(output_dir / "manifest.json")
    completion = _read_json(output_dir / "completed.json")
    episodes_path = output_dir / "episodes.csv"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing evaluation artifact: {episodes_path}")
    if completion.get("status") != "complete":
        raise ValueError(f"Job {expected_job_id} does not have a complete marker.")
    if completion.get("job_id") != expected_job_id or manifest.get("job_id") != expected_job_id:
        raise ValueError(f"Job ID mismatch in {output_dir}.")
    if completion.get("episodes_recorded") != expected_episodes:
        raise ValueError(f"Completion episode count mismatch in {output_dir}.")
    if summary.get("episodes_recorded") != expected_episodes:
        raise ValueError(f"Summary episode count mismatch in {output_dir}.")
    if manifest.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError(f"Checkpoint hash mismatch in {output_dir}.")
    target_set = manifest.get("target_set")
    actual_target_sha256 = target_set.get("sha256") if isinstance(target_set, dict) else None
    if actual_target_sha256 != expected_target_set_sha256:
        raise ValueError(f"Target-set hash mismatch in {output_dir}.")
    return EvaluationArtifacts(output_dir, summary, manifest, completion)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation artifact: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return value
