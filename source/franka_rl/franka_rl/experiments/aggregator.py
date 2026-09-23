"""Compile evaluator artifacts into suite-level CSV tables."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .artifact_reader import EvaluationArtifacts


METRICS = (
    "success_rate",
    "timeout_rate",
    "unsafe_failure_rate",
    "mean_episode_steps",
    "mean_time_to_success_s",
    "median_time_to_success_s",
    "p90_time_to_success_s",
)


@dataclass(frozen=True)
class CompletedJob:
    policy: str
    scenario: str
    seed: int
    job_id: str
    checkpoint_sha256: str
    artifacts: EvaluationArtifacts


def compile_results(
    jobs: Iterable[CompletedJob],
    destination: Path,
    *,
    baseline_policy: str,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    rows = [_job_row(job) for job in jobs]
    rows.sort(key=lambda row: (row["policy"], row["scenario"], int(row["seed"])))
    _write_csv(destination / "jobs.csv", rows, _job_fields())

    summary_rows = _scenario_rows(rows)
    _write_csv(destination / "scenario_summary.csv", summary_rows, _scenario_fields())

    comparison_rows = _comparison_rows(rows, baseline_policy)
    _write_csv(destination / "policy_comparison.csv", comparison_rows, _comparison_fields())


def _job_row(job: CompletedJob) -> dict[str, Any]:
    summary = job.artifacts.summary
    row: dict[str, Any] = {
        "policy": job.policy,
        "scenario": job.scenario,
        "seed": job.seed,
        "job_id": job.job_id,
        "checkpoint_sha256": job.checkpoint_sha256,
        "output_dir": str(job.artifacts.output_dir),
        "episodes_recorded": summary["episodes_recorded"],
        "successes": summary["successes"],
        "timeouts": summary["timeouts"],
        "unsafe_failures": summary["unsafe_failures"],
    }
    for metric in METRICS:
        row[metric] = summary.get(metric)
    return row


def _scenario_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["policy"]), str(row["scenario"]))].append(row)

    output: list[dict[str, Any]] = []
    for (policy, scenario), group in sorted(groups.items()):
        result: dict[str, Any] = {
            "policy": policy,
            "scenario": scenario,
            "num_seeds": len(group),
            "seeds": ";".join(str(row["seed"]) for row in sorted(group, key=lambda item: int(item["seed"]))),
            "episodes_recorded": sum(int(row["episodes_recorded"]) for row in group),
            "successes": sum(int(row["successes"]) for row in group),
            "timeouts": sum(int(row["timeouts"]) for row in group),
            "unsafe_failures": sum(int(row["unsafe_failures"]) for row in group),
        }
        for metric in METRICS:
            statistics_for_metric = _statistics([row[metric] for row in group])
            for statistic, value in statistics_for_metric.items():
                result[f"{metric}_{statistic}"] = value
        output.append(result)
    return output


def _comparison_rows(rows: list[dict[str, Any]], baseline_policy: str) -> list[dict[str, Any]]:
    lookup = {
        (str(row["policy"]), str(row["scenario"]), int(row["seed"])): row
        for row in rows
    }
    policies = sorted({str(row["policy"]) for row in rows} - {baseline_policy})
    scenarios = sorted({str(row["scenario"]) for row in rows})
    output: list[dict[str, Any]] = []
    for policy in policies:
        for scenario in scenarios:
            seeds = sorted(
                seed
                for base_policy, base_scenario, seed in lookup
                if base_policy == baseline_policy
                and base_scenario == scenario
                and (policy, scenario, seed) in lookup
            )
            if not seeds:
                continue
            result: dict[str, Any] = {
                "baseline_policy": baseline_policy,
                "candidate_policy": policy,
                "scenario": scenario,
                "paired_seeds": len(seeds),
                "seeds": ";".join(str(seed) for seed in seeds),
            }
            for metric in METRICS:
                deltas = [
                    _numeric(lookup[(policy, scenario, seed)].get(metric))
                    - _numeric(lookup[(baseline_policy, scenario, seed)].get(metric))
                    for seed in seeds
                    if lookup[(policy, scenario, seed)].get(metric) is not None
                    and lookup[(baseline_policy, scenario, seed)].get(metric) is not None
                ]
                stats = _statistics(deltas)
                for statistic, value in stats.items():
                    result[f"delta_{metric}_{statistic}"] = value
            output.append(result)
    return output


def _statistics(values: Iterable[Any]) -> dict[str, float | int | str]:
    numbers = [_numeric(value) for value in values if value is not None and value != ""]
    if not numbers:
        return {"mean": "", "std": "", "ci95_low": "", "ci95_high": "", "min": "", "max": ""}
    mean = statistics.fmean(numbers)
    if len(numbers) > 1:
        std = statistics.stdev(numbers)
        half_width = 1.96 * std / math.sqrt(len(numbers))
        low: float | str = mean - half_width
        high: float | str = mean + half_width
    else:
        std = 0.0
        low = ""
        high = ""
    return {
        "mean": mean,
        "std": std,
        "ci95_low": low,
        "ci95_high": high,
        "min": min(numbers),
        "max": max(numbers),
    }


def _numeric(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Expected numeric metric, got {value!r}.")
    return float(value)


def _job_fields() -> list[str]:
    return [
        "policy",
        "scenario",
        "seed",
        "job_id",
        "checkpoint_sha256",
        "output_dir",
        "episodes_recorded",
        "successes",
        "timeouts",
        "unsafe_failures",
        *METRICS,
    ]


def _scenario_fields() -> list[str]:
    fields = [
        "policy",
        "scenario",
        "num_seeds",
        "seeds",
        "episodes_recorded",
        "successes",
        "timeouts",
        "unsafe_failures",
    ]
    for metric in METRICS:
        fields.extend(f"{metric}_{statistic}" for statistic in ("mean", "std", "ci95_low", "ci95_high", "min", "max"))
    return fields


def _comparison_fields() -> list[str]:
    fields = ["baseline_policy", "candidate_policy", "scenario", "paired_seeds", "seeds"]
    for metric in METRICS:
        fields.extend(
            f"delta_{metric}_{statistic}"
            for statistic in ("mean", "std", "ci95_low", "ci95_high", "min", "max")
        )
    return fields


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
