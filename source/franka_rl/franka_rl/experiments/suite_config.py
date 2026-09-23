"""Strict configuration model for evaluation suites."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class PolicySpec:
    name: str
    checkpoint: Path
    description: str = ""


@dataclass(frozen=True)
class EvaluationSettings:
    task: str
    num_envs: int
    episodes_per_job: int
    success_threshold: float
    success_steps: int
    device: str
    deterministic: bool = True
    visualizer: str = "none"


@dataclass(frozen=True)
class EvaluationSuiteConfig:
    name: str
    policies: tuple[PolicySpec, ...]
    scenarios: tuple[str, ...]
    seeds: tuple[int, ...]
    evaluation: EvaluationSettings
    baseline_policy: str
    scenario_file: Path | None
    source: Path
    source_sha256: str
    version: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvaluationSuiteConfig:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Evaluation suite file not found: {source}")
        raw = source.read_bytes()
        document = yaml.safe_load(raw) or {}
        document = _expand_environment(document)
        root = _mapping(document, "suite document")
        _unknown(
            root,
            {
                "version",
                "name",
                "policies",
                "scenarios",
                "seeds",
                "evaluation",
                "baseline_policy",
                "scenario_file",
            },
            "suite document",
        )
        if root.get("version") != 1:
            raise ValueError(f"Unsupported suite version {root.get('version')!r}; expected 1.")

        name = _safe_name(root.get("name"), "suite name")
        raw_policies = _mapping(root.get("policies"), "policies")
        policies: list[PolicySpec] = []
        for policy_name, value in raw_policies.items():
            policy_name = _safe_name(policy_name, "policy name")
            policy = _mapping(value, f"policy {policy_name!r}")
            _unknown(policy, {"checkpoint", "description"}, f"policy {policy_name!r}")
            checkpoint = policy.get("checkpoint")
            if not isinstance(checkpoint, str) or not checkpoint:
                raise TypeError(f"Policy {policy_name!r} must define a checkpoint path.")
            policies.append(
                PolicySpec(
                    name=policy_name,
                    checkpoint=Path(checkpoint).expanduser().resolve(),
                    description=str(policy.get("description", "")),
                )
            )
        if not policies:
            raise ValueError("The suite must define at least one policy.")

        scenarios = _string_tuple(root.get("scenarios"), "scenarios")
        for scenario in scenarios:
            _safe_name(scenario, "scenario name")
        seeds = _integer_tuple(root.get("seeds"), "seeds")

        evaluation = _parse_evaluation(root.get("evaluation"))
        baseline_policy = root.get("baseline_policy", policies[0].name)
        if baseline_policy not in {policy.name for policy in policies}:
            raise ValueError(f"Unknown baseline policy {baseline_policy!r}.")

        scenario_file_value = root.get("scenario_file")
        if scenario_file_value is None:
            scenario_file = None
        elif isinstance(scenario_file_value, str) and scenario_file_value:
            scenario_file = Path(scenario_file_value).expanduser().resolve()
        else:
            raise TypeError("scenario_file must be null or a non-empty path string.")

        return cls(
            name=name,
            policies=tuple(policies),
            scenarios=scenarios,
            seeds=seeds,
            evaluation=evaluation,
            baseline_policy=baseline_policy,
            scenario_file=scenario_file,
            source=source,
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source"] = str(self.source)
        result["scenario_file"] = str(self.scenario_file) if self.scenario_file else None
        for policy in result["policies"]:
            policy["checkpoint"] = str(policy["checkpoint"])
        return result


def _parse_evaluation(value: Any) -> EvaluationSettings:
    mapping = _mapping(value, "evaluation")
    allowed = {
        "task",
        "num_envs",
        "episodes_per_job",
        "success_threshold",
        "success_steps",
        "device",
        "deterministic",
        "visualizer",
    }
    _unknown(mapping, allowed, "evaluation")
    task = mapping.get("task")
    device = mapping.get("device")
    visualizer = mapping.get("visualizer", "none")
    if not isinstance(task, str) or not task:
        raise TypeError("evaluation.task must be a non-empty string.")
    if not isinstance(device, str) or not device:
        raise TypeError("evaluation.device must be a non-empty string.")
    if not isinstance(visualizer, str) or not visualizer:
        raise TypeError("evaluation.visualizer must be a non-empty string.")
    num_envs = _positive_int(mapping.get("num_envs"), "evaluation.num_envs")
    episodes = _positive_int(mapping.get("episodes_per_job"), "evaluation.episodes_per_job")
    success_steps = _positive_int(mapping.get("success_steps"), "evaluation.success_steps")
    threshold = mapping.get("success_threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or threshold <= 0:
        raise ValueError("evaluation.success_threshold must be positive.")
    deterministic = mapping.get("deterministic", True)
    if not isinstance(deterministic, bool):
        raise TypeError("evaluation.deterministic must be boolean.")
    return EvaluationSettings(
        task=task,
        num_envs=num_envs,
        episodes_per_job=episodes,
        success_threshold=float(threshold),
        success_steps=success_steps,
        device=device,
        deterministic=deterministic,
        visualizer=visualizer,
    )


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ValueError(f"Unresolved environment variable in {value!r}.")
        return expanded
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping for {context}, got {type(value).__name__}.")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"All keys in {context} must be strings.")
    return value


def _unknown(mapping: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {context}: {', '.join(sorted(unknown))}.")


def _safe_name(value: Any, context: str) -> str:
    if not isinstance(value, str) or SAFE_NAME.fullmatch(value) is None:
        raise ValueError(f"Invalid {context} {value!r}.")
    return value


def _string_tuple(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{context} must be a non-empty list of strings.")
    if len(set(value)) != len(value):
        raise ValueError(f"{context} cannot contain duplicates.")
    return tuple(value)


def _integer_tuple(value: Any, context: str) -> tuple[int, ...]:
    valid = (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )
    if not valid:
        raise TypeError(f"{context} must be a non-empty list of integers.")
    if len(set(value)) != len(value):
        raise ValueError(f"{context} cannot contain duplicates.")
    return tuple(value)


def _positive_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{context} must be a positive integer.")
    return value
