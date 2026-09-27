"""Simulator-independent path catalog and 50 Hz waypoint state machine."""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathSpec:
    name: str
    type: str
    waypoints_m: tuple[tuple[float, float, float], ...]
    waypoint_timeout_s: float
    position_threshold_m: float
    target_orientation_xyzw: tuple[float, float, float, float]
    center_m: tuple[float, float, float] | None = None
    orientation_rpy_deg: tuple[float, float, float] | None = None
    radius_m: float | None = None
    waypoint_count: int = 0
    phase_deg: float = 0.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PathCatalog:
    def __init__(self, paths: dict[str, PathSpec], source: Path, digest: str, version: int):
        self.paths, self.source, self.sha256, self.version = paths, source, digest, version

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PathCatalog":
        source = Path(path)
        raw = source.read_bytes()
        document = yaml.safe_load(raw) or {}
        if not isinstance(document, dict) or set(document) != {"version", "paths"}:
            raise ValueError("path catalog must contain exactly version and paths")
        if document["version"] != 1:
            raise ValueError("unsupported path catalog version")
        values = document["paths"]
        if not isinstance(values, dict) or not values:
            raise ValueError("path catalog must contain paths")
        paths = {name: _parse_path(name, value) for name, value in values.items()}
        return cls(paths, source, hashlib.sha256(raw).hexdigest(), 1)

    def get(self, name: str) -> PathSpec:
        if name not in self.paths:
            raise ValueError(f"unknown path {name!r}")
        return self.paths[name]


def _vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must be finite")
    return result


def _rotation_xyz(roll: float, pitch: float, yaw: float) -> tuple[tuple[float, ...], ...]:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _parse_path(name: str, value: Any) -> PathSpec:
    if not isinstance(name, str) or not name or not isinstance(value, dict):
        raise ValueError("path name and mapping required")
    allowed = {"type", "description", "center_m", "orientation_rpy_deg", "radius_m",
               "waypoint_count", "phase_deg", "target_orientation_xyzw", "waypoints_m",
               "waypoint_timeout_s", "position_threshold_m"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"path {name!r} has unknown keys: {sorted(unknown)}")
    path_type = value.get("type")
    center = orientation = None
    radius = None
    phase = 0.0
    if path_type == "circle":
        center = _vector(value.get("center_m"), 3, "center_m")
        orientation = _vector(value.get("orientation_rpy_deg"), 3, "orientation_rpy_deg")
        radius = float(value.get("radius_m", 0))
        count = int(value.get("waypoint_count", 0))
        phase = float(value.get("phase_deg", 0))
        if not math.isfinite(radius) or radius <= 0 or count < 3 or not math.isfinite(phase):
            raise ValueError("invalid circle radius/count/phase")
        rotation = _rotation_xyz(*(math.radians(v) for v in orientation))
        points = []
        for index in range(count):
            angle = math.radians(phase) + 2 * math.pi * index / count
            local = (radius * math.cos(angle), radius * math.sin(angle), 0.0)
            offset = tuple(sum(rotation[r][c] * local[c] for c in range(3)) for r in range(3))
            points.append(tuple(center[i] + offset[i] for i in range(3)))
        waypoints = tuple(points)
    elif path_type == "waypoints":
        raw = value.get("waypoints_m")
        if not isinstance(raw, list):
            raise ValueError("waypoints_m must be a list")
        waypoints = tuple(_vector(point, 3, "waypoint") for point in raw)
    else:
        raise ValueError(f"unsupported path type {path_type!r}")
    if len(waypoints) < 3:
        raise ValueError("path must have at least three waypoints")
    timeout = float(value.get("waypoint_timeout_s", 0))
    threshold = float(value.get("position_threshold_m", 0))
    if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("path timeout and threshold must be positive")
    quaternion = _vector(value.get("target_orientation_xyzw"), 4, "target_orientation_xyzw")
    norm = math.sqrt(sum(v * v for v in quaternion))
    if norm <= 1e-12:
        raise ValueError("target quaternion must be nonzero")
    quaternion = tuple(v / norm for v in quaternion)
    return PathSpec(name, path_type, waypoints, timeout, threshold, quaternion,
                    center, orientation, radius, len(waypoints), phase,
                    str(value.get("description", "")))


class PathExecution:
    """Deterministic per-policy-step traversal state."""

    def __init__(self, spec: PathSpec, repetitions: int, policy_hz: float):
        if repetitions <= 0 or policy_hz <= 0:
            raise ValueError("positive repetitions and policy_hz required")
        self.spec, self.repetitions = spec, repetitions
        self.policy_hz = float(policy_hz)
        self.timeout_steps = max(1, math.ceil(spec.waypoint_timeout_s * policy_hz - 1e-9))
        self.traversal_index = 0
        self.waypoint_index = 0
        self.waypoint_steps = 0
        self.minimum_error_m = math.inf
        self.outcomes: list[dict[str, Any]] = []
        self.traversals: list[dict[str, Any]] = []
        self.complete = False

    @property
    def trial_id(self) -> int:
        return self.traversal_index * len(self.spec.waypoints_m) + self.waypoint_index + 1

    @property
    def target(self) -> tuple[float, float, float]:
        return self.spec.waypoints_m[self.waypoint_index]

    def update(self, flange_position_m: tuple[float, ...]) -> dict[str, Any] | None:
        if self.complete:
            return None
        distance = math.dist(self.target, flange_position_m)
        self.waypoint_steps += 1
        self.minimum_error_m = min(self.minimum_error_m, distance)
        reason = ""
        if distance <= self.spec.position_threshold_m:
            reason = "reached"
        elif self.waypoint_steps >= self.timeout_steps:
            reason = "timed_out"
        if not reason:
            return None
        result = {
            "traversal_index": self.traversal_index,
            "waypoint_index": self.waypoint_index,
            "trial_id": self.trial_id,
            "outcome": reason,
            "elapsed_policy_steps": self.waypoint_steps,
            "elapsed_s": self.waypoint_steps / self.policy_hz,
            "final_position_error_m": distance,
            "minimum_position_error_m": self.minimum_error_m,
            "target_position_base_m": list(self.target),
        }
        self.outcomes.append(result)
        self.waypoint_index += 1
        if self.waypoint_index == len(self.spec.waypoints_m):
            current = [x for x in self.outcomes if x["traversal_index"] == self.traversal_index]
            self.traversals.append({
                "traversal_index": self.traversal_index,
                "reached": sum(x["outcome"] == "reached" for x in current),
                "timed_out": sum(x["outcome"] == "timed_out" for x in current),
                "success": all(x["outcome"] == "reached" for x in current),
            })
            self.traversal_index += 1
            self.waypoint_index = 0
            if self.traversal_index == self.repetitions:
                self.complete = True
        self.waypoint_steps = 0
        self.minimum_error_m = math.inf
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "path_name": self.spec.name,
            "repetitions": self.repetitions,
            "complete": self.complete,
            "reached": sum(x["outcome"] == "reached" for x in self.outcomes),
            "timed_out": sum(x["outcome"] == "timed_out" for x in self.outcomes),
            "success": self.complete and all(x["outcome"] == "reached" for x in self.outcomes),
            "outcomes": self.outcomes,
            "traversals": self.traversals,
        }
