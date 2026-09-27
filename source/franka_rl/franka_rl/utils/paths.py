"""Named Cartesian paths for deterministic policy evaluation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml


DEFAULT_PATH_RESOURCE = "config/paths.yaml"


@dataclass(frozen=True)
class PathSpec:
    """Resolved path used by the waypoint command term."""

    name: str
    type: Literal["circle", "waypoints"]
    waypoints_m: tuple[tuple[float, float, float], ...]
    waypoint_timeout_s: float
    position_threshold_m: float
    description: str = ""

    def __post_init__(self) -> None:
        if len(self.waypoints_m) < 3:
            raise ValueError("path must contain at least three waypoints")
        if self.waypoint_timeout_s <= 0.0:
            raise ValueError("waypoint_timeout_s must be positive")
        if self.position_threshold_m <= 0.0:
            raise ValueError("position_threshold_m must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PathCatalog:
    """Strict loader for named evaluation paths."""

    def __init__(self, paths: dict[str, PathSpec], *, source: str, sha256: str, version: int):
        self._paths = paths
        self.source = source
        self.sha256 = sha256
        self.version = version

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> PathCatalog:
        if path is None:
            resource = files("franka_rl").joinpath(DEFAULT_PATH_RESOURCE)
            raw = resource.read_bytes()
            source = f"package:{DEFAULT_PATH_RESOURCE}"
        else:
            path_file = Path(path).expanduser().resolve()
            if not path_file.is_file():
                raise FileNotFoundError(f"Path catalog not found: {path_file}")
            raw = path_file.read_bytes()
            source = str(path_file)

        document = yaml.safe_load(raw) or {}
        if not isinstance(document, dict) or set(document) != {"version", "paths"}:
            raise ValueError("Path catalog must contain exactly 'version' and 'paths'.")
        if document["version"] != 1:
            raise ValueError(f"Unsupported path schema version {document['version']!r}.")
        raw_paths = document["paths"]
        if not isinstance(raw_paths, dict) or not raw_paths:
            raise ValueError("Path catalog must define at least one path.")
        paths = {name: _parse_path(name, value) for name, value in raw_paths.items()}
        return cls(
            paths,
            source=source,
            sha256=hashlib.sha256(raw).hexdigest(),
            version=1,
        )

    def get(self, name: str) -> PathSpec:
        try:
            return self._paths[name]
        except KeyError as error:
            available = ", ".join(sorted(self._paths))
            raise ValueError(f"Unknown path {name!r}. Available paths: {available}.") from error

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._paths))

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.version,
            "source": self.source,
            "sha256": self.sha256,
        }


def _parse_path(name: str, value: Any) -> PathSpec:
    if not isinstance(value, dict):
        raise ValueError(f"Path {name!r} must be a mapping.")
    allowed = {
        "type",
        "description",
        "center_m",
        "radius_m",
        "plane",
        "waypoint_count",
        "phase_rad",
        "waypoints_m",
        "waypoint_timeout_s",
        "position_threshold_m",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Path {name!r} has unknown keys: {sorted(unknown)}.")

    path_type = value.get("type")
    if path_type == "circle":
        waypoints = _circle_waypoints(value)
    elif path_type == "waypoints":
        raw_waypoints = value.get("waypoints_m", ())
        waypoints = tuple(tuple(float(component) for component in point) for point in raw_waypoints)
    else:
        raise ValueError(f"Path {name!r} has unsupported type {path_type!r}.")
    if any(len(point) != 3 for point in waypoints):
        raise ValueError(f"Path {name!r} waypoints must be 3D positions.")

    return PathSpec(
        name=name,
        type=path_type,
        waypoints_m=waypoints,
        waypoint_timeout_s=float(value["waypoint_timeout_s"]),
        position_threshold_m=float(value["position_threshold_m"]),
        description=str(value.get("description", "")),
    )


def _circle_waypoints(value: dict[str, Any]) -> tuple[tuple[float, float, float], ...]:
    center = tuple(float(component) for component in value["center_m"])
    if len(center) != 3:
        raise ValueError("circle center_m must have three values")
    radius = float(value["radius_m"])
    count = int(value["waypoint_count"])
    plane = value.get("plane", "xy")
    if radius <= 0.0 or count < 3:
        raise ValueError("circle radius must be positive and waypoint_count must be at least 3")
    if plane not in ("xy", "xz", "yz"):
        raise ValueError("circle plane must be one of xy, xz, yz")
    phase = float(value.get("phase_rad", 0.0))
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]

    points: list[tuple[float, float, float]] = []
    for index in range(count):
        angle = phase + 2.0 * math.pi * index / count
        point = list(center)
        point[axes[0]] += radius * math.cos(angle)
        point[axes[1]] += radius * math.sin(angle)
        points.append(tuple(point))
    return tuple(points)
