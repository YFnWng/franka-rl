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
    first_waypoint_timeout_s: float
    position_threshold_m: float
    target_orientation_xyzw: tuple[float, float, float, float]
    center_m: tuple[float, float, float] | None = None
    orientation_rpy_deg: tuple[float, float, float] | None = None
    radius_m: float | None = None
    waypoint_count: int = 0
    phase_deg: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if len(self.waypoints_m) < 3:
            raise ValueError("path must contain at least three waypoints")
        if self.waypoint_timeout_s <= 0.0:
            raise ValueError("waypoint_timeout_s must be positive")
        if self.first_waypoint_timeout_s <= 0.0:
            raise ValueError("first_waypoint_timeout_s must be positive")
        if self.position_threshold_m <= 0.0:
            raise ValueError("position_threshold_m must be positive")
        if len(self.target_orientation_xyzw) != 4:
            raise ValueError("target_orientation_xyzw must contain four values")

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
        "orientation_rpy_deg",
        "radius_m",
        "waypoint_count",
        "phase_deg",
        "target_orientation_xyzw",
        "waypoints_m",
        "waypoint_timeout_s",
        "first_waypoint_timeout_s",
        "position_threshold_m",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Path {name!r} has unknown keys: {sorted(unknown)}.")

    path_type = value.get("type")
    if path_type == "circle":
        waypoints = _circle_waypoints(value)
        center = _vector(value, "center_m", 3)
        orientation = _vector(value, "orientation_rpy_deg", 3)
        radius = float(value["radius_m"])
        phase_deg = float(value.get("phase_deg", 0.0))
    elif path_type == "waypoints":
        raw_waypoints = value.get("waypoints_m", ())
        waypoints = tuple(tuple(float(component) for component in point) for point in raw_waypoints)
        center = None
        orientation = None
        radius = None
        phase_deg = 0.0
    else:
        raise ValueError(f"Path {name!r} has unsupported type {path_type!r}.")
    if any(len(point) != 3 for point in waypoints):
        raise ValueError(f"Path {name!r} waypoints must be 3D positions.")

    target_orientation = _normalized_quaternion_xyzw(value["target_orientation_xyzw"])
    return PathSpec(
        name=name,
        type=path_type,
        waypoints_m=waypoints,
        waypoint_timeout_s=float(value["waypoint_timeout_s"]),
        first_waypoint_timeout_s=float(
            value.get("first_waypoint_timeout_s", value["waypoint_timeout_s"])
        ),
        position_threshold_m=float(value["position_threshold_m"]),
        target_orientation_xyzw=target_orientation,
        center_m=center,
        orientation_rpy_deg=orientation,
        radius_m=radius,
        waypoint_count=len(waypoints),
        phase_deg=phase_deg,
        description=str(value.get("description", "")),
    )


def _circle_waypoints(value: dict[str, Any]) -> tuple[tuple[float, float, float], ...]:
    center = _vector(value, "center_m", 3)
    roll_deg, pitch_deg, yaw_deg = _vector(value, "orientation_rpy_deg", 3)
    radius = float(value["radius_m"])
    count = int(value["waypoint_count"])
    if radius <= 0.0 or count < 3:
        raise ValueError("circle radius must be positive and waypoint_count must be at least 3")
    phase = math.radians(float(value.get("phase_deg", 0.0)))
    rotation = _rotation_matrix_xyz(
        math.radians(roll_deg),
        math.radians(pitch_deg),
        math.radians(yaw_deg),
    )

    points: list[tuple[float, float, float]] = []
    for index in range(count):
        angle = phase + 2.0 * math.pi * index / count
        local = (radius * math.cos(angle), radius * math.sin(angle), 0.0)
        offset = tuple(
            sum(rotation[row][column] * local[column] for column in range(3))
            for row in range(3)
        )
        points.append(tuple(center[axis] + offset[axis] for axis in range(3)))
    return tuple(points)


def _vector(value: dict[str, Any], name: str, length: int) -> tuple[float, ...]:
    vector = tuple(float(component) for component in value[name])
    if len(vector) != length or not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{name} must contain {length} finite values")
    return vector


def _normalized_quaternion_xyzw(value: Any) -> tuple[float, float, float, float]:
    quaternion = tuple(float(component) for component in value)
    if len(quaternion) != 4 or not all(math.isfinite(component) for component in quaternion):
        raise ValueError("target_orientation_xyzw must contain four finite values")
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm <= 1.0e-12:
        raise ValueError("target_orientation_xyzw must have nonzero norm")
    return tuple(component / norm for component in quaternion)


def _rotation_matrix_xyz(
    roll: float, pitch: float, yaw: float
) -> tuple[tuple[float, float, float], ...]:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll)."""

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
