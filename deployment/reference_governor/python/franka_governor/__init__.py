"""Public native API and strict configuration loading.

Restored on the simulation host after the initializer was omitted in transfer.
This loader is local source, not evidence of parity with an unshipped wheel.
"""
import hashlib
import json
from pathlib import Path

from ._core import Batch, Config, Feedback, Governor, Message, Reason, Status, TICK_NS, __version__

__all__ = ["Batch", "Config", "Feedback", "Governor", "Message", "Reason", "Status",
           "TICK_NS", "load_config", "policy_tick"]

_FIELDS = {
    "lower", "upper", "margin", "max_velocity", "max_acceleration", "max_jerk",
    "envelope_velocity", "envelope_offset", "envelope_deceleration", "default_position",
    "tracking_error", "desired_error", "desired_velocity_error", "desired_acceleration_error",
    "max_segment_distance", "max_projection", "horizon_ticks", "max_blocked_ticks",
    "max_projected_ticks", "action_timeout_ns", "observation_timeout_ns", "tick_tolerance_ns",
}


def load_config(path, *, allow_simulation=False):
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if document.get("schema_version") != 1 or document.get("algorithm") != "bounded_quintic_v1":
        raise ValueError("Unsupported governor schema or algorithm")
    # Deployment loading deliberately remains blocked until the RT-host loader
    # and approved review-record schema are supplied; no invented authorization.
    if document.get("purpose") != "simulation_fixture" or not allow_simulation:
        raise ValueError("This simulation-host loader requires explicit simulation-fixture opt-in")
    core = document.get("core")
    if not isinstance(core, dict) or set(core) != _FIELDS:
        raise ValueError("Missing or unexpected governor core fields")
    config = Config()
    try:
        for name, value in core.items():
            if value is None:
                raise ValueError(f"Unspecified governor parameter: {name}")
            setattr(config, name, value)
        config.validate()
    except (TypeError, OverflowError) as error:
        raise ValueError("Invalid governor parameter type") from error
    return config, hashlib.sha256(raw).hexdigest()


def policy_tick(index):
    """First millisecond boundary at or after the index-th 30 Hz update."""
    if not isinstance(index, int) or index < 0:
        raise ValueError("Policy index must be a nonnegative integer")
    return (index * 1000 + 29) // 30
