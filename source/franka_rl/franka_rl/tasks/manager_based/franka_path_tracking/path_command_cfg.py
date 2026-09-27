"""Lightweight configuration for deterministic waypoint-path commands."""

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils.configclass import configclass


@configclass
class WaypointPathCommandCfg(UniformPoseCommandCfg):
    """Configuration for a deterministic waypoint path."""

    class_type: str = "{DIR}.path_command:WaypointPathCommand"
    waypoints_m: tuple[tuple[float, float, float], ...] = ()
    waypoint_timeout_s: float = 1.0
    position_threshold_m: float = 0.01
    fixed_quaternion_xyzw: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
