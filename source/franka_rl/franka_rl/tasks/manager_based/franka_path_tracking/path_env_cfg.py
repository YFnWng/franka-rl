"""Evaluation-only waypoint path task for the six-action FR3 policy."""

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from franka_rl.utils.paths import PathCatalog

from ..franka_incremental_impedance.incremental_impedance_env_cfg import (
    Fr3FrankyIncremental6DImpedanceEnvCfg,
)
from .path_command_cfg import WaypointPathCommandCfg
from .path_terms import waypoint_path_failed, waypoint_path_succeeded


@configclass
class Fr3FrankyIncremental6DCirclePathEnvCfg(
    Fr3FrankyIncremental6DImpedanceEnvCfg
):
    """Run the trained point-reaching policy over a deterministic circle."""

    evaluation_protocol: str = "waypoint_path"
    path_name: str = "circle_xy"

    def __post_init__(self) -> None:
        super().__post_init__()
        catalog = PathCatalog.from_yaml()
        path = catalog.get(self.path_name)
        self.path_metadata = {
            **path.to_dict(),
            "catalog": catalog.metadata(),
        }

        self.commands.ee_pose = WaypointPathCommandCfg(
            asset_name="robot",
            body_name="fr3_flange",
            resampling_time_range=(
                path.waypoint_timeout_s,
                path.waypoint_timeout_s,
            ),
            debug_vis=True,
            position_success_threshold=None,
            ranges=UniformPoseCommandCfg.Ranges(
                pos_x=(0.0, 0.0),
                pos_y=(0.0, 0.0),
                pos_z=(0.0, 0.0),
                roll=(0.0, 0.0),
                pitch=(0.0, 0.0),
                yaw=(0.0, 0.0),
            ),
            waypoints_m=path.waypoints_m,
            waypoint_timeout_s=path.waypoint_timeout_s,
            position_threshold_m=path.position_threshold_m,
        )

        # A successful path reaches every waypoint. Completing the sequence
        # with one or more waypoint timeouts is a separate failure outcome.
        self.terminations.reached_target = DoneTerm(
            func=waypoint_path_succeeded,
            time_out=False,
            params={"command_name": "ee_pose"},
        )
        self.terminations.waypoint_path_failed = DoneTerm(
            func=waypoint_path_failed,
            time_out=False,
            params={"command_name": "ee_pose"},
        )

        # Permit every waypoint to consume its full timeout, plus one second
        # for the one-step completion termination and numerical margin.
        self.episode_length_s = (
            len(path.waypoints_m) * path.waypoint_timeout_s + 1.0
        )
