"""FR3 reaching with a hardware-matched velocity-reference impedance loop."""

import math

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from franka_rl.utils.paths import PathCatalog

from ..franka_incremental_impedance.incremental_impedance_env_cfg import (
    Fr3FrankyIncremental6DImpedanceEnvCfg,
)
from ..franka_path_tracking.path_command_cfg import WaypointPathCommandCfg
from ..franka_path_tracking.path_terms import waypoint_path_failed, waypoint_path_succeeded
from ..franka_rl import mdp


@configclass
class Fr3FrankyVelocityImpedanceEnvCfg(Fr3FrankyIncremental6DImpedanceEnvCfg):
    """Six normalized joint velocities with 1 kHz q-reference integration."""

    control_contract: str = "fr3_joint_velocity_impedance_29d_v1"
    controller_profile: str = "franky_k100_d20_velocity_reference_50hz_1khz_integration"
    required_action_delay_steps: int = 1
    evaluation_startup_auto_reset_prime: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.arm_action = mdp.FrankyVelocityReference6DImpedanceActionCfg(
            asset_name="robot",
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
            preserve_order=True,
        )

        # Exact 29D contract: measured q/dq (14), flange position error (3),
        # current integrated q_ref (6), and preceding normalized velocity (6).
        self.observations.policy.position_reference = ObsTerm(
            func=mdp.normalized_position_reference,
            params={"action_name": "arm_action"},
        )
        self.observations.policy.previous_increment_action = None
        self.observations.policy.previous_velocity_action = ObsTerm(
            func=mdp.normalized_joint_velocity_action,
            params={"action_name": "arm_action"},
        )

        # A small physical acceleration cost shapes ordinary motion. The
        # hinge remains zero inside the reviewed envelope and adds a weak
        # secondary preference against abrupt, instability-triggering
        # transitions without overwhelming early policy exploration.
        self.rewards.reference_acceleration = RewTerm(
            func=mdp.normalized_reference_acceleration_l2,
            weight=-1.0e-2,
            params={"action_name": "arm_action"},
        )
        self.rewards.reference_acceleration_excess = RewTerm(
            func=mdp.normalized_reference_acceleration_excess_l2,
            weight=-5.0e-4,
            params={"action_name": "arm_action"},
        )


@configclass
class Fr3FrankyVelocityImpedancePositionZAxisEnvCfg(
    Fr3FrankyVelocityImpedanceEnvCfg
):
    """Track Cartesian position and the flange z-axis with velocity actions."""

    control_contract: str = "fr3_joint_velocity_impedance_position_z_axis_31d_v1"
    z_axis_component_limit_rad: float = math.radians(30.0)

    def __post_init__(self) -> None:
        super().__post_init__()

        tilt = self.z_axis_component_limit_rad
        if not 0.0 < tilt < 0.5 * math.pi:
            raise ValueError("z_axis_component_limit_rad must be between 0 and pi/2")
        self.commands.ee_pose.ranges.roll = (-tilt, tilt)
        self.commands.ee_pose.ranges.pitch = (math.pi - tilt, math.pi + tilt)
        self.commands.ee_pose.ranges.yaw = (0.0, 0.0)

        flange = SceneEntityCfg("robot", body_names=["fr3_flange"])
        self.observations.policy.ee_z_axis_error = ObsTerm(
            func=mdp.ee_z_axis_error_b,
            params={"command_name": "ee_pose", "asset_cfg": flange},
        )
        self.rewards.z_axis_tracking = RewTerm(
            func=mdp.z_axis_tracking_exp,
            weight=0.5,
            params={
                "command_name": "ee_pose",
                "asset_cfg": flange,
                "sigma": 0.25,
            },
        )
        self.rewards.fine_z_axis_tracking = RewTerm(
            func=mdp.z_axis_tracking_exp,
            weight=0.5,
            params={
                "command_name": "ee_pose",
                "asset_cfg": flange,
                "sigma": 0.01,
            },
        )


@configclass
class Fr3FrankyVelocityImpedanceCirclePathEnvCfg(Fr3FrankyVelocityImpedanceEnvCfg):
    """Evaluate the velocity-reference policy over a named waypoint path."""

    evaluation_protocol: str = "waypoint_path"
    path_name: str = "circle_yz"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.configure_path(self.path_name)

    def configure_path(self, path_name: str, path_file: str | None = None) -> None:
        catalog = PathCatalog.from_yaml(path_file)
        path = catalog.get(path_name)
        self.path_name = path_name
        self.path_metadata = {**path.to_dict(), "catalog": catalog.metadata()}

        self.commands.ee_pose = WaypointPathCommandCfg(
            asset_name="robot",
            body_name="fr3_flange",
            resampling_time_range=(
                max(path.waypoint_timeout_s, path.first_waypoint_timeout_s),
                max(path.waypoint_timeout_s, path.first_waypoint_timeout_s),
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
            first_waypoint_timeout_s=path.first_waypoint_timeout_s,
            position_threshold_m=path.position_threshold_m,
            fixed_quaternion_xyzw=path.target_orientation_xyzw,
        )
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
        self.episode_length_s = (
            path.first_waypoint_timeout_s
            + (len(path.waypoints_m) - 1) * path.waypoint_timeout_s
            + 1.0
        )


@configclass
class Fr3FrankyVelocityImpedancePositionZAxisCirclePathEnvCfg(
    Fr3FrankyVelocityImpedancePositionZAxisEnvCfg
):
    """Evaluate the velocity position-plus-z-axis policy on a waypoint path."""

    evaluation_protocol: str = "waypoint_path"
    path_name: str = "circle_yz"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.configure_path(self.path_name)

    configure_path = Fr3FrankyVelocityImpedanceCirclePathEnvCfg.configure_path
