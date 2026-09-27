"""FR3 reaching task with position and tip-z-axis targets."""

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ..franka_incremental_impedance.incremental_impedance_env_cfg import (
    Fr3FrankyIncremental6DImpedanceEnvCfg,
)
from ..franka_rl import mdp


@configclass
class Fr3FrankyIncremental6DPositionZAxisEnvCfg(
    Fr3FrankyIncremental6DImpedanceEnvCfg
):
    """Track a Cartesian position and a two-DoF flange z-axis direction.

    Joint 7 remains held because rotation about the flange z-axis does not
    change the controlled direction. The target z-axis is sampled in a cone
    around the nominal downward direction.
    """

    control_contract: str = "fr3_franky_incremental_6d_position_z_axis_v1"
    controller_profile: str = "franky_k100_d20_incremental_6d_50hz_20pct_velocity"
    z_axis_component_limit_rad: float = math.radians(30.0)

    def __post_init__(self) -> None:
        super().__post_init__()

        tilt = self.z_axis_component_limit_rad
        if not 0.0 < tilt < 0.5 * math.pi:
            raise ValueError("z_axis_component_limit_rad must be between 0 and pi/2")
        self.commands.ee_pose.ranges.roll = (-tilt, tilt)
        self.commands.ee_pose.ranges.pitch = (math.pi - tilt, math.pi + tilt)
        # Rotation about target z does not change the controlled direction.
        self.commands.ee_pose.ranges.yaw = (0.0, 0.0)

        flange = SceneEntityCfg("robot", body_names=["fr3_flange"])
        self.observations.policy.ee_z_axis_error = ObsTerm(
            func=mdp.ee_z_axis_error_b,
            params={
                "command_name": "ee_pose",
                "asset_cfg": flange,
            },
        )

        # Broad and fine kernels mirror the position-reward structure. Sigma
        # is in rad^2: characteristic errors are 0.5 rad and 0.1 rad.
        self.rewards.z_axis_tracking = RewTerm(
            func=mdp.z_axis_tracking_exp,
            weight=1.0,
            params={
                "command_name": "ee_pose",
                "asset_cfg": flange,
                "sigma": 0.25,
            },
        )
        self.rewards.fine_z_axis_tracking = RewTerm(
            func=mdp.z_axis_tracking_exp,
            weight=1.0,
            params={
                "command_name": "ee_pose",
                "asset_cfg": flange,
                "sigma": 0.01,
            },
        )
