# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from . import mdp

##
# Pre-defined configs
##
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import FRANKA_PANDA_CFG  # isort:skip

FRANKA_RL_PANDA_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_RL_PANDA_CFG.spawn.usd_path = (
    f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
)

##
# Scene definition
##


@configclass
class FrankaRlSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = FRANKA_RL_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
        )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class CommandsCfg:
    """Cartesian targets for the Franka hand."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        position_success_threshold=0.03,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.60),
            pos_y=(-0.20, 0.20),
            pos_z=(0.20, 0.50),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(0.0, 0.0),
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
        )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["panda_joint.*"],
                )
            },
            )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["panda_joint.*"],
                )
            },
            )

        ee_position_error = ObsTerm(
            func=mdp.ee_position_error_b,
            params={
                "command_name": "ee_pose",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["panda_hand"],
                ),
            },
        )

        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["panda_joint.*"],
            ),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for Cartesian reaching."""

    position_tracking = RewTerm(
        func=mdp.position_tracking_exp,
        weight=1.0,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["panda_hand"],
            ),
            "sigma": 0.05,
        },
    )

    action_magnitude = RewTerm(
        func=mdp.action_l2,
        weight=-1.0e-4,
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-1.0e-3,
    )

    joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["panda_joint.*"],
            )
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Success, commented out to prevent reward hacking by staying out until timeout
    # reached_target = DoneTerm(
    #     func=mdp.SustainedPositionSuccess,
    #     time_out=False,
    #     params={
    #         "command_name": "ee_pose",
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=["panda_hand"],
    #         ),
    #         "distance_threshold": 0.03,
    #         "required_steps": 5,
    #     },
    # )

    # Failure
    joint_position_limit = DoneTerm(
        func=mdp.joint_pos_out_of_limit,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["panda_joint.*"],
            )
        },
    )

    joint_velocity_limit = DoneTerm(
        func=mdp.joint_vel_out_of_manual_limit,
        time_out=False,
        params={
            "max_velocity": 5.0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["panda_joint.*"],
            )
        },
    )

    non_finite_state = DoneTerm(
        func=mdp.non_finite_joint_state,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["panda_joint.*"],
            )
        },
    )

    # Timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class FrankaRlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: FrankaRlSceneCfg = FrankaRlSceneCfg(num_envs=4, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 6.0
        # viewer settings
        self.viewer.eye = (3.0, 3.0, 2.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
