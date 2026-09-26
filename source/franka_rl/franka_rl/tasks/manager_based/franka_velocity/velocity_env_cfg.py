"""FR3 reaching task with bounded, gravity-compensated joint velocity actions."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from ..franka_rl import mdp
from ..franka_rl.fr3_env_cfg import Fr3BareFlangeEnvCfg


@configclass
class Fr3JointVelocityEnvCfg(Fr3BareFlangeEnvCfg):
    """Provisional ideal velocity-servo task for simulation validation.

    The action contract is stable, but servo damping and command-transition
    parameters must be updated after the Franky hardware-response experiment.
    """

    control_contract: str = "fr3_reach_joint_velocity_v1"
    controller_profile: str = "ideal_velocity_provisional"

    def __post_init__(self) -> None:
        super().__post_init__()

        # Retain a 30 Hz policy while resolving the velocity servo at 120 Hz.
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation

        # Pure velocity drive plus explicit gravity feed-forward. These values
        # are placeholders pending tomorrow's hardware identification.
        self.scene.robot.actuators["panda_shoulder"].stiffness = 0.0
        self.scene.robot.actuators["panda_shoulder"].damping = 40.0
        self.scene.robot.actuators["panda_forearm"].stiffness = 0.0
        self.scene.robot.actuators["panda_forearm"].damping = 20.0

        self.actions.arm_action = mdp.GovernedJointVelocityActionCfg(
            asset_name="robot",
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
            preserve_order=True,
            scale=0.30,
            use_default_offset=False,
            normalized_clip=1.0,
            max_acceleration=1.0,
            braking_acceleration=1.0,
            joint_limit_margin=0.10,
            enable_gravity_compensation=True,
        )

        # The observation remains 24D. The final seven values now represent
        # accepted velocity actions rather than position actions.
        self.observations.policy.previous_action = ObsTerm(
            func=mdp.accepted_velocity_action,
            params={"action_name": "arm_action"},
        )

        self.rewards.action_magnitude = RewTerm(
            func=mdp.governed_velocity_l2,
            weight=-2.0e-3,
            params={"action_name": "arm_action"},
        )
        self.rewards.action_rate = RewTerm(
            func=mdp.governed_velocity_rate_l2,
            weight=-1.0e-2,
            params={"action_name": "arm_action"},
        )
        self.rewards.joint_velocity.weight = -2.0e-4
        self.rewards.settling_velocity = RewTerm(
            func=mdp.near_target_joint_velocity_l2,
            weight=-2.0e-3,
            params={
                "command_name": "ee_pose",
                "hand_asset_cfg": SceneEntityCfg(
                    "robot", body_names=["fr3_flange"]
                ),
                "joint_asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["panda_joint.*"]
                ),
                "sigma": 0.05,
            },
        )

        # The commanded cap is 0.3 rad/s. A 1 rad/s measured-speed tripwire
        # leaves transient headroom while still catching controller failures.
        self.terminations.joint_velocity_limit.params["max_velocity"] = 1.0
        self.terminations.velocity_action_fault = DoneTerm(
            func=mdp.velocity_action_fault,
            time_out=False,
            params={"action_name": "arm_action"},
        )
