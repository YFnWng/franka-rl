"""FR3 reaching with the reviewed Franky joint-impedance controller contract."""

from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from ..franka_rl import mdp
from ..franka_rl.fr3_env_cfg import Fr3BareFlangeEnvCfg


@configclass
class ImpedanceCurriculumCfg:
    """Delay full safety shaping until the bounded policy learns to reach."""

    position_command_difference = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "position_command_difference", "weight": -2.0e-3, "num_steps": 2000},
    )
    measured_velocity_envelope = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "measured_velocity_envelope", "weight": -2.0e-3, "num_steps": 2000},
    )


@configclass
class Fr3FrankyImpedanceEnvCfg(Fr3BareFlangeEnvCfg):
    """1 kHz torque simulation with direct held position-goal actions.

    The controller implements the measured K=100, D=20 nominal Franky profile.
    Normalized actions cover the complete reviewed soft joint range. Temporal
    smoothness is learned from consecutive physical position commands.
    """

    control_contract: str = "fr3_franky_direct_joint_impedance_v3_bounded_smooth"
    controller_profile: str = "franky_k100_d20_bounded_full_range_safety_rewards"

    curriculum: ImpedanceCurriculumCfg = ImpedanceCurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.dt = 0.001
        self.decimation = 20
        self.sim.render_interval = self.decimation

        # Use the measured deployment home as the reset center rather than
        # inheriting the Panda pose. The midpoint action map is independent.
        deployment_home = (0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.0)
        self.scene.robot.init_state.joint_pos = {
            f"panda_joint{index + 1}": value for index, value in enumerate(deployment_home)
        }

        # The action term owns the full non-gravity torque command.
        for actuator_name in ("panda_shoulder", "panda_forearm"):
            actuator = self.scene.robot.actuators[actuator_name]
            actuator.stiffness = 0.0
            actuator.damping = 0.0
            # The inherited Panda value (0.001 kg*m^2) makes the explicit
            # torque loop numerically unstable at joint 7. The DR lower bound
            # must also retain margin under simultaneous inertia/payload
            # variation, so use 0.003 (DR range: 0.0024--0.0036). This remains
            # a simulation stabilizer, not an identified FR3 motor parameter.
            actuator.armature = 0.003

        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.arm_contacts = PhysXContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(fr3_link[1-6]|fr3_flange)",
            update_period=self.sim.dt,
            history_length=self.decimation,
        )

        self.actions.arm_action = mdp.FrankyImpedanceActionCfg(
            asset_name="robot",
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
            preserve_order=True,
        )

        # Preserve the 24D observation shape and expose the normalized held command.
        self.observations.policy.previous_action = ObsTerm(
            func=mdp.held_position_command,
            params={"action_name": "arm_action"},
        )

        # Penalize only normalized violations of the 20-percent command and measured-speed envelopes.
        self.rewards.action_rate = None
        velocity_envelope = self.actions.arm_action.max_measured_velocity
        self.rewards.position_command_difference = RewTerm(
            func=mdp.normalized_command_step_violation_l2,
            weight=-2.0e-4,
            params={
                "action_name": "arm_action",
                "max_velocity": velocity_envelope,
            },
        )
        self.rewards.measured_velocity_envelope = RewTerm(
            func=mdp.normalized_measured_velocity_violation_l2,
            weight=-2.0e-4,
            params={
                "action_name": "arm_action",
                "max_velocity": velocity_envelope,
            },
        )
        self.rewards.action_clipping_overshoot = RewTerm(
            func=mdp.action_clipping_overshoot_l2,
            weight=-1.0e-1,
            params={"action_name": "arm_action"},
        )
        self.rewards.action_magnitude = None
        self.rewards.joint_velocity = None

        # In this free-space task, any moving-link contact is unsafe. The
        # aggregate signal includes self-contact and unexpected scene contact.
        self.rewards.self_collision = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("arm_contacts"), "threshold": 1.0},
        )

        # Position-only reaching leaves joint 7 nearly unconstrained. Select
        # the deployment-home posture among equivalent Cartesian solutions.
        # Even at the reviewed joint-7 soft limit, this remains secondary to
        # Cartesian tracking.
        self.rewards.joint_7_posture = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-1.0e-2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint7"])},
        )

        arm = SceneEntityCfg("robot", joint_names=["panda_joint.*"])
        cfg = self.actions.arm_action
        self.terminations.joint_position_limit = DoneTerm(
            func=mdp.soft_joint_position_violation,
            time_out=False,
            params={"asset_cfg": arm, "lower": cfg.soft_lower, "upper": cfg.soft_upper},
        )
        # The 20-percent speed envelope is a policy qualification threshold,
        # not an online controller or one-step training termination. Smoothness
        # is learned from command differences and measured joint velocity.
        self.terminations.joint_velocity_limit = None
        self.terminations.impedance_action_fault = DoneTerm(
            func=mdp.impedance_action_fault,
            time_out=False,
            params={"action_name": "arm_action"},
        )
