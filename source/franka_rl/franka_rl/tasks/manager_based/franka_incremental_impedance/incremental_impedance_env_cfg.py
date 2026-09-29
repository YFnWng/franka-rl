"""FR3 reaching with 50 Hz incremental position references."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ..franka_impedance.impedance_env_cfg import Fr3FrankyImpedanceEnvCfg
from ..franka_rl import mdp


@configclass
class Fr3FrankyIncrementalImpedanceEnvCfg(Fr3FrankyImpedanceEnvCfg):
    """Deployment task whose action is a bounded joint-position increment.

    The 50 Hz policy output is interpreted as a normalized reference velocity,
    integrated into a persistent position reference, and held by the same 1 kHz
    explicit Franky impedance torque controller as the hardware runtime.
    """

    control_contract: str = "fr3_franky_incremental_joint_impedance_v1"
    controller_profile: str = "franky_k100_d20_incremental_50hz_20pct_velocity"
    curriculum = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.arm_action = mdp.FrankyIncrementalImpedanceActionCfg(
            asset_name="robot",
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
            preserve_order=True,
        )

        # Markov state for the action integrator: measured state (14), Cartesian
        # error (3), held reference (7), and preceding increment action (7).
        self.observations.policy.previous_action = None
        self.observations.policy.position_reference = ObsTerm(
            func=mdp.normalized_position_reference,
            params={"action_name": "arm_action"},
        )
        self.observations.policy.previous_increment_action = ObsTerm(
            func=mdp.incremental_position_action,
            params={"action_name": "arm_action"},
        )

        # Increment size is hard-bounded. Smoothness is learned as physical
        # reference acceleration instead of penalizing raw actor magnitude.
        self.rewards.position_command_difference = None
        self.rewards.reference_acceleration = RewTerm(
            func=mdp.normalized_reference_acceleration_l2,
            weight=-1.0e-3,
            params={"action_name": "arm_action"},
        )
        self.rewards.near_target_reference_velocity = RewTerm(
            func=mdp.near_target_reference_velocity_l2,
            weight=-2.0e-3,
            params={
                "action_name": "arm_action",
                "threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", body_names=["fr3_flange"]),
            },
        )
        self.rewards.reference_projection = RewTerm(
            func=mdp.reference_projection_overshoot_l2,
            weight=-1.0e-1,
            params={"action_name": "arm_action"},
        )
        velocity_envelope = self.actions.arm_action.max_measured_velocity
        self.rewards.measured_velocity_envelope = RewTerm(
            func=mdp.normalized_measured_velocity_violation_l2,
            weight=-2.0e-3,
            params={
                "action_name": "arm_action",
                "max_velocity": velocity_envelope,
            },
        )


@configclass
class Fr3FrankyIncremental6DImpedanceEnvCfg(Fr3FrankyIncrementalImpedanceEnvCfg):
    """Six-action position reach task with a narrow terminal reward."""

    control_contract: str = "fr3_franky_incremental_6d_joint_impedance_v1"
    controller_profile: str = "franky_k100_d20_incremental_6d_50hz_20pct_velocity"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.arm_action = mdp.FrankyIncremental6DImpedanceActionCfg(
            asset_name="robot",
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
            preserve_order=True,
        )

        # The two action-dependent terms now expose six values each, yielding
        # a 29D policy observation. Joint 7 remains measured but not commanded.
        self.observations.policy.position_reference = ObsTerm(
            func=mdp.normalized_position_reference,
            params={"action_name": "arm_action"},
        )
        self.observations.policy.previous_increment_action = ObsTerm(
            func=mdp.incremental_position_action,
            params={"action_name": "arm_action"},
        )

        # Keep the original broad kernel for approach and add only one narrow
        # kernel for terminal precision: sqrt(0.0025) = 0.05 m.
        self.rewards.fine_position_tracking = RewTerm(
            func=mdp.position_tracking_exp,
            weight=1.0,
            params={
                "command_name": "ee_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names=["fr3_flange"]),
                "sigma": 0.0025,
            },
        )
        # Joint 7 is held by construction, so an uncontrollable posture reward
        # would add a reset-dependent constant rather than a learning signal.
        self.rewards.joint_7_posture = None



@configclass
class Fr3FrankyIncremental6DNoFineRewardEnvCfg(Fr3FrankyIncremental6DImpedanceEnvCfg):
    """Axis-2 ablation without the narrow terminal-position kernel."""

    reward_ablation: str = "no_fine_tracking"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.fine_position_tracking = None


@configclass
class Fr3FrankyIncremental6DNoReferenceAccelerationEnvCfg(Fr3FrankyIncremental6DImpedanceEnvCfg):
    """Axis-2 ablation without physical reference-acceleration shaping."""

    reward_ablation: str = "no_reference_acceleration"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.reference_acceleration = None


@configclass
class Fr3FrankyIncremental6DNoNearTargetVelocityEnvCfg(Fr3FrankyIncremental6DImpedanceEnvCfg):
    """Axis-2 ablation without the near-target settling objective."""

    reward_ablation: str = "no_near_target_velocity"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.near_target_reference_velocity = None


@configclass
class Fr3FrankyIncremental6DNoConstraintShapingEnvCfg(Fr3FrankyIncremental6DImpedanceEnvCfg):
    """Axis-2 ablation of velocity, clipping, and reference-bound shaping."""

    reward_ablation: str = "no_constraint_shaping"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.measured_velocity_envelope = None
        self.rewards.action_clipping_overshoot = None
        self.rewards.reference_projection = None



@configclass
class Fr3FrankyIncremental6DCurriculumWarmupEnvCfg(Fr3FrankyIncremental6DImpedanceEnvCfg):
    """Six-axis incremental task for the mild first DR curriculum stage."""

    required_training_scenario: str = "fr3_incremental_deployment_dr_warmup_v1"
    domain_randomization_contract: str = "gain_payload_curriculum_warmup_v1"


@configclass
class Fr3FrankyIncremental6DDeploymentDREnvCfg(Fr3FrankyIncremental6DImpedanceEnvCfg):
    """Six-axis incremental task reserved for the focused deployment DR scenario."""

    required_training_scenario: str = "fr3_incremental_deployment_dr_v1"
    domain_randomization_contract: str = "gain_delay_payload_v1"
