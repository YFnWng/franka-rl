"""Independent non-RL FR3 scene; no reach rewards, commands, or episode resets."""
import hashlib
import os
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets import FRANKA_PANDA_CFG
from isaaclab_physx.sensors import ContactSensorCfg
from franka_motion.config import load_motion_suite
from . import mdp


def make_fr3_robot_cfg(workspace=None):
    workspace = load_motion_suite().workspace.data if workspace is None else workspace
    root = Path(os.environ.get("FRANKA_RL_DATA_ROOT", "/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data"))
    path = Path(os.environ.get("FRANKA_RL_FR3_USD", str(root / workspace["model"]["asset_relative_to_data_root"])))
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != workspace["model"]["usd_sha256"]:
        raise FileNotFoundError(f"Missing/hash-mismatched audited FR3 live asset: {path}")
    robot = FRANKA_PANDA_CFG.copy()
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    robot.spawn.usd_path = str(path)
    robot.spawn.rigid_props.disable_gravity = False
    robot.spawn.articulation_props.enabled_self_collisions = True
    robot.spawn.activate_contact_sensors = True
    robot.init_state.joint_pos = dict(zip(workspace["joint_names"], workspace["initial_postures_rad"]["deployment_home"]))
    robot.init_state.joint_vel = {".*": 0.}
    robot.actuators.pop("panda_hand", None)
    for actuator in robot.actuators.values():
        actuator.stiffness = actuator.damping = 0.
        actuator.armature = workspace["model"]["armature_kg_m2"]
    return robot


@configclass
class Fr3MotionSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100., 100.)))
    robot: ArticulationCfg = make_fr3_robot_cfg()
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=500.))
    arm_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/(fr3_link[1-6]|fr3_flange)",
                                    update_period=.001, history_length=1)


@configclass
class MotionStateObservationsCfg:
    @configclass
    class StateCfg(ObservationGroupCfg):
        joints = ObservationTermCfg(func=mdp.measured_joint_state)
        reference = ObservationTermCfg(func=mdp.applied_motion_reference)
        tcp_pose = ObservationTermCfg(func=mdp.tcp_pose_base)
        tcp_twist = ObservationTermCfg(func=mdp.tcp_twist_base)
        fault_code = ObservationTermCfg(func=mdp.motion_trial_status)
        contact_force = ObservationTermCfg(func=mdp.contact_force)
        concatenate_terms: bool = False
        enable_corruption: bool = False
    state: StateCfg = StateCfg()


@configclass
class MotionEventsCfg:
    reset_state = EventTermCfg(func=mdp.reset_motion_state, mode="reset")


@configclass
class MotionActionsCfg:
    joint_reference = mdp.JointMotionActionCfg()


@configclass
class CartesianActionsCfg:
    cartesian_reference = mdp.CartesianMotionActionCfg()


@configclass
class Fr3MotionControlEnvCfg(ManagerBasedEnvCfg):
    scene: Fr3MotionSceneCfg = Fr3MotionSceneCfg(num_envs=1, env_spacing=2.5)
    observations: MotionStateObservationsCfg = MotionStateObservationsCfg()
    actions: MotionActionsCfg = MotionActionsCfg()
    events: MotionEventsCfg = MotionEventsCfg()
    decimation: int = 20
    seed: int = 42
    motion_action_name: str = "joint_reference"

    def __post_init__(self):
        self.sim.dt = .001
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.5, 1.5, 1.2)
        workspace = load_motion_suite().workspace.data
        action = getattr(self.actions, self.motion_action_name)
        action.lower, action.upper = (tuple(workspace["joint_bounds_rad"]["commanded"][x]) for x in ("lower", "upper"))
        action.abort_lower, action.abort_upper = (tuple(workspace["joint_bounds_rad"]["measured_abort"][x]) for x in ("lower", "upper"))


@configclass
class Fr3JointMotionEnvCfg(Fr3MotionControlEnvCfg):
    """M1 physical 14D joint reference task."""


@configclass
class Fr3CartesianMotionEnvCfg(Fr3MotionControlEnvCfg):
    actions: CartesianActionsCfg = CartesianActionsCfg()
    motion_action_name: str = "cartesian_reference"


@configclass
class PostureActionsCfg:
    cartesian_reference = mdp.CartesianPostureMotionActionCfg()


@configclass
class Fr3FixturePostureMotionEnvCfg(Fr3CartesianMotionEnvCfg):
    actions: PostureActionsCfg = PostureActionsCfg()
