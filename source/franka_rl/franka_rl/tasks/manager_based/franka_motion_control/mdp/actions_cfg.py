from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.configclass import configclass


@configclass
class JointMotionActionCfg(ActionTermCfg):
    class_type: str = "{DIR}.actions:JointMotionAction"
    asset_name: str = "robot"
    joint_names: list[str] = [f"panda_joint{i}" for i in range(1, 8)]
    stiffness: tuple = (100.,)*7
    damping: tuple = (20.,)*7
    lower: tuple = ()
    upper: tuple = ()
    abort_lower: tuple = ()
    abort_upper: tuple = ()
    max_reference_velocity: tuple = (.1,)*7
    # Simulation candidate abort threshold; not the former PPO 20% envelope.
    max_measured_velocity: tuple = (1.31,)*4+(2.63, 2.09, 2.63)
    effort_limits_nm: tuple = (87.,)*4+(12.,)*3
    position_error_clip: float = .5
    slew_rate_nm_s: float = 1000.
    cutoff_hz: float = 100.
    contact_threshold_n: float = 1.


@configclass
class CartesianMotionActionCfg(JointMotionActionCfg):
    class_type: str = "{DIR}.actions:CartesianMotionAction"
    stiffness: tuple = (100.,)*3+(10.,)*3
    damping: tuple = (20.,)*3+(6.5,)*3
    cartesian_lower: tuple = (.35,-.2,.2)
    cartesian_upper: tuple = (.6,.2,.55)
    max_reference_linear_velocity: float = .1
    max_reference_angular_velocity: float = .5
