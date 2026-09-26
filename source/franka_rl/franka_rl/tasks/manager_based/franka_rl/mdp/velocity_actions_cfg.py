"""Lightweight configuration for governed joint-velocity actions."""

from dataclasses import MISSING

from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg
from isaaclab.utils.configclass import configclass


@configclass
class GovernedJointVelocityActionCfg(JointVelocityActionCfg):
    """Configuration loaded before Kit; the action class is resolved lazily."""

    class_type: str = "{DIR}.velocity_actions:GovernedJointVelocityAction"
    joint_names: list[str] = MISSING
    use_default_offset: bool = False
    normalized_clip: float = 1.0
    max_acceleration: float = 1.0
    braking_acceleration: float = 1.0
    velocity_target_scale_range: tuple[float, float] | None = None
    acceleration_scale_range: tuple[float, float] | None = None
    joint_limit_margin: float = 0.10
    enable_gravity_compensation: bool = True
