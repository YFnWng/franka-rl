"""Lightweight configuration for the deployment-oriented impedance action."""

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.configclass import configclass


@configclass
class FrankyImpedanceActionCfg(ActionTermCfg):
    """Configuration for full-range held goals and explicit Franky-like torque."""

    class_type: str = "{DIR}.impedance_actions:FrankyImpedanceAction"
    joint_names: list[str] = MISSING
    preserve_order: bool = True
    nominal_stiffness: float = 100.0
    gain_alpha_range: tuple[float, float] | None = None
    position_error_clip: float = 0.5
    torque_slew_rate: float = 1000.0
    command_filter_cutoff_hz: float = 100.0

    model_lower: tuple[float, ...] = (
        -2.9007400166666666,
        -1.8360900166666667,
        -2.9007400166666666,
        -3.077020016666667,
        -2.87630335,
        0.43982265,
        -3.05083335,
    )
    model_upper: tuple[float, ...] = (
        2.9007400166666666,
        1.8360900166666667,
        2.9007400166666666,
        -0.11693708333333333,
        2.87630335,
        4.62163335,
        3.05083335,
    )
    soft_lower: tuple[float, ...] = (
        -2.610666015,
        -1.652481015,
        -2.610666015,
        -2.929015870,
        -2.588673015,
        0.648913185,
        -2.745750015,
    )
    soft_upper: tuple[float, ...] = (
        2.610666015,
        1.652481015,
        2.610666015,
        -0.264941230,
        2.588673015,
        4.412542815,
        2.745750015,
    )

    # Qualification limit on measured motion. This is not a command governor.
    max_measured_velocity: tuple[float, ...] = (
        0.435,
        0.435,
        0.435,
        0.435,
        0.522,
        0.522,
        0.522,
    )
    joint_limit_activation_distance: float = 0.1
    joint_limit_stiffness: float = 4.0
    joint_limit_damping: float = 1.0
    joint_limit_max_torque: float = 5.0

    compensate_coriolis: bool = True
    compensate_gravity: bool = True


@configclass
class FrankyIncrementalImpedanceActionCfg(FrankyImpedanceActionCfg):
    """50 Hz bounded position increments for the Franky torque servo."""

    class_type: str = "{DIR}.impedance_actions:FrankyIncrementalImpedanceAction"
    max_reference_velocity: tuple[float, ...] = (
        0.435, 0.435, 0.435, 0.435, 0.522, 0.522, 0.522,
    )
    # This normalizes a learned smoothness cost; it is not a hidden limiter.
    max_reference_acceleration: tuple[float, ...] = (
        3.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.0,
    )


@configclass
class FrankyIncremental6DImpedanceActionCfg(FrankyImpedanceActionCfg):
    """Six bounded increments for joints 1--6 while joint 7 is held."""

    class_type: str = "{DIR}.impedance_actions:FrankyIncremental6DImpedanceAction"
    max_reference_velocity: tuple[float, ...] = (
        0.435, 0.435, 0.435, 0.435, 0.522, 0.522,
    )
    max_reference_acceleration: tuple[float, ...] = (
        3.0, 1.5, 2.0, 2.5, 3.0, 4.0,
    )


@configclass
class FrankyVelocityReference6DImpedanceActionCfg(FrankyImpedanceActionCfg):
    """Six normalized velocity references integrated by the 1 kHz controller."""

    class_type: str = "{DIR}.impedance_actions:FrankyVelocityReference6DImpedanceAction"
    max_reference_velocity: tuple[float, ...] = (
        0.435, 0.435, 0.435, 0.435, 0.522, 0.522,
    )
    # Reward normalization only; this is deliberately not a hidden limiter.
    max_reference_acceleration: tuple[float, ...] = (
        3.0, 1.5, 2.0, 2.5, 3.0, 3.0,
    )
