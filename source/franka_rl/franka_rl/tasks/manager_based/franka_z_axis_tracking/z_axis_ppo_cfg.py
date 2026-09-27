"""PPO configuration for position plus tip-z-axis reaching."""

from isaaclab.utils.configclass import configclass

from ..franka_incremental_impedance.rsl_rl_incremental_impedance_ppo_cfg import (
    Incremental6DImpedancePPORunnerCfg,
)


@configclass
class Incremental6DPositionZAxisPPORunnerCfg(Incremental6DImpedancePPORunnerCfg):
    experiment_name = "fr3_incremental_6d_position_z_axis_reach"
