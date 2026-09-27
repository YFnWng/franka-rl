"""PPO defaults for the deployment velocity-reference impedance task."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from ..franka_rl.agents.rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class VelocitySquashedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    class_name: str = "franka_rl.utils.rsl_rl_distributions:SquashedGaussianDistribution"
    initial_action: tuple[float, ...] = (0.0,) * 6
    epsilon: float = 1.0e-6


@configclass
class VelocityImpedancePPORunnerCfg(PPORunnerCfg):
    num_steps_per_env = 40
    experiment_name = "fr3_velocity_impedance_reach"

    def __post_init__(self) -> None:
        self.actor.distribution_cfg = VelocitySquashedGaussianDistributionCfg(
            init_std=0.50,
            std_type="log",
        )
        self.algorithm.gamma = 0.99 ** (30.0 / 50.0)
        self.algorithm.lam = 0.95 ** (30.0 / 50.0)


@configclass
class VelocityImpedancePositionZAxisPPORunnerCfg(VelocityImpedancePPORunnerCfg):
    experiment_name = "fr3_velocity_impedance_position_z_axis_reach"
