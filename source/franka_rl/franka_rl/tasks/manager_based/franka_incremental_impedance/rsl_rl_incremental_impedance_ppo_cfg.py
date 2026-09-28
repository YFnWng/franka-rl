"""PPO defaults for the 50 Hz incremental-position impedance task."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from ..franka_rl.agents.rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class IncrementalSquashedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    class_name: str = "franka_rl.utils.rsl_rl_distributions:SquashedGaussianDistribution"
    initial_action: tuple[float, ...] = (0.0,) * 7
    epsilon: float = 1.0e-6


@configclass
class IncrementalImpedancePPORunnerCfg(PPORunnerCfg):
    num_steps_per_env = 40
    experiment_name = "fr3_incremental_impedance_reach"

    def __post_init__(self) -> None:
        self.actor.distribution_cfg = IncrementalSquashedGaussianDistributionCfg(
            init_std=0.50,
            std_type="log",
        )
        self.algorithm.gamma = 0.99 ** (30.0 / 50.0)
        self.algorithm.lam = 0.95 ** (30.0 / 50.0)


@configclass
class Incremental6DSquashedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    class_name: str = "franka_rl.utils.rsl_rl_distributions:SquashedGaussianDistribution"
    initial_action: tuple[float, ...] = (0.0,) * 6
    epsilon: float = 1.0e-6


@configclass
class Incremental6DImpedancePPORunnerCfg(PPORunnerCfg):
    num_steps_per_env = 40
    experiment_name = "fr3_incremental_6d_impedance_reach"

    def __post_init__(self) -> None:
        self.actor.distribution_cfg = Incremental6DSquashedGaussianDistributionCfg(
            init_std=0.50,
            std_type="log",
        )
        self.algorithm.gamma = 0.99 ** (30.0 / 50.0)
        self.algorithm.lam = 0.95 ** (30.0 / 50.0)



@configclass
class Incremental6DDeploymentDRPPORunnerCfg(Incremental6DImpedancePPORunnerCfg):
    """Separate logging namespace for the deployment-focused DR policy."""

    experiment_name = "fr3_incremental_6d_impedance_reach_dr"
