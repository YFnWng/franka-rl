"""PPO defaults for the 50 Hz deployment-faithful impedance task."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from ..franka_rl.agents.rsl_rl_ppo_cfg import PPORunnerCfg

HOME_ACTION = (
    0.0,
    -0.47528422793858743,
    0.0,
    -0.5699659677645853,
    0.0,
    -0.510109531263895,
    0.0,
)


@configclass
class SquashedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    class_name: str = "franka_rl.utils.rsl_rl_distributions:SquashedGaussianDistribution"
    initial_action: tuple[float, ...] = HOME_ACTION
    epsilon: float = 1.0e-6


@configclass
class ImpedancePPORunnerCfg(PPORunnerCfg):
    num_steps_per_env = 40
    experiment_name = "fr3_impedance_reach"

    def __post_init__(self) -> None:
        self.actor.distribution_cfg = SquashedGaussianDistributionCfg(
            init_std=0.10,
            std_type="log",
        )
        # Preserve approximately the original discount per wall-clock second.
        self.algorithm.gamma = 0.99 ** (30.0 / 50.0)
        self.algorithm.lam = 0.95 ** (30.0 / 50.0)
