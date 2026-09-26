"""RSL-RL PPO configuration for the FR3 joint-velocity reaching task."""

from isaaclab.utils.configclass import configclass

from ..franka_rl.agents.rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class VelocityPPORunnerCfg(PPORunnerCfg):
    experiment_name = "fr3_velocity_reach"
    clip_actions = 1.0
