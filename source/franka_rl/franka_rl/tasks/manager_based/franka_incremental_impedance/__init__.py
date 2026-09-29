"""Gym registration for the 50 Hz incremental-position FR3 task."""

import gymnasium as gym

gym.register(
    id="Franka-FR3v2-FrankyImpedance-IncrementalReach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.incremental_impedance_env_cfg:Fr3FrankyIncrementalImpedanceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.rsl_rl_incremental_impedance_ppo_cfg:IncrementalImpedancePPORunnerCfg"
        ),
    },
)


gym.register(
    id="Franka-FR3v2-FrankyImpedance-Incremental6DReach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.incremental_impedance_env_cfg:Fr3FrankyIncremental6DImpedanceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.rsl_rl_incremental_impedance_ppo_cfg:Incremental6DImpedancePPORunnerCfg"
        ),
    },
)



def _register_reward_ablation(task_id: str, cfg_class: str) -> None:
    """Register one Axis-2 reward ablation with the common PPO contract."""
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.incremental_impedance_env_cfg:{cfg_class}",
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.rsl_rl_incremental_impedance_ppo_cfg:Incremental6DImpedancePPORunnerCfg"
            ),
        },
    )


_register_reward_ablation(
    "Franka-FR3v2-FrankyImpedance-Incremental6DReach-NoFineReward-v0",
    "Fr3FrankyIncremental6DNoFineRewardEnvCfg",
)
_register_reward_ablation(
    "Franka-FR3v2-FrankyImpedance-Incremental6DReach-NoReferenceAcceleration-v0",
    "Fr3FrankyIncremental6DNoReferenceAccelerationEnvCfg",
)
_register_reward_ablation(
    "Franka-FR3v2-FrankyImpedance-Incremental6DReach-NoNearTargetVelocity-v0",
    "Fr3FrankyIncremental6DNoNearTargetVelocityEnvCfg",
)
_register_reward_ablation(
    "Franka-FR3v2-FrankyImpedance-Incremental6DReach-NoConstraintShaping-v0",
    "Fr3FrankyIncremental6DNoConstraintShapingEnvCfg",
)



gym.register(
    id="Franka-FR3v2-FrankyImpedance-Incremental6DReach-DR-Warmup-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.incremental_impedance_env_cfg:Fr3FrankyIncremental6DCurriculumWarmupEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.rsl_rl_incremental_impedance_ppo_cfg:Incremental6DDeploymentDRPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Franka-FR3v2-FrankyImpedance-Incremental6DReach-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.incremental_impedance_env_cfg:Fr3FrankyIncremental6DDeploymentDREnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.rsl_rl_incremental_impedance_ppo_cfg:Incremental6DDeploymentDRPPORunnerCfg"
        ),
    },
)
