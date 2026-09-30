"""Gym registration for the deployment-faithful FR3 impedance task."""

import gymnasium as gym


gym.register(
    id="Franka-FR3v2-FrankyImpedance-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.impedance_env_cfg:Fr3FrankyImpedanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_impedance_ppo_cfg:ImpedancePPORunnerCfg",
    },
)


gym.register(
    id="Franka-FR3v2-FrankyImpedance-Reach-FineSettle-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.impedance_env_cfg:Fr3FrankyImpedanceFineSettleEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.rsl_rl_impedance_ppo_cfg:FineSettleImpedancePPORunnerCfg"
        ),
    },
)

