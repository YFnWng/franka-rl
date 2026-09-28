"""Gym registration for deterministic FR3 waypoint-path evaluation."""

import gymnasium as gym

gym.register(
    id="Franka-FR3v2-FrankyImpedance-CirclePath-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (f"{__name__}.path_env_cfg:Fr3FrankyAbsolutePositionCirclePathEnvCfg"),
        "rsl_rl_cfg_entry_point": (
            "franka_rl.tasks.manager_based.franka_impedance.rsl_rl_impedance_ppo_cfg:ImpedancePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Franka-FR3v2-FrankyImpedance-Incremental6DCirclePath-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (f"{__name__}.path_env_cfg:Fr3FrankyIncremental6DCirclePathEnvCfg"),
        "rsl_rl_cfg_entry_point": (
            "franka_rl.tasks.manager_based.franka_incremental_impedance."
            "rsl_rl_incremental_impedance_ppo_cfg:Incremental6DImpedancePPORunnerCfg"
        ),
    },
)


gym.register(
    id="Franka-FR3v2-FrankyImpedance-Incremental6DPositionZAxisCirclePath-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (f"{__name__}.path_env_cfg:Fr3FrankyIncremental6DPositionZAxisCirclePathEnvCfg"),
        "rsl_rl_cfg_entry_point": (
            "franka_rl.tasks.manager_based.franka_z_axis_tracking.z_axis_ppo_cfg:Incremental6DPositionZAxisPPORunnerCfg"
        ),
    },
)
