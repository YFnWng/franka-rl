"""Non-RL motion tasks; Cartesian/fixture registrations arrive in M2/M3."""
import gymnasium as gym

gym.register(
    id="Franka-FR3v2-MotionControl-Joint-v0",
    entry_point=f"{__name__}.motion_env:Fr3MotionControlEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.motion_env_cfg:Fr3JointMotionEnvCfg"},
)
gym.register(
    id="Franka-FR3v2-MotionControl-Cartesian-v0",
    entry_point=f"{__name__}.motion_env:Fr3MotionControlEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.motion_env_cfg:Fr3CartesianMotionEnvCfg"},
)
