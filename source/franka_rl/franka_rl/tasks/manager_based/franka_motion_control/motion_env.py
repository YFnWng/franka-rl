import torch
import gymnasium as gym
import numpy as np
from isaaclab.envs import ManagerBasedEnv


class Fr3MotionControlEnv(ManagerBasedEnv, gym.Env):
    """Explicit-reset trial environment retaining terminal and first-fault feedback."""
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self.render_mode = None
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.num_envs, 14), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({"state": gym.spaces.Dict({
            key: gym.spaces.Box(-np.inf, np.inf, shape=(self.num_envs, size), dtype=np.float32)
            for key, size in (("joints", 14), ("reference", 14), ("tcp_pose", 7), ("tcp_twist", 6),
                              ("fault_code", 1), ("contact_force", 1))})})
        self.trial_id = None
        self.terminal_state = None

    def capture_state(self):
        robot = self.scene["robot"]
        action = self.action_manager.get_term("joint_reference")
        return {"q": robot.data.joint_pos.torch.clone(), "dq": robot.data.joint_vel.torch.clone(),
            "fault_code": action.fault_code.clone(), "sim_tick": self._sim_step_counter}

    def begin_trial(self, trial_id):
        if self.trial_id is not None:
            raise RuntimeError("End the active trial before starting another")
        self.trial_id = trial_id
        self.terminal_state = None

    def end_trial(self):
        self.action_manager.get_term("joint_reference").check_state()
        self.terminal_state = self.capture_state()
        self.trial_id = None
        return self.terminal_state

    def step(self, action):
        result = super().step(action)
        self.action_manager.get_term("joint_reference").check_state()
        return result

    def reset(self, *args, **kwargs):
        if getattr(self, "trial_id", None) is not None:
            raise RuntimeError("No reset allowed during an active trial")
        return super().reset(*args, **kwargs)
