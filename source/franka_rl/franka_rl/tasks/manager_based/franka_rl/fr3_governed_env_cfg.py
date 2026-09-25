"""Opt-in CPU reference governor task; existing FR3/Panda tasks are unchanged."""
import os
from pathlib import Path

from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass
from franka_governor.isaaclab import GovernedJointPositionActionCfg, governor_failed

from .fr3_env_cfg import Fr3BareFlangeEnvCfg


@configclass
class Fr3GovernedEnvCfg(Fr3BareFlangeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        path = os.environ.get("FRANKA_RL_GOVERNOR_CONFIG")
        if not path or not Path(path).is_file():
            raise ValueError("Set FRANKA_RL_GOVERNOR_CONFIG to a versioned governor JSON file")
        self.sim.dt = 1 / 3000
        self.decimation = 100
        self.sim.render_interval = 100
        self.actions.arm_action = GovernedJointPositionActionCfg(
            asset_name="robot",
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
            governor_config=str(Path(path).resolve()),
            allow_simulation_fixture=os.environ.get("FRANKA_RL_GOVERNOR_SIMULATION_FIXTURE") == "1",
        )
        self.terminations.governor_fault = TerminationTermCfg(func=governor_failed)
