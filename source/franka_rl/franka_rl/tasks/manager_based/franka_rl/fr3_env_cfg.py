"""FR3v2 bare flange, preserving the frozen Panda policy's 24D/7D interface."""

import os
from pathlib import Path

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .franka_rl_env_cfg import FrankaRlEnvCfg


@configclass
class Fr3BareFlangeEnvCfg(FrankaRlEnvCfg):
    robot_model: str = "fr3v2_bare_flange"
    dr_arm_body_pattern: str = "fr3_link[1-6]|fr3_flange"
    payload_reference_body_origin: bool = True

    def __post_init__(self):
        super().__post_init__()
        root = Path(os.environ.get("FRANKA_RL_DATA_ROOT", "/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data"))
        asset = Path(
            os.environ.get("FRANKA_RL_FR3_USD", str(root / "generated_assets/fr3v2_bare_flange_v1/fr3v2.usda"))
        )
        if not asset.is_file():
            raise FileNotFoundError(f"Build FR3 asset with scripts/deployment/build_fr3v2_asset.py first: {asset}")
        self.scene.robot = self.scene.robot.copy()
        self.scene.robot.spawn.usd_path = str(asset)
        self.scene.robot.init_state.joint_pos = {
            name: value
            for name, value in self.scene.robot.init_state.joint_pos.items()
            if name.startswith("panda_joint")
        }
        self.scene.robot.actuators.pop("panda_hand", None)
        self.commands.ee_pose.body_name = "fr3_flange"
        for group in [self.observations.policy, self.rewards, self.terminations]:
            for term in vars(group).values():
                if not hasattr(term, "params"):
                    continue
                for value in term.params.values():
                    if isinstance(value, SceneEntityCfg) and value.body_names == ["panda_hand"]:
                        value.body_names = ["fr3_flange"]
