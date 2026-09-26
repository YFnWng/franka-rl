"""Robot variant isolation and flange-relative payload scenario tests."""

from pathlib import Path

import pytest
from franka_rl.tasks.manager_based.franka_rl.fr3_env_cfg import Fr3BareFlangeEnvCfg
from franka_rl.tasks.manager_based.franka_rl.franka_rl_env_cfg import FrankaRlEnvCfg
from franka_rl.tasks.manager_based.franka_velocity.velocity_env_cfg import Fr3JointVelocityEnvCfg
from franka_rl.utils.scenarios import ScenarioCatalog, ScenarioModifier


@pytest.fixture
def fr3(tmp_path, monkeypatch):
    asset = tmp_path / "test.usda"
    asset.write_text("#usda 1.0\n")
    monkeypatch.setenv("FRANKA_RL_FR3_USD", str(asset))
    return Fr3BareFlangeEnvCfg()


def test_variant_preserves_policy_and_panda(fr3):
    panda = FrankaRlEnvCfg()
    assert fr3.commands.ee_pose.body_name == "fr3_flange"
    assert panda.commands.ee_pose.body_name == "panda_hand"
    assert len(fr3.scene.robot.init_state.joint_pos) == 7
    assert "panda_hand" not in fr3.scene.robot.actuators
    assert "panda_hand" in panda.scene.robot.actuators
    assert fr3.observations.policy.ee_position_error.params["asset_cfg"].body_names == ["fr3_flange"]
    assert panda.observations.policy.ee_position_error.params["asset_cfg"].body_names == ["panda_hand"]
    assert fr3.actions.arm_action.scale == 0.5
    assert fr3.decimation * fr3.sim.dt == pytest.approx(1 / 30)


def test_payload_scenarios_use_flange_origin(fr3):
    catalog = ScenarioCatalog.from_yaml(
        Path(__file__).resolve().parents[2] / "source/franka_rl/franka_rl/config/fr3_scenarios.yaml"
    )
    for name in ["fr3_payload_050", "fr3_payload_100_z005", "fr3_payload_dr"]:
        cfg = fr3.copy()
        ScenarioModifier(catalog.get(name), catalog).apply(cfg)
        params = cfg.events.scenario_payload_mass.params
        assert params["asset_cfg"].body_names == ["fr3_flange"]
        assert params["reference_body_origin"] is True
    panda = FrankaRlEnvCfg()
    ScenarioModifier(catalog.get("fr3_payload_050"), catalog).apply(panda)
    assert panda.events.scenario_payload_mass.params["reference_body_origin"] is False


def test_velocity_dr_configures_only_velocity_relevant_controls(tmp_path, monkeypatch):
    asset = tmp_path / "test.usda"
    asset.write_text("#usda 1.0\n")
    monkeypatch.setenv("FRANKA_RL_FR3_USD", str(asset))
    cfg = Fr3JointVelocityEnvCfg()
    catalog = ScenarioCatalog.from_yaml()
    modifier = ScenarioModifier(catalog.get("fr3_velocity_dr_train_v1"), catalog)

    metadata = modifier.apply(cfg)

    assert cfg.actions.arm_action.velocity_target_scale_range == (0.85, 1.15)
    assert cfg.actions.arm_action.acceleration_scale_range == (0.75, 1.25)
    assert getattr(cfg.events, "scenario_damping") is not None
    assert getattr(cfg.events, "scenario_actuator_gains", None) is None
    assert getattr(cfg.events, "scenario_effort_limits", None) is None
    assert metadata["configured_values"]["velocity_target_scale_range"] == (0.85, 1.15)
    assert metadata["configured_values"]["acceleration_scale_range"] == (0.75, 1.25)
