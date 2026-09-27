"""Robot variant isolation and flange-relative payload scenario tests."""

from pathlib import Path

import pytest
from franka_rl.tasks.manager_based.franka_impedance.impedance_env_cfg import Fr3FrankyImpedanceEnvCfg
from franka_rl.tasks.manager_based.franka_impedance.rsl_rl_impedance_ppo_cfg import (
    HOME_ACTION,
    ImpedancePPORunnerCfg,
)
from franka_rl.tasks.manager_based.franka_incremental_impedance.incremental_impedance_env_cfg import (
    Fr3FrankyIncremental6DImpedanceEnvCfg,
    Fr3FrankyIncrementalImpedanceEnvCfg,
)
from franka_rl.tasks.manager_based.franka_incremental_impedance.rsl_rl_incremental_impedance_ppo_cfg import (
    Incremental6DImpedancePPORunnerCfg,
    IncrementalImpedancePPORunnerCfg,
)
from franka_rl.tasks.manager_based.franka_rl import mdp
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


def test_impedance_task_uses_hardware_timing_limits_and_gain_dr(tmp_path, monkeypatch):
    asset = tmp_path / "test.usda"
    asset.write_text("#usda 1.0\n")
    monkeypatch.setenv("FRANKA_RL_FR3_USD", str(asset))
    cfg = Fr3FrankyImpedanceEnvCfg()

    assert cfg.sim.dt == pytest.approx(0.001)
    assert cfg.decimation == 20
    assert cfg.control_contract == "fr3_franky_direct_joint_impedance_v3_bounded_smooth"
    assert cfg.rewards.action_magnitude is None
    assert cfg.rewards.action_rate is None
    assert cfg.rewards.position_command_difference.weight == pytest.approx(-2.0e-4)
    assert cfg.rewards.measured_velocity_envelope.weight == pytest.approx(-2.0e-4)
    assert cfg.curriculum.position_command_difference.params == {
        "term_name": "position_command_difference", "weight": -2.0e-3, "num_steps": 2000
    }
    assert cfg.curriculum.measured_velocity_envelope.params == {
        "term_name": "measured_velocity_envelope", "weight": -2.0e-3, "num_steps": 2000
    }
    assert cfg.rewards.action_clipping_overshoot.weight == pytest.approx(-1.0e-1)
    assert cfg.rewards.joint_velocity is None
    assert cfg.actions.arm_action.max_measured_velocity[0] == pytest.approx(0.435)
    assert cfg.terminations.joint_velocity_limit is None
    assert cfg.scene.robot.actuators["panda_shoulder"].stiffness == 0.0
    assert cfg.scene.robot.actuators["panda_forearm"].damping == 0.0
    assert cfg.scene.robot.actuators["panda_shoulder"].armature == pytest.approx(0.003)
    assert cfg.scene.robot.actuators["panda_forearm"].armature == pytest.approx(0.003)
    assert cfg.scene.arm_contacts.history_length == cfg.decimation
    assert cfg.rewards.self_collision.weight == -1.0
    assert cfg.rewards.joint_7_posture.weight == pytest.approx(-1.0e-2)
    assert cfg.rewards.joint_7_posture.params["asset_cfg"].joint_names == ["panda_joint7"]

    runner_cfg = ImpedancePPORunnerCfg()
    distribution_cfg = runner_cfg.actor.distribution_cfg
    assert distribution_cfg.class_name.endswith(":SquashedGaussianDistribution")
    assert distribution_cfg.init_std == pytest.approx(0.10)
    assert distribution_cfg.std_type == "log"
    assert tuple(distribution_cfg.initial_action) == HOME_ACTION

    catalog = ScenarioCatalog.from_yaml()
    metadata = ScenarioModifier(catalog.get("fr3_impedance_gain_dr_v1"), catalog).apply(cfg)
    assert cfg.actions.arm_action.gain_alpha_range == (0.5, 2.0)
    assert metadata["configured_values"]["gain_alpha_range"] == (0.5, 2.0)


def test_incremental_impedance_task_uses_50hz_bounded_integrator(tmp_path, monkeypatch):
    asset = tmp_path / "test.usda"
    asset.write_text("#usda 1.0\n")
    monkeypatch.setenv("FRANKA_RL_FR3_USD", str(asset))
    cfg = Fr3FrankyIncrementalImpedanceEnvCfg()

    assert cfg.sim.dt == pytest.approx(0.001)
    assert cfg.decimation == 20
    assert cfg.control_contract == "fr3_franky_incremental_joint_impedance_v1"
    assert cfg.actions.arm_action.class_type.endswith(":FrankyIncrementalImpedanceAction")
    assert cfg.actions.arm_action.max_reference_velocity == pytest.approx(
        (0.435, 0.435, 0.435, 0.435, 0.522, 0.522, 0.522)
    )
    assert cfg.observations.policy.previous_action is None
    assert cfg.observations.policy.position_reference.func is mdp.normalized_position_reference
    assert cfg.observations.policy.previous_increment_action.func is mdp.incremental_position_action
    assert cfg.rewards.position_command_difference is None
    assert cfg.rewards.reference_acceleration.weight == pytest.approx(-1.0e-3)
    assert cfg.rewards.measured_velocity_envelope.weight == pytest.approx(-2.0e-3)
    assert cfg.curriculum is None

    runner = IncrementalImpedancePPORunnerCfg()
    distribution = runner.actor.distribution_cfg
    assert tuple(distribution.initial_action) == (0.0,) * 7
    assert distribution.init_std == pytest.approx(0.50)
    assert runner.experiment_name == "fr3_incremental_impedance_reach"


def test_incremental_6d_task_removes_joint_7_and_adds_fine_reward(tmp_path, monkeypatch):
    asset = tmp_path / "test.usda"
    asset.write_text("#usda 1.0\n")
    monkeypatch.setenv("FRANKA_RL_FR3_USD", str(asset))
    cfg = Fr3FrankyIncremental6DImpedanceEnvCfg()

    assert cfg.control_contract == "fr3_franky_incremental_6d_joint_impedance_v1"
    assert cfg.actions.arm_action.class_type.endswith(":FrankyIncremental6DImpedanceAction")
    assert len(cfg.actions.arm_action.max_reference_velocity) == 6
    assert cfg.rewards.fine_position_tracking.weight == pytest.approx(1.0)
    assert cfg.rewards.fine_position_tracking.params["sigma"] == pytest.approx(0.0025)
    assert cfg.rewards.joint_7_posture is None

    runner = Incremental6DImpedancePPORunnerCfg()
    distribution = runner.actor.distribution_cfg
    assert tuple(distribution.initial_action) == (0.0,) * 6
    assert distribution.init_std == pytest.approx(0.50)
    assert runner.experiment_name == "fr3_incremental_6d_impedance_reach"
