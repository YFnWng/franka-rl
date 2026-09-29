"""Tests for the reusable Axis-3 curriculum comparison generator."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import yaml

from franka_rl.experiments import EvaluationSuiteConfig
from franka_rl.tasks.manager_based.franka_incremental_impedance.incremental_impedance_env_cfg import (
    Fr3FrankyIncremental6DImpedanceEnvCfg,
)
from franka_rl.utils.scenarios import ScenarioCatalog, ScenarioModifier


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "experiments"
    / "run_axis3_curriculum_comparison.py"
)
SPEC = spec_from_file_location("run_axis3_curriculum_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_generates_valid_matched_suite_and_fixed_gain_scenarios(tmp_path):
    nominal = tmp_path / "model_149.pt"
    dr = tmp_path / "model_297.pt"
    run = tmp_path / "run"
    nominal.touch()
    dr.touch()
    run.mkdir()
    manifest_path = tmp_path / "curriculum_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "target_seeds": [42],
                "seeds": {
                    "42": {
                        "source_checkpoint": str(nominal),
                        "stages": {
                            "full": {
                                "complete": True,
                                "checkpoint": str(dr),
                                "run_dir": str(run),
                            }
                        },
                    }
                },
            }
        )
    )

    manifest = MODULE.load_curriculum_manifest(manifest_path)
    scenario_file = tmp_path / "scenarios.yaml"
    MODULE.write_scenario_catalog(scenario_file)
    document, identities = MODULE.suite_document(
        manifest,
        scenario_file,
        num_envs=8,
        episodes_per_job=16,
        evaluation_seed=123,
        device="cpu",
    )
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(yaml.safe_dump(document, sort_keys=False))

    scenarios = yaml.safe_load(scenario_file.read_text())["scenarios"]
    catalog = ScenarioCatalog.from_yaml(scenario_file)
    parsed = EvaluationSuiteConfig.from_yaml(suite_file)
    assert scenarios["gain_alpha_050"]["control"]["impedance_gain_alpha"] == 0.5
    assert scenarios["gain_alpha_200"]["control"]["impedance_gain_alpha"] == 2.0
    assert catalog.get("gain_alpha_050").control.impedance_gain_alpha == 0.5
    assert set(identities) == {"nominal_seed_42", "curriculum_dr_seed_42"}
    assert len(parsed.policies) == 2
    assert parsed.scenarios == MODULE.SCENARIOS
    assert parsed.evaluation.episodes_per_job == 16


def test_fixed_gain_scenario_reaches_incremental_action_cfg(tmp_path, monkeypatch):
    asset = tmp_path / "test.usda"
    asset.write_text("#usda 1.0\n")
    monkeypatch.setenv("FRANKA_RL_FR3_USD", str(asset))
    scenario_file = tmp_path / "scenarios.yaml"
    MODULE.write_scenario_catalog(scenario_file)
    catalog = ScenarioCatalog.from_yaml(scenario_file)
    cfg = Fr3FrankyIncremental6DImpedanceEnvCfg()

    metadata = ScenarioModifier(catalog.get("gain_alpha_050"), catalog).apply(cfg)

    assert cfg.actions.arm_action.gain_alpha_range == (0.5, 0.5)
    assert metadata["configured_values"]["gain_alpha_range"] == (0.5, 0.5)
