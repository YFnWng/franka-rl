"""Artifact validation tests for the Axis-3 curriculum coordinator."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "run_axis3_curriculum.py"
SPEC = spec_from_file_location("run_axis3_curriculum", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stage_completion_requires_expected_iteration(tmp_path):
    source = tmp_path / "model_198.pt"
    checkpoint = tmp_path / "model_250.pt"
    source.touch()
    checkpoint.touch()

    error = MODULE.stage_completion_error(tmp_path, checkpoint, source, 100)

    assert error == "last checkpoint is model_250.pt; expected at least model_297.pt"


def test_stage_completion_rejects_failure_artifact_even_at_target(tmp_path):
    source = tmp_path / "model_198.pt"
    checkpoint = tmp_path / "model_297.pt"
    source.touch()
    checkpoint.touch()
    failure = tmp_path / "ppo_numerical_failure.json"
    failure.write_text("{}\n")

    error = MODULE.stage_completion_error(tmp_path, checkpoint, source, 100)

    assert error == f"numerical failure artifact exists: {failure}"


def test_stage_completion_accepts_clean_target_checkpoint(tmp_path):
    source = tmp_path / "model_198.pt"
    checkpoint = tmp_path / "model_297.pt"
    source.touch()
    checkpoint.touch()

    assert MODULE.stage_completion_error(tmp_path, checkpoint, source, 100) is None
