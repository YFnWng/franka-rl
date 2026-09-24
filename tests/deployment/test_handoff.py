"""Fail closed on altered or escaping handoff inventory entries."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/deployment"))
from preflight_handoff import digest, preflight


def test_handoff_manifest_anchor(tmp_path):
    (tmp_path / "handoff_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="trust anchor"):
        preflight(tmp_path, "0" * 64)


def test_handoff_file_tampering(tmp_path):
    payload = tmp_path / "data.txt"
    payload.write_text("original")
    manifest = dict(schema_version=1, files={"data.txt": digest(payload)})
    (tmp_path / "handoff_manifest.json").write_text(json.dumps(manifest))
    payload.write_text("modified")
    with pytest.raises(ValueError, match="Hash mismatch"):
        preflight(tmp_path)


def test_handoff_path_escape(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    manifest = dict(schema_version=1, files={"../outside.txt": digest(outside)})
    (root / "handoff_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes handoff"):
        preflight(root)
