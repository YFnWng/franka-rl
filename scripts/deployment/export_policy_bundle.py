"""Offline deployment entry point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deployment"))
from franka_policy_bundle.bundle import export_main

if __name__ == "__main__":
    export_main()
