"""Offline deployment entry point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deployment"))
from franka_policy_bundle.bundle import verify_main

if __name__ == "__main__":
    verify_main()
