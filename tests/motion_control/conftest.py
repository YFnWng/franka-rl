"""Test the portable source tree without requiring an Isaac Lab installation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/franka_motion"))
