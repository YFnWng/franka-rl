"""Track overlay samples the actual geometry, not its approach corridor."""
import importlib.util
from pathlib import Path
import numpy as np
from franka_motion.config import load_motion_suite, expand_trials

ROOT = Path(__file__).resolve().parents[2]


def test_scene_path_points_match_circle_and_lissajous():
    path = ROOT/"source/franka_rl/franka_rl/experiments/motion_path_visualization.py"
    spec = importlib.util.spec_from_file_location("path_visualization",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = load_motion_suite(ROOT/"source/franka_motion/franka_motion/config/suites/simulation_smoke.yaml")
    for trial in expand_trials(suite):
        trajectory = trial.trajectory.build(trial.angular_speed_rad_s,[.3,0,.59])
        points = module.tracking_curve_points(trajectory)
        assert points.shape == (512,3) and np.isfinite(points).all()
        np.testing.assert_allclose(points[0],trajectory.path.evaluate(0).reference.position_m)
        np.testing.assert_allclose(points[-1],points[0],atol=1e-12)
        assert np.linalg.norm(points[0]-trajectory.initial_position) > .1
