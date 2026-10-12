"""Track overlay samples the actual geometry, not its approach corridor."""
import importlib.util
from pathlib import Path
import numpy as np
from franka_motion.config import load_motion_suite, expand_trials

ROOT = Path(__file__).resolve().parents[2]


def test_marker_visualizer_is_detected_even_when_legacy_gui_is_false():
    from types import SimpleNamespace
    path = ROOT/"source/franka_rl/franka_rl/experiments/motion_path_visualization.py"
    spec = importlib.util.spec_from_file_location("path_visualization_gate",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    viz = SimpleNamespace(supports_markers=lambda:True,cfg=SimpleNamespace(enable_markers=True))
    sim = SimpleNamespace(has_gui=False,visualizers=[viz])
    assert module.path_visualization_enabled(sim)
    viz.cfg.enable_markers = False
    assert not module.path_visualization_enabled(sim)
    sim.visualizers = []
    assert not module.path_visualization_enabled(sim)
    sim.has_gui = True
    assert module.path_visualization_enabled(sim)


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


def test_measured_frame_axes_transform_and_colors():
    import torch
    from isaaclab.utils.math import quat_apply
    path = ROOT/"source/franka_rl/franka_rl/experiments/motion_path_visualization.py"
    spec = importlib.util.spec_from_file_location("frame_visualization",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
    q = torch.tensor([[0.,0.,0.,1.],[1.,0.,0.,0.]])
    centers,orientations,indices = module.frame_marker_poses(p,q)
    direction = quat_apply(orientations,torch.tensor([[0.,0.,1.]]).repeat(6,1))
    torch.testing.assert_close(centers-p.repeat_interleave(3,dim=0),direction*.03,atol=1e-6,rtol=0)
    torch.testing.assert_close(direction[:3],torch.eye(3),atol=1e-6,rtol=0)
    torch.testing.assert_close(direction[3:],torch.diag(torch.tensor([1.,-1.,-1.])),atol=1e-6,rtol=0)
    assert indices.tolist() == [0,1,2,0,1,2]
