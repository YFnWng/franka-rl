import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from franka_motion.virtual_fixture import LineVirtualFixture,nullspace_projector,compute_posture_torque,load_demo_b

ROOT = Path(__file__).resolve().parents[2]


def test_finite_fixture_projection_twist_filter_and_endpoints():
    fixture = LineVirtualFixture([0,0,0],[0,0,1],-.03,.03)
    p,v = fixture.evaluate([.2,.1,.01],[1,1,.02])
    np.testing.assert_allclose(p,[0,0,.01])
    assert 0 < v[2] < .02 and not v[:2].any()
    p,v = fixture.evaluate([0,0,.04],[0,0,.1])
    np.testing.assert_allclose(p,[0,0,.03])
    assert not v.any()
    fixture.reset()
    assert fixture.velocity == 0
    with pytest.raises(ValueError):
        LineVirtualFixture([0,0,0],[0,0,2])


@pytest.mark.parametrize("rank_deficient",[False,True])
def test_numpy_torch_nullspace_projection_and_posture(rank_deficient):
    path = ROOT/"source/franka_rl/franka_rl/tasks/manager_based/franka_motion_control/mdp/controllers.py"
    spec = importlib.util.spec_from_file_location("m3_controllers",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    j = np.random.default_rng(3).normal(size=(6,7))
    if rank_deficient:
        j[-1] = j[0]
    n = nullspace_projector(j)
    np.testing.assert_allclose(n@n,n,atol=1e-12)
    np.testing.assert_allclose(j@n,0,atol=1e-12)
    actual = module.nullspace_projector_torch(torch.tensor(j)[None])[0].numpy()
    np.testing.assert_allclose(actual,n,atol=1e-12)
    torque = compute_posture_torque(j,np.zeros(7),np.ones(7),np.ones(7),2.,1.)
    np.testing.assert_allclose(torque,n@np.ones(7),atol=1e-12)


def test_demo_b_config_and_feasible_postures():
    from franka_motion.config import load_motion_suite
    from franka_motion.kinematics import UrdfKinematicsProvider,DampedLeastSquaresIK,pose_error_base
    config = load_demo_b(ROOT/"source/franka_motion/franka_motion/config/suites/demo_b_fixture_posture.yaml")
    suite = load_motion_suite(config["motion_suite"])
    q = np.array(suite.workspace.data["initial_postures_rad"]["deployment_home"])
    provider = UrdfKinematicsProvider(ROOT/suite.workspace.data["model"]["archived_urdf"])
    p,r,j = provider.evaluate(q)
    _,_,vh = np.linalg.svd(j,full_matrices=True)
    b = suite.workspace.data["joint_bounds_rad"]["commanded"]
    solver = DampedLeastSquaresIK(provider,b["lower"],b["upper"])
    for sign in (-1,1):
        target = solver.solve(p,r,q+sign*config["posture_offset_rad"]*vh[-1])
        assert np.linalg.norm(target-q) > .5*config["posture_offset_rad"]
        a,c,_ = provider.evaluate(target)
        assert np.linalg.norm(pose_error_base(p,r,a,c)) < .001


def test_fixture_numpy_torch_parity_and_posture_torque_processing():
    path = ROOT/"source/franka_rl/franka_rl/tasks/manager_based/franka_motion_control/mdp/controllers.py"
    spec = importlib.util.spec_from_file_location("m3_processing",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = LineVirtualFixture([0,0,0],[0,0,1])
    filtered = torch.zeros(1,dtype=torch.float64)
    rng = np.random.default_rng(2)
    for _ in range(40):
        p,v = rng.normal(size=(2,3))*.05
        expected = fixture.evaluate(p,v)
        a,b,filtered = module.line_fixture_reference_torch(torch.tensor(p)[None],torch.tensor(v)[None],
            torch.tensor(fixture.origin),torch.tensor(fixture.direction),filtered,fixture.alpha,fixture.lower,fixture.upper,fixture.max_speed)
        np.testing.assert_allclose(a[0].numpy(),expected[0],atol=1e-12)
        np.testing.assert_allclose(b[0].numpy(),expected[1],atol=1e-12)
    controller = module.BatchedCartesianImpedanceController(1,"cpu",[100.]*3+[10.]*3,[20.]*3+[6.5]*3)
    controller.components["posture"] = torch.ones(1,7)*.2
    pose = torch.tensor([[0.,0.,0.,0.,0.,0.,1.]])
    ref = torch.cat((pose,torch.zeros(1,6)),dim=1)
    controller.compute(pose,torch.zeros(1,6),ref,torch.zeros(1,6,7),torch.zeros(1,7),torch.zeros(1,7))
    torch.testing.assert_close(controller.components["requested"],torch.full((1,7),.2))
    controller.reset(slice(None))
    assert not controller.components["posture"].any()
