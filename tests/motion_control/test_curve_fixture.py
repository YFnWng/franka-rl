"""Endpoint-free projection, dissipative wrench, and batched controller parity."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from franka_motion.virtual_fixture import CurveVirtualFixture, load_curve_fixture

ROOT = Path(__file__).resolve().parents[2]


def controllers():
    path = ROOT/"source/franka_rl/franka_rl/tasks/manager_based/franka_motion_control/mdp/controllers.py"
    spec = importlib.util.spec_from_file_location("curve_controllers",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind",["z_line","yz_circle"])
def test_projection_force_parity_and_dissipation(kind):
    fixture = CurveVirtualFixture(kind,[.3,0,.5],.05)
    module = controllers()
    c = module.BatchedCurveImpedanceController(2,"cpu",[100.]*3+[10.]*3,[20.]*3+[6.5]*3)
    c.kind,c.radius = kind,fixture.radius
    c.origin.copy_(torch.tensor(fixture.origin))
    p = np.array([[.31,.01,.55],[.3,-.05,.5]],dtype=np.float32)
    v = np.array([[.01,.02,.03],[-.02,.03,.01]],dtype=np.float32)
    pose = torch.tensor(np.c_[p,np.tile([1,0,0,0],(2,1))],dtype=torch.float32)
    ref = torch.zeros(2,13)  # Orientation cannot be overridden by caller.
    twist = torch.tensor(np.c_[v,np.zeros((2,3))],dtype=torch.float32)
    c.compute(pose,twist,ref,torch.zeros(2,6,7),torch.zeros(2,7),torch.zeros(2,7))
    for i in range(2):
        target,tangent = fixture.project(p[i])
        np.testing.assert_allclose(c.reference[i,:3],target,atol=1e-7)
        np.testing.assert_allclose(c.wrench[i,:3],fixture.force(p[i],v[i]),atol=2e-6)
        damping = fixture.force(target,v[i])
        assert damping@v[i] < 0
        assert abs(fixture.force(target,np.zeros(3))@tangent) < 1e-8
    assert not c.error[:,3:].any()
    c.reference.fill_(4.)
    c.components["filtered"].fill_(3.)
    c.reset(torch.tensor([0]))
    assert not c.reference[0].any() and not c.components["filtered"][0].any()
    assert (c.reference[1]==4.).all() and (c.components["filtered"][1]==3.).all()


def test_endpoint_free_geometry_and_projection_rejection():
    line = CurveVirtualFixture("z_line",[0,0,0])
    np.testing.assert_array_equal(line.project([1,2,100])[0],[0,0,100])
    circle = CurveVirtualFixture("yz_circle",[0,0,0],.1)
    for angle in np.linspace(0,2*np.pi,50):
        p = np.array([0,.1*np.sin(angle),.1*np.cos(angle)])
        q,t = circle.project(p)
        np.testing.assert_allclose(p,q,atol=1e-15)
        assert abs(t@q) < 1e-15 and np.isclose(t@t,1.)
    with pytest.raises(ValueError,match="undefined"):
        circle.project([0,0,0])
    q,_ = controllers().curve_projection_torch(torch.zeros(1,3),"yz_circle",torch.zeros(3),.1)
    assert not torch.isfinite(q).any()  # Routes through existing nonfinite fault stop.
    with pytest.raises(ValueError):
        CurveVirtualFixture("unknown",[0,0,0])


def test_curve_config():
    config = load_curve_fixture(ROOT/"source/franka_motion/franka_motion/config/suites/curve_fixture.yaml")
    assert config["tangent_damping"] > 0 and len(config["experiments"]) == 4
    assert Path(config["motion_suite"]).is_file()
