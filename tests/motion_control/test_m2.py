import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from franka_motion.config import load_motion_suite, expand_trials, ControllerProfile
from franka_motion.kinematics import (UrdfKinematicsProvider, DampedLeastSquaresIK, validate_path_ik,
    pose_error_base, quat_exp, quat_multiply, rotation_vector)
from franka_motion.impedance import CartesianImpedanceController

ROOT = Path(__file__).resolve().parents[2]


def setup():
    suite = load_motion_suite(ROOT/"source/franka_motion/franka_motion/config/suites/demo_a_tracking.yaml")
    provider = UrdfKinematicsProvider(ROOT/suite.workspace.data["model"]["archived_urdf"])
    home = np.array(suite.workspace.data["initial_postures_rad"]["deployment_home"])
    return suite, provider, home


def test_fk_jacobian_finite_difference_and_quaternion_sign():
    _, provider, q = setup()
    p,r,j = provider.evaluate(q)
    for i in range(7):
        displaced = q.copy()
        displaced[i] += 1e-6
        p1,r1,_ = provider.evaluate(displaced)
        np.testing.assert_allclose(pose_error_base(p,r,p1,r1)/1e-6,j[:,i],atol=1e-6)
    np.testing.assert_allclose(pose_error_base(p,r,p,-r),0,atol=1e-14)
    np.testing.assert_allclose(rotation_vector(quat_exp([0,np.pi,0])),[0,np.pi,0],atol=1e-14)


def test_continuous_full_pose_dls_and_unreachable_rejection():
    suite,provider,q = setup()
    p,r,_ = provider.evaluate(q)
    trial = expand_trials(suite)[0]
    path = trial.trajectory.build(.4,p,r)
    b = suite.workspace.data["joint_bounds_rad"]["commanded"]
    solver = DampedLeastSquaresIK(provider,b["lower"],b["upper"])
    report,trace = validate_path_ik(path,solver,q,sample_hz=5.)
    assert report["passed"] and report["minimum_jacobian_singular_value_mixed_units"] > .02
    assert len(trace["ik_q"]) == report["sample_count"]
    with pytest.raises(ValueError,match="did not converge"):
        solver.solve([3,3,3],r,q)
    assert len(expand_trials(suite)) == 8


def test_orientation_entry_twist_is_derivative_and_joins():
    suite,provider,q = setup()
    p,r,_ = provider.evaluate(q)
    trial = expand_trials(suite)[0]
    changed = quat_multiply(quat_exp([0,.2,0]),r)
    path = trial.trajectory.build(.4,p,changed)
    for t in (.1,2.,5.9,path.duration_s-2.):
        a = path.evaluate(t).reference
        b = path.evaluate(t+1e-6).reference
        velocity = pose_error_base(a.position_m,a.quaternion_xyzw,b.position_m,b.quaternion_xyzw)/1e-6
        np.testing.assert_allclose(velocity,np.r_[a.linear_velocity_m_s,a.angular_velocity_rad_s],atol=1e-6)
    np.testing.assert_allclose(path.evaluate(0).reference.quaternion_xyzw,changed,atol=1e-12)
    np.testing.assert_allclose(pose_error_base(p,changed,path.evaluate(path.duration_s).reference.position_m,
        path.evaluate(path.duration_s).reference.quaternion_xyzw),0,atol=1e-12)


def test_numpy_torch_cartesian_processing_parity_and_partial_reset():
    path = ROOT/"source/franka_rl/franka_rl/tasks/manager_based/franka_motion_control/mdp/controllers.py"
    spec = importlib.util.spec_from_file_location("motion_m2_test",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module.BatchedCartesianImpedanceController
    batch = cls(2,"cpu",[100.]*3+[10.]*3,[20.]*3+[6.5]*3)
    controllers = [CartesianImpedanceController([100.]*3+[10.]*3,[20.]*3+[6.5]*3) for _ in range(2)]
    rng = np.random.default_rng(42)
    for _ in range(40):
        pose = np.c_[rng.normal(size=(2,3)),np.array([quat_exp(v) for v in rng.normal(size=(2,3))*.3])]
        ref = np.c_[rng.normal(size=(2,3)),np.array([quat_exp(v) for v in rng.normal(size=(2,3))*.3]),rng.normal(size=(2,6))]
        inputs = [pose,rng.normal(size=(2,6)),ref,rng.normal(size=(2,6,7)),rng.normal(size=(2,7)),rng.normal(size=(2,7))]
        batch.compute(*(torch.tensor(x,dtype=torch.float32) for x in inputs))
        for i in range(2):
            expected = controllers[i].compute(*(x[i] for x in inputs))
            for key,value in expected.items():
                actual = batch.error[i] if key=="error" else batch.wrench[i] if key=="wrench" else batch.components[key][i]
                np.testing.assert_allclose(actual.numpy(),value,atol=3e-4,rtol=1e-4)
    neighbor = batch.components["filtered"][1].clone()
    batch.reset(torch.tensor([0]))
    assert not batch.error[0].any() and not batch.components["filtered"][0].any()
    torch.testing.assert_close(batch.components["filtered"][1],neighbor,rtol=0,atol=0)


def test_profile_rejects_invalid_cartesian_servo():
    import copy
    suite,_,_ = setup()
    data = copy.deepcopy(suite.controllers["cartesian_soft"].data)
    data["cartesian_servo"]["max_reference_linear_velocity_m_s"] = -1
    with pytest.raises(ValueError):
        ControllerProfile("bad",data)


def test_circle_contouring_ignores_phase_but_includes_plane_distance():
    from franka_motion.metrics import circle_contour_error,sampled_curve_contour_error
    points = np.array([[1,0,0],[0,1,0],[1,0,.2],[1.1,0,0]])
    np.testing.assert_allclose(circle_contour_error(points,[0,0,0],1,np.eye(3)),[0,0,.2,.1],atol=1e-12)
    with pytest.raises(ValueError):
        circle_contour_error(points,[0,0,0],1,np.zeros((3,3)))
    np.testing.assert_allclose(sampled_curve_contour_error([[0,0,.2],[1,0,0]],[[0,0,0],[1,0,0]]),[.2,0],atol=1e-12)


def test_twist_observation_uses_actor_not_com_origin():
    from types import SimpleNamespace as NS
    path = ROOT/"source/franka_rl/franka_rl/tasks/manager_based/franka_motion_control/mdp/observations.py"
    spec = importlib.util.spec_from_file_location("motion_observation_test",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = torch.tensor([[[.1,.2,.3,.4,.5,.6]]])
    data = NS(root_quat_w=NS(torch=torch.tensor([[0.,0.,0.,1.]])),body_link_vel_w=NS(torch=expected))
    env = NS(scene={"robot":NS(data=data)},motion_action=NS(flange_id=0))
    torch.testing.assert_close(module.tcp_twist_base(env),expected[:,0],atol=0,rtol=0)


def test_coordinator_rejects_success_index_without_finalized_artifacts(tmp_path):
    import json
    path = ROOT/"scripts/motion_control/run_sim_suite.py"
    spec = importlib.util.spec_from_file_location("motion_suite_test",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path/"index.json").write_text(json.dumps({"all_accepted":True,"results":[{"trial_id":"missing"}]}))
    with pytest.raises(OSError):
        module.validate_phase_output(tmp_path,["missing"])
    with pytest.raises(ValueError,match="exact accepted trial set"):
        module.validate_phase_output(tmp_path,["other"])
