import importlib.util
from pathlib import Path
import torch
import numpy as np
from franka_motion.impedance import dynamic_torque_projector


def test_dynamic_torque_projector_decouples_task_acceleration():
    path=Path(__file__).resolve().parents[2]/"source/franka_rl/franka_rl/tasks/manager_based/franka_motion_control/mdp/controllers.py"
    spec=importlib.util.spec_from_file_location("dynamic_posture",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator=torch.Generator().manual_seed(18)
    j=torch.randn(3,6,7,generator=generator,dtype=torch.float64)
    a=torch.randn(3,7,7,generator=generator,dtype=torch.float64)
    m=a@a.transpose(-1,-2)+torch.eye(7)*.2
    for jacobian in (j,torch.cat((j[:,:5],j[:,:1]),dim=1)):
        n=module.dynamic_torque_projector_torch(jacobian,m)
        for i in range(3):
            np.testing.assert_allclose(n[i].numpy(),dynamic_torque_projector(jacobian[i].numpy(),m[i].numpy()),atol=1e-10)
        torch.testing.assert_close(jacobian@torch.linalg.solve(m,n),torch.zeros(3,6,7,dtype=torch.float64),atol=1e-10,rtol=0)
        torch.testing.assert_close(n@n,n,atol=1e-10,rtol=0)
        torque=torch.randn(3,7,1,generator=generator,dtype=torch.float64)
        expected=torque-jacobian.transpose(-1,-2)@torch.linalg.pinv(jacobian@torch.linalg.solve(m,jacobian.transpose(-1,-2)),rtol=1e-6)@jacobian@torch.linalg.solve(m,torque)
        torch.testing.assert_close(n@torque,expected,atol=1e-10,rtol=0)
