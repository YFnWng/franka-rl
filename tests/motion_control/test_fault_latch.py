"""Test the actual action latch without starting Kit or constructing a scene."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_fault_latch_preserves_first_snapshot_and_unfaulted_neighbors():
    path = Path(__file__).resolve().parents[2]/"source/franka_rl/franka_rl/tasks/manager_based/franka_motion_control/mdp/actions.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "JointMotionAction")
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_latch")
    namespace = {"torch":torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]),str(path),"exec"),namespace)
    latch = namespace["_latch"]
    state = SimpleNamespace(_env=SimpleNamespace(_sim_step_counter=12),
        fault_code=torch.zeros(3,dtype=torch.int32),first_fault_tick=torch.full((3,),-1,dtype=torch.long),
        first_fault_code=torch.zeros(3,dtype=torch.int32),contact_force=torch.arange(3,dtype=torch.float32),
        first_fault_contact_force=torch.zeros(3),first_fault_q=torch.zeros(3,7),
        first_fault_dq=torch.zeros(3,7),first_fault_reference=torch.zeros(3,13),
        fault_hold_q=torch.zeros(3,7),_raw=torch.arange(39,dtype=torch.float32).reshape(3,13))
    q = torch.arange(21,dtype=torch.float32).reshape(3,7)
    dq = -q
    latch(state,torch.tensor([0,2,0]),q,dq)
    assert state.first_fault_tick.tolist() == [-1,12,-1]
    torch.testing.assert_close(state.first_fault_q[1],q[1],rtol=0,atol=0)
    snapshot = {k:v.clone() for k,v in vars(state).items() if k.startswith("first_fault")}
    state._env._sim_step_counter = 13
    state._raw.add_(100)
    q2 = q+100
    q2[2,0] = float("nan")
    latch(state,torch.tensor([0,8,4]),q2,dq+100)
    assert state.fault_code.tolist() == [0,10,4]
    assert state.first_fault_tick.tolist() == [-1,12,13]
    for key,value in snapshot.items():
        torch.testing.assert_close(getattr(state,key)[:2],value[:2],rtol=0,atol=0)
    assert torch.isnan(state.first_fault_q[2,0])
    assert state.fault_hold_q[2,0] == 0
    torch.testing.assert_close(state.first_fault_reference[2],state._raw[2],rtol=0,atol=0)
    saved = {k:v.clone() for k,v in vars(state).items() if isinstance(v,torch.Tensor)}
    latch(state,torch.zeros(3,dtype=torch.int32),q,dq)
    for key,value in saved.items():
        torch.testing.assert_close(getattr(state,key),value,rtol=0,atol=0,equal_nan=True)
