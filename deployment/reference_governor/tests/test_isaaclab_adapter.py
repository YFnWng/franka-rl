"""Action-hook conformance with lightweight API stubs; NOT an Isaac runtime test."""
import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from test_governor import ROOT, cfg


class Tensor(np.ndarray):
    def clone(self): return self.copy().view(Tensor)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return np.asarray(self)


def tensor(x, **kwargs): return np.asarray(x, dtype=kwargs.get('dtype')).view(Tensor)


@pytest.fixture
def adapter(monkeypatch):
    torch=ModuleType('torch');torch.bool=np.bool_
    torch.zeros=lambda shape,**kw: np.zeros(shape,dtype=kw.get('dtype',np.float32)).view(Tensor)
    torch.as_tensor=tensor
    class Base:
        def __init__(self,cfg,env): self.cfg,self._env,self._asset=cfg,env,env.scene[cfg.asset_name]
    for name in ['isaaclab','isaaclab.managers','isaaclab.managers.action_manager','isaaclab.utils','isaaclab.utils.configclass']:
        monkeypatch.setitem(sys.modules,name,ModuleType(name))
    sys.modules['isaaclab.managers.action_manager'].ActionTerm=Base
    sys.modules['isaaclab.managers.action_manager'].ActionTermCfg=type('Cfg',(),{})
    sys.modules['isaaclab.utils.configclass'].configclass=lambda cls:cls
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.delitem(sys.modules,'franka_governor.isaaclab',raising=False)
    module=importlib.import_module('franka_governor.isaaclab')
    yield module
    sys.modules.pop('franka_governor.isaaclab',None)


def make(adapter):
    names=[f'panda_joint{i}' for i in range(1,8)]
    q=tensor(np.tile(cfg().default_position,(2,1)),dtype=np.float32)
    robot=SimpleNamespace(data=SimpleNamespace(default_joint_pos=SimpleNamespace(torch=q.clone()),
        joint_pos=SimpleNamespace(torch=q.clone()),joint_vel=SimpleNamespace(torch=tensor(np.zeros((2,7)))))),
    robot=robot[0]
    robot.find_joints=lambda names,preserve_order: (list(range(7)),names)
    robot.set_joint_position_target_index=lambda target,joint_ids: setattr(robot,'target',target.clone())
    env=SimpleNamespace(scene={'robot':robot},num_envs=2,device='cpu',physics_dt=1/3000,
                        cfg=SimpleNamespace(decimation=100),_physics_handles_decimation=False)
    c=SimpleNamespace(asset_name='robot',joint_names=names,governor_config=str(ROOT/'configs/simulation.json'),
                      allow_simulation_fixture=True)
    return adapter.GovernedJointPositionAction(c,env),env,robot


def test_exact_hook_timing_partial_reset_and_raw_action(adapter):
    term,env,robot=make(adapter)
    assert term.action_dim==7
    for policy in range(5):
        raw=tensor(np.full((2,7),.1),dtype=np.float32)
        term.process_actions(raw)
        for _ in range(100):
            term.apply_actions()
            robot.data.joint_pos.torch=robot.target.clone()
            robot.data.joint_vel.torch=tensor([o.dq for o in term._governor.outputs])
        assert not term.failed.any()
        np.testing.assert_array_equal(term.raw_actions,raw)
        assert term._governor.outputs[1].accepted_sequence==policy+1
        if policy==1:
            robot.data.joint_pos.torch[0]=cfg().default_position
            robot.data.joint_vel.torch[0]=0
            term.reset([0])
    assert term._physics_tick==500
    assert term._governor.sessions.tolist()==[2,1]
    assert term._governor.outputs[0].accepted_sequence==3
    np.testing.assert_array_equal(term._governor.previous_raw,raw)


def test_invalid_command_is_not_written(adapter):
    term, env, robot = make(adapter)
    term.process_actions(tensor(np.zeros((2, 7)), dtype=np.float32))
    term.apply_actions()  # reset at tick zero
    robot.data.joint_vel.torch[1, 3] = 0.31
    term.apply_actions()
    term.apply_actions()
    writes = []
    robot.set_joint_position_target_index = lambda **kwargs: writes.append(kwargs)
    with pytest.raises(RuntimeError, match="TRACKING"):
        term.apply_actions()
    assert not writes


def test_reject_wrong_timebase_and_internal_decimation(adapter):
    term,env,robot=make(adapter)
    env.physics_dt=1/60
    with pytest.raises(ValueError): adapter.GovernedJointPositionAction(term.cfg,env)
    env.physics_dt=1/3000;env._physics_handles_decimation=True
    with pytest.raises(ValueError): adapter.GovernedJointPositionAction(term.cfg,env)


def test_fault_and_pending_overwrite(adapter):
    term,env,robot=make(adapter)
    raw=tensor(np.full((2,7),np.nan))
    term.process_actions(raw)
    with pytest.raises(RuntimeError): term.process_actions(raw)
    term.apply_actions()
    assert term.failed.all()
