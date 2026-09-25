import numpy as np
import pytest
from franka_governor import Status
from franka_governor.simulation import SimulationBatch
from test_governor import cfg


def test_parallel_scheduling_and_selective_episode_reset():
    c=cfg(); batch=SimulationBatch(c,3)
    q=np.tile(c.default_position,(3,1));dq=np.zeros_like(q)
    batch.reset(range(3),q,dq,0)
    seq=0
    for physics in range(1,601):
        if physics%100==0:
            t=physics*1000000//3
            assert batch.submit(np.full((3,7),.1),t,t,t).all()
            seq+=1
        if physics%3==0:
            t=physics//3*1000000
            q=batch.step(t,q,dq)
            dq=np.array([o.dq for o in batch.outputs])
            assert not batch.failed.any()
    assert seq==6
    before=batch.outputs[1].q
    batch.reset([0],q,dq,200000000)
    np.testing.assert_array_equal(batch.outputs[1].q,before)
    assert batch.sessions.tolist()==[2,1,1]
    np.testing.assert_array_equal(batch.previous_raw[0],0)
    assert batch.outputs[1].status==Status.RUNNING


def test_sim_batch_input_validation():
    b=SimulationBatch(cfg(),2)
    with pytest.raises(ValueError): b.reset([0],np.zeros((1,7)),np.zeros((1,7)),0)
    with pytest.raises(RuntimeError): b.step(1000000,np.zeros((2,7)),np.zeros((2,7)))
