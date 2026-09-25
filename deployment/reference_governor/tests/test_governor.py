import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from franka_governor import (Batch, Config, Feedback, Governor, Message, Reason, Status,
                             TICK_NS, load_config, policy_tick)

ROOT = Path(__file__).resolve().parents[1]


def cfg():
    return load_config(ROOT/'configs/simulation.json', allow_simulation=True)[0]


def feedback(o, t, session=1):
    f = Feedback()
    f.q = f.desired_q = o.q
    f.dq = f.desired_dq = o.dq
    f.desired_ddq = o.ddq
    f.session, f.observed_ns = session, t
    return f


def fresh(c=None, q=None, velocity=None):
    c = c or cfg()
    g = Governor(c)
    f = Feedback()
    f.q = f.desired_q = q if q is not None else c.default_position
    f.dq = f.desired_dq = velocity if velocity is not None else [0]*7
    f.session = 1
    assert g.reset(1, 0, f)
    return g


def message(seq=1, now=0, action=None, session=1):
    m = Message()
    m.sequence, m.session = seq, session
    m.observation_ns = m.completed_ns = now
    m.action = action if action is not None else [.2]*7
    return m


def advance(g, t):
    return g.step(t, feedback(g.output, t))


def test_config_rejects_unreviewed_and_missing(tmp_path):
    with pytest.raises(ValueError): load_config(ROOT/'configs/simulation.json')
    with pytest.raises(ValueError): load_config(ROOT/'configs/deployment.template.json')
    with pytest.raises(ValueError): Governor(Config())
    c = cfg(); c.max_jerk = [float('nan')]*7
    with pytest.raises(ValueError): Governor(c)
    d = json.loads((ROOT/'configs/simulation.json').read_text())
    del d['core']['margin']
    p = tmp_path/'missing.json'; p.write_text(json.dumps(d))
    with pytest.raises(ValueError): load_config(p, allow_simulation=True)


def test_rational_schedule():
    assert [policy_tick(i) for i in range(4)] == [0,34,67,100]
    assert policy_tick(30000) == 1000000
    with pytest.raises(ValueError): policy_tick(-1)


@pytest.mark.parametrize('reason', ['nan','duplicate','future','old','session','mailbox'])
def test_bad_policy_stops_and_latches(reason):
    g = fresh(); m = message(now=TICK_NS)
    if reason == 'nan': m.action = [float('nan')]*7
    if reason == 'duplicate': m.sequence = 0
    if reason == 'future': m.completed_ns = 2*TICK_NS
    if reason == 'old': m.observation_ns = -1
    if reason == 'session': m.session = 2
    if reason == 'mailbox': assert g.submit(m,TICK_NS)
    assert not g.submit(m,TICK_NS)
    assert g.output.status == Status.STOPPING
    for t in range(1,252): advance(g,t*TICK_NS)
    assert g.output.status == Status.TERMINAL
    assert not g.submit(message(),252*TICK_NS)
    assert not g.reset(1,252*TICK_NS,feedback(g.output,252*TICK_NS))
    assert g.reset(2,252*TICK_NS,feedback(g.output,252*TICK_NS,2))
    assert g.output.previous_raw == [0]*7


@pytest.mark.parametrize('kind', ['clock','duplicate_state','stale_state','future_state','nan','unhealthy','tracking','desired','desired_velocity','desired_acceleration','session'])
def test_hardware_fault_is_invalid_command(kind):
    g = fresh(); f = feedback(g.output,TICK_NS); now=TICK_NS
    if kind == 'clock': now=2*TICK_NS
    if kind == 'duplicate_state': f.observed_ns=0
    if kind == 'stale_state': f.observed_ns=-100000000
    if kind == 'future_state': f.observed_ns=2*TICK_NS
    if kind == 'nan': f.dq=[float('nan')]*7
    if kind == 'unhealthy': f.healthy=False
    if kind == 'tracking': f.q=[x+.21 for x in f.q]
    if kind == 'desired': f.desired_q=[x+.002 for x in f.q]
    if kind == 'desired_velocity': f.desired_dq=[.02]*7
    if kind == 'desired_acceleration': f.desired_ddq=[.2]*7
    if kind == 'session': f.session=2
    o=g.step(now,f)
    assert o.status==Status.FAULT and not o.command_valid
    assert advance(g,3*TICK_NS).status==Status.FAULT


def test_nonzero_activation_and_infeasible_boundary():
    g=fresh(velocity=[.01]*7)
    g.stop()
    for i in range(1,251): advance(g,i*TICK_NS)
    np.testing.assert_allclose(g.output.dq,0,atol=1e-12)
    c=cfg(); f=Feedback(); f.session=1
    f.q=f.desired_q=[x-.1 for x in c.upper]
    f.dq=f.desired_dq=[.29]*7
    g=Governor(c)
    assert not g.reset(1,0,f)
    assert not g.output.command_valid


def test_raw_action_and_projection_separate():
    c=cfg();c.max_projection=100
    g=fresh(c);raw=[10.]*7
    assert g.submit(message(now=TICK_NS,action=raw),TICK_NS)
    o=advance(g,TICK_NS)
    np.testing.assert_array_equal(o.previous_raw,raw)
    np.testing.assert_array_equal(o.mapped_target,(np.asarray(c.default_position,dtype=np.float32)+np.float32(.5)*np.asarray(raw,dtype=np.float32)).astype(float))
    np.testing.assert_allclose(o.projected_target,np.asarray(c.upper)-np.asarray(c.margin))
    assert o.projection>0
    assert max(abs(np.array(o.q)-c.default_position))<1e-10


def test_intervention_budget_stops():
    c=cfg();c.max_projection=.01
    g=fresh(c);assert g.submit(message(now=TICK_NS,action=[100.]*7),TICK_NS)
    assert advance(g,TICK_NS).reason==Reason.INTERVENTION
    assert g.output.status==Status.STOPPING


def trajectory(g, ticks=5000):
    rows=[];seq=0
    for tick in range(1,ticks+1):
        t=tick*TICK_NS
        if tick<=4000 and tick%34==1:
            seq+=1
            assert g.submit(message(seq,t,[.2 if tick<2000 else -.2]*7),t)
        o=advance(g,t)
        rows.append([tick,*o.q,*o.dq,*o.ddq,int(o.status),int(o.reason)])
    return np.array(rows)


def test_full_trajectory_and_native_parity():
    c=cfg();g=fresh(c);a=trajectory(g)
    np.testing.assert_array_equal(a,trajectory(fresh(c)))
    q,v,acc=a[:,1:8],a[:,8:15],a[:,15:22]
    assert np.all(q>=np.asarray(c.lower)+c.margin)
    assert np.all(q<=np.asarray(c.upper)-c.margin)
    assert np.max(np.abs(v))<=.3
    assert np.max(np.abs(acc))<=2
    assert np.max(np.abs(np.diff(acc,axis=0))/.001)<=30+1e-8
    assert np.max(np.abs(np.diff(q,axis=0))/.001)<=.3
    assert np.max(np.abs(np.diff(q,n=2,axis=0))/.001**2)<=2+1e-7
    assert np.max(np.abs(np.diff(q,n=3,axis=0))/.001**3)<=30+1e-5
    assert g.output.status==Status.TERMINAL
    np.testing.assert_allclose(g.output.dq,0,atol=1e-12)
    np.testing.assert_allclose(a[99::100],np.loadtxt(ROOT/'fixtures/native_trace.csv',delimiter=','),rtol=0,atol=1e-12)
    executable=os.environ.get('GOVERNOR_NATIVE_TEST')
    if executable:
        from io import StringIO
        native=np.loadtxt(StringIO(subprocess.check_output([executable,'--trace'],text=True)),delimiter=',')
        np.testing.assert_allclose(a[99::100],native,rtol=0,atol=1e-12)


def test_boundaries_and_reversals():
    c=cfg();c.max_projection=100
    for initial in [np.asarray(c.lower)+.10001,np.asarray(c.upper)-.10001]:
        g=fresh(c,initial.tolist());seq=0
        prev=np.array(g.output.ddq)
        for tick in range(1,801):
            t=tick*TICK_NS
            if tick%34==1:
                seq+=1;assert g.submit(message(seq,t,[(-1 if tick<400 else 1)*20.]*7),t)
            o=advance(g,t)
            assert o.command_valid
            assert np.all(np.array(o.q)>=np.array(c.lower)+c.margin)
            assert np.all(np.array(o.q)<=np.array(c.upper)-c.margin)
            assert np.max(abs(np.array(o.ddq)-prev)/.001)<=30+1e-7
            prev=np.array(o.ddq)


def test_batch_selective_reset():
    c=cfg();b=Batch(c,2);g=fresh(c)
    for i in range(2): assert b.reset(i,1,0,feedback(g.output,0))
    assert b.submit(0,message(now=TICK_NS),TICK_NS)
    outputs=b.step(TICK_NS,[feedback(b.output(i),TICK_NS) for i in range(2)])
    assert outputs[0].accepted_sequence==1 and outputs[1].accepted_sequence==0
    other=b.output(1)
    assert b.reset(0,2,TICK_NS,feedback(outputs[0],TICK_NS,2))
    np.testing.assert_array_equal(b.output(1).q,other.q)
    with pytest.raises(ValueError): b.step(2*TICK_NS,[])


def test_nonzero_acceleration_reset_has_continuous_stop():
    c=cfg();g=Governor(c);f=Feedback();f.session=1
    f.q=f.desired_q=c.default_position
    f.dq=f.desired_dq=[.01]*7;f.desired_ddq=[.02]*7
    assert g.reset(1,0,f)
    np.testing.assert_array_equal(g.output.ddq,[.02]*7)
    g.stop();prior=np.array(g.output.ddq)
    for k in range(1,c.horizon_ticks+1):
        o=advance(g,k*TICK_NS)
        assert max(abs(np.array(o.ddq)-prior))/.001<=30+1e-8
        prior=np.array(o.ddq)
    assert o.status==Status.TERMINAL
    np.testing.assert_allclose(o.ddq,0,atol=1e-12)


def test_random_retarget_stop_has_bounded_continuation():
    rng=np.random.default_rng(42);c=cfg()
    for stop_tick in [2,37,301,731]:
        g=fresh(c);seq=0
        for k in range(1,stop_tick+1):
            if k%34==1:
                seq+=1;assert g.submit(message(seq,k*TICK_NS,rng.uniform(-.4,.4,7).tolist()),k*TICK_NS)
            advance(g,k*TICK_NS)
        start=np.array(g.output.q);g.stop()
        for k in range(stop_tick+1,stop_tick+c.horizon_ticks+1):
            o=advance(g,k*TICK_NS)
            assert max(abs(np.array(o.q)-start))<=2*c.max_segment_distance+1e-12
        assert o.status==Status.TERMINAL
        np.testing.assert_allclose(o.dq,0,atol=1e-12)
