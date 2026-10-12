import importlib.util
from pathlib import Path
import numpy as np
import pytest
from franka_motion.dls_tracking import resolved_rate
ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0,str(ROOT/'deployment/franky_runtime'))
spec=importlib.util.spec_from_file_location('m4_runner',ROOT/'deployment/franky_runtime/run_m4_hardware.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def success():return dict(terminal_state='complete',smooth_stop_ok=True,stop_fault=0,native_fault=0,dropped_samples=0)


def test_three_trials_and_fault_prevents_restart():
    calls=[];pauses=[]
    reports=runner.run_repetitions(3,lambda i:(calls.append(i) or success()),lambda i:(pauses.append(i) or True))
    assert calls==[1,2,3] and pauses==[1,2] and len(reports)==3
    calls=[]
    def trial(i):
        calls.append(i);r=success()
        if i==2:r.update(terminal_state='fault',native_fault=8)
        return r
    assert len(runner.run_repetitions(3,trial,lambda i:True))==2
    assert calls==[1,2]


@pytest.mark.parametrize('field,value',[('smooth_stop_ok',False),('stop_fault',18),('dropped_samples',1),('terminal_state','stopped')])
def test_any_bad_outcome_prevents_restart(field,value):
    r=success();r[field]=value
    assert len(runner.run_repetitions(3,lambda i:r,lambda i:True))==1


def test_operator_stops_between_trials():
    assert len(runner.run_repetitions(3,lambda i:success(),lambda i:False))==1


def test_dls_twist_nullspace_and_scaling():
    j=np.random.default_rng(3).normal(size=(6,7));v=np.full(6,.01)
    args=dict(pose_gain=np.ones(6)*2,task_scale=np.array([1,1,1,.2,.2,.2]),lambda_value=1e-8)
    dq,secondary,factor,rank=resolved_rate(j,np.zeros(6),v,np.ones(7),np.ones(7)*100,**args)
    np.testing.assert_allclose(j@dq,v,atol=1e-12)
    np.testing.assert_allclose(j@secondary,0,atol=1e-12)
    assert rank==6 and factor==1
    dq,_,factor,_=resolved_rate(j,np.ones(6),v,np.ones(7),np.ones(7)*.01,**args)
    assert max(abs(dq))<=.010000001 and factor<1


def test_native_dls_portable_parity():
    f=pytest.importorskip('franky')
    if not hasattr(f,'M4DlsParams'):pytest.skip('comparison wheel not loaded')
    j=np.random.default_rng(7).normal(size=(6,7));p=f.M4DlsParams();p.task_scale=[1,1,1,.2,.2,.2]
    e=np.arange(6)*.001;v=e*.2;posture=np.arange(7)*.01;bounds=np.ones(7)*.4
    actual=f.compute_m4_dls_step(j,e,v,posture,bounds,p)
    expected=resolved_rate(j,e,v,posture,bounds,pose_gain=p.pose_gain,task_scale=p.task_scale,
                          lambda_value=p.lambda_value,rank_tolerance=p.rank_tolerance,posture_gain=p.posture_gain)
    np.testing.assert_allclose(actual[0],expected[0],atol=1e-12)
    np.testing.assert_allclose(actual[1],expected[1],atol=1e-12)
    assert actual[3]==expected[3]


def test_aggregation_counts_faults_but_does_not_average_them():
    spec=importlib.util.spec_from_file_location('m4_report',ROOT/'deployment/franky_runtime/report_m4_comparisons.py')
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    rows=[dict(task='tracking',profile_id='nominal',completed=True,position_rms_mm=2.),
          dict(task='tracking',profile_id='nominal',completed=True,position_rms_mm=4.),
          dict(task='tracking',profile_id='nominal',completed=False,position_rms_mm=100.)]
    group=report.aggregate(rows)[0]
    assert group['trial_count']==3 and group['completed_count']==2
    assert group['metrics']['position_rms_mm']['mean']==3.
    assert group['metrics']['position_rms_mm']['count']==2


def test_native_fault_is_stopped_and_persisted_before_next_trial(tmp_path,monkeypatch):
    from dataclasses import dataclass
    from types import SimpleNamespace
    import json
    @dataclass
    class Snapshot:
        q:tuple=(0.,)*7
        flange_position_m:tuple=(.3,0.,.5)
        flange_orientation_xyzw:tuple=(1.,0.,0.,0.)
    class Backend:
        motion=None
        stopped=False
        def preflight(self):return Snapshot()
        def plan(self,s):return None
        def start(self,s):self.motion=SimpleNamespace(fault=8,stop_fault=0,dropped_samples=0)
        def stop(self,log):self.stopped=True;return True
    monkeypatch.setattr(runner,'operator_stop_pending',lambda:False)
    c={'task':{'kind':'hold','duration_s':.1},'bundle_manifest_sha256':'fake'}
    config=tmp_path/'config.yaml';config.write_text('fake')
    backend=Backend()
    report=runner.run_trial(backend,c,config,tmp_path/'trial-01',1)
    assert backend.stopped and report['native_fault']==8 and report['terminal_reason']=='pose_error'
    assert not runner.trial_can_repeat(report)
    assert json.loads((tmp_path/'trial-01/report.json').read_text())['terminal_state']=='fault'


def test_posture_profiles_reuse_exact_tested_targets_across_routes():
    import json
    from types import SimpleNamespace
    from franky_experiment.motion_demos.hardware import FrankyMotionBackend,load_hardware_config
    base=Path('/home/chen-lab/franka_ros2_ws/hardware_inventory/2026-10-12/m4/posture.20261012T150500Z')
    if not (base/'start.json').exists():pytest.skip('Recorded hardware posture fixture is unavailable')
    snapshot=SimpleNamespace(**json.loads((base/'start.json').read_text()))
    expected=json.loads((base/'posture-plan.json').read_text())
    for profile in ('cartesian.posture.k1.d1','joint.posture.k100.d1'):
        backend=object.__new__(FrankyMotionBackend)
        backend.config=load_hardware_config(ROOT/'deployment/franky_runtime/config/comparisons'/f'{profile}.pending.yaml')
        assert backend.plan(snapshot) is None
        assert backend.posture_plan['targets_rad']==expected['targets_rad']
        assert backend.posture_plan['fixed_position_m']==expected['fixed_position_m']
        assert backend.posture_plan['source_sha256']
