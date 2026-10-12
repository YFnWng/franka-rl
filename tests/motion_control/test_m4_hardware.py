"""No FCI connections: contract rejection and native publication adapter checks."""
from pathlib import Path
import sys
import json
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deployment/franky_runtime'))
from franky_experiment.motion_demos.hardware import load_hardware_config, FrankyMotionBackend
from franky_experiment.core import ConfigError
CONFIG=ROOT/'deployment/franky_runtime/config/m4.hold.pending.yaml'


@pytest.mark.parametrize('fail',[False,True])
def test_native_spawn_priority_scope_restores_python_scheduler(monkeypatch,fail):
    from franky_experiment.motion_demos import hardware
    state={'policy':0,'priority':0}
    monkeypatch.setattr(hardware.os,'sched_getscheduler',lambda tid:state['policy'])
    monkeypatch.setattr(hardware.os,'sched_getparam',lambda tid:hardware.os.sched_param(state['priority']))
    monkeypatch.setattr(hardware.os,'sched_get_priority_max',lambda policy:99)
    def change(tid,policy,param):state.update(policy=policy,priority=param.sched_priority)
    monkeypatch.setattr(hardware.os,'sched_setscheduler',change)
    try:
        with hardware.preserve_caller_scheduler(realtime=True):
            assert state=={'policy':hardware.os.SCHED_FIFO,'priority':99}
            if fail:raise RuntimeError('native startup failed')
    except RuntimeError:
        assert fail
    assert state=={'policy':0,'priority':0}


@pytest.mark.parametrize('name,value', [('approach_duration_s', 0),
                                      ('exit_duration_s', -1),
                                      ('settle_duration_s', float('nan'))])
def test_invalid_tracking_timing_rejected(tmp_path, name, value):
    c=yaml.safe_load(CONFIG.read_text())
    c['task'].update(kind='tracking', trajectory='circle_yz_m2', angular_speed_rad_s=.2)
    c['task'][name]=value
    p=tmp_path/'timing.yaml';p.write_text(yaml.safe_dump(c))
    with pytest.raises(ConfigError, match=name):load_hardware_config(p)


def test_local_tracking_timing_preserves_bundle_and_circle(monkeypatch):
    from franky_experiment.motion_demos import hardware
    c=load_hardware_config(CONFIG)
    c['task'].update(kind='tracking', trajectory='circle_yz_m2', angular_speed_rad_s=.2)
    c['limits']['workspace_lower']=[.276891,-.18,.17]
    c['limits']['workspace_upper']=[.505,.18,.620282]
    snapshot=SimpleNamespace(flange_position_m=[.306891,0.,.590282],
                             flange_orientation_xyzw=[1.,0.,0.,0.],
                             q=[0.,-.785398,0.,-2.356194,0.,1.570796,0.])
    # This test checks timing composition; full IK is checked separately in design audits.
    monkeypatch.setattr(hardware, 'validate_path_ik',
                        lambda *args, **kwargs: ({'minimum_jacobian_singular_value_mixed_units':.2}, None))
    bundle_catalog=Path(c['bundle'])/'config/trajectories.yaml'
    before=bundle_catalog.read_bytes()
    backend=SimpleNamespace(config=c)
    original=FrankyMotionBackend.plan(backend,snapshot)
    c['task'].update(approach_duration_s=18.,exit_duration_s=18.,settle_duration_s=4.)
    slower=FrankyMotionBackend.plan(backend,snapshot)
    assert slower.duration_s == pytest.approx(original.duration_s+26.)
    for offset in (0.,5.,15.):
        np.testing.assert_allclose(original.evaluate(8.+offset).reference.position_m,
                                   slower.evaluate(22.+offset).reference.position_m)
    assert bundle_catalog.read_bytes()==before


def test_pending_validates_offline_but_cannot_execute():
    c=load_hardware_config(CONFIG)
    assert c['task']['kind']=='hold'
    with pytest.raises(ConfigError,match='approved'):load_hardware_config(CONFIG,execute=True)


@pytest.mark.parametrize('name,value',[('joint_velocity',[0.1]*6),('reference_timeout_s',float('nan')),
                                      ('workspace_upper',[0.,0.,0.]),('minimum_command_success_rate',2.)])
def test_bad_guard_config_rejected(tmp_path,name,value):
    c=yaml.safe_load(CONFIG.read_text());c['limits'][name]=value
    p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(c))
    with pytest.raises(ConfigError):load_hardware_config(p)


def test_native_adapter_publishes_sequence_and_pose_without_robot():
    f=pytest.importorskip('franky')
    if not hasattr(f,'M4RuntimeMotion'):pytest.skip('native M4 runtime not installed')
    c=load_hardware_config(CONFIG)
    limits=f.M4RuntimeLimits()
    for name,value in c['limits'].items():setattr(limits,name,value)
    backend=object.__new__(FrankyMotionBackend)
    backend.config=c;backend.f=f;backend.sequence=0
    backend.motion=f.M4RuntimeMotion(f.CartesianImpedanceParams(),limits,f.TorqueStopParams())
    backend.publish([.4,0.,.4],[0.,0.,0.,1.],np.zeros(6))
    assert backend.sequence==1 and backend.motion.fault==0
    assert not backend.motion.drain()
    with pytest.raises(ConfigError):backend.start(SimpleNamespace())


def test_nonfinite_fault_telemetry_remains_valid_json():
    from franky_experiment.motion_demos.hardware import SAMPLE_FIELDS, serialize_runtime_sample
    values={name: 0 for name in SAMPLE_FIELDS}
    values.update(q=np.array([float('nan')]*7), fault=4, min_singular_value=0.,
                  pose=SimpleNamespace(matrix=np.eye(4)), reference_pose=SimpleNamespace(matrix=np.eye(4)))
    result=serialize_runtime_sample(SimpleNamespace(**values))
    assert result['q']==[None]*7 and result['pose'] is None
    assert not result['ee_feedback_valid'] and 'q' in result['nonfinite_fields']
    json.dumps(result,allow_nan=False)


def test_protocol_evidence_handles_robot_binding_without_version():
    from franky_experiment.motion_demos.hardware import connection_protocol_evidence
    evidence=connection_protocol_evidence(SimpleNamespace())
    assert evidence['compatibility_verified'] and evidence['client_protocol_version']==10
    assert evidence['server_version_from_python_api'] is None
    assert not evidence['python_server_version_available']
    with pytest.raises(ConfigError,match='FCI server'):
        connection_protocol_evidence(SimpleNamespace(server_version=9))


def test_preflight_uses_handshake_when_robot_version_api_is_absent(monkeypatch):
    from franky_experiment.motion_demos import hardware
    c=load_hardware_config(CONFIG)
    s=SimpleNamespace(q=[0.,-.785398,0.,-2.356194,0.,1.570796,0.],dq=[0.]*7,
                      has_errors=False,is_in_control=False,f_t_ee=np.eye(4),load_mass_kg=0.,
                      encoder_fk_reported_position_error_m=0.,flange_position_m=[.306891,0.,.590282])
    state=SimpleNamespace(m_ee=0.,tau_ext_hat_filtered=np.zeros(7),joint_collision=np.zeros(7),cartesian_collision=np.zeros(6))
    robot=SimpleNamespace(state=state,model_urdf=(Path(c['bundle'])/'model/robot_returned_model.urdf').read_text(),
                          model=SimpleNamespace(zero_jacobian=lambda *args:np.c_[np.eye(6),np.zeros(6)]))
    backend=object.__new__(FrankyMotionBackend)
    backend.config=c;backend.robot=robot;backend.encoder=None;backend.f=SimpleNamespace(Frame=SimpleNamespace(Flange=0))
    monkeypatch.setattr(hardware,'_snapshot',lambda *args:s)
    assert backend.preflight() is s
    assert backend.connection_evidence['server_version_from_python_api'] is None
