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
