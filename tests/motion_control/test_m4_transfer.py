from pathlib import Path
import sys
import json
import numpy as np
import pytest
from franka_motion.transfer import load_transfer_suite,prepare_task,sha256,verify_motion_bundle

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deployment/franky_runtime'))
from franky_experiment.motion_demos.backend import FakeMotionBackend,FrankyMotionBackend


def test_five_contracts_and_fixtures_are_portable():
    config=ROOT/'source/franka_motion/franka_motion/config'
    suite=load_transfer_suite(config/'suites/hardware_transfer.yaml')
    model=ROOT/'deployment/hardware_control_audit/2026-09-25/robot_returned_model.urdf'
    assert suite['mode']=='fake' and not suite['hardware_approved']
    for task in suite['tasks']:
        metadata,arrays=prepare_task(task,config,model)
        assert metadata['kind']==task['kind']
        assert all(np.isfinite(v).all() for v in arrays.values() if v.dtype.kind in 'fci')
        if task['kind']=='tracking':assert metadata['ik_validation']['minimum_interior_joint_margin_rad']>.01
        if task['kind']=='posture':assert arrays['reference'].shape[1]==20
    import subprocess
    subprocess.run([sys.executable,'-c',"import franka_motion.transfer,sys; assert not any(n.startswith(('isaaclab','isaacsim')) for n in sys.modules)"],check=True)


def test_fake_lifecycle_and_hardware_fail_closed():
    backend=FakeMotionBackend()
    with pytest.raises(RuntimeError):backend.set_reference(0,[0]*13)
    backend.start({'name':'circle'})
    with pytest.raises(RuntimeError):backend.start({'name':'other'})
    backend.set_reference(0,[0]*13);backend.stop();backend.close()
    assert [e[0] for e in backend.events]==['start','reference','stop']
    with pytest.raises(RuntimeError,match='Hardware mode unavailable'):FrankyMotionBackend()


def test_manifest_detects_missing_extra_and_mutated_files(tmp_path):
    data=tmp_path/'reference.txt';data.write_text('golden')
    (tmp_path/'manifest.json').write_text(json.dumps({'files':{'reference.txt':sha256(data)}}))
    verify_motion_bundle(tmp_path)
    data.write_text('wrong')
    with pytest.raises(ValueError,match='integrity'):verify_motion_bundle(tmp_path)
    data.write_text('golden');(tmp_path/'extra').write_text('extra')
    with pytest.raises(ValueError,match='inventory'):verify_motion_bundle(tmp_path)
