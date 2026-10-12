import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('m4_metrics',ROOT/'deployment/franky_runtime/summarize_m4.py')
metrics=importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def test_geometric_pose_errors():
    pose=np.tile(np.eye(4),(2,1,1));ref=pose.copy()
    ref[0,0,3]=.03
    ref[1,:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    e,r=metrics.pose_errors(pose,ref)
    np.testing.assert_allclose(e,[.03,0])
    np.testing.assert_allclose(r,[0,np.pi/2])


def test_fault_and_stop_do_not_pollute_active_metrics(tmp_path):
    config=yaml.safe_load((ROOT/'deployment/franky_runtime/config/m4.hold.pending.yaml').read_text())
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(config))
    (tmp_path/'report.json').write_text(json.dumps({'terminal_state':'fault'}))
    samples=[]
    for i in range(4):
        pose=np.eye(4);reference=pose.copy();reference[0,3]=.001 if i<2 else .1
        samples.append(dict(robot_time_s=i*.001,monotonic_ns=i*1000000,
            fault=8 if i>=2 else 0,stopping=i>=2,ee_feedback_valid=True,
            pose=pose.tolist(),reference_pose=reference.tolist(),nonfinite_fields=[],
            twist=[0.]*6,reference_twist=[0.]*6,q=[0.,-.8,0.,-2.3,0.,1.5,0.],
            command_success_rate=1.,reference_age_s=.005))
    (tmp_path/'servo.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
    summary,records=metrics.summarize(tmp_path)
    assert not summary['completed']
    assert summary['phases']['all_active']['position_error_m']['rms']==pytest.approx(.001)
    assert len(records)==2 and summary['excluded_fault_stop_invalid_samples']==2
    assert summary['first_fault']['position_error_m']==pytest.approx(.1)
    assert summary['planned_track_duration_s'] is None
    json.dumps(summary,allow_nan=False)
