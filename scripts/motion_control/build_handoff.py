"""Build an immutable, offline-validated M4 transfer; never access a robot."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np

from franka_motion.transfer import load_transfer_suite,prepare_task,sha256,verify_motion_bundle

REPO=Path(__file__).resolve().parents[2]


def build_motion_handoff(output,evidence=()):
    output=Path(output).resolve()
    data=Path(os.environ.get('FRANKA_RL_DATA_ROOT','/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data')).resolve()
    if not output.is_relative_to(Path('/tmp')):
        if not output.is_relative_to(data) or not any(os.path.ismount(p) for p in (data,*data.parents) if str(p)!='/'):
            raise ValueError('Bundle output must be on mounted FRANKA_RL_DATA_ROOT or bounded /tmp')
    parent=next(p for p in (output.parent,*output.parents) if p.exists())
    if not os.access(parent,os.W_OK) or shutil.disk_usage(parent).free<128*1024**2:
        raise ValueError('Bundle filesystem unavailable or insufficient free space')
    output.mkdir(parents=True,exist_ok=False)
    package=REPO/'source/franka_motion'
    shutil.copytree(package/'franka_motion/config',output/'config')
    shutil.copytree(REPO/'deployment/franky_runtime/franky_experiment/motion_demos',output/'runtime/franky_experiment/motion_demos',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(REPO/'deployment/franky_runtime/run_motion_suite.py',output/'runtime/run_motion_suite.py')
    (output/'model').mkdir()
    shutil.copy2(REPO/'deployment/hardware_control_audit/2026-09-25/robot_returned_model.urdf',output/'model/robot_returned_model.urdf')
    # Source snapshot supports inspection/parity independent of the installed wheel.
    shutil.copytree(package/'franka_motion',output/'source/franka_motion',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (output/'wheels').mkdir()
    with tempfile.TemporaryDirectory(prefix='franka-motion-wheel-') as staging:
        stage=Path(staging)/'package'
        shutil.copytree(package,stage,ignore=shutil.ignore_patterns('__pycache__','*.pyc','build','*.egg-info'))
        subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation',str(stage),'-w',str(output/'wheels')],check=True)
    suite=load_transfer_suite(output/'config/suites/hardware_transfer.yaml')
    (output/'fixtures').mkdir()
    # Recurrence-only golden trace: replay tau_J_d seed, saturation, reversal,
    # and simulator gravity/effort guard without importing Isaac Lab.
    from franka_motion.impedance import TorqueCommandProcessor
    processor=TorqueCommandProcessor()
    seed=np.linspace(-.7,.7,7)
    processor.reset(seed)
    raw=np.repeat(np.array([20.]*10+[-20.]*10+[.2]*10)[:,None],7,axis=1)
    gravity=np.repeat(np.arange(7,dtype=float)[None,:],len(raw),axis=0)
    torque_records=[processor.compute(r,g) for r,g in zip(raw,gravity)]
    np.savez_compressed(output/'fixtures/torque_processing_v2.npz',initial_command=seed,
        **{key:np.stack([row[key] for row in torque_records]) for key in torque_records[0]})
    (output/'fixtures/torque_processing_v2.json').write_text(json.dumps({
        'version':'shared_previous_output_v2','dt_s':processor.dt,'max_delta_nm':processor.step,
        'alpha':processor.alpha,'cutoff_hz':100.,'hardware_limit_rate':False,
        'seed_source':'synthetic tau_J_d for replay test; simulation resets to zero',
        'comparison_channel':'filtered non-gravity outgoing command',
        'effort_clamp':'simulator guard only, not a Franky client clamp'},indent=2)+'\n')
    for task in suite['tasks']:
        metadata,arrays=prepare_task(task,output/'config',output/'model/robot_returned_model.urdf')
        np.savez_compressed(output/'fixtures'/(task['name']+'.npz'),**arrays)
        (output/'fixtures'/(task['name']+'.json')).write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    evidence_index=[]
    for i,root in enumerate(evidence):
        root=Path(root).resolve()
        summaries=sorted(root.rglob('summary.json'))
        if not summaries:raise ValueError(f'No simulation evidence: {root}')
        for summary_path in summaries:
            from importlib.util import spec_from_file_location,module_from_spec
            spec=spec_from_file_location('motion_compile',REPO/'scripts/motion_control/compile_cartesian_results.py')
            module=module_from_spec(spec);spec.loader.exec_module(module)
            summary,_=module.validate(summary_path.parent)
            if not summary['acceptance_passed']:raise ValueError(f'Unaccepted simulation evidence: {summary_path}')
            if summary.get('torque_processing_version')!='shared_previous_output_v2':
                raise ValueError(f'Evidence belongs to an older torque processor: {summary_path}')
            dest=output/'evidence'/str(i)/summary_path.parent.name
            dest.mkdir(parents=True)
            shutil.copy2(summary_path,dest/'summary.json')
            shutil.copy2(summary_path.parent/'FINALIZED.json',dest/'FINALIZED.json')
            evidence_index.append({'source':str(summary_path.parent),'copied_summary':str((dest/'summary.json').relative_to(output)),
                'trace_sha256':summary['trace_sha256'],'trace_included':False,
                'interpretation':'validated accepted run; not automatically matched to current changed YAML'})
    versions={name:importlib.metadata.version(name) for name in ('numpy','PyYAML','setuptools')}
    (output/'requirements-workstation.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items() if k!='setuptools')+'\n')
    (output/'README.md').write_text((REPO/'deployment/MOTION_M4_HANDOFF.md').read_text())
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()
    files={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    manifest={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'git_revision':revision,
        'source_state':'snapshot may include uncommitted changes; file hashes authoritative',
        'hardware_approved':False,'hardware_adapter_ready':False,'tasks':[t['name'] for t in suite['tasks']],
        'torque_processing_version':'shared_previous_output_v2','non_gravity_reset_seed':'zero',
        'versions':versions,'simulation_evidence':evidence_index,'files':files}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    verify_motion_bundle(output)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--simulation-evidence',type=Path,action='append',default=[])
    a=p.parse_args()
    data=Path(os.environ.get('FRANKA_RL_DATA_ROOT','/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data'))
    output=a.output_dir or data/'motion_control/handoff_bundles'/datetime.now().strftime('%Y-%m-%d_%H-%M-%S_m4')
    print('M4 bundle:',build_motion_handoff(output,a.simulation_evidence))


if __name__=='__main__':main()
