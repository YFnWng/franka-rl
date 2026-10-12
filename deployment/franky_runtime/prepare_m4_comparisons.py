"""Generate local comparison candidates from executed configs; immutable bundle untouched.

--approve-profile creates an executable copy of one explicitly selected profile.
No FCI connection. New DLS route should first be commissioned with --repeats 1.
"""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import math
import yaml
from franky_experiment.motion_demos.hardware import load_hardware_config

HERE=Path(__file__).resolve().parent
INVENTORY=Path('/home/chen-lab/franka_ros2_ws/hardware_inventory')
VERSION='2.0.1.dev66+gm4comparefix.libfranka.0.21.2'


def profiles():
    base=INVENTORY/'2026-10-12/m4'
    templates={'circle':yaml.safe_load((base/'tracking.20261012T142902Z/config.yaml').read_text()),
               'posture':yaml.safe_load((base/'posture.20261012T150500Z/config.yaml').read_text())}
    old=yaml.safe_load((INVENTORY/'2026-09-25/franky_reference/reference.j3-j7-k100-smoke.20260926171741.pending.yaml').read_text())
    nominal_damping=old['robot']['impedance_controller']['damping_nms_rad']
    for task in ('circle','posture'):
        cases=[('cartesian',k,d) for k,d in ((.5,1),(1,1),(1.5,1),(1,.75),(1,1.25))]
        cases += [('joint',k,d) for k,d in ((50,1),(100,1),(200,1),(100,.75),(100,1.25))]
        if task=='posture':cases += [('cartesian_dynamic',1,1)]
        for route,k,d in cases:
            c=copy.deepcopy(templates[task]);profile=f'{route}.{task}.k{k:g}.d{d:g}'
            c.update(status='pending',executable=False,motion_authorized=False)
            c['approval']={'approved_by':'','approved_at':'','physical_stop_procedure':'Terminal stop/Ctrl-C; reachable physical E-stop.',
                'workspace_review':'Same catalog path / feasible posture schedule as successful source run; live checks retained. Comparison profile '+profile}
            c['experiment']={'profile_id':profile,'repeats':3,'inter_trial_s':5.,
                'repeat_start_tolerance_rad':.1,'repeat_position_tolerance_m':.02,'repeat_orientation_tolerance_rad':.1}
            if task=='posture':c['experiment']['posture_plan_path']=str(base/'posture.20261012T150500Z/posture-plan.json')
            c['robot']['expected_franky_version']=VERSION
            # Fixed nominal Cartesian gains also serve as stop gains for the joint route.
            if route.startswith('cartesian'):
                c['controller']['kind']='cartesian_impedance'
                for i in range(3):
                    c['controller']['stiffness'][i]*=k
                    c['controller']['damping'][i]*=math.sqrt(k)*d
                # Decouple stop gains from each experiment's active gains.
            else:
                c['controller']['kind']='dls_joint_pd'
                c['controller']['dls']={'stiffness':[float(k)]*7,'damping':[v*math.sqrt(k/100)*d for v in nominal_damping],
                    'reference_error_limit':[.15]*7,'pose_gain':[2.]*6,'task_scale':[1.,1.,1.,.2,.2,.2],
                    'lambda_value':.01,'rank_tolerance':1e-6,'posture_gain':.5 if task=='posture' else 0.}
            c['controller']['projector']='dynamic' if route=='cartesian_dynamic' or task=='circle' and route=='cartesian' else 'kinematic'
            c['controller']['stop_stiffness']=[800.,800.,800.,40.,40.,40.]
            c['controller']['stop_damping']=[56.568542494923804]*3+[13.]*3
            c['required_comparison_runtime_version']=1
            yield profile,c
    c=copy.deepcopy(templates['circle']);c['task']={'kind':'hold','duration_s':5.}
    c['controller'].update(kind='dls_joint_pd',projector='kinematic',posture_stiffness=0.,posture_damping=0.,
        dls={'stiffness':[100.]*7,'damping':nominal_damping,'reference_error_limit':[.15]*7,
             'pose_gain':[2.]*6,'task_scale':[1.,1.,1.,.2,.2,.2],'lambda_value':.01,'rank_tolerance':1e-6,'posture_gain':0.})
    c['controller'].update(stop_stiffness=[800.,800.,800.,40.,40.,40.],stop_damping=[56.568542494923804]*3+[13.]*3)
    c['robot']['expected_franky_version']=VERSION;c.update(status='pending',executable=False,motion_authorized=False)
    c['approval']={'approved_by':'','approved_at':'','physical_stop_procedure':'Terminal stop/Ctrl-C; physical E-stop.',
                   'workspace_review':'DLS matched-start hold commissioning; no planned motion.'}
    c['experiment']={'profile_id':'joint.hold.k100.d1','repeats':3,'inter_trial_s':5.}
    c['required_comparison_runtime_version']=1
    yield 'joint.hold.k100.d1',c


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=HERE/'config/comparisons')
    p.add_argument('--approve-profile');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if a.approve_profile:
        src=a.output_dir/(a.approve_profile+'.pending.yaml');c=yaml.safe_load(src.read_text())
        c.update(status='approved',executable=True,motion_authorized=True)
        c['approval'].update(approved_by='Operator explicitly selected profile via --approve-profile',approved_at=datetime.now(timezone.utc).isoformat())
        dst=a.output_dir/(a.approve_profile+'.approved.yaml')
        if dst.exists():raise FileExistsError(dst)
        dst.write_text(yaml.safe_dump(c,sort_keys=False));load_hardware_config(dst,execute=True);print(dst);return
    index=[]
    for profile,c in profiles():
        dst=a.output_dir/(profile+'.pending.yaml')
        if dst.exists():raise FileExistsError(dst)
        dst.write_text(yaml.safe_dump(c,sort_keys=False));load_hardware_config(dst)
        index.append({'profile_id':profile,'config':dst.name,'repeats':3})
    (a.output_dir/'profiles.json').write_text(json.dumps({'profiles':index},indent=2)+'\n')
    print(f'{len(index)} candidates saved under {a.output_dir}; no robot connection')


if __name__=='__main__':main()
