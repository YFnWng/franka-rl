"""Portable five-demo transfer contracts. No Isaac, Franky, network or robot imports."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .config import _load, _keys, TrajectorySpec
from .kinematics import UrdfKinematicsProvider, DampedLeastSquaresIK, validate_path_ik
from .trajectories import validate_trajectory, sample_trajectory
from .time_scaling import QuinticSegment
from .virtual_fixture import CurveVirtualFixture, load_curve_fixture, load_demo_b


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_transfer_suite(path):
    data=_load(Path(path))
    fields={'schema_version','mode','hardware_approved','controller','reference_hz','servo_hz','tracking_posture','tasks'}
    _keys(data,fields,fields,'hardware transfer')
    if data['mode'] not in ('fake','no_motion','hardware') or data['hardware_approved'] is not False:
        raise ValueError('Transfer candidates must be explicitly unapproved')
    if data['reference_hz']!=50 or data['servo_hz']!=1000 or data['controller']!='cartesian_soft':
        raise ValueError('Transfer requires audited candidate 50 Hz / 1 kHz contract')
    expected={'circle_tracking':'tracking','lissajous_tracking':'tracking','static_posture':'posture','z_impedance':'curve','circle_impedance':'curve'}
    tasks=data['tasks']
    if len(tasks)!=5 or {t['name']:t['kind'] for t in tasks}!=expected:
        raise ValueError('Expected exactly the five transfer demos')
    for task in tasks:
        allowed={'name','kind','trajectory','angular_speed_rad_s','twist_feedforward'} if task['kind']=='tracking' else {'name','kind','configuration'}|({'curve'} if task['kind']=='curve' else set())
        _keys(task,allowed,allowed,'transfer task')
        if task['kind']=='tracking' and task['trajectory']!=({'circle_tracking':'circle_yz_m2','lissajous_tracking':'lissajous_yz_small'}[task['name']]):
            raise ValueError('Unexpected transfer trajectory')
        if task['kind']=='posture' and task['configuration']!='demo_b_fixture_posture.yaml':
            raise ValueError('Unexpected posture configuration')
        if task['kind']=='curve' and (task['configuration']!='curve_fixture.yaml' or task['curve']!=({'z_impedance':'z_line','circle_impedance':'yz_circle'}[task['name']])):
            raise ValueError('Unexpected curve configuration')
    for key in ('stiffness','damping','rcond'):
        if not np.isfinite(data['tracking_posture'][key]) or data['tracking_posture'][key]<=0:
            raise ValueError('Invalid posture profile')
    return data


def prepare_task(task,config_dir,urdf):
    """Deterministic startup/validation fixtures, not executable hardware trajectories."""
    config_dir=Path(config_dir)
    workspace=_load(config_dir/'fr3_workspace.yaml')['workspace']
    q=np.array(workspace['initial_postures_rad']['deployment_home'])
    if sha256(urdf)!=workspace['model']['urdf_sha256']:
        raise ValueError('Robot-returned URDF hash mismatch')
    model=UrdfKinematicsProvider(urdf)
    p,r,j=model.evaluate(q)
    bounds=workspace['joint_bounds_rad']['commanded']
    solver=DampedLeastSquaresIK(model,bounds['lower'],bounds['upper'])
    if task['kind']=='tracking':
        catalog=_load(config_dir/'trajectories.yaml')['trajectories']
        spec=TrajectorySpec(task['trajectory'],catalog[task['trajectory']])
        path=spec.build(task['angular_speed_rad_s'],p,r)
        box=workspace['cartesian_bounds_m']; lim=workspace['reference_limits']
        report=validate_trajectory(path,max_speed_m_s=lim['speed_m_s'],max_acceleration_m_s2=lim['acceleration_m_s2'],
            max_jerk_m_s3=lim['jerk_m_s3'],lower_m=np.minimum(box['lower'],p)-1e-6,upper_m=np.maximum(box['upper'],p)+1e-6)
        ik,_=validate_path_ik(path,solver,q,sample_hz=5.)
        if ik['minimum_jacobian_singular_value_mixed_units']<.02 or ik['minimum_interior_joint_margin_rad']<.01:
            raise ValueError('Tracking IK margin/singularity gate failed')
        samples=sample_trajectory(path,sample_hz=50.)
        if np.linalg.norm(samples['angular_velocity_rad_s'],axis=1).max()>.5:
            raise ValueError('Angular reference speed exceeds candidate limit')
        return {'kind':'tracking','trajectory':spec.data,'validation':report,'ik_validation':ik,
            'initial_q':q.tolist(),'initial_pose':np.r_[p,r].tolist(),'twist_feedforward':task['twist_feedforward']},samples
    if task['kind']=='posture':
        c=load_demo_b(config_dir/'suites'/task['configuration'])
        _,_,vh=np.linalg.svd(j,full_matrices=True)
        targets=[solver.solve(p,r,q+sign*c['posture_offset_rad']*vh[-1]) for sign in (-1,1)]
        duration=c['hold_s']+2*(c['transition_s']+c['posture_settle_s'])
        times=np.arange(int(np.ceil(duration*50)))/50
        segments=[QuinticSegment(q,targets[0],c['transition_s']),QuinticSegment(targets[0],targets[1],c['transition_s'])]
        refs=[]
        for t in times:
            elapsed=t-c['hold_s']
            index=0 if elapsed<c['transition_s']+c['posture_settle_s'] else 1
            local=elapsed-index*(c['transition_s']+c['posture_settle_s'])
            posture=q if elapsed<0 else segments[index].evaluate(min(c['transition_s'],max(0.,local))).position
            refs.append(np.r_[p,r,np.zeros(6),posture])
        return {'kind':'posture','config':c,'initial_q':q.tolist(),'initial_pose':np.r_[p,r].tolist(),
            'targets_rad':[a.tolist() for a in targets],'projector':'I-J.T pinv(J.T); original M3 kinematic law',
            'sequence':'hold -> target A quintic -> settle -> target B quintic -> settle'}, {'posture_targets_rad':np.array(targets),'reference':np.array(refs),'time_s':times}
    c=load_curve_fixture(config_dir/'suites'/task['configuration'])
    origin=c['line_origin_m'] if task['curve']=='z_line' else c['circle_center_m']
    curve=CurveVirtualFixture(task['curve'],origin,c['circle_radius_m'])
    closest,_=curve.project(p)
    if np.linalg.norm(closest-p)>.001:
        raise ValueError('Curve does not pass deployment home TCP; no automatic entry')
    # Full-circle / bounded-line IK gate; independent workspace guard still mandatory.
    lower,upper=np.array(c['workspace_lower_m']),np.array(c['workspace_upper_m'])
    angles=np.linspace(0,2*np.pi,65)
    points=(np.array(origin)+np.c_[np.zeros(65),curve.radius*np.sin(angles),curve.radius*np.cos(angles)]
            if task['curve']=='yz_circle' else np.array([[origin[0],origin[1],z] for z in np.linspace(lower[2],upper[2],17)]))
    seed=q.copy()
    for point in points:
        if np.any(point<lower) or np.any(point>upper):raise ValueError('Curve outside candidate workspace')
        seed=solver.solve(point,[1,0,0,0],seed)
    projections=np.array([curve.project(point)[0] for point in points])
    forces=np.array([curve.force(point,[.01,.02,.03],c['normal_stiffness'],c['normal_damping'],c['tangent_damping']) for point in points])
    return {'kind':'curve','curve':task['curve'],'config':c,'initial_q':q.tolist(),'initial_pose':np.r_[p,r].tolist(),
        'hardware_external_force':'human hand only; never replay simulated injected force',
        'projection_rate_hz':1000,'orientation_xyzw':[1,0,0,0],
        'posture_projector':'full six-row dynamically consistent torque projector'}, {'curve_points_m':points,'projected_points_m':projections,'fixture_force_n':forces}


def verify_motion_bundle(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    expected=set(manifest['files'])
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
    if expected!=actual:raise ValueError('Bundle file inventory mismatch')
    for name,digest in manifest['files'].items():
        path=root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or sha256(path)!=digest:
            raise ValueError(f'Bundle integrity failure: {name}')
    return manifest
