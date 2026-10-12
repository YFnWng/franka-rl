"""Offline hardware M4 metrics from saved config, report and servo.jsonl.

Logged reference/state pairs define tracking error. Phase boundaries use elapsed
robot time from the first servo sample, not a recorded publisher trajectory clock;
exclude 25 ms around boundaries to avoid publication alignment ambiguity.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from franka_motion.config import TrajectorySpec
from franka_motion.metrics import circle_contour_error, sampled_curve_contour_error
from franka_motion.trajectories import rotation_matrix_rpy_deg


def stats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return None
    return dict(count=int(x.size), minimum=float(x.min()), mean=float(x.mean()),
                rms=float(np.sqrt(np.mean(x*x))), p95=float(np.percentile(x, 95)),
                maximum=float(x.max()))


def pose_errors(pose, reference):
    position = np.linalg.norm(pose[:, :3, 3]-reference[:, :3, 3], axis=1)
    relative = np.swapaxes(pose[:, :3, :3], 1, 2) @ reference[:, :3, :3]
    rotation = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1))
    return position, rotation


def lissajous_contour_errors(positions, specification, sample_count=4096):
    """Nearest sampled curve upper approximation with a discretization bound."""
    theta = np.arange(sample_count)*2*np.pi/sample_count
    amplitudes = np.asarray(specification['amplitudes_m'])
    harmonics = np.asarray(specification['harmonics'])
    offsets = np.asarray(specification['phase_offsets_rad'])
    local = np.column_stack((amplitudes*np.sin(theta[:,None]*harmonics+offsets),
                             np.zeros(sample_count)))
    rotation = rotation_matrix_rpy_deg(specification['plane_rpy_deg'])
    curve = np.asarray(specification['center_m']) + local @ rotation.T
    error = sampled_curve_contour_error(positions, curve)
    # ||dp/dtheta|| <= ||A*h||; nearest parameter grid point is <= pi/N away.
    bound = float(np.linalg.norm(amplitudes*harmonics)*np.pi/sample_count)
    return error, bound


def summarize(directory):
    c = yaml.safe_load((directory/'config.yaml').read_text())
    report = json.loads((directory/'report.json').read_text())
    rows = [json.loads(line) for line in (directory/'servo.jsonl').open() if line.strip()]
    if not rows:
        raise ValueError('No servo samples')
    elapsed = np.array([s['robot_time_s']-rows[0]['robot_time_s'] for s in rows])
    phase = np.full(len(rows), c['task']['kind'], dtype=object)
    boundary = np.zeros(len(rows), dtype=bool)
    spec = None
    planned_track_duration = None
    if c['task']['kind'] == 'tracking':
        catalog = yaml.safe_load((Path(c['bundle'])/'config/trajectories.yaml').read_text())
        spec = dict(catalog['trajectories'][c['task']['trajectory']])
        for key in ('approach_duration_s', 'settle_duration_s', 'exit_duration_s'):
            if key in c['task']:
                spec[key] = c['task'][key]
        start = json.loads((directory/'start.json').read_text())
        path = TrajectorySpec(c['task']['trajectory'], spec).build(
            c['task']['angular_speed_rad_s'], start['flange_position_m'], start['flange_orientation_xyzw'])
        a, b = spec['approach_duration_s'], path.track_start_s
        end = b + path.path.duration_s
        planned_track_duration = path.path.duration_s
        phase[:] = 'exit'
        phase[elapsed < end] = 'track'
        phase[elapsed < b] = 'settle'
        phase[elapsed < a] = 'approach'
        for time in (a, b, end):
            boundary |= np.abs(elapsed-time) <= .025
    elif c['task']['kind']=='posture' and 'posture_offset_rad' in c['task']:
        task=c['task']; end=task['posture_hold_s']
        phase[:]='initial_hold'
        for label in ('a','b',*(['return'] if task['posture_return_to_start'] else [])):
            boundary |= np.abs(elapsed-end)<=.025
            transition_end=end+task['posture_transition_s']
            settle_end=transition_end+task['posture_settle_s']
            phase[(elapsed>=end)&(elapsed<transition_end)]=label+'_transition'
            phase[(elapsed>=transition_end)&(elapsed<=settle_end)]=label+'_settle'
            boundary |= np.abs(elapsed-transition_end)<=.025
            end=settle_end
    fault_indices = [i for i, s in enumerate(rows) if s['fault']]
    fault = None
    if fault_indices:
        i = fault_indices[0]
        fault = dict(elapsed_s=float(elapsed[i]), code=rows[i]['fault'],
                     phase=str(phase[i]), command_success_rate=rows[i]['command_success_rate'],
                     position_error_limit_m=c['limits']['position_error_m'],
                     command_success_limit=c['limits']['minimum_command_success_rate'])
    valid = np.array([s.get('ee_feedback_valid', False) and
                      s.get('pose') is not None and s.get('reference_pose') is not None and
                      not s.get('nonfinite_fields') for s in rows])
    valid &= np.array([np.isfinite(s['robot_time_s']) for s in rows])
    stopping = np.array([bool(s['stopping']) for s in rows])
    phase[stopping] = 'stop'
    valid &= ~np.array([bool(s['fault']) for s in rows]) & ~stopping
    indices = np.flatnonzero(valid)
    records, summaries = [], {}
    contour_approximation = None
    if len(indices):
        samples = [rows[i] for i in indices]
        poses = np.array([s['pose'] for s in samples])
        references = np.array([s['reference_pose'] for s in samples])
        error, orientation = pose_errors(poses, references)
        speed = np.linalg.norm([s['twist'][:3] for s in samples], axis=1)
        angular = np.linalg.norm([s['twist'][3:] for s in samples], axis=1)
        speed_ref = np.linalg.norm([s['reference_twist'][:3] for s in samples], axis=1)
        q = np.array([s['q'] for s in samples])
        margin = np.minimum(q-c['limits']['joint_lower'], c['limits']['joint_upper']-q).min(axis=1)
        displacement = np.linalg.norm(q-np.array(rows[0]['q']), axis=1)
        posture_error = np.linalg.norm(q-np.asarray([s['posture'] for s in samples]),axis=1) \
            if c['task']['kind']=='posture' else None
        contour = None
        lissajous_contour = None
        if spec and spec['type'] == 'circle':
            contour = circle_contour_error(poses[:, :3, 3], spec['center_m'],
                                          spec['radius_m'], rotation_matrix_rpy_deg(spec['plane_rpy_deg']))
        elif spec and spec['type'] == 'lissajous':
            track_mask = (phase[indices]=='track') & ~boundary[indices]
            lissajous_contour = np.full(len(indices),np.nan)
            if track_mask.any():
                values,bound = lissajous_contour_errors(poses[track_mask,:3,3],spec)
                lissajous_contour[track_mask] = values
                contour_approximation = {'method':'nearest of 4096 analytic curve samples; upper approximation',
                                         'maximum_distance_overestimate_m':bound}
        for label in ('all_active', *dict.fromkeys(phase[indices])):
            mask = np.ones(len(indices), dtype=bool) if label == 'all_active' else (
                (phase[indices] == label) & ~boundary[indices])
            summaries[str(label)] = {
                'sample_count':int(mask.sum()), 'position_error_m':stats(error[mask]),
                'orientation_error_rad':stats(orientation[mask]),
                'measured_linear_speed_m_s':stats(speed[mask]),
                'measured_angular_speed_rad_s':stats(angular[mask]),
                'reference_linear_speed_m_s':stats(speed_ref[mask]),
                'speed_excess_over_reference_m_s':stats(np.maximum(0, speed[mask]-speed_ref[mask])),
                'joint_displacement_norm_rad':stats(displacement[mask]),
                'posture_reference_error_norm_rad':stats(posture_error[mask]) if posture_error is not None else None,
                'joint_limit_margin_rad':stats(margin[mask]),
                # Geometric circle distance is meaningful only on the circle phase.
                'circle_contour_error_m':stats(contour[mask]) if contour is not None and label == 'track' else None,
                'sampled_lissajous_contour_error_m':stats(lissajous_contour[mask])
                    if lissajous_contour is not None and label == 'track' else None}
        for k, i in enumerate(indices):
            records.append(dict(elapsed_s=float(elapsed[i]), phase=str(phase[i]),
                                boundary_excluded=bool(boundary[i]), position_error_m=float(error[k]),
                                orientation_error_rad=float(orientation[k]),
                                circle_contour_error_m=float(contour[k]) if contour is not None and phase[i]=='track' else None,
                                sampled_lissajous_contour_error_m=float(lissajous_contour[k])
                                    if lissajous_contour is not None and np.isfinite(lissajous_contour[k]) else None,
                                measured_speed_m_s=float(speed[k]), reference_speed_m_s=float(speed_ref[k])))
            records[-1].update({f'measured_{axis}_m':float(poses[k,j,3]) for j,axis in enumerate('xyz')})
            records[-1].update({f'reference_{axis}_m':float(references[k,j,3]) for j,axis in enumerate('xyz')})
        if fault:
            fs = rows[fault_indices[0]]
            if fs.get('pose') is not None and fs.get('reference_pose') is not None:
                e, o = pose_errors(np.array([fs['pose']]), np.array([fs['reference_pose']]))
                fault.update(position_error_m=float(e[0]), orientation_error_rad=float(o[0]))
    dt = np.diff([s['robot_time_s'] for s in rows])
    host_dt = np.diff([s['monotonic_ns'] for s in rows])/1e9
    track = (phase=='track') & valid & ~boundary
    summary = dict(schema_version=1, source_directory=str(directory.resolve()), terminal_report=report,
        effective_limits=json.loads((directory/'effective-limits.json').read_text())
            if (directory/'effective-limits.json').exists() else None,
        completed=report['terminal_state']=='complete', first_fault=fault,
        contour_approximation=contour_approximation,
        sample_count=len(rows), excluded_fault_stop_invalid_samples=int((~valid).sum()),
        boundary_excluded_samples=int((boundary & valid).sum()), phases=summaries,
        observed_track_span_s=float(np.ptp(elapsed[track])) if track.sum()>1 else 0.,
        planned_track_duration_s=planned_track_duration,
        timing=dict(robot_sample_interval_s=stats(dt), host_callback_interval_s=stats(host_dt),
                    host_interval_error_from_1ms_s=stats(host_dt-.001),
                    reference_age_s=stats([s['reference_age_s'] for s in rows]),
                    command_success_rate_after_first_tick=stats([s['command_success_rate'] for s in rows[1:]]),
                    active_reference_age_s=stats([rows[i]['reference_age_s'] for i in indices]),
                    active_command_success_rate_after_first_tick=stats([
                        rows[i]['command_success_rate'] for i in indices if i>0])),
        limitations=['Phase boundaries inferred from first servo robot time; 25 ms boundary exclusion.',
                     'Errors use applied logged references and encoder/model FK, not external pose metrology.',
                     'RMS is sample-weighted; faulted/partial runs are not successful full-circle trials.',
                     'No acceleration/jerk differentiation or calibrated external-force/stiffness estimate.',
                     'Fixture/posture-specific curve progress and manipulability are not implemented here.'])
    return summary, records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path, help='Default: RUN_DIR/m4-metrics.json')
    args=parser.parse_args()
    summary, records=summarize(args.run_dir)
    output=args.output or args.run_dir/'m4-metrics.json'
    output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if records:
        with output.with_suffix('.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
