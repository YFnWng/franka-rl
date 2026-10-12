"""Validate or run repeated M4 trials. No FCI connection unless explicitly selected."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import select
import sys
import time
import numpy as np

from franky_experiment.core import sha256
from franky_experiment.motion_demos.hardware import load_hardware_config, FrankyMotionBackend, FAULTS
from franka_motion.transfer import verify_motion_bundle


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def reference_phase(c, t, path=None):
    if path:
        if t < path.approach_duration_s: return 'approach'
        if t < path.track_start_s: return 'settle'
        return 'track' if t < path.track_start_s+path.path.duration_s else 'exit'
    task=c['task']
    if task['kind']=='posture' and 'posture_offset_rad' in task:
        end=task['posture_hold_s']
        if t<end: return 'initial_hold'
        for label in ['a','b']+(['return'] if task['posture_return_to_start'] else []):
            end+=task['posture_transition_s']
            if t<end: return label+'_transition'
            end+=task['posture_settle_s']
            if t<end: return label+'_settle'
        return 'return_settle' if task['posture_return_to_start'] else 'b_settle'
    return task['kind']


def operator_stop_pending():
    if select.select([sys.stdin], [], [], 0)[0]:
        command=sys.stdin.readline()
        # EOF means the operator channel is gone; do not restart a new trial.
        return not command or command.strip().lower()=='stop'
    return False


def trial_can_repeat(report):
    return (report.get('terminal_state')=='complete' and report.get('smooth_stop_ok') is True
            and not report.get('stop_fault') and not report.get('native_fault')
            and report.get('dropped_samples')==0)


def run_trial(backend, c, config_path, output, index, frozen_posture=None, repeat_anchor=None):
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.yaml').write_bytes(config_path.read_bytes())
    report={'hardware_executed':False,'terminal_state':'fault','terminal_reason':'startup',
            'trial_index':index,'config_sha256':sha256(config_path),
            'bundle_manifest_sha256':c['bundle_manifest_sha256'],
            'runtime_source_sha256':sha256(Path(__file__)),
            'adapter_source_sha256':sha256(Path(__file__).parent/'franky_experiment/motion_demos/hardware.py')}
    # Do not accidentally report the previous trial's native evidence after preflight failure.
    backend.motion=None
    backend._last_sample=None
    try:
        snapshot=backend.preflight()
        if repeat_anchor is not None:
            experiment=c.get('experiment',{})
            if np.max(np.abs(np.asarray(snapshot.q)-repeat_anchor.q))>experiment.get('repeat_start_tolerance_rad',.1):
                raise ValueError('repeat start joint drift exceeds tolerance; no automatic homing')
            if np.linalg.norm(np.asarray(snapshot.flange_position_m)-repeat_anchor.flange_position_m)>experiment.get('repeat_position_tolerance_m',.02):
                raise ValueError('repeat start Cartesian drift exceeds tolerance')
            dot=abs(np.dot(snapshot.flange_orientation_xyzw,repeat_anchor.flange_orientation_xyzw))
            if 2*np.arccos(np.clip(dot,0,1))>experiment.get('repeat_orientation_tolerance_rad',.1):
                raise ValueError('repeat start orientation drift exceeds tolerance')
        path=backend.plan(snapshot)
        if frozen_posture is not None: backend.posture_plan=copy.deepcopy(frozen_posture)
        write_json(output/'start.json',asdict(snapshot))
        for name in ('posture_plan','fixture_plan'):
            if hasattr(backend,name): write_json(output/(name.replace('_','-')+'.json'),getattr(backend,name))
        duration=path.duration_s if path else c['task']['duration_s']
        with (output/'servo.jsonl').open('w') as log, (output/'phase-events.jsonl').open('w') as events:
            try:
                backend.start(snapshot)
                report['hardware_executed']=True
                started=time.monotonic();next_tick=started;last_phase=None
                while time.monotonic()-started < duration:
                    if operator_stop_pending():
                        report.update(terminal_state='stopped',terminal_reason='operator_stop');break
                    if backend.motion.fault: raise RuntimeError(FAULTS.get(backend.motion.fault,'native_fault'))
                    if not backend.robot.is_in_control: raise RuntimeError('unexpected_control_exit')
                    t=min(duration,max(0.,time.monotonic()-started))
                    if path:
                        sample=path.evaluate(t).reference
                        backend.publish(sample.position_m,sample.quaternion_xyzw,
                                        [*sample.linear_velocity_m_s,*sample.angular_velocity_rad_s])
                    else:
                        posture=backend.posture_target(t,snapshot)
                        anchor=getattr(backend,'posture_plan',{}) if c['task']['kind']=='posture' else {}
                        backend.publish(anchor.get('fixed_position_m',snapshot.flange_position_m),
                                        anchor.get('fixed_orientation_xyzw',snapshot.flange_orientation_xyzw),
                                        [0.]*6,posture)
                    phase=reference_phase(c,t,path)
                    if phase!=last_phase:
                        events.write(json.dumps({'phase':phase,'elapsed_s':t,'monotonic_ns':time.monotonic_ns(),
                                                 'first_reference_sequence':backend.sequence})+'\n')
                        events.flush();last_phase=phase
                    backend.drain(log)
                    next_tick+=1/c['reference_hz'];time.sleep(max(0.,next_tick-time.monotonic()))
                if report['terminal_reason']=='startup': report.update(terminal_state='complete',terminal_reason='')
            except KeyboardInterrupt:
                report.update(terminal_state='stopped',terminal_reason='operator_interrupt')
            except Exception as error:
                report.update(terminal_state='fault',terminal_reason=str(error))
            finally:
                events.write(json.dumps({'phase':'stop','monotonic_ns':time.monotonic_ns()})+'\n')
                report['smooth_stop_ok']=backend.stop(log)
                report['stop_evidence']=getattr(backend,'stop_evidence',{})
                report['stop_fault']=getattr(backend.motion,'stop_fault',0) if backend.motion else 0
                report['native_fault']=backend.motion.fault if backend.motion else None
                report['dropped_samples']=backend.motion.dropped_samples if backend.motion else 0
                if report['terminal_state']=='complete' and not trial_can_repeat(report):
                    report.update(terminal_state='fault',terminal_reason=FAULTS.get(report['stop_fault'],'stop_or_logging_fault'))
    except KeyboardInterrupt:
        report.update(terminal_state='stopped',terminal_reason='operator_interrupt')
    except Exception as error:
        report.update(terminal_state='fault',terminal_reason=str(error))
    finally:
        for attr,name in [('effective_limit_evidence','effective-limits'),('scheduling_evidence','scheduling-evidence'),
                          ('connection_evidence','connection-evidence')]:
            if hasattr(backend,attr): write_json(output/(name+'.json'),getattr(backend,attr))
        if hasattr(backend,'preflight_snapshot'):write_json(output/'last-preflight.json',asdict(backend.preflight_snapshot))
        write_json(output/'report.json',report)
    print(f"TRIAL {index}: {report['terminal_state']} {report['terminal_reason']}\nData: {output}",flush=True)
    return report


def run_repetitions(count, run_one, between):
    """Pure sequencing: only a completed, settled, lossless trial permits another."""
    reports=[]
    for index in range(1,count+1):
        report=run_one(index);reports.append(report)
        if not trial_can_repeat(report):break
        if index<count and not between(index):break
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--repeats',type=int,help='Default: experiment.repeats, otherwise 3; 1 for commissioning')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--preflight',action='store_true',help='FCI inspection only; no motion API')
    group.add_argument('--execute',action='store_true',help='approved config, one terminal start, then repeated trials')
    args=parser.parse_args();c=load_hardware_config(args.config,execute=args.execute)
    count=args.repeats if args.repeats is not None else c.get('experiment',{}).get('repeats',3)
    if not 1<=count<=10:parser.error('repeats must be 1..10')
    verify_motion_bundle(c['bundle'])
    if sha256(Path(c['bundle'])/'manifest.json')!=c['bundle_manifest_sha256']:raise ValueError('Transfer manifest hash mismatch')
    if not (args.execute or args.preflight):print('Config and immutable transfer verified. No FCI connection.');return 0
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=Path(c['output_root'])/(c['task']['kind']+'.'+stamp);output.mkdir(parents=True,exist_ok=False)
    (output/'config.yaml').write_bytes(args.config.read_bytes())
    session={'terminal_state':'fault','terminal_reason':'startup','requested_repeats':count,'trials':[]}
    try:
        backend=FrankyMotionBackend(c);snapshot=backend.preflight();backend.plan(snapshot)
        write_json(output/'preflight.json',asdict(snapshot))
        for name in ('posture_plan','fixture_plan'):
            if hasattr(backend,name):write_json(output/(name.replace('_','-')+'.json'),getattr(backend,name))
        if args.preflight:
            session.update(terminal_state='complete',terminal_reason='no_motion_preflight');return 0
        frozen=None
        anchor=None
        print(f"READY {c['task']['kind']}: {count} trials; type 'start' then Enter, or 'stop'.",flush=True)
        while True:
            command=input().strip().lower()
            if command=='stop':session.update(terminal_state='stopped',terminal_reason='operator_before_start');return 0
            if command=='start':break
            print("Expected 'start' or 'stop'.",flush=True)
        def between(index):
            end=time.monotonic()+c.get('experiment',{}).get('inter_trial_s',5.)
            print(f'Trial {index} complete. Next trial starts automatically after the interval; type stop to end.',flush=True)
            while time.monotonic()<end:
                if operator_stop_pending():return False
                time.sleep(.05)
            return True
        def run_one(i):
            nonlocal frozen,anchor
            result=run_trial(backend,c,args.config,output/f'trial-{i:02d}',i,frozen,anchor)
            if i==1 and trial_can_repeat(result):
                from types import SimpleNamespace
                anchor=SimpleNamespace(**json.loads((output/'trial-01/start.json').read_text()))
                frozen=copy.deepcopy(getattr(backend,'posture_plan',None))
                if frozen is not None and c['task'].get('posture_return_to_start'):
                    anchor.q=frozen['anchor_q_rad']
                    anchor.flange_position_m=frozen['fixed_position_m']
                    anchor.flange_orientation_xyzw=frozen['fixed_orientation_xyzw']
            session['trials'].append({'index':i,'directory':f'trial-{i:02d}','report':result})
            write_json(output/'session-report.json',session)
            return result
        reports=run_repetitions(count,run_one,between)
        session['trials']=[{'index':i+1,'directory':f'trial-{i+1:02d}','report':r} for i,r in enumerate(reports)]
        if reports and not trial_can_repeat(reports[-1]):
            session.update(terminal_state=reports[-1]['terminal_state'],terminal_reason=reports[-1]['terminal_reason'])
        elif len(reports)==count:session.update(terminal_state='complete',terminal_reason='')
        else:session.update(terminal_state='stopped',terminal_reason='operator_between_trials')
    except KeyboardInterrupt:session.update(terminal_state='stopped',terminal_reason='operator_interrupt')
    except Exception as error:session.update(terminal_state='fault',terminal_reason=str(error))
    finally:
        write_json(output/'session-report.json',session)
        print(f"SESSION {session['terminal_state']}: {session['terminal_reason']}\nData: {output}",flush=True)
    return 1 if session['terminal_state']=='fault' else 0


if __name__=='__main__':raise SystemExit(main())
