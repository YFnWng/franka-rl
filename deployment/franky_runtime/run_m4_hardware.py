"""Validate or run one local M4 commissioning task. No connection by default."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import select
import sys

from franky_experiment.core import sha256
from franky_experiment.motion_demos.hardware import load_hardware_config, FrankyMotionBackend, FAULTS
from franka_motion.transfer import verify_motion_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--preflight', action='store_true', help='FCI inspection only; no motion API. Franky constructor configures collision thresholds.')
    group.add_argument('--execute', action='store_true', help='requires approved local config and terminal start')
    args = parser.parse_args()
    c = load_hardware_config(args.config, execute=args.execute)
    manifest = verify_motion_bundle(c['bundle'])
    if sha256(Path(c['bundle'])/'manifest.json') != c['bundle_manifest_sha256']:
        raise ValueError('Transfer manifest hash mismatch')
    if not (args.execute or args.preflight):
        print('Config and immutable transfer verified. No FCI connection.'); return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path(c['output_root'])/(c['task']['kind']+'.'+stamp)
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.yaml').write_bytes(args.config.read_bytes())
    backend = None
    report = {'hardware_executed':False, 'terminal_state':'fault', 'terminal_reason':'startup',
              'config_sha256':sha256(args.config), 'bundle_manifest_sha256':c['bundle_manifest_sha256']}
    try:
        backend = FrankyMotionBackend(c)
        snapshot = backend.preflight()
        (output/'preflight.json').write_text(json.dumps(asdict(snapshot),indent=2)+'\n')
        if args.preflight:
            report.update(terminal_state='complete',terminal_reason='no_motion_preflight')
            return
        path = backend.plan(snapshot)
        duration = path.duration_s if path else c['task']['duration_s']
        print(f"READY {c['task']['kind']} for {duration:.1f}s; type 'start' then Enter, or 'stop'.", flush=True)
        while True:
            command = input().strip().lower()
            if command == 'stop':
                report.update(terminal_state='stopped',terminal_reason='operator_before_start'); return
            if command == 'start': break
            print("Expected 'start' or 'stop'.", flush=True)
        # Recheck and rebuild entry after the operator wait, using current encoder feedback.
        snapshot = backend.preflight()
        path = backend.plan(snapshot)
        duration = path.duration_s if path else c['task']['duration_s']
        (output/'start.json').write_text(json.dumps(asdict(snapshot),indent=2)+'\n')
        if hasattr(backend, 'posture_plan'):
            (output/'posture-plan.json').write_text(json.dumps(backend.posture_plan,indent=2)+'\n')
        with (output/'servo.jsonl').open('w') as log:
            try:
                backend.start(snapshot)
                report['hardware_executed']=True
                started=time.monotonic(); next_tick=started
                while time.monotonic()-started < duration:
                    if select.select([sys.stdin], [], [], 0)[0]:
                        command = sys.stdin.readline().strip().lower()
                        if command == 'stop':
                            report.update(terminal_state='stopped', terminal_reason='operator_stop')
                            break
                    if backend.motion.fault:
                        raise RuntimeError(FAULTS.get(backend.motion.fault,'native_fault'))
                    if not backend.robot.is_in_control:
                        raise RuntimeError('unexpected_control_exit')
                    t=min(duration,max(0.,time.monotonic()-started))
                    if path:
                        sample=path.evaluate(t).reference
                        backend.publish(sample.position_m,sample.quaternion_xyzw,
                                        [*sample.linear_velocity_m_s,*sample.angular_velocity_rad_s])
                    else:
                        posture=backend.posture_target(t,snapshot)
                        backend.publish(snapshot.flange_position_m,snapshot.flange_orientation_xyzw,[0.]*6,posture)
                    backend.drain(log)
                    next_tick+=1/c['reference_hz']
                    time.sleep(max(0.,next_tick-time.monotonic()))
                if report['terminal_reason']=='startup':
                    report.update(terminal_state='complete',terminal_reason='')
            except KeyboardInterrupt:
                report.update(terminal_state='stopped',terminal_reason='operator_interrupt')
            except Exception as error:
                report.update(terminal_state='fault',terminal_reason=str(error))
            finally:
                report['smooth_stop_ok']=backend.stop(log)
                report['native_fault']=backend.motion.fault if backend.motion else None
                report['dropped_samples']=backend.motion.dropped_samples if backend.motion else 0
                if (not report['smooth_stop_ok'] or report['native_fault'] or report['dropped_samples']) and report['terminal_state']=='complete':
                    report.update(terminal_state='fault',terminal_reason='stop_or_logging_fault')
    except Exception as error:
        report.update(terminal_state='fault',terminal_reason=str(error))
    finally:
        if backend is not None and hasattr(backend, 'effective_limit_evidence'):
            (output/'effective-limits.json').write_text(json.dumps(backend.effective_limit_evidence, indent=2)+'\n')
        if backend is not None and hasattr(backend, 'scheduling_evidence'):
            (output/'scheduling-evidence.json').write_text(json.dumps(backend.scheduling_evidence, indent=2)+'\n')
        if backend is not None and hasattr(backend, 'connection_evidence'):
            (output/'connection-evidence.json').write_text(json.dumps(backend.connection_evidence, indent=2)+'\n')
        if backend is not None and hasattr(backend, 'preflight_snapshot'):
            (output/'last-preflight.json').write_text(json.dumps(asdict(backend.preflight_snapshot), indent=2)+'\n')
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f"SESSION {report['terminal_state']}: {report['terminal_reason']}\nData: {output}", flush=True)
    return 0 if report['terminal_state']!='fault' else 1


if __name__=='__main__': raise SystemExit(main())
