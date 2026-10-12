"""Index saved trials and produce a brief repeat-aware report; offline only.

Pass session directories or an --index JSON with a trials list of run directories.
Unclassified old runs are exploratory, never silently grouped into matched repeats.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import yaml

HERE=Path(__file__).resolve().parent


def aggregate(trials):
    groups={}
    for row in trials:groups.setdefault((row['task'],row['profile_id']),[]).append(row)
    result=[]
    for (task,profile),rows in groups.items():
        completed=[r for r in rows if r['completed']]
        out={'task':task,'profile_id':profile,'trial_count':len(rows),'completed_count':len(completed),'metrics':{}}
        for field in ('position_rms_mm','position_max_mm','orientation_rms_deg','contour_rms_mm','posture_error_rms_rad'):
            vals=[r[field] for r in completed if r.get(field) is not None]
            out['metrics'][field]=None if not vals else {'mean':float(np.mean(vals)),
                'std_between_trials':float(np.std(vals,ddof=1)) if len(vals)>1 else None,'count':len(vals)}
        result.append(out)
    return result


def plot(data_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads(data_path.read_text());out=data_path.parent
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    shown_reference=set()
    for row in data['trials']:
        f=Path(row['run_dir'])/'m4-metrics.csv'
        if not f.exists():continue
        records=list(csv.DictReader(f.open()))
        records=[r for r in records if r['phase']=='track'] if row['task']=='tracking' else records
        if not records:continue
        label=f"{row['profile_id']} / {Path(row['run_dir']).name}"
        if row['task'] not in shown_reference:
            axes[0].plot([float(r['reference_y_m']) for r in records],[float(r['reference_z_m']) for r in records],
                         '--',color='black',linewidth=1,label='Applied reference: '+row['task'])
            shown_reference.add(row['task'])
        axes[0].plot([float(r['measured_y_m']) for r in records],[float(r['measured_z_m']) for r in records],label=label,alpha=.7)
        axes[1].plot([float(r['elapsed_s']) for r in records],[float(r['position_error_m'])*1000 for r in records],label=label,alpha=.7)
        axes[2].plot([float(r['elapsed_s']) for r in records],[float(r['joint_displacement_norm_rad']) for r in records],label=label,alpha=.7)
    axes[0].set(xlabel='Base Y (m)',ylabel='Base Z (m)',title='Measured encoder-FK path');axes[0].axis('equal')
    axes[2].set(xlabel='Elapsed time (s)',ylabel='Joint excursion norm (rad)',title='Realized arm movement')
    axes[1].set(xlabel='Elapsed time (s)',ylabel='Position error (mm)',title='Applied-reference error')
    if data['trials'] and len(data['trials'])<=12:axes[1].legend(fontsize=6)
    for a in axes:a.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'paths-and-error.png',dpi=180);fig.savefig(out/'paths-and-error.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,max(4,len(data['groups'])*.45)))
    labels=[g['task']+' / '+g['profile_id'] for g in data['groups']]
    for axis,field,title in zip(axes,('position_rms_mm','orientation_rms_deg'),('Position RMS (mm)','Orientation RMS (deg)')):
        for i,g in enumerate(data['groups']):
            m=g['metrics'][field]
            if m is not None:axis.errorbar(m['mean'],i,xerr=m['std_between_trials'] or 0,fmt='o',capsize=3)
        axis.set_yticks(range(len(labels)));axis.set_yticklabels(labels,fontsize=7)
        axis.set_xlabel(title);axis.grid(axis='x',alpha=.25)
    fig.suptitle('Completed-trial means; error bars show between-trial standard deviation')
    fig.tight_layout();fig.savefig(out/'profile-errors.png',dpi=180);fig.savefig(out/'profile-errors.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('sessions',nargs='*',type=Path)
    p.add_argument('--index',type=Path);p.add_argument('--output-dir',type=Path)
    p.add_argument('--plot-only',type=Path);p.add_argument('--plots-python',help='Optional plotting interpreter with matplotlib')
    args=p.parse_args()
    if args.plot_only:plot(args.plot_only);return
    if not args.output_dir:p.error('--output-dir required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    dirs=set()
    for session in args.sessions:
        if (session/'servo.jsonl').exists():dirs.add(session.resolve())
        else:dirs.update(f.parent.resolve() for f in session.rglob('report.json') if (f.parent/'config.yaml').exists())
    if args.index:
        value=json.loads(args.index.read_text())
        dirs.update((args.index.parent/Path(r['run_dir'])).resolve() for r in value['trials'])
    if not dirs:p.error('No servo.jsonl trials found')
    trials=[]
    for directory in sorted(dirs):
        report=json.loads((directory/'report.json').read_text());c=yaml.safe_load((directory/'config.yaml').read_text())
        # Regenerate metrics with current phase/event and torque handling.
        result=subprocess.run([sys.executable,str(HERE/'summarize_m4.py'),str(directory)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        m=json.loads((directory/'m4-metrics.json').read_text()) if result.returncode==0 else {
            'completed':False,'phases':{},'analysis_error':result.stderr[-2000:]}
        h=m['phases'].get('track',m['phases'].get('all_active',{}))
        def val(key,stat):return None if not h.get(key) else h[key][stat]
        def converted(key,stat,scale):
            v=val(key,stat);return None if v is None else v*scale
        trials.append({'run_dir':str(directory),'profile_id':c.get('experiment',{}).get('profile_id','exploratory:'+directory.name),
            'task':c['task']['kind'],'controller':c['controller'],'config_sha256':report.get('config_sha256'),
            'bundle_manifest_sha256':report.get('bundle_manifest_sha256'),
            'completed':m['completed'],'terminal_reason':report.get('terminal_reason',''),
            'smooth_stop_ok':report.get('smooth_stop_ok'), 'position_rms_mm':converted('position_error_m','rms',1000),
            'position_max_mm':converted('position_error_m','maximum',1000),'orientation_rms_deg':converted('orientation_error_rad','rms',180/np.pi),
            'contour_rms_mm':converted('circle_contour_error_m','rms',1000),
            'posture_error_rms_rad':val('posture_reference_error_norm_rad','rms'),
            'posture_excursion_max_rad':val('joint_displacement_norm_rad','maximum'),
            'analysis_error':m.get('analysis_error'),'phase_source':m.get('phase_source'),'slew_active_fraction':m.get('slew_active_fraction')})
    data={'schema_version':1,'trials':trials,'groups':aggregate(trials)}
    index=args.output_dir/'trial-index.json';index.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    fields=[k for k in trials[0] if k!='controller']
    with (args.output_dir/'trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in trials)
    lines=['# M4 hardware controller comparison','',
           '| Task / profile | Completed / total | Position RMS (mm) | Orientation RMS (deg) |',
           '|---|---:|---:|---:|']
    for group in data['groups']:
        def fmt(field):
            v=group['metrics'][field]
            if v is None:return '—'
            return f"{v['mean']:.3f}"+(f" ± {v['std_between_trials']:.3f}" if v['std_between_trials'] is not None else ' (one trial)')
        lines.append(f"| {group['task']} / {group['profile_id']} | {group['completed_count']} / {group['trial_count']} | {fmt('position_rms_mm')} | {fmt('orientation_rms_deg')} |")
    lines+=['','Errors use applied references and encoder/model FK. Tracking rows use the track phase; other tasks use all active samples.',
            'Spread is between completed trials; faults remain indexed. Exploratory runs are not matched controller comparisons.',
            'Controller configurations, hashes, phase source, torque-slew fraction and source directories are in trial-index.json.',
            'Gains and manual-force differences are not causal evidence from a single trial. No external pose/force metrology.','',
            '![Paths and error](paths-and-error.png)','', '![Profile errors](profile-errors.png)','', '## Source trials','']
    lines.extend(f"- [{r['profile_id']}]({r['run_dir']}/m4-metrics.json): {r['terminal_reason'] or 'complete'}" for r in trials)
    (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
    if args.plots_python:
        subprocess.run([args.plots_python,str(Path(__file__).resolve()),'--plot-only',str(index)],check=True)
    else:
        try:plot(index)
        except ImportError:print('Report and CSV saved. Use --plots-python with an interpreter containing matplotlib.')
    print(args.output_dir/'report.md')


if __name__=='__main__':main()
