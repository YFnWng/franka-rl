"""Generate consolidated hardware report figures from saved data; no FCI access."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/m4-report-matplotlib')
import csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'hardware_control_audit/2026-10-12-m4-results/figures'
INV=Path('/home/chen-lab/franka_ros2_ws/hardware_inventory/2026-10-12/m4')
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25,'savefig.dpi':180})
OUT.mkdir(parents=True,exist_ok=True)
provenance={}

def read(run,phase=None):
    rows=list(csv.DictReader((run/'m4-metrics.csv').open()))
    rows=[r for r in rows if (phase is None or r['phase']==phase) and r['boundary_excluded'].lower()!='true']
    return {k:np.array([float(r[k]) for r in rows]) for k in rows[0] if k not in ('phase','boundary_excluded') and rows[0][k]!=''}

def save(fig,name,sources):
    fig.tight_layout();fig.savefig(OUT/(name+'.png'));fig.savefig(OUT/(name+'.pdf'));plt.close(fig)
    provenance[name]=[str(p) for p in sources]

def tracking(runs,name):
    data=[read(p,'track') for _,p in runs]
    colors=['#2563eb','#16a34a','#dc2626','#7c3aed','#0891b2','#ea580c']
    fig,ax=plt.subplots(figsize=(8,6))
    d=data[0];ax.plot(d['reference_y_m'],d['reference_z_m'],'k--',lw=2,label='Target path',zorder=10)
    for i,((label,_),d) in enumerate(zip(runs,data)):
        ax.plot(d['measured_y_m'][::10],d['measured_z_m'][::10],color=colors[i],lw=1.3,label=label,alpha=.85)
    ax.set(xlabel='Base Y (m)',ylabel='Base Z (m)',title=name.capitalize()+' tracking: encoder-FK paths');ax.set_aspect('equal');ax.legend(fontsize=8)
    save(fig,name+'-paths',[p for _,p in runs])
    fig,ax=plt.subplots(figsize=(10,4.5))
    for i,((label,_),d) in enumerate(zip(runs,data)):
        t=d['elapsed_s']-22.0
        ax.plot(t[::10],d['position_error_m'][::10]*1000,color=colors[i],lw=1,label=label)
    ax.set(xlabel='Time from scheduled tracking start (s)',ylabel='Position error (mm)',title=name.capitalize()+' tracking: error to applied timed reference');ax.legend(fontsize=8,ncol=2)
    save(fig,name+'-position-error',[p for _,p in runs])

selection=json.loads((INV/'circle-comparison-report/gain-selection.json').read_text())['representative_trials']
wanted=[('cartesian.circle.k0.5.d1','Cartesian K400'),('cartesian.circle.k1.d1','Cartesian K800'),('cartesian.circle.k1.5.d1','Cartesian K1200'),('joint.circle.k50.d1','DLS K50'),('joint.circle.k100.d1','DLS K100'),('joint.circle.k200.d1','DLS K200')]
tracking([(label,Path(next(r['run_dir'] for r in selection if r['profile_id']==profile))) for profile,label in wanted],'circle')
tracking([('Cartesian K1200',INV/'tracking.20261012T225714.056239Z/trial-01'),('DLS K200',INV/'tracking.20261012T225905.208292Z/trial-01')],'lissajous')
shutil.copy2(INV/'posture-comparison-report/paths-and-error.png',OUT/'posture-paths-and-error.png')
shutil.copy2(INV/'posture-comparison-report/paths-and-error.pdf',OUT/'posture-paths-and-error.pdf')
provenance['posture-paths-and-error']=[str(INV/'posture-comparison-report/paths-and-error.png')]
fixtures=[('Z line',INV/'z_line.20261012T161636Z'),('YZ circle',INV/'yz_circle.20261012T164958Z')]
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,(label,run) in zip(axes,fixtures):
    d=read(run);ax.plot(d['measured_y_m'][::10],d['measured_z_m'][::10],color='#2563eb',lw=1.4,label='Actual encoder-FK path')
    if label=='Z line':
        ax.plot([d['reference_y_m'][0]]*2,[d['measured_z_m'].min(),d['measured_z_m'].max()],'k--',lw=2,label='Target Z line')
    else:
        start=json.loads((run/'start.json').read_text());c=np.array(start['flange_position_m']);c[2]-=.1
        theta=np.linspace(0,2*np.pi,600);ax.plot(c[1]+.1*np.cos(theta),c[2]+.1*np.sin(theta),'k--',lw=2,label='Target circle (r = 0.10 m)')
    ax.set(xlabel='Base Y (m)',ylabel='Base Z (m)',title=label+' fixture');ax.set_aspect('equal');ax.legend(fontsize=8)
save(fig,'fixture-paths',[p for _,p in fixtures])
fig,axes=plt.subplots(2,2,figsize=(12,8))
for col,(label,run) in enumerate(fixtures):
    d=read(run);axes[0,col].plot(d['elapsed_s'][::10],d['position_error_m'][::10]*1000,color='#2563eb')
    axes[0,col].set(title=label+' normal deviation',xlabel='Elapsed time (s)',ylabel='Distance to curve (mm)')
    times=[];torques=[]
    with (run/'servo.jsonl').open() as f:
        initial=None
        for i,line in enumerate(f):
            r=json.loads(line)
            if initial is None:initial=r['robot_time_s']
            if r['fault'] or r['stopping']:continue
            if i%10==0:times.append(r['robot_time_s']-initial);torques.append(r['torque_measured'])
    torque=np.abs(np.asarray(torques))
    for j in range(7):axes[1,col].plot(times,torque[:,j],lw=.9,label=f'J{j+1}')
    axes[1,col].set(title=label+' measured joint torque magnitudes',xlabel='Elapsed time (s)',ylabel=r'$|\tau_{J,i}|$ (Nm)');axes[1,col].legend(ncol=4,fontsize=8)
save(fig,'fixture-deviation-and-torque',[p for _,p in fixtures])
(OUT/'figure-sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(OUT)
