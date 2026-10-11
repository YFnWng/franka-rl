"""Verify finalized M2 telemetry and compile tracking/contouring/dynamics figures.

No Isaac/Kit imports. Never treats two deterministic repeats as independent
statistical evidence. Faulted trials remain visible and cannot pass acceptance.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

PARITY_LIMITS = {"q_rad":.001,"dq_rad_s":.001,"position_m":.0001,"rotation_rad":.001,
                 "linear_twist_m_s":.001,"angular_twist_rad_s":.001}


def differences(trace,other):
    a,b = trace["pose"][...,3:].astype(float),other["pose"][...,3:].astype(float)
    dots = np.abs(np.sum(a*b,axis=-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)))
    return {"q_rad":float(np.max(np.abs(trace["q"]-other["q"]))),
        "dq_rad_s":float(np.max(np.abs(trace["dq"]-other["dq"]))),
        "position_m":float(np.linalg.norm(trace["pose"][...,:3]-other["pose"][...,:3],axis=-1).max()),
        "rotation_rad":float((2*np.arccos(np.clip(dots,0,1))).max()),
        "linear_twist_m_s":float(np.linalg.norm(trace["twist"][...,:3]-other["twist"][...,:3],axis=-1).max()),
        "angular_twist_rad_s":float(np.linalg.norm(trace["twist"][...,3:]-other["twist"][...,3:],axis=-1).max())}


def validate(directory):
    summary_path = directory/"summary.json"
    summary = json.loads(summary_path.read_text())
    marker = json.loads((directory/"FINALIZED.json").read_text())
    if marker["summary_sha256"] != hashlib.sha256(summary_path.read_bytes()).hexdigest() or marker["status"] != summary["status"]:
        raise ValueError("Summary finalization mismatch")
    if summary["trace_sha256"] != hashlib.sha256((directory/"trace.npz").read_bytes()).hexdigest():
        raise ValueError("Trace hash mismatch")
    with np.load(directory/"trace.npz",allow_pickle=False) as archive:
        trace = {key:archive[key] for key in archive.files}
    n = summary["physics_samples"]
    if n < 1 or summary["schema_version"] != 1:
        raise ValueError("Empty/unsupported telemetry")
    if trace["pose"].shape != (n,summary["num_envs"],7):
        raise ValueError("Trace shape mismatch")
    for key,size in (("q",7),("dq",7),("reference",13),("twist",6),("pose_error",6),("wrench",6),("applied",7)):
        if trace[key].shape != (n,summary["num_envs"],size):
            raise ValueError(f"Trace shape mismatch: {key}")
    np.testing.assert_allclose(trace["time_s"],np.arange(n)*.001,atol=1e-12,rtol=0)
    if summary["acceptance_passed"] and (trace["fault_code"].any() or not all(np.isfinite(v).all() for v in trace.values() if v.dtype.kind in "fc")):
        raise ValueError("Accepted trace contains faults/nonfinite data")
    if summary["acceptance_passed"] and "kinematic_twist" in trace and np.max(np.abs(trace["twist"]-trace["kinematic_twist"])) > 1e-4:
        raise ValueError("Flange-origin twist / J*dq parity failed")
    return summary,trace


def compile_results(suite_dirs,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=False)
    rows,previous,parity,suites_complete = [],{},[],[]
    for suite_dir in suite_dirs:
        index = json.loads((suite_dir/"index.json").read_text())
        suites_complete.append(bool(index["all_accepted"]))
        for item in index["results"]:
            name = item["trial_id"]
            summary,trace = validate(suite_dir/name)
            row = {"suite":suite_dir.name,"trial_id":name,"controller":summary["controller"]["stiffness"],
                "angular_speed_rad_s":summary["angular_speed_rad_s"],"twist_feedforward":summary["twist_feedforward"],
                "acceptance_passed":summary["acceptance_passed"],**summary["metrics"]}
            rows.append(row)
            cell = (summary["controller"]["stiffness"].__repr__(),name.rsplit("__repeat_",1)[0])
            finite = all(np.isfinite(trace[k]).all() for k in ("q","dq","pose","twist"))
            clone = differences(trace,{k:trace[k][:,:1] for k in ("q","dq","pose","twist")}) if finite else None
            repeat = None
            if cell in previous:
                old = previous[cell]
                if finite and old["q"].shape == trace["q"].shape and all(np.isfinite(old[k]).all() for k in ("q","dq","pose","twist")):
                    repeat = differences(trace,old)
            previous[cell] = trace
            parity.append({"trial_id":name,"clone_max_differences":clone,
                "repeat_max_differences":repeat,"parity_passed":clone is not None and all(v<=PARITY_LIMITS[k] for k,v in clone.items())
                    and (repeat is None or all(v<=PARITY_LIMITS[k] for k,v in repeat.items()))})
            t,p,r = trace["time_s"],trace["pose"][:,0,:3],trace["reference"][:,0,:3]
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(221,projection="3d")
            track = trace["phase"]=="track"
            mask = track if track.any() else np.ones(len(t),dtype=bool)
            ax.plot(*r[mask].T,label="desired")
            ax.plot(*p[mask].T,label="measured")
            points = np.vstack((r[mask],p[mask]))
            center = (points.min(axis=0)+points.max(axis=0))/2
            half_span = max(float(np.ptp(points,axis=0).max()),.002)*.55
            ax.set_xlim(center[0]-half_span,center[0]+half_span)
            ax.set_ylim(center[1]-half_span,center[1]+half_span)
            ax.set_zlim(center[2]-half_span,center[2]+half_span)
            ax.set_box_aspect((1,1,1))  # Do not visually magnify sub-mm out-of-plane error.
            ax.view_init(elev=20,azim=25)
            ax.set(xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)",title="Desired / measured (base frame)")
            ax.legend()
            ax = fig.add_subplot(222)
            ax.plot(t,1000*np.linalg.norm(trace["pose_error"][:,0,:3],axis=1),label="tracking")
            if track.any():
                ax.plot(t[track],1000*trace["contour_error_m"][:,0],label="contouring")
                ax.axvspan(t[track][0],t[track][-1],color="gray",alpha=.12,label="track phase")
            ax.set(xlabel="Time (s)",ylabel="Error (mm)")
            ax.legend()
            ax = fig.add_subplot(223)
            for joint in range(7):
                ax.plot(t,trace["dq"][:,0,joint],label=f"j{joint+1}")
            ax.set(xlabel="Time (s)",ylabel="Measured joint speed (rad/s)")
            ax.legend(ncol=4,fontsize=7)
            ax = fig.add_subplot(224)
            for joint in range(7):
                ax.plot(t,trace["applied"][:,0,joint],label=f"j{joint+1}")
            ax.set(xlabel="Time (s)",ylabel="Applied joint torque (Nm)")
            ax.legend(ncol=4,fontsize=7)
            fig.suptitle(f"{name}: Kx={summary['controller']['stiffness'][0]}, w={summary['angular_speed_rad_s']} rad/s, twist={summary['twist_feedforward']}; accepted={summary['acceptance_passed']}")
            fig.subplots_adjust(left=.10,right=.97,top=.89,bottom=.08,hspace=.30,wspace=.30)
            fig.savefig(output/f"{suite_dir.name}__{name}.png",dpi=130)
            plt.close(fig)
    if not rows:
        raise ValueError("No finalized trials to compile")
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (output/"metrics.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {"all_accepted":all(suites_complete) and all(r["acceptance_passed"] for r in rows),"all_parity_passed":all(r["parity_passed"] for r in parity),
        "parity_limits":PARITY_LIMITS,
        "trials":rows,"parity":parity,"repeat_interpretation":"deterministic reproducibility, not confidence intervals"}
    (output/"compiled.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    fig,ax = plt.subplots(figsize=(max(8,len(rows)*.9),4))
    ax.bar(np.arange(len(rows)),[1000*r["tracking_rms_error_m"] if r["tracking_rms_error_m"] is not None else np.nan for r in rows])
    ax.set_xticks(np.arange(len(rows)),[f"Kx={r['controller'][0]}, {r['angular_speed_rad_s']} rad/s\nFF={r['twist_feedforward']}\n{r['trial_id'].split('__')[-1]}" for r in rows],rotation=30,ha="right")
    ax.set(ylabel="Tracking RMS (mm)",title="M2 speed × feedforward cells (individual repeats)")
    fig.tight_layout()
    fig.savefig(output/"tracking_summary.png",dpi=130)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir",type=Path,action="append",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    report = compile_results(args.suite_dir,args.output_dir)
    print(json.dumps({"all_accepted":report["all_accepted"],"all_parity_passed":report["all_parity_passed"],"compiled":str(args.output_dir)},indent=2))
    return 0 if report["all_accepted"] and report["all_parity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
