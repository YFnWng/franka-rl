"""Serial isolated-profile M2 coordinator: commissioning gate -> tracking -> plots.

Inherit the repository direnv environment. This script never connects to a robot
or installs/upgrades libraries. Each process initializes/reset its own simulator.
"""
import argparse
import importlib.util
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from franka_motion.config import load_motion_suite,expand_trials


def validate_phase_output(directory,expected_ids):
    """A zero exit code/index flag alone is never evidence of a finished trial."""
    path = Path(__file__).resolve().with_name("compile_cartesian_results.py")
    spec = importlib.util.spec_from_file_location("motion_artifact_check",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index = json.loads((directory/"index.json").read_text())
    if not index["all_accepted"] or [r["trial_id"] for r in index["results"]] != expected_ids:
        raise ValueError("Phase did not finish the exact accepted trial set")
    for name in expected_ids:
        summary,_ = module.validate(directory/name)
        if not summary["acceptance_passed"] or summary["status"] != "complete":
            raise ValueError("Commissioning/tracking acceptance failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--num_envs",type=int,default=2)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--controller",help="Optional named profile subset; all profiles by default")
    parser.add_argument("--output-dir",type=Path)
    args = parser.parse_args()
    suite = load_motion_suite(args.suite)
    trials = [t for t in expand_trials(suite) if args.controller is None or t.controller.name==args.controller]
    if not trials:
        raise ValueError("No trials match the selected controller")
    if suite.backend != "simulation" or any(t.controller.data["mode"] != "cartesian" for t in trials):
        raise ValueError("M2 coordinator accepts Cartesian simulation suites only")
    root = Path(os.environ.get("FRANKA_RL_DATA_ROOT","/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data")).resolve()
    output = (args.output_dir or root/"motion_control"/(datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")+"_"+suite.name)).resolve()
    if not (output.is_relative_to(root) or output.is_relative_to(Path("/tmp"))):
        raise ValueError("Output must be on data volume or /tmp for bounded verification")
    if output.is_relative_to(root) and not any(os.path.ismount(p) for p in (root,*root.parents) if str(p)!="/"):
        raise RuntimeError("Data volume not mounted; no fallback")
    parent = next(p for p in output.parents if p.exists())
    if not os.access(parent,os.W_OK) or shutil.disk_usage(parent).free < 128*1024**2:
        raise RuntimeError("Output not writable or less than 128 MiB free")
    output.mkdir(parents=True,exist_ok=False)
    profiles = list(dict.fromkeys(t.controller.name for t in trials))
    tracking,manifest = [],[]
    scripts = Path(__file__).resolve().parent
    for profile in profiles:
        for phase in ("commissioning","tracking"):
            directory = output/profile/phase
            command = [sys.executable,"-u",str(scripts/"run_sim_demo.py"),"--task","Franka-FR3v2-MotionControl-Cartesian-v0",
                "--suite",str(args.suite.resolve()),"--controller",profile,"--num_envs",str(args.num_envs),
                "--device",args.device,"--viz","none","--output-dir",str(directory)]
            if phase == "commissioning":
                command.append("--commissioning-only")
            log = output/f"{profile}_{phase}.log"
            print(f"{profile}: {phase}",flush=True)
            with log.open("w") as stream:
                result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False)
            expected_ids = [t.trial_id for t in trials if t.controller.name==profile]
            if phase == "commissioning":
                expected_ids = expected_ids[:1]
            artifact_error = None
            try:
                validate_phase_output(directory,expected_ids)
            except (ValueError,KeyError,OSError,AssertionError) as error:
                artifact_error = str(error)
            passed = result.returncode==0 and artifact_error is None
            manifest.append({"profile":profile,"phase":phase,"command":command,"returncode":result.returncode,
                             "passed":passed,"artifact_error":artifact_error,"stdout_log":str(log)})
            (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
            if not passed:
                print(f"STOP: {phase} did not pass; see {log}",flush=True)
                return 1
            if phase == "tracking":
                tracking.append(directory)
    command = [sys.executable,str(scripts/"compile_cartesian_results.py"),"--output-dir",str(output/"compiled")]
    for directory in tracking:
        command.extend(("--suite-dir",str(directory)))
    return subprocess.run(command,check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
