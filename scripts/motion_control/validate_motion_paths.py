"""Offline M2 complete-reference/full-pose DLS guard; no Kit or hardware imports."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from franka_motion.config import load_motion_suite,expand_trials
from franka_motion.kinematics import UrdfKinematicsProvider,DampedLeastSquaresIK,validate_path_ik
from franka_motion.trajectories import sample_trajectory,validate_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,required=True)
    args = parser.parse_args()
    suite = load_motion_suite(args.suite)
    workspace = suite.workspace.data
    path = Path(__file__).resolve().parents[2]/workspace["model"]["archived_urdf"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != workspace["model"]["urdf_sha256"]:
        raise ValueError("Kinematic model hash mismatch")
    frames = workspace["frames"]
    if frames["base"] != "fr3_link0" or frames["tcp"] != "fr3_flange" or frames["flange_to_tcp_position_m"] != [0.,0.,0.] or frames["flange_to_tcp_quaternion_xyzw"] != [0.,0.,0.,1.]:
        raise ValueError("M2 currently checks fixed-base identity flange TCP only")
    provider = UrdfKinematicsProvider(path)
    bounds,limits,box = workspace["joint_bounds_rad"]["commanded"],workspace["reference_limits"],workspace["cartesian_bounds_m"]
    cache,rows = {},[]
    for trial in expand_trials(suite):
        if trial.controller.data["mode"] != "cartesian":
            raise ValueError("Use joint planner validation for joint suites")
        key = (trial.trajectory.name,trial.angular_speed_rad_s,trial.initial_posture)
        if key not in cache:
            home = workspace["initial_postures_rad"][trial.initial_posture]
            p,r,_ = provider.evaluate(home)
            trajectory = trial.trajectory.build(trial.angular_speed_rad_s,p,r)
            reference = validate_trajectory(trajectory,max_speed_m_s=limits["speed_m_s"],max_acceleration_m_s2=limits["acceleration_m_s2"],
                max_jerk_m_s3=limits["jerk_m_s3"],lower_m=np.minimum(p,box["lower"])-1e-6,upper_m=np.maximum(p,box["upper"])+1e-6)
            ik,_ = validate_path_ik(trajectory,DampedLeastSquaresIK(provider,bounds["lower"],bounds["upper"]),home)
            if ik["minimum_jacobian_singular_value_mixed_units"] < .02:
                raise ValueError("Mixed-unit Jacobian singularity guard failed")
            angular = float(np.linalg.norm(sample_trajectory(trajectory)["angular_velocity_rad_s"],axis=1).max())
            cache[key] = {"reference":reference,"ik":ik,"max_angular_reference_speed_rad_s":angular}
        servo = trial.controller.data["cartesian_servo"]
        if cache[key]["max_angular_reference_speed_rad_s"] > servo["max_reference_angular_velocity_rad_s"]:
            raise ValueError("Angular reference limit failed")
        if cache[key]["reference"]["peaks"]["linear_velocity_m_s"] > servo["max_reference_linear_velocity_m_s"]:
            raise ValueError("Controller linear reference limit failed")
        rows.append({"trial_id":trial.trial_id,**cache[key]})
    print(json.dumps({"passed":True,"scope":"sampled references/full-pose kinematics only; no dynamics or mesh clearance certificate",
                      "file_hashes":suite.file_hashes,"trials":rows},indent=2,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
