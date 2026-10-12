"""Finite Cartesian trials and commissioning; no robot SDK or learned policy."""
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import warp as wp
from franka_motion.kinematics import UrdfKinematicsProvider, DampedLeastSquaresIK, validate_path_ik, pose_error_base
from franka_motion.trajectories import validate_trajectory, sample_trajectory
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from ..tasks.manager_based.franka_motion_control.mdp.observations import tcp_pose_base, tcp_twist_base


class IsaacCartesianMotionRunner:
    def __init__(self, env, suite, *, profile_steps=0):
        self.env, self.suite = env.unwrapped, suite
        self.profile_steps = profile_steps
        self.path_visualization = None
        repo = Path(__file__).resolve().parents[4]
        path = repo/suite.workspace.data["model"]["archived_urdf"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != suite.workspace.data["model"]["urdf_sha256"]:
            raise ValueError("Archived kinematic model hash mismatch")
        frames = suite.workspace.data["frames"]
        if frames["base"] != "fr3_link0" or frames["tcp"] != "fr3_flange" or frames["flange_to_tcp_position_m"] != [0.,0.,0.] or frames["flange_to_tcp_quaternion_xyzw"] != [0.,0.,0.,1.]:
            raise ValueError("M2 currently verifies fixed-base, identity flange TCP only")
        self.kinematics = UrdfKinematicsProvider(path)
        shared = repo/"source/franka_motion/franka_motion"
        task = Path(__file__).resolve().parents[1]/"tasks/manager_based/franka_motion_control/mdp"
        sources = [Path(__file__),Path(__file__).with_name("motion_path_visualization.py"),*(task/name for name in ("controllers.py","actions.py","observations.py")),
                   *(shared/name for name in ("kinematics.py","trajectories.py","time_scaling.py","metrics.py","impedance.py"))]
        self.source_hashes = {str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}

    def reset(self, trial):
        env = self.env
        q0 = np.array(self.suite.workspace.data["initial_postures_rad"][trial.initial_posture])
        env.scene["robot"].data.default_joint_pos.torch[:] = torch.tensor(q0,device=env.device)
        env.reset(seed=42)
        pose = tcp_pose_base(env).detach().cpu().numpy()[0].astype(float)
        pose[3:] /= np.linalg.norm(pose[3:])  # Feedback roundoff only, never normalize invalid commands.
        p,r,j = self.kinematics.evaluate(q0)
        error = pose_error_base(p,r,pose[:3],pose[3:])
        actual_j = env.motion_action.jacobian_base().detach().cpu().numpy()[0]
        if np.linalg.norm(error[:3]) > 1e-5 or np.linalg.norm(error[3:]) > 1e-5 or np.max(np.abs(j-actual_j)) > 1e-5:
            raise ValueError(f"URDF/Isaac FK/Jacobian mismatch: {error}, J={np.max(np.abs(j-actual_j))}")
        # Cross-check the official absolute-pose DLS update, without commanding motion.
        ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose",use_relative_mode=False,ik_method="dls",ik_params={"lambda_val":.01}),env.num_envs,env.device)
        target = torch.tensor(pose,dtype=torch.float32,device=env.device).repeat(env.num_envs,1)
        target[:,0] += .001
        ik.set_command(target)
        measured = tcp_pose_base(env)
        q_tensor = env.scene["robot"].data.joint_pos.torch
        delta = ik.compute(measured[:,:3],measured[:,3:],env.motion_action.jacobian_base(),q_tensor)-q_tensor
        e = pose_error_base(pose[:3],pose[3:],target[0,:3].cpu().numpy(),target[0,3:].cpu().numpy()/np.linalg.norm(target[0,3:].cpu().numpy()))
        expected = j.T@np.linalg.solve(j@j.T+.01**2*np.eye(6),e)
        peak = float(np.max(np.abs(delta[0].cpu().numpy()-expected)))
        if peak > 1e-5:
            raise ValueError("Official DLS cross-check failed")
        return q0,pose,{"fk_position_error_m":float(np.linalg.norm(error[:3])),"fk_rotation_error_rad":float(np.linalg.norm(error[3:])),
                       "jacobian_max_error":float(np.max(np.abs(j-actual_j))),"official_dls_max_update_error_rad":peak}

    def commissioning(self, trial, output):
        """2 s hold, 2 mm smooth step, +10% K step, terminal hold; simulation only."""
        from franka_motion.time_scaling import QuinticSegment
        q0,pose,parity = self.reset(trial)
        times = np.arange(500)*self.env.step_dt
        segment = QuinticSegment(0.,.002,1.)
        refs,phases = [],[]
        for t in times:
            row = np.r_[pose,np.zeros(6)]
            if 2 <= t < 3:
                law = segment.evaluate(t-2)
                row[0] += float(law.position)
                row[7] = float(law.velocity)
            elif t >= 3:
                row[0] += .002
            refs.append(row)
            phases.append("hold" if t < 2 else "step")
        result = self._execute(trial,output,np.array(refs),phases,{"parity":parity,"commissioning":True,
            "commissioning_commanded_twist_feedforward":True,
            "gain_step": {"time_s":4.,"stiffness_multiplier":1.1}},gain_step_index=200)
        return result

    def run_trial(self, trial, output):
        q0,pose,parity = self.reset(trial)
        path = trial.trajectory.build(trial.angular_speed_rad_s,pose[:3],pose[3:])
        w = self.suite.workspace.data
        # The preview box is track-only. Validate the measured-home entry corridor
        # as its union with home, not by silently rejecting/teleporting home.
        box = w["cartesian_bounds_m"]
        lower,upper = np.minimum(box["lower"],pose[:3])-1e-6,np.maximum(box["upper"],pose[:3])+1e-6
        limits = w["reference_limits"]
        validation = validate_trajectory(path,max_speed_m_s=limits["speed_m_s"],max_acceleration_m_s2=limits["acceleration_m_s2"],
            max_jerk_m_s3=limits["jerk_m_s3"],lower_m=lower,upper_m=upper)
        sampled = sample_trajectory(path)
        angular_peak = float(np.linalg.norm(sampled["angular_velocity_rad_s"],axis=1).max())
        if angular_peak > self.env.motion_action.cfg.max_reference_angular_velocity:
            raise ValueError("Orientation approach exceeds angular reference bound")
        b = w["joint_bounds_rad"]["commanded"]
        feasibility,ik_trace = validate_path_ik(path,DampedLeastSquaresIK(self.kinematics,b["lower"],b["upper"]),q0)
        if feasibility["minimum_jacobian_singular_value_mixed_units"] < .02:
            raise ValueError("Path enters configured mixed-unit Jacobian singularity guard")
        from .motion_path_visualization import MotionPathVisualization, path_visualization_enabled
        if path_visualization_enabled(self.env.sim):
            if self.path_visualization is None:
                self.path_visualization = MotionPathVisualization()
            self.path_visualization.show(path,self.env.scene["robot"])
            print(f"[MotionPath] Showing {self.path_visualization.markers.count} cyan target-path markers",flush=True)
        times = np.arange(math.ceil((2+path.duration_s)/self.env.step_dt))*self.env.step_dt
        refs,phases = [],[]
        for t in times:
            if t < 2:
                refs.append(np.r_[pose,np.zeros(6)])
                phases.append("hold")
            else:
                sample = path.evaluate(min(t-2,path.duration_s))
                refs.append(sample.reference.as_action(twist_feedforward=trial.twist_feedforward))
                phases.append(sample.phase)
        return self._execute(trial,output,np.array(refs),phases,{"parity":parity,"reference_validation":validation,
            "ik_validation":feasibility,"entry_corridor_m":{"lower":lower.tolist(),"upper":upper.tolist()},
            "max_analytic_angular_reference_speed_rad_s":angular_peak},ik_trace=ik_trace)

    def _execute(self,trial,output,refs,phases,metadata,*,ik_trace=None,gain_step_index=None,
                 reference_update=None,before_step=None,cleanup=None,analyze_trace=None):
        env,action = self.env,self.env.motion_action
        from .motion_path_visualization import MotionPathVisualization,path_visualization_enabled
        if path_visualization_enabled(env.sim) and self.path_visualization is None:
            self.path_visualization = MotionPathVisualization()
        # Include every channel, with margin; reject oversized allocations before motion.
        ticks = len(refs)*env.cfg.decimation
        if ticks*env.num_envs*600 > 256*1024**2:
            raise ValueError("Cartesian trace exceeds 256 MiB GPU budget")
        action.allocate_trace(ticks)
        commands = torch.as_tensor(refs,dtype=torch.float32,device=env.device)
        output = Path(output)
        output.mkdir(parents=True,exist_ok=False)
        initial = env.capture_state()
        initial_pose = tcp_pose_base(env).clone()
        k0 = action.controller.k.clone()
        env.begin_trial(trial.trial_id)
        runtime_error = None
        step_wall_times = []
        profiler = None
        profiling_active = False
        rollout_started = time.perf_counter()
        try:
            for i,row in enumerate(commands):
                if self.profile_steps and i == 20:
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if str(env.device).startswith("cuda"):
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    profiler = torch.profiler.profile(activities=activities)
                    profiler.start()
                    profiling_active = True
                if gain_step_index is not None and i == gain_step_index:
                    action.controller.k.mul_(1.1)
                started = time.perf_counter()
                command = row.expand(env.num_envs,-1)
                if before_step is not None:
                    before_step(i)
                if reference_update is not None:
                    command = reference_update(i,command)
                if self.path_visualization is not None:
                    self.path_visualization.update_frame(env.scene["robot"],action.flange_id)
                env.step(command)
                faulted = bool((action.fault_code!=0).any())
                elapsed = time.perf_counter()-started
                if i >= 20 and not profiling_active:
                    step_wall_times.append(elapsed)
                if profiling_active and i == 20+self.profile_steps-1:
                    profiler.stop()
                    profiling_active = False
                if faulted:
                    break
        except Exception as error:
            runtime_error = f"{type(error).__name__}: {error}"
        finally:
            if profiling_active:
                profiler.stop()
            if cleanup is not None:
                cleanup()
            terminal = env.end_trial()
            terminal_pose = tcp_pose_base(env).clone()
            action.controller.k.copy_(k0)
        rollout_wall_s = time.perf_counter()-rollout_started
        performance = {"rollout_wall_s":rollout_wall_s,
            "simulation_to_wall_ratio":action.trace_index*env.physics_dt/rollout_wall_s,
            "reference_step_wall_median_ms":float(np.median(step_wall_times)*1000) if step_wall_times else None,
            "reference_step_wall_p90_ms":float(np.percentile(step_wall_times,90)*1000) if step_wall_times else None,
            "timing_scope":"step plus existing fault synchronization; excludes first 20 steps and profiler window; includes rendering when enabled",
            "profile_reference_steps_requested":self.profile_steps}
        if profiler is not None:
            profiler.export_chrome_trace(str(output/"step_profile.json"))
            (output/"step_profile.txt").write_text(profiler.key_averages().table(sort_by="self_cpu_time_total",row_limit=40))
        trace = {key:value[:action.trace_index].detach().cpu().numpy() for key,value in action.trace.items()}
        trace["time_s"] = np.arange(action.trace_index)*env.physics_dt
        trace["phase"] = np.array(phases)[np.arange(action.trace_index)//env.cfg.decimation]
        trace["initial_pose"] = initial_pose.cpu().numpy()
        trace["terminal_pose"] = terminal_pose.cpu().numpy()
        for prefix,state in (("initial",initial),("terminal",terminal)):
            for key in ("q","dq","fault_code"):
                trace[f"{prefix}_{key}"] = state[key].cpu().numpy()
        for key in ("first_fault_q","first_fault_dq","first_fault_reference","first_fault_tick","first_fault_code","first_fault_contact_force"):
            trace[key] = getattr(action,key).cpu().numpy()
        if ik_trace:
            trace.update(ik_trace)
        robot = env.scene["robot"]
        for key,getter in (("body_mass_kg","get_masses"),("body_com_pose_xyz_xyzw","get_coms"),
                           ("body_inertia_at_com_column_major_kg_m2","get_inertias")):
            value = getattr(robot.root_view,getter)()
            trace[key] = (wp.to_torch(value) if isinstance(value,wp.array) else value).cpu().numpy()
        fault = terminal["fault_code"].cpu().numpy()
        finite = all(np.isfinite(value).all() for value in trace.values() if value.dtype.kind in "fc")
        position_error = np.linalg.norm(trace["pose_error"][...,:3],axis=-1)
        rotation_error = np.linalg.norm(trace["pose_error"][...,3:],axis=-1)
        hold = trace["phase"]=="hold"
        track = trace["phase"]=="track"
        mask = track if track.any() else ~hold
        if not mask.any():
            mask = np.ones(len(hold),dtype=bool)
        metrics = {"hold_max_position_error_m":float(position_error[hold].max()),
            "hold_max_rotation_error_rad":float(rotation_error[hold].max()),
            "tracking_rms_error_m":float(np.sqrt(np.mean(position_error[mask]**2))),
            "tracking_max_error_m":float(position_error[mask].max()),
            "tracking_rms_rotation_error_rad":float(np.sqrt(np.mean(rotation_error[mask]**2))),
            "max_measured_joint_speed_rad_s":float(np.abs(trace["dq"]).max()),
            "max_applied_torque_nm":float(np.abs(trace["applied"]).max()),
            "max_contact_proxy_n":float(trace["contact_force_n"].max())}
        if track.any():
            from franka_motion.trajectories import rotation_matrix_rpy_deg
            from franka_motion.metrics import circle_contour_error,sampled_curve_contour_error
            data = trial.trajectory.data
            rotation = rotation_matrix_rpy_deg(data["plane_rpy_deg"])
            if data["type"] == "circle":
                contour = circle_contour_error(trace["pose"][track,:,:3],data["center_m"],data["radius_m"],rotation)
                metadata["contouring_method"] = "exact distance to analytic 3D circle"
            else:
                phase = np.linspace(0,2*np.pi,2049)
                uv = np.array(data["amplitudes_m"])*np.sin(phase[:,None]*data["harmonics"]+data["phase_offsets_rad"])
                curve = np.c_[uv,np.zeros(len(uv))]@rotation.T+data["center_m"]
                contour = sampled_curve_contour_error(trace["pose"][track,:,:3],curve)
                metadata["contouring_method"] = "nearest of 2049 sampled curve points; upper approximation"
            trace["contour_error_m"] = contour
            metrics["contouring_rms_error_m"] = float(np.sqrt(np.mean(contour**2)))
        accepted = finite and not fault.any() and runtime_error is None and metrics["hold_max_position_error_m"] <= .002 and metrics["hold_max_rotation_error_rad"] <= .01 and metrics["tracking_max_error_m"] <= .03
        if analyze_trace is not None:
            additional, passed = analyze_trace(trace)
            metrics.update(additional)
            accepted = accepted and passed
        metrics["max_joint_displacement_from_initial_rad"] = float(np.linalg.norm(
            trace["q"]-trace["initial_q"][None],axis=-1).max())
        metrics["terminal_joint_displacement_from_initial_rad"] = float(np.linalg.norm(
            trace["terminal_q"]-trace["initial_q"],axis=-1).max())
        if "posture" in trace:
            metrics["max_posture_torque_nm"] = float(np.abs(trace["posture"]).max())
        metrics = {key:value if np.isfinite(value) else None for key,value in metrics.items()}
        summary = {"schema_version":1,"trial_id":trial.trial_id,"status":"fault" if fault.any() or runtime_error else "complete",
            "acceptance_passed":bool(accepted),"runtime_error":runtime_error,"metrics":metrics,"fault_codes":fault.tolist(),
            "acceptance_limits":{"hold_position_m":.002,"hold_rotation_rad":.01,"track_max_position_m":.03},
            "physics_samples":action.trace_index,"duration_s":action.trace_index*env.physics_dt,"num_envs":env.num_envs,
            "telemetry_state_timing":"pre-physics; terminal post-physics state separate",
            "model":self.suite.workspace.data["model"],"frames":self.suite.workspace.data["frames"],
            "controller":trial.controller.data,"trajectory":trial.trajectory.data,"angular_speed_rad_s":trial.angular_speed_rad_s,
            "twist_feedforward":trial.twist_feedforward,"file_hashes":self.suite.file_hashes,
            "processing_order":"J^T [K*base_pose_log_error + D*(desired_twist-measured_twist)] + PhysX C -> non-gravity slew -> first-order lowpass -> PhysX G -> effort clamp",
            "tracking_posture":{"enabled":action.tracking_posture_enabled,
                "stiffness_nm_rad":action.cfg.tracking_posture_stiffness,
                "damping_nm_s_rad":action.cfg.tracking_posture_damping,
                "anchor_rad":action.posture_anchor.cpu().tolist(),
                "projector":"I - J.T (J M^-1 J.T)^+ J M^-1",
                "rcond":action.cfg.tracking_projector_rcond},
            "posture_torque":"see tracking_posture or task-specific metadata",
            "fault_stop":"gravity-compensated velocity damper candidate; not hardware-qualified",
            "collision_validation":"runtime moving-link contact proxy only; no mesh clearance certificate",
            "measured_abort_limits":{key:getattr(action.cfg,key) for key in ("abort_lower","abort_upper","max_measured_velocity")},
            "contact_proxy_threshold_n":action.cfg.contact_threshold_n,
            "source_sha256":self.source_hashes,
            "performance":performance,
            "hardware_execution_authorized":False,"versions":{k:importlib.metadata.version(k) for k in ("numpy","torch","isaaclab","isaacsim")},**metadata}
        np.savez_compressed(output/"trace.npz",**trace)
        summary["trace_sha256"] = hashlib.sha256((output/"trace.npz").read_bytes()).hexdigest()
        (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
        (output/"FINALIZED.json").write_text(json.dumps({"status":summary["status"],"summary_sha256":hashlib.sha256((output/"summary.json").read_bytes()).hexdigest()})+"\n")
        action.trace = None
        return summary
