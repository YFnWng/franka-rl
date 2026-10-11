"""Known-force verification of endpoint-free curve impedance, simulation only."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from .cartesian_motion_runner import IsaacCartesianMotionRunner
from franka_motion.virtual_fixture import CurveVirtualFixture
from franka_motion.kinematics import DampedLeastSquaresIK


class IsaacCurveFixtureRunner(IsaacCartesianMotionRunner):
    def run_curve(self,trial,output,config):
        q0,pose,parity = self.reset(trial)
        env,action = self.env,self.env.motion_action
        kind = "z_line" if trial.trial_id.startswith("line_") else "yz_circle"
        origin = config["line_origin_m"] if kind == "z_line" else config["circle_center_m"]
        fixture = CurveVirtualFixture(kind,origin,config["circle_radius_m"])
        lower,upper = np.array(config["workspace_lower_m"]),np.array(config["workspace_upper_m"])
        target,tangent = fixture.project(pose[:3])
        if np.linalg.norm(target-pose[:3]) > .001 or np.any(pose[:3]<lower) or np.any(pose[:3]>upper):
            raise ValueError("Initial FK must be within 1 mm of curve and inside the independent workspace")
        bounds = self.suite.workspace.data["joint_bounds_rad"]["commanded"]
        solver = DampedLeastSquaresIK(self.kinematics,bounds["lower"],bounds["upper"])
        seed = q0.copy()
        points = (np.array([origin])+np.c_[np.zeros(65),config["circle_radius_m"]*np.sin(np.linspace(0,2*np.pi,65)),
            config["circle_radius_m"]*np.cos(np.linspace(0,2*np.pi,65))]) if kind=="yz_circle" else np.array([[origin[0],origin[1],z] for z in np.linspace(lower[2],upper[2],17)])
        for point in points:
            if np.any(point<lower) or np.any(point>upper):
                raise ValueError("Curve validation point outside workspace")
            seed = solver.solve(point,[1.,0.,0.,0.],seed)
        c = action.controller
        c.kind,c.radius = kind,fixture.radius
        c.origin.copy_(torch.tensor(origin,dtype=torch.float32,device=env.device))
        for key in ("normal_stiffness","normal_damping","tangent_damping"):
            setattr(c,key,config[key])
        action.cart_lower.copy_(torch.tensor(lower,device=env.device))
        action.cart_upper.copy_(torch.tensor(upper,device=env.device))
        times = np.arange(int(np.ceil((config["hold_s"]+config["push_s"]+config["release_s"])/env.step_dt)))*env.step_dt
        refs = np.tile(np.r_[pose[:3],[1.,0.,0.,0.],np.zeros(6)],(len(times),1))
        phases = ["hold" if t<config["hold_s"] else "curve_fixture" for t in times]
        active = (times>=config["hold_s"])&(times<config["hold_s"]+config["push_s"])
        force = config["force_n"]*(tangent if trial.trial_id.endswith("tangent") else np.array([1.,0.,0.]))
        force_tensor = torch.tensor(force,dtype=torch.float32,device=env.device).expand(env.num_envs,1,3)
        robot = env.scene["robot"]
        np.testing.assert_allclose(robot.data.root_quat_w.torch.cpu().numpy(),np.tile([0,0,0,1],(env.num_envs,1)),atol=1e-6)
        composer = robot.permanent_wrench_composer
        body_ids = torch.tensor([action.flange_id],device=env.device)
        previous = False
        def before_step(i):
            nonlocal previous
            if bool(active[i]) != previous:
                composer.reset()
                if active[i]:
                    composer.add_forces_and_torques_index(forces=force_tensor,body_ids=body_ids,is_global=True)
                previous = bool(active[i])
        def analyze(trace):
            p = trace["pose"][...,:3]
            if kind == "z_line":
                distance = np.linalg.norm(p[...,:2]-fixture.origin[:2],axis=-1)
            else:
                distance = np.sqrt((p[...,0]-fixture.origin[0])**2+(np.linalg.norm(p[...,1:]-fixture.origin[1:],axis=-1)-fixture.radius)**2)
            trace["curve_distance_m"] = distance
            trace["external_force_world_n"] = np.repeat(active[:,None]*force,env.cfg.decimation,axis=0)[:len(p)]
            travel = np.linalg.norm(p-pose[:3],axis=-1)
            orientation = np.linalg.norm(trace["pose_error"][...,3:],axis=-1)
            final_speed = float(np.linalg.norm(trace["twist"][-1,:,:3],axis=-1).max())
            metrics = {"curve_distance_max_m":float(distance.max()),"curve_distance_rms_m":float(np.sqrt(np.mean(distance**2))),
                "tip_travel_max_m":float(travel.max()),"orientation_error_max_rad":float(orientation.max()),"final_tip_speed_m_s":final_speed}
            passed = distance.max()<=.01 and orientation.max()<=.02 and (not trial.trial_id.endswith("tangent") or travel.max()>=.002) and final_speed<=.01
            return metrics,bool(passed)
        path = Path(__file__)
        self.source_hashes[str(path.relative_to(path.resolve().parents[4]))] = hashlib.sha256(path.read_bytes()).hexdigest()
        import franka_motion.virtual_fixture as shared
        shared_path = Path(shared.__file__)
        self.source_hashes[str(shared_path.relative_to(path.resolve().parents[4]))] = hashlib.sha256(shared_path.read_bytes()).hexdigest()
        metadata = {"curve_fixture_config":config,"kinematic_parity":parity,"trajectory":{"type":kind},"angular_speed_rad_s":None,
            "curve_kind":kind,"curve_origin_m":origin,"curve_radius_m":fixture.radius,"curve_validation_points":len(points),
            "orientation_contract":"fixed xyzw [1,0,0,0]; tip z=base -z, fixed roll",
            "wrench_processing_order":"normal spring + normal damping + positive tangent damping; fixed orientation impedance; J.T wrench -> same M2 torque processing",
            "force_application":"world-frame force at flange-body CoM; no applied torque",
            "workspace_fault_bit":32,"workspace_guard":"measured TCP box independent of endpoint-free curve",
            "curve_acceptance":{"max_curve_error_m":.01,"max_orientation_error_rad":.02,"min_tangent_travel_m":.002,"max_final_tip_speed_m_s":.01},
            "collision_validation":"sampled IK and runtime contact checks; NOT full-curve mesh clearance certification"}
        return self._execute(trial,output,refs,phases,metadata,before_step=before_step,cleanup=composer.reset,analyze_trace=analyze)
