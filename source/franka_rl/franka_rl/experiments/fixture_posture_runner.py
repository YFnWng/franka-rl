"""Finite M3 known-force fixtures and fixed-pose secondary objectives."""
import numpy as np
import torch
import hashlib
from pathlib import Path
from .cartesian_motion_runner import IsaacCartesianMotionRunner
from ..tasks.manager_based.franka_motion_control.mdp.observations import tcp_pose_base,tcp_twist_base
from ..tasks.manager_based.franka_motion_control.mdp.controllers import line_fixture_reference_torch
from franka_motion.kinematics import DampedLeastSquaresIK,pose_error_base
from franka_motion.virtual_fixture import LineVirtualFixture,nullspace_projector


class IsaacFixturePostureRunner(IsaacCartesianMotionRunner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        import franka_motion.virtual_fixture as shared
        for path in (Path(__file__),Path(shared.__file__)):
            self.source_hashes[str(path.relative_to(Path(__file__).resolve().parents[4]))] = hashlib.sha256(path.read_bytes()).hexdigest()

    def run_demo_b(self,trial,output,config):
        q0,pose,parity = self.reset(trial)
        env,action = self.env,self.env.motion_action
        if action.action_dim != 20:
            raise ValueError("Demo B requires the independent 20D fixture/posture task")
        name = trial.trial_id
        fixture = name.startswith("fixture_")
        dt = env.step_dt
        duration = config["hold_s"]+(config["force_duration_s"]+config["release_s"] if fixture else config["transition_s"]+config["posture_settle_s"])
        times = np.arange(int(np.ceil(duration/dt)))*dt
        refs = np.tile(np.r_[pose,np.zeros(6),q0],(len(times),1))
        phases = ["hold" if t < config["hold_s"] else "fixture" if fixture else "posture" for t in times]
        metadata = {"demo_b_config":config,"kinematic_parity":parity,"trajectory":{"type":name},
            "angular_speed_rad_s":None,"posture_projector":"I-J.T@pinv(J.T), orthogonal kinematic; NOT dynamically consistent",
            "posture_stiffness":0. if fixture else config["weak_stiffness"] if name == "posture_weak" else config["posture_stiffness"],
            "posture_damping":0. if fixture else config["posture_damping"],
            "force_application":"fr3_flange body CoM; world-frame force; no applied torque; fixture direction in robot base",
            "fixture_acceptance":{"minimum_axial_travel_m":.002,"max_transverse_error_m":.01,"max_orientation_error_rad":.02,"segment_overshoot_m":.01},
            "posture_acceptance":{"minimum_joint_change_rad":.01,"max_position_error_m":.005,"max_orientation_error_rad":.02}}
        action.cfg.posture_stiffness = metadata["posture_stiffness"]
        action.cfg.posture_damping = metadata["posture_damping"]
        action.cfg.projector_rcond = config["projector_rcond"]
        before_step,reference_update = None,None
        force_schedule = np.zeros((len(times),3))
        robot = env.scene["robot"]
        composer = robot.permanent_wrench_composer
        body_ids = torch.tensor([action.flange_id],device=env.device)
        if fixture:
            np.testing.assert_allclose(robot.data.root_quat_w.torch.detach().cpu().numpy(),
                np.tile([0.,0.,0.,1.],(env.num_envs,1)),atol=1e-6,rtol=0)
            model = LineVirtualFixture(pose[:3],config["direction_base"],*config["segment_m"],dt=dt,
                cutoff_hz=config["cutoff_hz"],max_speed=config["max_reference_speed_m_s"])
            endpoints = model.origin+np.array(config["segment_m"])[:,None]*model.direction
            # Entry corridor is explicit because home FK lies outside the M2 tracking box.
            lower = np.minimum(self.suite.workspace.data["cartesian_bounds_m"]["lower"],pose[:3]-.04)
            upper = np.maximum(self.suite.workspace.data["cartesian_bounds_m"]["upper"],pose[:3]+.04)
            if np.any(endpoints < lower) or np.any(endpoints > upper):
                raise ValueError("Fixture segment outside bounded home entry corridor")
            action.cart_lower.copy_(torch.tensor(lower,device=env.device))
            action.cart_upper.copy_(torch.tensor(upper,device=env.device))
            direction = model.direction
            transverse = np.cross(direction,np.eye(3)[np.argmin(np.abs(direction))])
            transverse /= np.linalg.norm(transverse)
            force = config["force_n"]*(direction if name == "fixture_axial" else transverse)
            active = (times >= config["hold_s"]) & (times < config["hold_s"]+config["force_duration_s"])
            force_schedule[active] = force
            force_tensor = torch.tensor(force,dtype=torch.float32,device=env.device).expand(env.num_envs,1,3)
            was_active = False
            def before_step(i):
                nonlocal was_active
                if bool(active[i]) != was_active:
                    composer.reset()
                    if active[i]:
                        composer.add_forces_and_torques_index(forces=force_tensor,body_ids=body_ids,is_global=True)
                    was_active = bool(active[i])
            origin = torch.tensor(pose[:3],dtype=torch.float32,device=env.device)
            axis = torch.tensor(direction,dtype=torch.float32,device=env.device)
            filtered = torch.zeros(env.num_envs,device=env.device)
            def reference_update(i,row):
                if times[i] < config["hold_s"]:
                    return row
                p,v = tcp_pose_base(env)[:,:3],tcp_twist_base(env)[:,:3]
                position,velocity,next_filtered = line_fixture_reference_torch(p,v,origin,axis,filtered,
                    model.alpha,model.lower,model.upper,model.max_speed)
                filtered.copy_(next_filtered)
                command = row.clone()
                command[:,:3],command[:,7:10] = position,velocity
                return command
            metadata.update(fixture_origin_m=pose[:3].tolist(),fixture_direction_base=direction.tolist(),fixture_segment_m=config["segment_m"],
                fixture_filter_alpha=model.alpha,reference_corridor_m={"lower":lower.tolist(),"upper":upper.tolist()})
        else:
            _,_,j = self.kinematics.evaluate(q0)
            _,_,vh = np.linalg.svd(j,full_matrices=True)
            direction = vh[-1]
            # Fix SVD sign so posture labels remain deterministic across machines.
            direction *= 1 if direction[np.argmax(np.abs(direction))] >= 0 else -1
            sign = -1 if name == "posture_a" else 1
            b = self.suite.workspace.data["joint_bounds_rad"]["commanded"]
            solver = DampedLeastSquaresIK(self.kinematics,b["lower"],b["upper"])
            target = solver.solve(pose[:3],pose[3:],q0+sign*config["posture_offset_rad"]*direction)
            # Solver returns the joint configuration, not arbitrary seven-joint target.
            target = np.asarray(target)
            u = np.clip((times-config["hold_s"])/config["transition_s"],0,1)
            blend = 10*u**3-15*u**4+6*u**5
            refs[:,13:] = q0+blend[:,None]*(target-q0)
            p,r,_ = self.kinematics.evaluate(target)
            metadata.update(posture_target_rad=target.tolist(),target_pose_error=pose_error_base(pose[:3],pose[3:],p,r).tolist(),
                projector_leakage=float(np.linalg.norm(j@nullspace_projector(j,config["projector_rcond"]))),
                minimum_target_limit_margin_rad=float(np.minimum(target-b["lower"],np.array(b["upper"])-target).min()))
        def analyze(trace):
            n = len(trace["q"])
            trace["external_force_base_n"] = np.repeat(force_schedule,env.cfg.decimation,axis=0)[:n]
            delta = trace["pose"][...,:3]-pose[:3]
            orient = np.linalg.norm(trace["pose_error"][...,3:],axis=-1)
            if fixture:
                s = delta@model.direction
                transverse_error = np.linalg.norm(delta-s[...,None]*model.direction,axis=-1)
                trace["fixture_axial_m"],trace["fixture_transverse_error_m"] = s,transverse_error
                excursion = float(np.abs(s).max())
                lateral = float(transverse_error.max())
                over = float(np.maximum(np.maximum(model.lower-s,s-model.upper),0).max())
                metrics = {"axial_travel_max_m":excursion,"transverse_error_max_m":lateral,
                    "orientation_error_max_rad":float(orient.max()),"segment_overshoot_max_m":over}
                passed = lateral<=.01 and orient.max()<=.02 and over<=.01 and (name!="fixture_axial" or excursion>=.002)
            else:
                change = np.linalg.norm(trace["terminal_q"]-q0,axis=-1)
                drift = np.linalg.norm(delta,axis=-1)
                metrics = {"terminal_joint_change_min_rad":float(change.min()),"fixed_pose_position_max_m":float(drift.max()),
                    "orientation_error_max_rad":float(orient.max()),"posture_target_error_mean_rad":float(np.linalg.norm(trace["terminal_q"]-target,axis=-1).mean())}
                passed = change.min()>=.01 and drift.max()<=.005 and orient.max()<=.02
            return metrics,bool(passed)
        return self._execute(trial,output,refs,phases,metadata,before_step=before_step,reference_update=reference_update,
            cleanup=composer.reset,analyze_trace=analyze)
