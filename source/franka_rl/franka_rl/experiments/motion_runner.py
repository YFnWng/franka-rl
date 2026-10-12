"""M1 finite joint trials, GPU-resident references/telemetry and buffered artifacts."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import numpy as np
import torch
from franka_motion.joint_planning import plan_joint_ptp, validate_joint_trajectory


class IsaacMotionRunner:
    def __init__(self, env, suite):
        self.env = env.unwrapped
        self.suite = suite

    def run_trial(self, trial, output):
        env, suite = self.env, self.suite
        data = trial.trajectory.data
        if data["type"] not in ("joint_hold", "joint_ptp"):
            raise ValueError("M1 runner supports joint hold/PTP only")
        q0 = np.array(suite.workspace.data["initial_postures_rad"][trial.initial_posture])
        action = env.action_manager.get_term("joint_reference")
        env.scene["robot"].data.default_joint_pos.torch[:] = torch.tensor(q0, device=env.device)
        env.reset(seed=42)
        initial = env.capture_state()
        bounds = suite.workspace.data["joint_bounds_rad"]["commanded"]
        target = q0 if data["type"] == "joint_hold" else np.array(data["target_position_rad"])
        path = None
        planner_report = {"planner": "stationary_hold", "passed": True}
        if data["type"] == "joint_ptp":
            path = plan_joint_ptp(q0, target, lower=bounds["lower"], upper=bounds["upper"],
                max_velocity=np.array(data["max_velocity_rad_s"])*trial.speed_scale,
                max_acceleration=np.array(data["max_acceleration_rad_s2"])*trial.speed_scale,
                max_jerk=np.array(data["max_jerk_rad_s3"])*trial.speed_scale)
            planner_report = validate_joint_trajectory(path)
        duration = data["hold_duration_s"] + (path.duration_s if path else 0.) + data["settle_duration_s"]
        steps = math.ceil(duration / env.step_dt)
        references, phases = [], []
        for step in range(steps):
            t = step*env.step_dt - data["hold_duration_s"]
            if t < 0:
                q, dq, phase = q0, np.zeros(7), "hold"
            elif path and t < path.duration_s:
                ref = path.evaluate(t)
                q, dq, phase = ref.position_rad, ref.velocity_rad_s, "ptp"
            else:
                q, dq, phase = target, np.zeros(7), "settle"
            references.append(np.r_[q, dq if trial.twist_feedforward else np.zeros(7)])
            phases.append(phase)
        references = torch.as_tensor(np.array(references), device=env.device, dtype=torch.float32)
        # Memory is bounded before allocating; hot path never writes files or copies state to CPU.
        ticks = steps*env.cfg.decimation
        estimated_bytes = ticks*env.num_envs*(12*7*4+8)
        if estimated_bytes > 256*1024**2:
            raise ValueError("M1 trace exceeds the 256 MiB GPU budget; reduce batch/duration")
        action.allocate_trace(ticks)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        env.begin_trial(trial.trial_id)
        completed_steps = 0
        runtime_error = None
        try:
            for row in references:
                env.step(row.expand(env.num_envs, -1))
                completed_steps += 1
                # One batch-level synchronization at reference rate, never a per-environment loop.
                if bool((action.fault_code != 0).any()):
                    break
        except Exception as error:
            runtime_error = f"{type(error).__name__}: {error}"
        finally:
            terminal = env.end_trial()
        trace = {key: value[:action.trace_index].detach().cpu().numpy() for key, value in action.trace.items()}
        trace["time_s"] = np.arange(action.trace_index)*env.physics_dt
        trace["phase"] = np.array(phases)[np.arange(action.trace_index)//env.cfg.decimation]
        trace["initial_q"] = initial["q"].cpu().numpy()
        trace["initial_dq"] = initial["dq"].cpu().numpy()
        trace["terminal_q"] = terminal["q"].cpu().numpy()
        trace["terminal_dq"] = terminal["dq"].cpu().numpy()
        trace["first_fault_q"] = action.first_fault_q.cpu().numpy()
        trace["first_fault_dq"] = action.first_fault_dq.cpu().numpy()
        trace["first_fault_reference"] = action.first_fault_reference.cpu().numpy()
        trace["first_fault_tick"] = action.first_fault_tick.cpu().numpy()
        trace["first_fault_code"] = action.first_fault_code.cpu().numpy()
        trace["first_fault_contact_force_n"] = action.first_fault_contact_force.cpu().numpy()
        import warp as wp
        robot = env.scene["robot"]
        for key, getter in (("body_mass_kg", "get_masses"), ("body_com_pose_xyz_xyzw", "get_coms"),
                            ("body_inertia_at_com_column_major_kg_m2", "get_inertias")):
            value = getattr(robot.root_view, getter)()
            trace[key] = (wp.to_torch(value) if isinstance(value, wp.array) else value).cpu().numpy()
        hold = trace["phase"] == "hold"
        fault = terminal["fault_code"].cpu().numpy()
        finite = np.isfinite(trace["q"]).all() and np.isfinite(trace["dq"]).all()
        hold_error = float(np.max(np.abs(trace["q"][hold]-q0))) if finite and hold.any() else None
        end_error = float(np.max(np.abs(trace["terminal_q"]-target))) if finite else None
        speed = float(np.max(np.abs(trace["dq"]))) if finite and action.trace_index else None
        # Numerical commissioning criteria, not hardware readiness or a robot guarantee.
        accepted = finite and not fault.any() and runtime_error is None and hold_error is not None and hold_error <= .002 and end_error <= .01
        summary = {"schema_version": 1, "trial_id": trial.trial_id, "status": "fault" if fault.any() or runtime_error else "complete",
            "runtime_error": runtime_error,
            "acceptance_passed": bool(accepted), "acceptance_limits": {"hold_max_error_rad": .002, "terminal_max_error_rad": .01},
            "duration_s": (terminal["sim_tick"]-initial["sim_tick"])*env.physics_dt, "planned_duration_s": duration, "num_envs": env.num_envs,
            "start_sim_tick": initial["sim_tick"],
            "physics_samples": action.trace_index, "telemetry_state_timing": "pre-physics state; final post-physics state stored separately",
            "hold_max_error_rad": hold_error, "terminal_max_error_rad": end_error, "max_measured_speed_rad_s": speed,
            "fault_codes": fault.tolist(), "fault_bits": {"1": "measured_position", "2": "measured_velocity", "4": "nonfinite",
                "8": "moving-link contact proxy", "16": "invalid_reference"}, "planner": planner_report,
            "controller": trial.controller.data, "controller_processing": {
                "non_gravity_slew_nm_s": action.cfg.slew_rate_nm_s, "lowpass_cutoff_hz": action.cfg.cutoff_hz,
                "position_error_clip_rad": action.cfg.position_error_clip, "effort_limits_nm": action.cfg.effort_limits_nm,
                "version": "shared_previous_output_v2", "non_gravity_reset_seed": "zero",
                "order": "PD + PhysX coriolis -> slew candidate relative to previous filtered output -> lowpass against same previous output -> PhysX gravity -> simulator-only effort clamp",
                "reference_hold": "zero_order_hold", "delay_steps": 0,
                "fault_transition": "latched measured-position hold, zero desired velocity, damped gravity-compensated stop candidate; not hardware-qualified"},
            "measured_abort_limits": {key: getattr(action.cfg, key) for key in ("abort_lower", "abort_upper", "max_measured_velocity")},
            "contact_proxy_threshold_n": action.cfg.contact_threshold_n,
            "model": suite.workspace.data["model"], "file_hashes": suite.file_hashes,
            "joint_names": suite.workspace.data["joint_names"], "twist_feedforward": trial.twist_feedforward,
            "body_names": robot.body_names,
            "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__), Path(__file__).resolve().parents[1]/"tasks/manager_based/franka_motion_control/mdp/controllers.py",
                             Path(__file__).resolve().parents[1]/"tasks/manager_based/franka_motion_control/mdp/actions.py")},
            "trajectory": data, "speed_scale": trial.speed_scale, "frames": suite.workspace.data["frames"],
            "armature_kg_m2": suite.workspace.data["model"]["armature_kg_m2"], "hardware_execution_authorized": False,
            "versions": {key: importlib.metadata.version(key) for key in ("torch", "numpy", "ruckig", "isaaclab", "isaacsim")}}
        np.savez_compressed(output / "trace.npz", **trace)
        summary["trace_sha256"] = hashlib.sha256((output/"trace.npz").read_bytes()).hexdigest()
        (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        # Last-write finalization marker: complete artifact, even when the recorded trial faulted.
        (output/"FINALIZED.json").write_text(json.dumps({"status": summary["status"],
            "summary_sha256": hashlib.sha256((output/"summary.json").read_bytes()).hexdigest()})+"\n")
        action.trace = None
        return summary


def verify_reset_parity(env):
    """Simulation-only reference rejection, fault retention and selective-reset check.

    This submits a NaN reference, not a plant impulse. The manager rejects it and
    retains the held reference. It never tests on hardware.
    """
    env = env.unwrapped
    term = env.motion_action
    if env.num_envs < 2:
        raise ValueError("Selective-reset verification needs at least two clones")
    before = env.capture_state()
    command = term.processed_actions.clone()
    command[0, 0] = float("nan")
    env.begin_trial("invalid_reference_verification")
    env.step(command)
    terminal = env.end_trial()
    fault_q = term.first_fault_q[0].clone()
    torch.testing.assert_close(fault_q, before["q"][0], rtol=0, atol=0)
    assert int(terminal["fault_code"][0]) == 16
    assert not terminal["fault_code"][1:].any()
    assert term.first_fault_tick[0] >= 0
    neighbor_q = terminal["q"][1:].clone()
    neighbor_controller = {key: value[1:].clone() for key, value in term.controller.components.items()}
    env.reset(env_ids=torch.tensor([0], device=env.device, dtype=torch.int32))
    torch.testing.assert_close(env.capture_state()["q"][1:], neighbor_q, rtol=0, atol=0)
    for key, previous in neighbor_controller.items():
        torch.testing.assert_close(term.controller.components[key][1:], previous, rtol=0, atol=0)
    assert not term.fault_code[0]
    assert term.first_fault_tick[0] == -1
    if term.action_dim == 14:
        torch.testing.assert_close(term.processed_actions[0, :7], env.scene["robot"].data.default_joint_pos.torch[0], rtol=0, atol=0)
    else:
        from ..tasks.manager_based.franka_motion_control.mdp.observations import tcp_pose_base
        torch.testing.assert_close(term.processed_actions[0,:7],tcp_pose_base(env)[0],rtol=0,atol=0)
    return {"passed": True, "invalid_reference_rejected": True, "fault_state_retained_before_reset": True,
            "first_fault_feedback_exact": True, "selective_reset_neighbor_state_and_filter_unchanged": True}
