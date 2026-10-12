"""M4 native Cartesian adapter. Python publishes references, never servo torques."""
from __future__ import annotations
import importlib.metadata
import json
import hashlib
import math
from pathlib import Path
import time
import numpy as np
from ..backend import EncoderForwardKinematics, _snapshot, _matrix
from ..core import _approval, require, vector, sha256
from franka_motion.config import _load, TrajectorySpec
from franka_motion.trajectories import validate_trajectory
from franka_motion.kinematics import UrdfKinematicsProvider, DampedLeastSquaresIK, validate_path_ik

LIMIT_VECTORS = {'joint_lower': 7, 'joint_upper': 7, 'joint_velocity': 7,
                 'external_torque': 7, 'command_torque': 7, 'workspace_lower': 3, 'workspace_upper': 3}
LIMIT_SCALARS = ('reference_timeout_s', 'callback_gap_s', 'position_error_m', 'orientation_error_rad',
                'linear_speed_m_s', 'angular_speed_rad_s', 'minimum_jacobian_singular_value',
                'minimum_command_success_rate')
SAMPLE_FIELDS = ('tick', 'reference_sequence', 'monotonic_ns', 'robot_time_s', 'dt_s', 'reference_age_s',
                 'min_singular_value', 'q', 'dq', 'torque_measured', 'torque_previous', 'external_torque',
                 'twist', 'reference_twist', 'posture', 'raw', 'task', 'nullspace', 'limit', 'coriolis',
                 'slew', 'filtered_prediction', 'command_success_rate', 'fault', 'stopping',
                 'stop_finished', 'stop_at_rest')
FAULTS = {1:'reference_timeout', 2:'reference_sequence', 3:'callback_gap', 4:'nonfinite', 5:'joint_bounds',
          6:'joint_velocity', 7:'workspace', 8:'pose_error', 9:'cartesian_speed', 10:'external_torque',
          11:'contact_collision', 12:'singularity', 13:'robot_error', 14:'command_success',
          15:'command_torque', 16:'logger_overflow', 17:'controller_projection', 18:'stop_timeout'}


def serialize_runtime_sample(sample):
    record = {name: getattr(sample, name) for name in SAMPLE_FIELDS}
    feedback_valid = sample.min_singular_value > 0 and sample.fault not in (4, 17)
    for name in ('pose', 'reference_pose'):
        value = getattr(sample, name).matrix
        record[name] = np.asarray(value() if callable(value) else value).tolist()
    nonfinite = []
    def clean(value):
        if isinstance(value, np.ndarray): value = value.tolist()
        if isinstance(value, (list, tuple)): return [clean(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value): return None
        return value
    for name, value in list(record.items()):
        converted = clean(value)
        # Missing scalar entries preserve evidence of nonfinite samples without invalid JSON.
        if 'null' in json.dumps(converted, allow_nan=False): nonfinite.append(name)
        record[name] = converted
    record['nonfinite_fields'] = nonfinite
    record['ee_feedback_valid'] = feedback_valid
    if not feedback_valid: record['pose'] = None; record['twist'] = None
    record['ee_feedback_source'] = 'measured_joint_encoder_libfranka_fk'
    record['ee_feedback_frame'] = 'fr3_flange'
    record['filtered_prediction_role'] = 'predicted_libfranka_100hz_output; torque_previous is actual prior tau_J_d'
    return record


def load_hardware_config(path, *, execute=False):
    c = _load(Path(path))
    require(c.get('runtime') == 'franky_m4_cartesian_v1', 'unsupported M4 runtime')
    require(c.get('execution_context') == 'hardware', 'M4 requires hardware context')
    _approval(c, execute)
    require(c['task']['kind'] in ('hold', 'tracking', 'posture', 'z_line', 'yz_circle'), 'unsupported task')
    for name, size in LIMIT_VECTORS.items():
        values = vector(c['limits'][name], size, name)
        if name in ('joint_velocity', 'external_torque', 'command_torque'):
            require(all(v > 0 for v in values), f'invalid {name}')
    for name in LIMIT_SCALARS:
        value = c['limits'][name]
        require(type(value) in (float, int) and math.isfinite(value) and value > 0, f'invalid {name}')
    for name in ('joint', 'workspace'):
        require(np.all(np.array(c['limits'][name+'_lower']) < c['limits'][name+'_upper']), 'unordered bounds')
    require(type(c['limits']['abort_on_contact']) is bool, 'abort_on_contact must be bool')
    require(c['limits']['minimum_command_success_rate'] <= 1, 'invalid success rate')
    for name in ('stiffness', 'damping'):
        require(all(v >= 0 for v in vector(c['controller'][name], 6, name)), 'negative gains')
    for name in ('posture_stiffness', 'posture_damping'):
        require(math.isfinite(c['controller'][name]) and c['controller'][name] >= 0, 'invalid posture gains')
    require(math.isfinite(c['task']['duration_s']) and 0 < c['task']['duration_s'] <= 600, 'invalid duration')
    require(math.isfinite(c['robot']['max_start_velocity_rad_s']) and c['robot']['max_start_velocity_rad_s'] > 0, 'invalid start velocity')
    require(c['controller']['projector'] in ('kinematic', 'dynamic'), 'invalid projector')
    require(c['controller']['projector'] == ('kinematic' if c['task']['kind']=='posture' else 'dynamic'), 'wrong task projector')
    require(c['controller']['cutoff_hz'] == 100 and c['controller']['max_delta_tau'] == 1, 'preserve tested torque pipeline')
    require(c['reference_hz'] == 50, 'M4 reference rate must be 50 Hz')
    require(c['limits']['reference_timeout_s'] >= 2 / c['reference_hz'], 'watchdog below two publication periods')
    require(type(c['log_capacity']) is int and c['log_capacity'] >= 1024, 'invalid log capacity')
    require(all(type(c['stop'][name]) in (float, int) and math.isfinite(c['stop'][name])
                for name in ('max_duration', 'ramp_duration')), 'nonfinite stop duration')
    require(c['stop']['max_duration'] > c['stop']['ramp_duration'] >= 0, 'invalid stop duration')
    require(all(v > 0 for v in vector(c['stop']['damping'], 7, 'stop damping')), 'invalid stop damping')
    for name in ('velocity_epsilon', 'max_delta_tau'):
        require(math.isfinite(c['stop'][name]) and c['stop'][name] > 0, 'invalid stop threshold')
    if c['task']['kind']=='tracking':
        require(isinstance(c['task']['trajectory'], str), 'trajectory required')
        require(math.isfinite(c['task']['angular_speed_rad_s']) and c['task']['angular_speed_rad_s'] > 0, 'invalid angular speed')
    if c['task']['kind']=='posture':
        vector(c['task']['posture_target_rad'], 7, 'posture target')
        require(c['task']['posture_transition_s'] > 0, 'invalid posture transition')
    return c


def verify_runtime_dependencies(c, f):
    package = Path(f.__file__).parent
    for filename, digest in c['dependencies'].items():
        require(Path(filename).name == filename and sha256(package.parent/'franky_control.libs'/filename)==digest,
                f'dependency mismatch: {filename}')


class FrankyMotionBackend:
    """One native motion, native watchdog/guards/stop, native bounded 1 kHz log."""
    def __init__(self, config=None):
        if config is None:
            raise RuntimeError('Hardware mode unavailable without a local M4 commissioning configuration')
        self.config = config
        require(importlib.metadata.version('franky-control') == config['robot']['expected_franky_version'], 'Franky version mismatch')
        import franky as f
        verify_runtime_dependencies(config, f)
        require(hasattr(f, 'M4RuntimeMotion'), 'wheel lacks M4 native runtime')
        self.f = f
        self.robot = f.Robot(config['robot']['host'], realtime_config=f.RealtimeConfig.Enforce,
                            default_torque_threshold=config['robot']['collision_torque_nm'],
                            default_force_threshold=config['robot']['collision_force_n'])
        self.encoder = EncoderForwardKinematics(self.robot, f)
        self.motion = None
        self.sequence = 0
        self._last_sample = None

    def preflight(self):
        state = self.robot.state
        s = _snapshot(self.robot, state, self.encoder)
        self.preflight_snapshot = s
        c = self.config
        require(self.robot.server_version == 10, 'FCI server must be 10')
        expected_model = Path(c['bundle'])/'model/robot_returned_model.urdf'
        require(hashlib.sha256(self.robot.model_urdf.encode()).hexdigest() == sha256(expected_model), 'live robot model differs from transfer')
        require(not s.has_errors and not s.is_in_control, 'robot has errors or active controller')
        require(np.allclose(s.f_t_ee, np.eye(4), atol=1e-8, rtol=0), 'requires bare flange F_T_EE identity')
        require(abs(s.load_mass_kg) < 1e-8 and abs(float(state.m_ee)) < 1e-8, 'requires zero configured payload')
        require(np.max(np.abs(s.dq)) <= c['robot']['max_start_velocity_rad_s'], 'robot must be at rest')
        require(np.all(np.abs(state.tau_ext_hat_filtered) <= c['limits']['external_torque']), 'external torque at start exceeds limits')
        require(not np.any(state.joint_collision) and not np.any(state.cartesian_collision), 'collision at start')
        require(s.encoder_fk_reported_position_error_m < 1e-5, 'encoder-model feedback frame mismatch')
        q = np.array(s.q)
        require(np.all(q >= c['limits']['joint_lower']) and np.all(q <= c['limits']['joint_upper']), 'start joints outside bounds')
        p = np.array(s.flange_position_m)
        require(np.all(p >= c['limits']['workspace_lower']) and np.all(p <= c['limits']['workspace_upper']), 'start pose outside workspace')
        j = np.asarray(self.robot.model.zero_jacobian(self.f.Frame.Flange, state))
        require(np.linalg.svd(j, compute_uv=False).min() >= c['limits']['minimum_jacobian_singular_value'], 'start Jacobian singular')
        return s

    def plan(self, snapshot):
        """Construct tracking entry from the measured start; fixtures never auto-enter."""
        c = self.config
        if c['task']['kind'] != 'tracking': return None
        bundle = Path(c['bundle'])
        catalog = _load(bundle/'config/trajectories.yaml')['trajectories']
        path = TrajectorySpec(c['task']['trajectory'], catalog[c['task']['trajectory']]).build(
            c['task']['angular_speed_rad_s'], snapshot.flange_position_m, snapshot.flange_orientation_xyzw)
        validate_trajectory(path, max_speed_m_s=c['limits']['linear_speed_m_s'], max_acceleration_m_s2=c['reference_limits']['acceleration_m_s2'],
                            max_jerk_m_s3=c['reference_limits']['jerk_m_s3'], lower_m=c['limits']['workspace_lower'], upper_m=c['limits']['workspace_upper'])
        model = UrdfKinematicsProvider(bundle/'model/robot_returned_model.urdf')
        solver = DampedLeastSquaresIK(model, c['limits']['joint_lower'], c['limits']['joint_upper'])
        report, _ = validate_path_ik(path, solver, np.array(snapshot.q), sample_hz=5)
        require(report['minimum_jacobian_singular_value_mixed_units'] >= c['limits']['minimum_jacobian_singular_value'], 'planned Jacobian singular')
        return path

    def start(self, snapshot):
        _approval(self.config, True)
        f, c = self.f, self.config
        params = f.CartesianImpedanceParams()
        params.stiffness = np.diag(c['controller']['stiffness'])
        params.damping = np.diag(c['controller']['damping'])
        params.nullspace_projector = (f.NullspaceProjectorType.DynamicallyConsistent if c['controller']['projector']=='dynamic'
                                      else f.NullspaceProjectorType.Kinematic)
        params.nullspace_tasks = [f.PostureTask(snapshot.q, c['controller']['posture_stiffness'], c['controller']['posture_damping'])]
        safety = params.safety
        safety.lower_joint_limits = c['limits']['joint_lower']; safety.upper_joint_limits = c['limits']['joint_upper']
        safety.max_delta_tau = 1.; params.safety = safety
        limits = f.M4RuntimeLimits()
        for name in (*LIMIT_VECTORS, *LIMIT_SCALARS, 'abort_on_contact'): setattr(limits, name, c['limits'][name])
        stop = f.TorqueStopParams()
        for name, value in c['stop'].items(): setattr(stop, name, value)
        fixture = None
        if c['task']['kind'] in ('z_line', 'yz_circle'):
            fixture = f.CurveFixtureParams()
            fixture.type = f.CurveFixtureType.ZLine if c['task']['kind']=='z_line' else f.CurveFixtureType.YZCircle
            for name, value in c['fixture'].items(): setattr(fixture, name, value)
            # Native initialization validation runs on the actual first servo state.
        self.motion = f.M4RuntimeMotion(params, limits, stop, fixture, c['log_capacity'])
        self.publish(snapshot.flange_position_m, snapshot.flange_orientation_xyzw, np.zeros(6))
        self.robot.move(self.motion, asynchronous=True, limit_rate=False, cutoff_frequency=100.)

    def publish(self, p, q, twist, posture=None):
        self.sequence += 1
        ref = self.f.M4Reference()
        ref.sequence = self.sequence
        ref.cartesian = self.f.CartesianReference(self.f.Affine(p, q), self.f.Twist(twist[:3], twist[3:]))
        if posture is not None:
            ref.posture = posture; ref.has_posture = True
            ref.posture_transition_s = self.config['task']['posture_transition_s']
        self.motion.publish(ref)

    def drain(self, stream):
        result = [] if self.motion is None else self.motion.drain()
        for s in result:
            self._last_sample = s
            stream.write(json.dumps(serialize_runtime_sample(s), allow_nan=False)+'\n')
        return result

    def stop(self, stream):
        if self.motion is None: return True
        self.motion.request_stop()
        end = time.monotonic() + self.config['stop']['max_duration'] + 2
        while self.robot.is_in_control and time.monotonic() < end:
            self.drain(stream); time.sleep(.01)
        joined = self.robot.join_motion(.1)
        samples = self.drain(stream)
        if not joined: self.robot.stop() # Explicit emergency fallback if native stop cannot finish.
        return bool(joined) and self._last_sample is not None and self._last_sample.stop_finished and self._last_sample.stop_at_rest
