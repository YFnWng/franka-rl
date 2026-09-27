# Simulation workstation handoff: velocity-reference impedance PPO

**Status:** position-only nominal policy trained and exported; receiving-machine qualification pending. This document is
the canonical next-step contract. It supersedes the earlier custom jerk-limited
governor proposal and the experimental Franky/Ruckig position-policy adapter.
Neither the existing position-policy ONNX bundles nor the Ruckig adapter should
be reused for this training generation.

## Workstation implementation status — 2026-09-27

The parallel simulation route is implemented without changing the completed
incremental-position tasks:

- training task: `Franka-FR3v2-FrankyVelocityImpedance-Reach-v0`
- path task: `Franka-FR3v2-FrankyVelocityImpedance-CirclePath-v0`
- action: `FrankyVelocityReference6DImpedanceAction`
- bundle contract: `fr3_joint_velocity_impedance_29d_v1`

The task integrates `q_ref` and projects soft limits at every 1 ms controller
tick, supplies `dq_ref` to the damping term, holds joint 7 at its reset
reference, and automatically installs the required one-policy-step delay in the
training and evaluation scripts unless a scenario explicitly overrides it.
The old position-reference controller retains a zero velocity-reference default.

A four-environment, one-iteration Isaac/RSL-RL smoke run completed with a 29D
observation, six-action actor, finite rewards, and the delay recorded as
`source: task_contract`. The complete deployment unit suite passed (34 tests).
A temporary smoke checkpoint also exported and independently verified as a
schema-2 ONNX bundle over 64 parity vectors (maximum absolute error
`1.49e-08`). This validates workstation integration only; production policy
training, path evaluation, a production bundle, native parity, and hardware
commissioning remain outstanding.

## Objective

Train a six-action PPO policy for `circle_yz` path tracking against the explicit
controller that will run on the FR3. The actor commands normalized joint
reference velocity for joints 1--6. Joint 7 is held at its measured reset
position. The hardware controller is a persistent torque-control motion; it does
not use firmware position/velocity control or Ruckig.

Use these identifiers in code and exported artifacts:

```text
runtime:     franky_joint_velocity_impedance_tracking_v1
contract_id: fr3_joint_velocity_impedance_29d_v1
action:      normalized_joint_velocity
integration: controller_1khz_forward_euler
```

## Rates, state, and update order

Use a 1 kHz physics/controller step and a 50 Hz policy step:

```text
physics dt = 0.001 s
policy dt  = 0.020 s
decimation = 20
```

The controller state is `q_ref`. At reset:

```text
q_ref = measured/reset q
v_target = 0
previous_action = 0
joint-7 reference = reset q7
```

At policy boundary `k`:

1. Observe measured `q`, measured `dq`, flange position, the current applied
   `q_ref`, and `previous_action = a[k-1]`.
2. Evaluate `a[k] = actor(observation[k])`.
3. Preserve the measured fixed one-policy-step delay: interval `k` receives
   `a[k-1]`; queue `a[k]` for interval `k+1`.
4. Hold the selected velocity target for all 20 physics ticks in that interval.

At every 1 ms physics tick, for controlled joints:

```text
dq_ref = clip(a_delayed, -1, 1) * [0.435, 0.435, 0.435, 0.435, 0.522, 0.522]
q_ref  = clamp(q_ref + dq_ref * dt, soft_lower, soft_upper)
```

If projection reaches a soft bound, set an outward `dq_ref` to zero for that
joint. For joint 7, retain its reset `q_ref` and use `dq_ref=0`.

Do not add Ruckig, acceleration/jerk limiting, interpolation, a 50 Hz position
jump, or a reset-to-zero reference velocity at ordinary policy boundaries. A
policy may reverse velocity at a boundary; training rewards can discourage
unnecessary reversals, but that behavior must not be hidden by an unmodeled
runtime filter.

## Torque controller

At every physics tick evaluate the hardware-equivalent law:

```text
position_error = clamp(q_ref - q, -0.5, 0.5)
tau_raw = K * position_error + D * (dq_ref - dq)
          + coriolis + joint_limit_torque
tau_slew = previous_tau + clamp(tau_raw - previous_tau, -1.0, 1.0)  # Nm per 1 ms
tau_filtered = 100 Hz command low-pass(tau_slew)
tau_applied = tau_filtered + simulator_gravity_compensation
```

Nominal values:

```text
K = [100] * 7 Nm/rad
D = [20] * 7 Nms/rad
torque slew = 1000 Nm/s = 1 Nm/ms
position-error clip = 0.5 rad
command filter = 100 Hz
joint-limit activation distance = 0.1 rad
joint-limit K/D/max torque = 4 Nm/rad, 1 Nms/rad, 5 Nm
friction and feed-forward torque = zero
```

Libfranka/FR3 supplies gravity compensation below the commanded-torque
interface; Isaac Sim must continue adding simulator gravity compensation. The
existing `FrankyImpedanceController.compute()` currently implements
`-D*dq`. Extend it to accept a velocity reference and compute
`D*(dq_ref-dq)`. Preserve the old position-route call with a zero default so its
behavior and tests remain stable.

## Exact 29D observation

Keep this ordering and dimension:

```text
0:7    q_measured - q_default
7:14   dq_measured
14:17  target_position_base - measured_fr3_flange_position_base
17:23  2 * (q_ref[0:6] - soft_lower[0:6]) /
           (soft_upper[0:6] - soft_lower[0:6]) - 1
23:29  previous_normalized_joint_velocity[0:6]
```

`q_ref` is the controller state at the observation boundary after the preceding
interval. It is not a projection using the action the actor is about to produce.
The Cartesian feedback body/frame remain `fr3_flange` in `fr3_link0`.

Freeze these reference values in the exported contract:

```text
q_default = [0, -0.7853981633974483, 0, -2.356194490192345,
             0, 1.5707963267948966, 0]
soft_lower = [-2.610666015, -1.652481015, -2.610666015, -2.929015870,
              -2.588673015, 0.648913185, -2.745750015]
soft_upper = [ 2.610666015,  1.652481015,  2.610666015, -0.264941230,
               2.588673015, 4.412542815,  2.745750015]
```

## Repository changes on the workstation

The closest existing task is
`franka_incremental_impedance`, but its action term integrates once per 20 ms
and its controller omits desired velocity. Add a separate task/action instead
of silently changing the existing position-policy contract.

Required changes:

1. Add a six-action velocity-reference impedance action beside
   `FrankyIncremental6DImpedanceAction`. Store `v_target` at `process_actions()`;
   integrate and project `q_ref` in every `apply_actions()` call.
2. Extend `FrankyImpedanceController.compute()` with `velocity_reference`, using
   a zero default for existing callers.
3. Add a task config with `sim.dt=0.001`, `decimation=20`, the 29D observation
   above, K=100/D=20, and the existing bare-flange FR3 model and `circle_yz`
   command/path definitions.
4. Apply `FixedActionDelayWrapper(delay_steps=1)` during training and final
   evaluation. Its reset fill must be a six-vector of zeros.
5. Version the exporter/preflight schema for
   `fr3_joint_velocity_impedance_29d_v1`; do not label it as
   `fr3_incremental_position_29d_v1`.
6. Add tests for reset, 1 ms integration, soft-bound projection/outward velocity,
   held J7, desired-velocity damping, exact observation order/size, and one-step
   delay ordering.

Hardware-side reference files shipped in this repository:

- `deployment/franky_runtime/VELOCITY_IMPEDANCE_ROUTE.md`
- `deployment/franky_runtime/franky_velocity_impedance.patch`
- `deployment/franky_runtime/config/ppo_velocity_impedance_path.pending.yaml`
- `deployment/franky_runtime/franky_experiment/core.py`

The Franky patch adds the C++ `JointVelocityImpedanceTrackingMotion`. Its full
C++ core and Python binding already compile against libfranka 0.21.2, Ruckig
0.17.3, pybind11 3.0.4, and Python 3.12; its offline binding smoke test passed.

## Training and evaluation

Train the nominal policy first with fixed K=100/D=20. Retain rewards for EE path
tracking, reference-velocity changes, near-target settling, measured velocity,
and soft-bound projection. Report each reward coefficient with the checkpoint.

Evaluate the deterministic actor on only the approved `circle_yz` path first:

```text
path: circle_yz from source/franka_rl/franka_rl/config/paths.yaml
policy rate: 50 Hz
physics/controller rate: 1 kHz
fixed delay: 1 policy step
controller: K=100, D=20
initial state: the deployment home/reset distribution used by the existing task
```

Run at least 16 deterministic evaluation episodes with no unsafe termination,
non-finite value, action clipping, or persistent soft-bound projection. Report
path completion, waypoint timeouts, EE RMS/peak error, joint tracking error,
velocity/acceleration, torque, action reversal rate, and projection counts.
Plot at least the initial approach and one full circle for `q`, `dq`, `q_ref`,
`dq_ref`, action, torque, and EE target/measured path.

A DR policy may follow after nominal closure. If produced, randomize controller
gain consistently over the previously evaluated K=50/100/200 range and report
its exact K/D sampling rule. Do not delay the nominal bundle for DR training.

## Return package

Return one immutable nominal bundle containing:

- deterministic-tanh ONNX actor with input `[1,29]` and output `[1,6]`;
- `manifest.json` and SHA-256;
- `policy_contract.yaml` with schema version 2 and the identifiers above;
- exact observation layout, action scales, soft limits, default joint position,
  rates, fixed delay, controller/filter/torque-slew fields, and frames;
- checkpoint/config/source revision provenance;
- `circle_yz` evaluation summary, per-episode table, and plots;
- an ONNX-versus-native inference parity report.

Place the transfer under a new dated directory outside the repository. Add its
path, manifest hash, policy hash, evaluation result, and any remaining mismatch
to the end of `deployment/HANDOFF_AGENT.md`. Do not overwrite the 2026-09-27
position-policy bundles; they remain evidence for the completed staircase demo.

## Hardware boundary

Workstation completion authorizes no robot motion. The real-time machine will
build the patched Franky wheel in a separate environment, verify the returned
bundle, run fake and FCI shadow sessions, create a new reviewed motion YAML, and
commission a reduced initial approach before attempting one `circle_yz`
traversal.


## Workstation return — 2026-09-27

The selected position-only velocity policy is model 149 from run
`2026-09-27_17-35-37_fr3_velocity_impedance_accel_objective_v2`. Its checkpoint
SHA-256 is `9b013893830353108534f5d388f308dde6031db0c640e263719febc47f18f01a`.
Do not substitute either position-plus-z-axis experiment: those policies retained
roughly 7 cm deterministic position error and are deferred.

Transfer the complete immutable bundle directory
`fr3_velocity_impedance_position_only_nominal_model149_v1`. Trust anchors:

- manifest: `8d19cb788f43530241dffba708cb47ca964e469ecaa085a51bb4ddc09cb5982c`
- ONNX actor: `d7c888a926a8ccdd7557880c2926b12f480f6e187fb0fdf492da798776421ef2`
- policy contract: `d191cca5d5d65de29516c6a7787079f2db2300d33a22712d586455a3e135fe60`
- native/ONNX parity: 1,024 vectors, maximum absolute error `6.780028343200684e-7`

The contract is schema 2, `fr3_joint_velocity_impedance_29d_v1`, deterministic
tanh input `[batch,29]` and output `[batch,6]`, 50 Hz actor, 1 kHz integration,
one policy-step delay, K=100/D=20, J1--J6 normalized velocity action and held J7.

Random-point evaluation passed 5,120/5,120 episodes with no unsafe failure,
mean final error 0.0241 m and mean success time 2.224 s. Final `circle_yz`
qualification did **not** pass: 0/16 complete paths, all 16 eventually terminated
on the simulated joint-position limit after a mean 5.94/24 waypoints. Motion was
visually smooth, but smoothness is not path qualification. Therefore the bundle
is released only for offline verification, fake backend, no-motion FCI shadow,
and a separately reviewed reduced point-to-point commissioning trial. Do not
approve a full circle traversal from this evidence. Hardware timeout handling
must remain abort-on-timeout and all faults terminal.

Full workstation evidence:

- random point: data-root `evaluations/2026-09-27_17-43-28-360600_nominal`
  (`summary.json` SHA-256 `7c94b6d5cefa08116d0c3ec78d2df12b9a6cecfa772a62f4c6796d932e618ed7`)
- circle: data-root `evaluations/2026-09-27_velocity_position_only_circle_yz_handoff_v1`
  (`summary.json` SHA-256 `1b3f02408a33bbdc3162e8e5cf1a903159fd307f79cc6a46a50b62a687a8f21c`)
- active path catalog SHA-256:
  `39455c82dfec508cdac0d26befda5a701501006c3758d1c115adc868a1ff6c69`

Receiving gates are: verify the trusted bundle manifest and C++ fixtures; pass
fake-backend lifecycle, delay, reset, watchdog and terminal-fault tests; pass a
no-motion FCI shadow using measured-joint libfranka flange FK; then prepare one
immutable reduced point-to-point YAML for separate lab review and motion
authorization. This handoff itself authorizes no robot motion.
