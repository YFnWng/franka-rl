# Simulation workstation handoff: velocity-reference impedance PPO

**Status:** ready for workstation implementation and training. This document is
the canonical next-step contract. It supersedes the earlier custom jerk-limited
governor proposal and the experimental Franky/Ruckig position-policy adapter.
Neither the existing position-policy ONNX bundles nor the Ruckig adapter should
be reused for this training generation.

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
