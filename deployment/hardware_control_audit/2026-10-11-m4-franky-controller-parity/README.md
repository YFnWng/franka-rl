# M4 Franky controller parity audit

Date: 2026-10-11 (America/New_York)

## Scope and pinned sources

This audit compares the controllers in `franka_rl` with the exact Franky and
libfranka versions selected for the real-time FR3 machine.

- Installed Franky version:
  `2.0.1.dev60+gf88f0e9bvelimpzdt.libfranka.0.21.2`
- Installed Franky wheel SHA-256:
  `8590e1241a8e305cf046c148968b7dc9055d2f58bcb944ddb5db33ebda70c00d`
- Franky source: official commit
  `f88f0e9b8f598d7c537b4e727e4411e67c95e017`, plus local
  `franky_velocity_impedance.patch`
- Recovered source tree: `/home/chen-lab/yifan/franky-source-f88f0e9b`
- Exact libfranka tag/commit: `0.21.2` /
  `9f9304ec0ac897eff3219a67f612b959948535e2`
- Recovered libfranka source tree:
  `/home/chen-lab/yifan/libfranka-source-0.21.2`
- Simulation controller source:
  `source/franka_rl/franka_rl/tasks/manager_based/franka_motion_control/mdp/controllers.py`
  and `mdp/actions.py`

The transferred M4 bundle remains immutable. This audit records what can be
reused and what the hardware adapter still has to implement.

## Result

Franky's Cartesian controller is a good base for M4, but it is **not an exact
implementation of the simulation controller**. The Cartesian spring-damper
law, model Coriolis term, measured Jacobian/twist, 1 kHz callback and 50 Hz
zero-order-held reference all align. Four differences are material for the M4
demos:

1. Franky's posture task uses the M3 kinematic projector. M4 tracking and curve
   demos use a mass-weighted dynamically consistent torque projector.
2. The current Franky plus libfranka torque-processing cascade is not the same
   state machine as simulation's slew limiter followed by low-pass filter.
3. Franky has a Python-side Z-line virtual-fixture example built on its generic
   tracker, but no native callback-synchronous implementation of M4's
   Z-line/YZ-circle fixture law.
4. Simulation and Franky use different clipping and final effort-guard layers.

Therefore the existing `CartesianImpedanceTrackingMotion` should be extended,
not used unchanged as the final M4 hardware controller. It is already suitable
as the common base for task torque, Coriolis, reference transport and the M3
static-posture projector.

## Controller equations

### Joint impedance

With optional features disabled, both simulation and Franky implement

```text
e_q  = clip(q_ref - q, -0.5, 0.5)
e_dq = dq_ref - dq
tau_raw = Kq e_q + Dq e_dq + C(q,dq)
```

Simulation then processes the non-gravity command and adds simulated gravity.
Franky sends a non-gravity FCI torque command; the robot supplies gravity.
Franky additionally supports feedforward torque, friction, soft-limit
repulsion, gain smoothing and optional Cartesian shaping. Those terms must be
zero/disabled when simulation does not contain them.

The local `JointVelocityImpedanceTrackingMotion` is the same joint impedance
law. It integrates the velocity reference into `q_ref` at each native callback,
clamps that reference to configured joint bounds, and supplies both integrated
`q_ref` and requested `dq_ref` to the PD law. It corresponds to the earlier
velocity-impedance policy route; it is not the Cartesian M4 controller.

### Cartesian impedance

The common task-space law is

```text
v_measured = J(q) dq
e_x = [p_des - p; Log(R_des R^T) expressed in the base frame]
wrench = Kx e_x + Dx (v_des - v_measured)
tau_task = J^T wrench
tau_raw = tau_task + tau_posture + C(q,dq)
```

Franky's implementation forms the opposite-signed pose error internally and
then multiplies by `-K`; the resulting torque sign and base-frame rotational
error agree with the simulation convention. It obtains `J` at the Franka
end-effector frame and computes measured twist as `J*dq`. It always adds model
Coriolis in the Cartesian controller. Franky's optional desired-acceleration
feedforward is absent from the current M4 simulation and should remain unset.

Gravity is not a missing Franky term. In simulation, PhysX needs the explicit
gravity torque that `TorqueCommandProcessor` adds after filtering. On the FCI
torque path, the robot supplies gravity compensation, so the client command
contains Coriolis but no gravity term.

## Feature-by-feature comparison

| Property | `franka_rl` simulation | Pinned Franky | Assessment |
|---|---|---|---|
| Servo rate | 1 kHz physics/controller | Native FCI callback, nominally 1 kHz | Match |
| Reference rate | 50 Hz, zero-order hold | Thread-safe latest reference read each callback; last value held | Match, but Franky has no timestamp watchdog |
| Cartesian pose error | full 6D position/SO(3) axis-angle | full 6D position/SO(3) angle-axis in base frame | Match in law/sign |
| Desired twist | used in damping error | `target_twist` used in damping error | Match |
| Measured twist | `J*dq` | `J*dq` | Match |
| Task torque | `J.T @ wrench` | `J.transpose() * wrench` | Match |
| Coriolis | added before processing | added before callback slew limit | Match |
| Gravity | added by simulation model after non-gravity processing | supplied by robot/FCI | Same physical responsibility, different layer |
| Tracking posture | dynamic projector `I-J.T (J M^-1 J.T)^+ J M^-1` | kinematic projector `I-J^+J` | **Mismatch** |
| M3 static posture | kinematic projector `I-J.T pinv(J.T)` | kinematic projector `I-J^+J` | Match |
| Curve fixture | measured-state line/circle projection inside the 1 kHz controller | Python Z-line example streams projected pose/twist to the generic tracker; no corresponding native line/circle motion type | Useful prototype, **not M4 parity** |
| Cartesian error clip | no controller-level component clip | default 0.10 m translation and 0.25 rad rotation per component | **Mismatch unless configured** |
| Gain updates | fixed per environment | runtime updates smoothed with 0.1 s time constant | Equal after settling; transient updates differ |
| Slew/filter | independent 1 Nm/tick limiter state, then independent 100 Hz filter state | Franky limits against prior applied `tau_J_d`; libfranka then filters against the same `tau_J_d` | **Mismatch** |
| Effort clamp | total simulated torque, including gravity, clipped to `[87 x4, 12 x3]` Nm | no identical client-side final clamp in Cartesian motion; robot protections remain | **Mismatch** |
| Joint-limit repulsion | absent from M4 equation | optional | Disable for parity or add the same term to simulation |
| Friction compensation | absent | optional, zero by default | Keep zero for parity |
| Reference freshness | simulation/session-side checks | latest-value handle only | Hardware adapter needs native watchdog and applied sequence |
| Stop | simulation damping surrogate; explicitly not a hardware stop | ramp to joint damping + Coriolis, slew limited, rest/timeout finish | Franky stop is the hardware base, with local guards still required |

## Posture projector evidence

Franky computes a raw posture torque

```text
tau_p = Kp (q_p - q) - Dp dq
```

then applies

```text
tau_null = tau_p - J^+ (J tau_p) = (I - J^+J) tau_p.
```

This is algebraically the same as M3's
`(I - J.T @ pinv(J.T)) tau_p`. It is generally different from M4's
mass-weighted torque projector. The deterministic numerical check reports a
maximum element difference of about `0.226` for one representative full-rank
`J,M` pair, while Franky versus M3 differs only at floating-point roundoff
(`1.93e-15`). These are structural results, not identified robot parameters.

### Why one projector is called kinematic

For a redundant arm, an infinitesimal joint velocity is invisible to the
Cartesian task when `J qdot_null = 0`. The Euclidean orthogonal projector onto
that velocity nullspace is

```text
P_kin = I - J^+ J.
```

Franky applies this matrix to a candidate **joint torque** even though the
projector itself comes only from the kinematic Jacobian. It guarantees
`J tau_null = 0`; it does not generally guarantee that the torque produces no
Cartesian acceleration, because acceleration depends on `M^-1 tau`.

M4 instead uses

```text
P_dyn = I - J^T (J M^-1 J^T)^+ J M^-1.
```

For a full-row-rank Jacobian,
`J M^-1 P_dyn tau = 0`. Thus the projected secondary torque has no
instantaneous task-space acceleration in the rigid-body model. `P_kin` is a
symmetric Euclidean projector; `P_dyn` is generally an oblique, mass-dependent
torque projector. They coincide only in special cases, such as suitably
isotropic inertia and compatible Jacobian geometry.

## What PR #79's virtual fixture provides

Franky PR #79 added the generic `CartesianImpedanceTrackingMotion`, its Python
`CartesianImpedanceTracker` wrapper, and
`examples/cartesian_impedance_tracker.py`. The example:

1. reads the measured end-effector position in Python;
2. projects it onto an infinite world-Z line;
3. estimates desired line velocity from consecutive projected positions;
4. publishes the projected pose and twist through the tracker;
5. lets the native C++ impedance callback continue holding the latest target at
   the FCI rate.

This demonstrates the same high-level virtual-fixture idea and means Franky
already provides the main Cartesian impedance building block. It is not the M4
fixture controller for four concrete reasons:

- the projection and finite difference run in a Python loop scheduled with
  `time.sleep`, rather than in the native FCI callback;
- it implements only the infinite Z line, not the YZ circle or center
  singularity rejection;
- tangent compliance emerges by moving the target with the measured position
  and matching its estimated velocity, whereas M4 explicitly computes normal
  spring/damping and a separate 20 N-s/m tangent damper;
- it uses Franky's kinematic posture projector if a posture task is enabled,
  whereas M4 fixtures require the mass-weighted dynamic projector.

The example can guide API and lifecycle design, but copying it unchanged would
change M4's controller and real-time contract.

## Torque-processing evidence

The simulation has two independent states:

```text
tau_limited[k]  = tau_limited[k-1]
                + clip(tau_raw[k] - tau_limited[k-1], +/-1 Nm)
tau_filtered[k] = tau_filtered[k-1]
                + alpha (tau_limited[k] - tau_filtered[k-1])
```

with `alpha = 0.3858695451` at 1 ms and 100 Hz.

The current Franky callback first limits against the previously applied
`robot_state.tau_J_d`. The runtime calls `Robot.move(... limit_rate=False,
cutoff_frequency=100)`, so libfranka 0.21.2 then filters that callback output
against the same previous `tau_J_d`. For a sustained large step this caps the
applied increase at approximately `alpha * 1 = 0.386 Nm/ms`. The simulation
filter transient lags its limiter, but its sustained slope approaches
`1 Nm/ms`. The reproducible step check below compares the simulation's filtered
non-gravity state before gravity and final effort clipping with the command
leaving libfranka's client-side filter:

| elapsed | simulation filtered non-gravity state | Franky + libfranka command |
|---:|---:|---:|
| 10 ms | 8.421 Nm | 3.859 Nm |
| 20 ms | 18.409 Nm | 7.717 Nm |
| 100 ms | 98.408 Nm | 38.587 Nm |

The simulation's later gravity addition and effort clamp are intentionally not
included in these numbers. This is a command-transient mismatch. It does not
change the steady-state PD equation, but it can change path lag and stop/entry
transients.

## Stop behavior

`TorqueStopMotion` captures the preceding command, ramps over 0.2 s by default
toward

```text
tau_stop = -D_stop dq + C(q,dq),
```

applies the configured per-tick torque slew cap, and finishes after every joint
is below the velocity threshold or the maximum duration expires. It is a sound
base for a reviewed hardware stop. The adapter still needs fault latching,
watchdog transition, stop-result telemetry and a policy for a stop timeout.
With libfranka's 100 Hz command filter enabled, the actual stop torque is also
downstream-filtered.

## Implementation decision

Use the pinned Franky Cartesian implementation for:

- 1 kHz FCI torque callback;
- FK/model Jacobian, measured twist and Coriolis;
- SO(3) Cartesian impedance and desired-twist feedforward;
- thread-safe 50 Hz latest-reference transport;
- M3 static-posture kinematic projection;
- torque-stop motion as the stop primitive.

Add or change the following in the native extension:

1. Add a selectable mass-weighted dynamic posture projector using hardware
   `M(q)` for tracking and curve demos.
2. Add native measured-state line and circle fixture reference generation.
3. Retain the present Franky/libfranka cascade and reproduce it in simulation:
   Franky limits the raw non-gravity command against the previous `tau_J_d`,
   then libfranka applies its 100 Hz filter against the same previous
   `tau_J_d`. Keep `limit_rate=false` and `cutoff_frequency=100`. This is now the
   selected contract because the simulation was intended to mimic the existing
   hardware path. Re-run the M4 simulation evaluations after changing it.
4. Configure Cartesian error clipping explicitly. Disabling it gives numerical
   parity; a finite hardware guard should remain separate and logged.
5. Add the reviewed final effort/non-finite/slew/workspace/joint/contact guards,
   native reference watchdog, applied-reference sequence and bounded telemetry.

No controller-identification experiment is required for these explicit
client-side equations. Commissioning is still required to validate the robot
model, payload/TCP, unmodeled friction/compliance, delay and stability on the
physical arm.

## Reproduction

Run:

```bash
python3 deployment/hardware_control_audit/2026-10-11-m4-franky-controller-parity/verify_controller_parity.py
```

The script rewrites `numerical_results.json`. It checks the two projector
forms and the two torque-processing state machines without connecting to FCI.

