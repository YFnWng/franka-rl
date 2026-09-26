# Deployment limits and timing contract

This file is the index for limits used by the FR3 Franky PPO path. It separates
robot/model constraints, lab operating limits, controller limits, and
experiment-only checks. The reviewed YAML and each run's `resolved_config.json`
remain the executable records.

## Timing

The characterized hardware controller runs at 1 kHz. Completed reference tests
used 30 Hz held position targets, a 75 ms action watchdog, a 50 ms state/callback
budget, and a 25 ms PPO inference budget. No PPO bundle has yet been qualified
on hardware, so 30 Hz is measured reference-test history rather than a frozen
policy interface.

For new training, use this candidate clock contract:

- PhysX and the Franky torque law: 1 kHz.
- PPO observations/actions: 50 Hz, or one update every 20 physics steps.
- Hold each position target between policy updates; evaluate impedance torque
  every physics step.
- Qualify policy inference and scheduling below 20 ms before freezing 50 Hz for
  hardware. Choose tighter inference/watchdog thresholds from measured latency.
- Train and deploy at the same policy rate. Do not convert an already-trained
  30 Hz policy to 50 Hz without retraining and reevaluation.

This removes the 3 kHz common clock previously needed for exact 30 Hz and 1 kHz
integer ratios. A 1 kHz/50 Hz implementation should first reproduce the recorded
0.1 Hz K=50/100/200 reference responses. Increase the physics rate only if that
comparison or contact-stability tests show a material integration error.

## Joint position limits

The model bounds and the lab soft bounds are in radians. The soft range is the
model range inset by 5% of each joint's full range.

| Joint | Model lower | Model upper | 5% margin | Lab soft lower | Lab soft upper |
|---|---:|---:|---:|---:|---:|
| J1 | -2.900740 | 2.900740 | 0.290074 | -2.610666 | 2.610666 |
| J2 | -1.836090 | 1.836090 | 0.183609 | -1.652481 | 1.652481 |
| J3 | -2.900740 | 2.900740 | 0.290074 | -2.610666 | 2.610666 |
| J4 | -3.077020 | -0.116937 | 0.148004 | -2.929016 | -0.264941 |
| J5 | -2.876303 | 2.876303 | 0.287630 | -2.588673 | 2.588673 |
| J6 | 0.439823 | 4.621633 | 0.209091 | 0.648913 | 4.412543 |
| J7 | -3.050833 | 3.050833 | 0.305083 | -2.745750 | 2.745750 |

Every mapped PPO target and measured/desired active state is checked against the
soft bounds. Franky's joint-limit repulsion starts another 0.1 rad inside these
bounds, uses stiffness 4.0, damping 1.0, and is capped at 5 Nm.

## Velocity, acceleration, and jerk

The installed Franky build reports these motion-generator maxima. They are
Franky trajectory limits, not the robot-description velocity envelope or the
libfranka interface constants:

| Joint | Velocity rad/s | Acceleration rad/s^2 | Jerk rad/s^3 |
|---|---:|---:|---:|
| J1 | 2.175 | 15.0 | 7500 |
| J2 | 2.175 | 7.5 | 3750 |
| J3 | 2.175 | 10.0 | 5000 |
| J4 | 2.175 | 12.5 | 6250 |
| J5 | 2.610 | 15.0 | 7500 |
| J6 | 2.610 | 20.0 | 10000 |
| J7 | 2.610 | 20.0 | 10000 |

The live robot-returned FR3v2.1 description separately defines maximum velocity
`[2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]` rad/s and a position-dependent
near-bound envelope. For each joint:

```text
v_upper(q) = min(vmax, max(0, -offset + sqrt(max(0, 2*d*(qmax-q))))) - 0.001
v_lower(q) = max(-vmax, min(0,  offset - sqrt(max(0, 2*d*(q-qmin))))) + 0.001
```

The corresponding velocity offsets are
`[0.652000, 0.249969, 0.200531, 0.354218, 0.573832, 0.488465, 0.459195]`
rad/s and deceleration parameters are `[6.0, 2.585, 3.5, 4.0, 17.0, 5.5,
17.0]` rad/s^2. The archived libfranka interface constants are 9.999 rad/s^2
acceleration, 4999.999 rad/s^3 jerk, and 999.999 Nm/s torque slew for every
joint. These layers must not be substituted for one another.

Reference experiments use 20% of the Franky motion-generator values as
conservative analytic waveform design caps:

| Joint | Velocity rad/s | Acceleration rad/s^2 | Jerk rad/s^3 |
|---|---:|---:|---:|
| J1 | 0.435 | 3.0 | 1500 |
| J2 | 0.435 | 1.5 | 750 |
| J3 | 0.435 | 2.0 | 1000 |
| J4 | 0.435 | 2.5 | 1250 |
| J5 | 0.522 | 3.0 | 1500 |
| J6 | 0.522 | 4.0 | 2000 |
| J7 | 0.522 | 4.0 | 2000 |

These 20% caps reject an unsafe sine design before execution. They are not
continuous PPO state limits. The PPO torque path holds position references with
desired velocity zero and currently has no velocity/acceleration/jerk governor.
The separate startup-velocity check is 5% of model maximum:
`[0.10875, 0.10875, 0.10875, 0.10875, 0.1305, 0.1305, 0.1305]` rad/s.

## Controller and torque limits

- Nominal impedance: K=100 Nm/rad and D=20 Nms/rad for every joint.
- Gain test/DR endpoints: K=50/100/200 with D=2*sqrt(K).
- Position error entering the tracker is clipped to 0.5 rad.
- Coriolis compensation is enabled; constant and friction feedforward are zero.
- Commanded torque change is limited to 1 Nm per 1 ms callback.
- The robot description effort fields are 87 Nm for J1-J4 and 12 Nm for J5-J7.
  These are robot-description/interface constraints, not extra coordinator clamps
  or measurements of actuator saturation.
- Constructing `franky.Robot` sets collision thresholds of 20 Nm per joint and
  30 N per Cartesian axis. These are collision thresholds, not command limits.
- The torque-stop controller ramps for 0.2 s, may run for at most 2 s, and uses
  damping `[2, 2, 2, 1, 1, 1, 0.5]` Nms/rad with Coriolis compensation.

## Tracking and lifecycle limits

- Start pose tolerance: 0.005 rad per joint in the pending PPO template.
- Active tracking-error fault threshold:
  `[0.08, 0.08, 0.08, 0.08, 0.06, 0.06, 0.04]` rad.
- Current 30 Hz template: 50 ms state timeout, 50 ms callback-gap limit, 75 ms
  action watchdog, and 25 ms inference timeout. These must be requalified and
  tightened if the policy interface moves to 50 Hz.
- Robot errors, invalid mode, soft-position violation, tracking error, callback
  loss, inference timeout, and watchdog expiry are terminal for a session.

## Authoritative locations

- Pending executable values: `franky_runtime/config/ppo.pending.yaml`.
- Config validation and FR3 dynamic constants: `franky_runtime/franky_experiment/core.py`.
- Controller construction and 1 kHz command path: `franky_runtime/franky_experiment/backend.py`.
- Exact values used by a run: its immutable `resolved_config.json`.
- Hardware response evidence: `hardware_control_audit/2026-09-26-franky-impedance/comparison.json`.
- Parameter provenance: `franky_runtime/IMPEDANCE_PARAMETER_SURVEY.md`.
- Robot-description envelope and libfranka constants:
  `hardware_control_audit/2026-09-25/limits.json` and `FINDINGS.md`.

The proposed 50 Hz clock is not yet an executable hardware setting. Freeze it
only after simulator replay and host inference/scheduling qualification.
