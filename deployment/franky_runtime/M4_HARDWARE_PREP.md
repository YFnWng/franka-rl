# M4 hardware preparation — 2026-10-12

The next hardware step is a 10-second **measured-pose hold**, not a transferred
trajectory replay. The controller does not home the arm or command the fixture
home position. The first configuration has Cartesian K=[100,100,100,10,10,10],
D=[20,20,20,6.5,6.5,6.5], zero posture gains, 1 Nm/tick slew and libfranka's
100 Hz filter with limit_rate=false. Gravity remains supplied by the robot;
Coriolis comes from the libfranka model inside Franky.

## Installed stack

Use `~/yifan/venvs/franky-m4-baseline-server10/bin/python`.
Franky version: `2.0.1.dev63+gm4baseline.libfranka.0.21.2`.
Wheel and dependency receipt:
`~/yifan/deployment_bundles/2026-10-12/franky-m4-baseline/`.
Every one of its 21 bundled native dependency files is byte-identical to
`franky-velocity-impedance-zdt-server10`, including libfranka and Ruckig.
Only the Franky core and Python extension were rebuilt. The tested environments
were not modified. The previous dev62 M4 wheel is superseded for commissioning.

## Software implemented

`M4RuntimeMotion` is a persistent native Cartesian torque motion. The Python
adapter publishes pose/twist and monotonically increasing sequence numbers at
50 Hz; the native controller timestamps receipt and checks freshness at each
servo tick. The source also supports reset-anchored posture with native quintic
transitions and the native Z-line/YZ-circle fixture laws. A different local
configuration is required for each task.

Native faults latch and enter the existing `TorqueStopMotion` within the same
motion object. Guards cover reference timeout/sequence, robot and host callback
gaps, finite inputs/Jacobian/command, measured joints and velocity, measured and
reference workspace, Cartesian error/speed, external torque, contact/collision,
Jacobian conditioning, command success rate, raw command torque and bounded log
overflow. A fixture can allow contact for hand guidance; collision always faults.
The controller/projection exception path requests the same native stop. A failed
libfranka/model operation can still terminate control through libfranka's own
exception/reflex path rather than complete the damping stop.

A preallocated SPSC ring records servo samples at 1 kHz, drained by Python.
`servo.jsonl` contains q/dq, encoder-model flange pose and J*dq twist, applied
reference pose/twist and sequence, posture, raw/task/nullspace/limit/Coriolis
command components, Franky slew output, actual previous tau_J_d, measured and
external torque, timing, singular value, faults and stop state. The predicted
post-filter command is explicitly labeled a prediction; it is not a measured
current command. Gains, model hash, dependency hashes and fixture parameters
are recorded in the copied configuration/preflight evidence. This is not yet a
per-tick full J/M matrix dump.

## Before connecting

Validate the config and immutable transfer without FCI:

```bash
PY=/home/chen-lab/yifan/venvs/franky-m4-baseline-server10/bin/python
RUNTIME=/home/chen-lab/yifan/franka-rl/deployment/franky_runtime
"$PY" "$RUNTIME/run_m4_hardware.py" \
  --config "$RUNTIME/config/m4.hold.pending.yaml"
```

The pending config uses the existing soft joint interval. Its new Cartesian
workspace and abort thresholds are local commissioning candidates. The hold
workspace is a +/-30 mm box around the transferred home FK
[0.306891, 0, 0.590282] m. It intentionally does not automatically expand to
include an arbitrary robot start. Review the box against the current arm pose;
no automatic approach or homing is performed.

With FCI open, collect a **no-motion preflight**:

```bash
"$PY" "$RUNTIME/run_m4_hardware.py" \
  --config "$RUNTIME/config/m4.hold.pending.yaml" --preflight
```

This calls no motion API. Franky's existing Robot constructor **does configure
collision thresholds** (20 Nm / 30 N in this config); consequently this is named
preflight, not read-only. It checks FCI server 10, exact robot-returned URDF,
identity flange-to-EE transform, zero payload, rest state, joint/workspace bounds,
external torque and the start Jacobian. The recorded encoder snapshot is saved
even when a subsequent preflight check fails. Inspect the output directory
printed by the command. Do not recover errors or change the TCP/payload silently.

## First hold

After the no-motion preflight, copy the pending YAML to a new local reviewed
file. Use the existing coordinator approval fields (`status: approved`,
`executable: true`, `motion_authorized: true`, and the four `approval` strings).
The transfer bundle stays immutable; approval applies only to this local task.
The config's guard values and physically clear hold box must reflect that review.

```bash
"$PY" "$RUNTIME/run_m4_hardware.py" --config /absolute/path/to/m4.hold.approved.yaml --execute
```

It rechecks the start after the terminal `start` confirmation. During motion,
type `stop` then Enter or press Ctrl-C to request the native damping stop. The
physical stop remains the operator's immediate intervention. Sessions do not
chain automatically. The runner distinguishes successful stop at rest from
maximum-duration termination, and records native faults and dropped samples.
Output is a fresh task/timestamp directory beneath
`~/franka_ros2_ws/hardware_inventory/2026-10-12/m4/`.

## Validation and remaining scope

- Native C++ tests: 13 cases / 41 assertions passed, including projector/fixture
  math, guard boundaries, hand-guidance contact semantics and publication checks.
- Installed Python bindings and adapter/config tests: 9 passed.
- All five transferred portable fixtures passed in the new environment's fake
  lifecycle; fake mode does not emulate robot dynamics.
- Native curve projection/force helpers match transfer fixtures to <=1.28e-15.
- Immediate ELF symbol binding and dependency resolution passed; all bundled
  dependencies match the tested baseline byte-for-byte.
- No FCI connection or physical commissioning has been performed by this work.

Full tracking/posture torque replay through the compiled controller, real-time
performance/stop commissioning and the simulation torque-processor correction
remain separate acceptance work. The tests above do not establish those results.
A passed hold is followed by a small translation/return and small-circle test,
then posture/fixture commissioning; the full five-demo suite is not yet validated.
