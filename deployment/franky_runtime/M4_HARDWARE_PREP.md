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
preflight, not read-only. It checks compatibility through the pinned libfranka protocol-10 handshake, exact robot-returned URDF,
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


## Current receiving status

Use bundle `~/yifan/2026-10-12_m4_torque_v2_verified`; its portable wheel is
installed and the pending hold config already references its verified manifest.
The simulation torque-processing correction is now verified (seeded 30-tick
fixture matches exactly). Franky and its native dependency files did not change.
Live no-motion preflight passed in `hardware_inventory/2026-10-12/m4/
hold.20261012T133558Z/`. `Robot.server_version` is not part of the installed
Python binding; the adapter uses the native compatibility handshake and records
that distinction in `connection-evidence.json`. The next task remains the
pending measured-pose hold, followed by physical controller commissioning.


## Hold passed and circle prepared — 2026-10-12

Hold `hold.20261012T134119Z` completed with native fault 0, no dropped
samples, and smooth stop at rest. Active position error max 1.85 mm,
RMS 1.79 mm; maximum robot callback period 1 ms. See its `hold-summary.json`.

The user requested circle tracking. Local approved configuration:
`config/m4.circle_yz.approved.yaml`. It uses immutable `circle_yz_m2`:
center [0.475, 0, 0.35] m, YZ-plane radius 0.15 m, one turn at 0.2 rad/s,
6-second quintic approach, 2-second settle and 6-second return to the measured
start. Total duration 47.416 s; peak reference speed from the recorded start
0.05985 m/s. Pose/twist feedforward is published at 50 Hz. Cartesian gains remain
K=[100,100,100,10,10,10], D=[20,20,20,6.5,6.5,6.5]; posture gains remain zero.
The workspace was expanded to include approach/circle/return with 30 mm margin:
[0.276891,-0.18,0.17] to [0.505,0.18,0.620282] m. All other guards and stop
settings are unchanged. Offline trajectory derivative bounds and path IK passed
using the completed hold start; `config/m4.circle_yz.design-audit.json` records
that check. The runner rechecks and replans from the live measured start before
motion. No circle motion has yet been executed.

```bash
/home/chen-lab/yifan/venvs/franky-m4-baseline-server10/bin/python \
  /home/chen-lab/yifan/franka-rl/deployment/franky_runtime/run_m4_hardware.py \
  --config /home/chen-lab/yifan/franka-rl/deployment/franky_runtime/config/m4.circle_yz.approved.yaml \
  --execute
```

Type `start` at READY. Type `stop` or press Ctrl-C for native stop.


## First circle approach fault — 2026-10-12

`tracking.20261012T134442Z` stopped at 2.118 s, before reaching the circle.
Native pose-error guard tripped on translation: 30.234 mm versus 30 mm;
orientation error was 0.03727 rad versus 0.2 rad. Reference linear speed was
0.04955 m/s, measured speed 0.03393 m/s. Raw command peak per joint remained
below 0.90 Nm; no logged torque saturation is indicated. Native stop finished
at rest, fault 8, zero dropped samples. The run does not validate circle tracking.
See `tracking-summary.json` in the session directory.

Prepared `config/m4.circle_yz.slow-approach.approved.yaml`: same circle, angular
speed, controller gains, workspace, 30 mm guard, native binaries and stop profile.
Only approach/return durations increase from 6 to 18 seconds and settle from
2 to 4 seconds. Total 73.416 s, peak approach reference speed ~0.01977 m/s from
the recorded start. This tests speed-related lag; it is not a confirmed fix for
low-stiffness/friction/model mismatch. The adapter supports explicit local task
approach/exit/settle timing overrides without modifying the immutable bundle.
Trajectory derivative checks and IK passed offline; the runner checks the actual
current start again. Adapter regression/timing tests: 13 passed. No retry motion
was executed by the agent.


## Slow approach still faults — 2026-10-12

`tracking.20261012T134815Z` reached translation guard at 7.299 s during
approach: 30.084 mm position error, 0.05402 rad orientation error. Circle was
not reached. Reference speed 0.01847 m/s, encoder-derived speed 0.01293 m/s.
Native fault 8, smooth stop at rest, zero dropped samples. Minimum Jacobian
sigma 0.22435; maximum reference age 19.89 ms. Raw nongravity commands stayed
below 0.9271 Nm per joint. Offline replay of 73 samples using the transferred
URDF Jacobian and pose/twist PD law reproduced logged task torque with maximum
absolute error 6.67e-16 Nm. This verifies task torque mapping on logged samples,
not the entire physical dynamics or total command pipeline.

Slowing the approach delayed the fault but did not resolve lag. Insufficient
Cartesian stiffness against friction/model residuals is plausible; this is not
uniquely identified by the logs. Suggested next comparison: double Cartesian
K to [200,200,200,20,20,20], scale D by sqrt(2) to
[28.2843,28.2843,28.2843,9.1924,9.1924,9.1924], retain slow approach and 30 mm
pose-error guard. These are Cartesian gains (N/m, Nm/rad), not the earlier
joint K=200 gains. No gain change or retry was executed in response to this
fault. Session audit is saved as `tracking-summary.json` in the run directory.


## Cartesian K200 run — 2026-10-12

`tracking.20261012T135441Z` faulted during approach at 10.559 s on position
error 30.075 mm; orientation error only 0.003325 rad. Smooth stop finished at
rest, zero dropped samples. Peak raw nongravity torque 1.652 Nm (J4), minimum
Jacobian sigma 0.22415, maximum reference age 20.00 ms. No circle reached.
The start pose differed from previous runs, so fault time alone is not a
controlled gain comparison. See session `tracking-summary.json`.
Suggested next diagnostic: translation K400/D40, retain rotation K20/D9.1924,
slow approach, same geometry and 30 mm guard. This targets translation while
leaving the already-small orientation error alone. This suggestion has not
been applied or executed. Residual friction/model effects remain hypotheses;
no native dynamic profile or Ruckig limiter is active on this route.


## Active circle configuration — K400 translation

User authorized translation K=[400,400,400], D=[40,40,40]; rotation remains
K=[20,20,20], D=[9.1923881554,9.1923881554,9.1923881554]. The active approved
file is now `config/m4.circle_yz.approved.yaml`, with 18-second approach/return
and 4-second settle. Geometry, speed, workspace, 30 mm pose guard, all other
limits and stop profile remain unchanged. Config/bundle validation passed
without FCI. No motion was executed by the agent.

Per user request, superseded `m4.circle_yz.slow-approach.approved.yaml` and
`m4.circle_yz.k200.slow-approach.approved.yaml` were removed. Their exact executed
configurations remain in the timestamped hardware session directories. Earlier
commands referencing those removed files are historical; use the active file.


## Automatic hardware metrics

`deployment/franky_runtime/summarize_m4.py RUN_DIR` processes saved
`config.yaml`, `start.json` (tracking), `report.json` and `servo.jsonl` offline.
Run with the M4 baseline Python; no FCI connection or Isaac dependency.
It writes `m4-metrics.json` and `m4-metrics.csv` beside the input data.
Use `--output /path/name.json` for another location.

Implemented: sample-weighted RMS/mean/p95/max translation and SO(3) orientation
errors against actual logged applied references, exact geometric circle contour
error, linear/angular speeds, positive speed excess over reference, joint travel
and soft-limit margins, robot/host timing, reference age, command success, first
fault and completion status. Metrics are separated into approach, settle,
track and exit; stop/fault/invalid feedback is excluded from active metrics.
Circle contour metrics apply only to the track phase. CSV includes measured and
reference XYZ for path overlays. No calibrated external-force/stiffness claims,
finite-difference acceleration/jerk, full fixture/posture-specific metrics or
automatic plots are implemented by this hardware adapter yet. Simulation's
`compile_cartesian_results.py` expects finalized simulation NPZ artifacts and
does not directly accept hardware JSONL. The shared metrics module currently
provides geometric contour helpers, not every planned metric API.

Phase labels are inferred from robot elapsed time relative to the first servo
sample and the saved trajectory settings. The publisher trajectory clock was
not logged, so exclude 25 ms around phase boundaries. Errors use the same-sample
logged reference; no ideal-reference time realignment is silently imposed.
A faulted partial circle is never treated as a successful completed trial.

K400 trial `tracking.20261012T135719Z` reached tracking and faulted at 51.851 s
on command success 0.949999988 versus threshold 0.95. Floating-point
representation puts nominal 95% just below the decimal threshold; success had
already decreased through 98%, 97% and 96%. This is a command-delivery/timing
indicator, not a pose error. No threshold change was applied. Position error
at fault was 15.564 mm; smooth stop succeeded with no dropped telemetry.


## Active circle comparison — K800 translation

User authorized the next comparison. `config/m4.circle_yz.approved.yaml`
now uses translation K800/D56.5685424949 and rotation K20/D9.1923881554.
The 18-second approach/return, 4-second settle, 0.2 rad/s circle, all guards
(including command success 0.95 and position error 30 mm), stop settings and
native dependencies remain unchanged. Offline config/bundle validation passed;
no robot connection or motion was performed. K400 executed configuration remains
in `tracking.20261012T135719Z/config.yaml`. Compare using the saved measured
starts: execution does not home automatically, and differing starts confound
approach comparisons. The command-success issue remains unresolved.


## K800 tracking audit — 2026-10-12

`tracking.20261012T140931Z` reached the circle, then faulted at 41.622 s on
command success 0.949999988 (<0.95); pose error at fault 5.729 mm, orientation
0.04622 rad. Native stop finished at rest, no dropped telemetry. libfranka defines
success as the proportion of the last 100 control commands received by the robot;
this is separate from logger drops and reference freshness. Reference age at
fault 4.541 ms. Maximum observed active host callback interval 2.294 ms does not
identify the cause of command-delivery failures; send-time/network/RT scheduling
need further investigation. Nominal 95% is represented slightly below 0.95; the
strict comparison trips at that boundary. No guard or native binary changed.

Observed partial-circle position RMS/max 6.713/8.402 mm; contour RMS/max
3.882/6.842 mm; orientation RMS/max 0.06649/0.08229 rad. Matched nominal circle
window (elapsed ~22.025 to 41.621 s) K400 position/contour RMS 13.818/8.585 mm,
K800 6.713/3.882 mm. This supports a gain-dependent translational error, but the
starting postures differ and these are partial, faulted trials, not full-circle
acceptance. Preserve K800 while investigating command delivery; increasing gains
will not address this failure. JSON/CSV metrics and `gain-comparison.json` are
saved in the K800 session directory.


## Command-success investigation and scheduler correction

See `deployment/franky_runtime/M4_COMMAND_SUCCESS_INVESTIGATION.md`.
Confirmed libfranka constructor promotes Python to FIFO99; adapter now restores
caller scheduling after construction and native control-thread spawn while
preserving FIFO99 for the native thread. New sessions record scheduling evidence.
17 tests and circle no-motion preflight passed. No native dependency, gain or
guard changed; no motion performed. Retry K800 to test whether this resolves
command delivery. CPU governor/IRQ/NIC settings remain unchanged; root cause
is not proven until a new trial confirms the effect.


## Scheduler-fixed retry and float32 guard boundary

`tracking.20261012T141804Z` records Python OTHER/0 and actual new native
thread FIFO99: priority separation works. It completed circle tracking and
faulted during return at 62.544 s on command success 0.949999988. Smooth stop
succeeded, zero dropped samples. Full observed track RMS/max: position
6.883/8.867 mm, contour 3.451/6.436 mm, orientation 0.06670/0.07743 rad.
Host callback-entry max 4.692 ms (earlier during track); in the 100 ms before
fault max 1.939 ms. Entry timestamps do not reveal send deadlines or callback
completion time. The scheduling fix did not eliminate the fault; do not claim
confirmed communication root cause or completed full-session acceptance.

Confirmed robot wire `rbk_types.h` uses float for control_command_success_rate,
while libfranka widens it to double. Native threshold was a double 0.95, causing
nominal 95% to fault on strict comparison. Adapter now rounds only this threshold
to wire float32 representation (0.949999988079071), so reported 95% meets the
configured minimum and 94% still faults. New sessions save effective-limits.json;
the summary includes this evidence. No minimum-success intent relaxation,
watchdog change, gain change or native-wheel rebuild. Threshold boundary regression
covers 94/95/96%. This change has not been tested in another hardware motion.
Real delivery misses still exist; allowing equality will not cure their cause.
