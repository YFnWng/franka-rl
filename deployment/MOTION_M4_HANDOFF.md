# M4 — five non-RL FR3 demos

Transfer candidate, **not hardware approval**. All execution stays on the
real-time machine; no workstation connection, ONNX, PPO, interactive targets or
Python 1 kHz torque loop. `hardware_transfer.yaml` selects exactly five demos.

| Demo | Source / control contract |
|---|---|
| Circle tracking | `circle_yz_m2`, 0.2 rad/s, desired twist feedforward |
| Lissajous tracking | `lissajous_yz_small`, 0.2 rad/s, desired twist feedforward; current YAML amplitudes preserved |
| Static EE / varying posture | Home flange pose; feasible DLS-corrected posture A/B, 0.60 rad null seed, 6 s quintic transitions, 4 s settling; original M3 kinematic torque projection |
| Z impedance | Infinite base Z line through home; hand-guided tangent motion |
| Circle impedance | Closed base YZ circle through home; hand-guided tangent motion |

The trajectory catalog is authoritative for geometry, orientation, entry and
return times. Runtime must construct tracking entry from **measured** initial
FK and revalidate limits/IK; bundled fixtures use deployment home and are
numerical parity evidence, not blindly replayable hardware commands. Standard
tracking and curve impedance have reset-anchored posture K=2 Nm/rad,
D=1 Nm·s/rad and full-six-row dynamically consistent torque projection:

$$
N_\tau=I-J^T(JM^{-1}J^T)^+JM^{-1}.
$$

Circle/line fixtures retain their home-on-curve initialization check (1 mm).
Hold fixed full xyzw `[1,0,0,0]` orientation, not merely an axis-only constraint.
Normal stiffness 100 N/m, normal/tangent damping 20 N·s/m. No tangent spring,
no time-driven motion and no simulated 0.5 N force injection on hardware.
Projection and posture torque must run **natively at 1 kHz**, using measured
flange pose/twist and hardware model M/J. Circle-center projection is rejected;
independent measured workspace guards remain required for endpoint-free curves.

## Transfer contents and offline checks

The bundle contains a `franka-motion` wheel and source snapshot, YAML catalogs,
robot-returned URDF, five numerical fixtures, runtime adapter scaffold,
dependency versions, optional accepted simulation summaries and SHA-256 manifest.
Large traces remain at their source paths; summary metadata explicitly says when
traces are not included. Evidence from old/smaller YAML is not approval of the
current larger Lissajous path. No simulation results are invented by the fake mode.

On the real-time machine, use its separate Python environment, not system Python
or the Isaac environment. Install the local wheel with its NumPy/PyYAML
dependencies reviewed for that machine; `requirements-workstation.txt` records
workstation versions, not a forced hardware upgrade. Example:

```bash
python -m pip install /path/to/bundle/wheels/franka_motion-0.1.0-py3-none-any.whl
python /path/to/bundle/runtime/run_motion_suite.py \
  --bundle /path/to/bundle --output-dir /path/to/new-offline-artifacts
```

Default mode is fake. It validates shared contracts and reference publication
lifecycle without importing a robot SDK or connecting to FCI. It does not emulate
robot dynamics or synthesize measured feedback. `no_motion` is also offline,
not a shadow robot connection. Configuration edits invalidate the manifest;
create a new reviewed bundle rather than editing an immutable transfer.

## Receiving-agent implementation gates

`FrankyMotionBackend` deliberately fails closed. The repository contains no
audited native continuous Cartesian/curve torque adapter; simply finding
`CartesianImpedanceMotion` or `PostureTask` does not establish these semantics.
Do not replace it with repeated waypoint moves, guessed constructor signatures,
silent SDK upgrades or a Python 1 kHz callback. Before any movement:

1. Record installed Franky/libfranka versions and exact native Cartesian,
   posture and torque-stop code. Confirm robot/tool/payload model and TCP parity.
2. Implement native SO(3)-log Cartesian impedance with timestamped pose/twist
   updates, dynamic tracking posture projection, and the separate M3 kinematic
   posture law. Native feedback-dependent fixtures must use the portable
   `CurveVirtualFixture`/`dynamic_torque_projector` as parity references.
3. Match non-gravity torque processing: impedance + posture + model Coriolis →
   1 Nm/ms slew → 100 Hz first-order filter. Hardware FCI supplies gravity;
   **do not add gravity twice**. Audit any additional libfranka filtering,
   friction compensation, effort/slew constraints and armature/model mismatch.
4. Implement startup pose/reference matching, reference watchdog, joint/contact/
   velocity/workspace guards, circle projection rejection, singularity handling,
   fault latching and locally reviewed torque stop. Simulation fault-stop speed
   spikes were observed previously; do not assume that surrogate is safe hardware.
5. Native buffered logs: robot/monotonic time, published/applied sequence, phase,
   measured q/dq and encoder-FK pose/twist, desired pose/twist/posture, gains,
   model/payload, measured/commanded/external torque with compensation conventions,
   fault and stop events. Never label echoed references as feedback.
6. Run offline golden fixtures and no-motion audits, then obtain local approval
   and commission short hold/small step, weak posture/short transitions, reduced
   path amplitude and brief hand guidance. Inspect tool/cable/self/environment
   clearance. Do not directly execute the 0.60 rad posture candidate.

Only then create a new hardware-approved configuration and functional native
adapter. Hardware results and simulation results must remain separately labeled.

## M4 real-time receiving plan — 2026-10-11

The real-time machine verified the immutable manifest and all five offline
golden fixtures without connecting to FCI. Hardware remains blocked at the
native-adapter gate. The receiving architecture, guards, telemetry contract and
stepwise commissioning sequence are recorded in
`deployment/MOTION_M4_HARDWARE_DEPLOYMENT_PLAN.md`. Receipt evidence is under
`/home/chen-lab/franka_ros2_ws/hardware_inventory/2026-10-11/m4/`.

## Native projector/fixture receipt — 2026-10-11

Franky branch `m4-dynamic-projector-fixtures` now implements the opt-in M4
dynamic projector and native 1 kHz Z-line/YZ-circle fixture at commit
`057a0646`. The default projector remains kinematic for the M3 static-posture
route. The repaired Python 3.12/libfranka 0.21.2 wheel has SHA-256
`a06c85249f705aa1ff9a8540dfe1b611098b2c77c85834886a3622d485a54e79`.
The compiled controller matches both transferred curve fixtures to below
`1.3e-15` maximum absolute error.

The simulation workstation must preserve the M4 controller equations and
change its torque processor to the hardware recurrence:

```text
previous = prior filtered/output non-gravity command
limited_candidate = previous + clip(raw - previous, -1 Nm, +1 Nm)
output = previous + 0.385869545095 * (limited_candidate - previous)
```

Use one persistent `output` state; do not retain an independently advancing
limiter state. Add simulated gravity once after this recurrence. On hardware,
Franky supplies the clip, libfranka supplies the 100 Hz filter with
`limit_rate=false`, and FCI supplies gravity. Full workstation instructions and
required reruns are in `FR3_Industrial_Motion_Control_Demos_HANDOFF.md`.

The workstation now implements `shared_previous_output_v2` in NumPy and Torch.
New manifests identify this version and include a seeded torque-only golden
fixture (`fixtures/torque_processing_v2.npz` and `.json`). `limited` is retained
as a compatibility telemetry alias of `limited_candidate`, not persistent
limiter history. Normal simulation resets seed zero; replay may seed the prior
non-gravity `tau_J_d`. Compare hardware with `filtered`, not `applied` (the
latter includes simulated gravity and an effort guard).

The portable wheel remains version 0.1.0; verify bundle hashes and reinstall
that wheel explicitly with `--force-reinstall --no-deps` in the receiving
environment. Do not replace the audited Franky/libfranka dependencies. Prior
bundle acceptance summaries remain historical; a new builder refuses to attach
old-processor summaries as corrected-controller evidence.


## Hardware runtime and dependency preservation — 2026-10-12

The receiver added native `M4RuntimeMotion` watchdog/guard/stop dispatch and a
bounded 1 kHz telemetry ring, plus a hardware adapter and single-task runner
`deployment/franky_runtime/run_m4_hardware.py`. It supports pose/twist tracking,
measured-pose hold, native quintic posture references, and line/circle fixtures.
The first local candidate is `config/m4.hold.pending.yaml`: 10-second hold with
zero posture gains. It is pending local review, not automatically authorized.

The new wheel `2.0.1.dev63+gm4baseline.libfranka.0.21.2` bundles byte-identical
copies of all 21 native dependencies from the tested RL velocity runtime,
including libfranka and Ruckig. The earlier dev62 dependency rebuild is
superseded for commissioning. Installed environment:
`~/yifan/venvs/franky-m4-baseline-server10`. Artifact and dependency/source hash
receipt: `~/yifan/deployment_bundles/2026-10-12/franky-m4-baseline/`.

Native tests (13 cases/41 assertions), installed binding/adapter tests (9),
portable five-fixture replay and native curve helper parity passed offline.
There has been no FCI connection or hardware motion. Full compiled tracking/
posture torque replay and physical timing/stop commissioning remain unvalidated;
the simulation torque-processing correction requested above also remains open.
Detailed commands, telemetry semantics and guard provenance are in
`deployment/franky_runtime/M4_HARDWARE_PREP.md`.


## Torque v2 verified receipt and live preflight — 2026-10-12

The receiver now selects `~/yifan/2026-10-12_m4_torque_v2_verified`, manifest
SHA-256 `79c3ea84734761e46bc4954320555f5c8bf70a4b14060f43c621aec972938816`.
All 74 files passed integrity verification. The seeded 30-tick torque fixture
matches the installed portable processor and the independent hardware recurrence
exactly in limited/filtered/gravity-added/effort-guarded channels. All five
portable task fixtures pass offline. All 11 included simulation summaries are
complete/accepted and identify `shared_previous_output_v2`; circle/Lissajous
geometry and Cartesian profile explicitly match the included catalogs.
No geometry/gain catalog changes from the prior bundle were found.

The new `franka-motion` wheel (still version 0.1.0) was force-reinstalled with
`--no-deps` in `franky-m4-baseline-server10`. The local `m4.hold.pending.yaml`
bundle path and manifest hash were updated. All 21 bundled Franky native
dependencies remain byte-identical to the tested baseline. No Franky C++ wheel,
libfranka, Ruckig, robot firmware, gains or guard thresholds needed changing.
The simulation torque-processor correction previously listed as open is now
received and verified; physical commissioning is still separate.

The preflight error `Robot has no attribute server_version` was an adapter API
mistake, not a firmware incompatibility. Only the Gripper exposes that Python
property in this wheel. The fixed adapter records the successful checksum-pinned
libfranka protocol-10 handshake as compatibility evidence; it does not fabricate
a Python-measured server version. A regression test covers the missing binding.

Live no-motion preflight passed at
`~/franka_ros2_ws/hardware_inventory/2026-10-12/m4/hold.20261012T133558Z/`.
It checked live URDF, payload/frame, rest state, joint/workspace limits and the
start Jacobian. `connection-evidence.json` explicitly records the unavailable
Python server-version value as null. No motion API was called; the Franky
constructor applied the configured 20 Nm/30 N collision thresholds. An earlier
restricted-sandbox retry's I/O error is separate from the successful permitted
retry. Receipt: `hardware_inventory/2026-10-12/m4/torque-v2-receipt.json`.
The pending 10-second measured-pose hold remains the next commissioning task.


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


## First completed hardware circle session — 2026-10-12

`hardware_inventory/2026-10-12/m4/tracking.20261012T142209Z` completed the
approach, circle and return with native fault 0, smooth stop and zero dropped
telemetry samples. Bundle torque-v2 verified, Cartesian translation K800/D56.5685,
rotation K20/D9.1924, zero posture gains, circle center [0.475,0,0.35] m,
radius 0.15 m, speed 0.2 rad/s, 18 s approach/return and 4 s settle.
Native FIFO99/Python OTHER0 are confirmed in scheduling-evidence.json.
Configured 95% command-success minimum uses wire float32 representation;
this run's minimum active success was 97%, mean 99.5805%. This successful
run does not independently prove the threshold fix or scheduler change was the
sole cause of earlier failures. No further guard/gain/native dependency change.

Track-phase metrics (25 ms boundary exclusions): position RMS/max
7.115/9.004 mm; geometric contour RMS/max 3.776/6.777 mm; orientation
RMS/max 0.06710/0.08008 rad (about 3.84/4.59 degrees). Approach position
RMS/max 5.968/8.704 mm; return position RMS/max 9.221/10.963 mm.
Active reference age p95/max 19.045/22.263 ms. Metrics use encoder/model FK,
not independent external metrology. Saved m4-metrics.json and m4-metrics.csv
include terminal status and phase exclusions. Completed-circle recording spans
33.365 s after 25 ms boundary exclusion versus planned track 33.416 s.

This establishes one successful hardware tracking trial, not repeatability,
full-suite acceptance, posture/fixture commissioning or identified dynamics.
Next comparison: repeat the same configuration without changing gains, geometry,
speed or guards, preferably from the same recorded start. Keep start snapshots
for any repeatability comparison; the runner does not home automatically.


## Second completed circle and rotation-gain comparison

`tracking.20261012T142630Z` completed with native fault 0, smooth stop and
zero dropped samples, providing a second successful full session at translation
K800/D56.5685 and rotation K20/D9.1924. Track position RMS/max
7.050/9.295 mm;
contour RMS/max 3.720/6.558 mm;
orientation RMS/max 0.06625/0.08077 rad.
Metrics JSON/CSV saved in its session directory.

User requested higher rotational gains. Active `config/m4.circle_yz.approved.yaml`
now uses rotation K40/D13, translation K800/D56.5685 unchanged. Geometry,
speed, approach/return timings, workspace, all guards, torque pipeline and
native dependencies unchanged. Offline config/bundle validation passed; no
new motion was performed by the agent. Previous executed gain configuration
remains in both completed sessions. This comparison targets orientation error;
check translation errors and torque/timing as well because the task axes couple
through the arm dynamics. No claim of improved orientation until the next audit.


## Completed rotation K40 comparison — 2026-10-12

`tracking.20261012T142902Z` completed with native fault 0, smooth stop and
zero dropped samples. Translation K800/D56.5685, rotation K40/D13, other
settings unchanged. Track position RMS/max 7.193/9.457 mm; contour RMS/max
3.713/6.970 mm; orientation RMS/max 2.013/2.447 degrees. Against the immediately
preceding K20 repeat, orientation RMS decreased ~47% (3.796 to 2.013 degrees),
while translation RMS changed ~2% (7.050 to 7.193 mm); contour RMS essentially
unchanged (3.720 to 3.713 mm). Initial measured pose/posture differed, so this
is not an isolated controlled gain comparison. Only one K40 repeat is completed.
Active command-success min/mean ~95%/99.5505%; nominal 95% now passes the
corrected float32 boundary, with no intent to allow 94%. No further tuning was
applied. Keep K40 as the current candidate and repeat unchanged to check stability
and repeatability before stronger rotational gains. Metrics JSON/CSV and
rotation-gain-comparison.json are saved in the run directory.


## Rotation K40 repeat completed; Lissajous prepared

`tracking.20261012T143526Z` completed without native faults or dropped samples,
with smooth stop. Track RMS: position 7.195 mm,
contour 3.679 mm,
orientation 1.945 degrees.
Metrics JSON/CSV saved; two full circle sessions now completed at rotation K40.

User requested Lissajous with the same parameters. Approved local config:
`config/m4.lissajous_yz.approved.yaml`, immutable bundled `lissajous_yz_small`:
center [0.475,0,0.35] m, plane [0,90,0] deg, amplitude [0.08,0.08] m,
harmonics [1,2], speed 0.2 rad/s, fixed quaternion [1,0,0,0]. Translation
K800/D56.5685, rotation K40/D13, zero posture gains, 18-second approach/return,
4-second settle and all guards/stop/native dependencies unchanged. Total
73.416 s, sampled peak reference speed 0.03578 m/s from the recorded stop pose.
Trajectory derivative limits and full path IK passed offline; runner rechecks
live measured start. Config/bundle validation passed with no FCI. No Lissajous
motion executed by the agent. Circle successful config remains available.
Current hardware summarizer computes Lissajous position/orientation/timing
metrics but does not yet compute sampled Lissajous contour error; circle-contour
metric remains null rather than applying an incorrect circle distance.


## First completed Lissajous hardware session

`tracking.20261012T144044Z` completed with native fault 0, smooth stop and
zero dropped samples; native FIFO99/Python OTHER0 recorded. Same controller:
translation K800/D56.5685, rotation K40/D13, 0.2 rad/s, 18-second
approach/return, 4-second settle. Track position RMS/max
6.709/9.046 mm;
sampled Lissajous contour RMS/max
3.926/6.672 mm;
orientation RMS/max
1.745/2.173 degrees.
Minimum active command success 95.00%.

Hardware summarizer now includes sampled Lissajous contour distance through the
shared sampled_curve_contour_error helper. It uses 4096 points on the analytic
periodic curve and reports an upper approximation, not exact geometric distance;
maximum distance overestimate bound 0.1372 mm
from the analytic derivative norm bound and phase-grid spacing. Tests cover
known off-plane error and discretization bound; 3 metrics tests passed.
JSON/CSV saved in the session directory. Prior note that Lissajous contour was
unimplemented is superseded. One completed Lissajous run establishes initial
feasibility; repeat unchanged before treating performance as repeatable.
No hardware setting/gain/native binary was changed during this audit.


## Lissajous repeat completed; stationary EE posture prepared

`tracking.20261012T144855Z` completed: no native fault, smooth stop, zero drops.
Track RMS position 6.686 mm,
sampled contour 3.909 mm,
orientation 1.714 degrees.
Two full Lissajous sessions now completed at translation K800 / rotation K40.

User requested stationary EE posture. Approved config `config/m4.posture.approved.yaml`:
same Cartesian K800/D56.5685 and K40/D13, existing workspace and guards,
original M3 kinematic torque projector, posture K2/D1 Nm/rad and Nm*s/rad.
Sequence: initial measured-pose hold 2 s, A transition 6 s/settle 4 s,
B transition 6 s/settle 4 s, return posture-reference transition 6 s/settle 4 s.
Total 32 s. Targets are generated from the actual start Jacobian null vector
with +/-0.60 rad seeds and DLS-corrected to the same full measured flange pose.
Do not replay the bundle's home-dependent target values. Native C++ handles
quintic posture transitions at servo rate; Python only selects endpoint goals.
The original absolute posture target route remains supported. Runtime saves
posture-plan.json after the live start recheck.

Offline endpoint IK, bounds, endpoint conditioning and desired quintic posture
reference velocity checks passed using last measured stop state, recorded in
config/m4.posture.design-audit.json. These checks do not certify swept-arm
collision clearance or predict actual nullspace dynamics. No approach/homing,
Cartesian trajectory, external force injection or return-to-home motion;
fixed EE target is the current measured pose. Posture return is a target, not
a guarantee of exact measured joint return with compliant K2. Preflight and
servo guards remain active. No posture motion was executed by the agent.
Summarizer now splits A/B/return transition and settle phases and reports
joint travel, bounds margin and measured joint-to-posture-reference error.


## First posture run completed, insufficient measured posture motion

`posture.20261012T145424Z` completed its 32-second lifecycle with native fault 0,
smooth stop and no telemetry drops. This does not establish successful posture
reconfiguration: J1 peak measured displacement 0.642 degrees versus requested
24.45 degrees; J3 0.216 versus 15.48 degrees; J7 0.384 versus 13.51 degrees.
The live posture goals and native quintic reference changed correctly through
A/B/return. Logged projected nullspace torque was nonzero but small: peak
J1 0.865 Nm, J3 0.561 Nm, J5 0.394 Nm, J7 0.468 Nm. At A/B settle the
reference error stayed around 0.59-0.60 rad norm while measured motion stalled.
Posture K2/D1 is too weak for a visible reconfiguration in this trial; residual
friction/model effects are plausible, not uniquely identified. This is not a
missing target-publication problem. No joint velocity/torque/pose guard tripped.
Maximum fixed-EE position error 1.974 mm;
maximum orientation error 0.188 degrees.

Suggested next comparison: posture K10/Dsqrt(5)=2.236 Nm/rad and Nm*s/rad,
keeping current Cartesian gains, kinematic projector, targets, six-second
transitions, timing and guards unchanged. This scales only the nullspace gains,
not task-space tracking gains. Not applied or executed by this audit. Metrics
JSON/CSV saved; measure actual posture excursion and fixed-EE drift, not just
session completion.


## Active posture comparison: stronger gains and dynamic projector

User authorized `next increase KD and use the dynamic projector`.
`config/m4.posture.approved.yaml` now uses posture K10 Nm/rad and Dsqrt(5)
=2.23606798 Nm*s/rad, with full-six-row dynamically consistent torque projector
N_tau = I - J.T pinv(J M^-1 J.T) J M^-1 (relative cutoff 1e-6, native default).
Cartesian K800/K40, D56.5685/D13, live-start IK target generation, +/-0.60 rad
null seeds, 6-second transitions/4-second settles, 32-second A/B/return sequence,
workspace, all guards and torque/stop pipeline are unchanged. This intentionally
changes both posture gain and projector, not a single-variable ablation of the
original simulation M3 kinematic profile. Bundle remains immutable.

Adapter config validation now allows explicit kinematic/dynamic choices for
posture only; other tasks continue to require dynamic. Existing installed C++
controller already supports this branch: no Franky wheel or dependency rebuild.
Original weak K2 kinematic executed config remains at
posture.20261012T145424Z/config.yaml. Offline config/bundle validation passed;
no FCI connection or motion performed by this update. Dynamically consistent
projection removes ideal task-acceleration contribution of secondary torque
with an exact full-rank model, not a guarantee of zero measured EE drift.


## Dynamic K10 posture result; 0.90 rad excursion prepared

`posture.20261012T150157Z` completed with native fault 0, smooth stop and
zero dropped samples. Peak measured excursions from the start: J1 18.01 deg,
J3 11.98 deg, J5 7.16 deg, J7 8.27 deg. Fixed-EE maximum position/orientation
error 5.163 mm / 1.947 deg. Peak projected nullspace torque J1 1.828 Nm,
J3 1.809 Nm. Unlike K2/kinematic trial this produced visible reconfiguration;
both projector and gains changed, so it is not a single-variable attribution.
Metrics JSON/CSV saved in the session directory.

User requested larger excursions. Active `config/m4.posture.approved.yaml`
nullspace seed increases 0.60 to 0.90 rad; IK-corrected targets regenerated
from each live start. Offline check from previous stop pose passed endpoint
IK, interior joint bounds, endpoint conditioning and desired native quintic
posture reference velocity. Requested J1 excursion ~36.15 deg, J3 ~22.82 deg,
J7 ~20.33 deg from that recorded stop, not a promise of measured tracking.
Posture K10/D2.236, full-6D dynamic torque projector, Cartesian gains,
six-second transitions/four-second settling, 32-second sequence, workspace,
all servo/stop guards and native dependencies unchanged. Larger seed is a local
hardware comparison, not a modification to the immutable simulation bundle.
Removed adapter's fixed 0.6 rad catalog ceiling; positive finite seeds must
still pass measured-start IK/bounds/conditioning/reference-velocity checks.
Swept-arm collision clearance is not established by those endpoint checks.
No new FCI connection/motion executed by the agent. Design receipt updated at
config/m4.posture.design-audit.json; previous actual config preserved in session.


## Larger dynamic posture excursion completed (2026-10-12 15:05 UTC)

`posture.20261012T150500Z` completed with native fault 0, zero dropped samples
and successful smooth stop. Executed seed 0.90 rad, posture K10/D2.236,
dynamic projector, Cartesian K800/K40 and D56.5685/D13; timing unchanged.
Measured peak displacement from initial joints: J1 29.45 deg, J3 19.34 deg,
J5 12.62 deg, J7 14.34 deg. Full A-to-B measured peak-to-peak ranges:
J1 56.49 deg, J3 36.83 deg, J5 24.51 deg, J7 27.68 deg.
Fixed-EE error over active phases: position RMS 3.506 mm, maximum 5.338 mm;
orientation RMS 1.805 deg, maximum 2.211 deg. Previous 0.60 rad trial
maximum errors were 5.163 mm / 1.947 deg: larger visible reconfiguration
with a modest increase in EE drift. Starts differ and targets are regenerated,
so this is an operational comparison, not a controlled identical-start study.

Return is compliant rather than exact homing: final measured J1/J3/J7 offsets
from initial were +6.10/-3.46/+4.71 deg; return-settle posture error norm
was approximately 0.163 rad. Do not equate session completion with exact
posture tracking. Current profile is a useful visible stationary-EE demo
candidate; next recommendation is repeat unchanged to check reproducibility.
No config, native controller or dependencies changed during this audit.
Raw data, executed config, actual posture plan, and generated `m4-metrics.json`
/ `m4-metrics.csv` are in
`/home/chen-lab/franka_ros2_ws/hardware_inventory/2026-10-12/m4/posture.20261012T150500Z/`.


## First Z-axis fixture smoke (prepared 2026-10-12)

User requested curve-constrained Cartesian impedance, starting with Z axis.
`config/m4.z_line.approved.yaml` runs 30 seconds of native Z-line impedance,
anchored to the flange position at the actual measured start, not a catalog
home. Added an opt-in `task.fixture_anchor_to_start` for Z line only; Python
sets the native fixture origin once before starting. Native wheel/dependencies
and torque/stop processing unchanged. Orientation is held at measured start
with K40/D13; posture gains zero. Fixture translation replaces Cartesian
translation PD: normal K100 N/m, normal D20 N*s/m, tangent D20 N*s/m, as in
the transferred fixture suite. X/Y are elastically constrained; Z has damping
but no restoring stiffness. No automatic push, homing or path traversal.

Gentle manual guidance along base-frame Z, initially within +/-2 cm. The
native fixture projects the position reference onto the line every tick: the
3 cm position-error guard limits perpendicular distance, not Z displacement.
Workspace and speed limits still bound permitted Z motion. Contact abort
is disabled for hand guidance; native collision, external torque, workspace,
speed, singularity and all other guards remain enabled. Start pose must remain
inside the existing workspace. `start.json` records actual line origin and
orientation; executed configuration is preserved with the servo log.
Offline config/bundle verification and native fixture construction passed;
Adapter tests: 22 passed, 1 skipped in system Python (native module unavailable
in that interpreter); native fixture construction checked separately in the
hardware venv. No robot connection or motion by this update.

Run:
```bash
/home/chen-lab/yifan/venvs/franky-m4-baseline-server10/bin/python \
  /home/chen-lab/yifan/franka-rl/deployment/franky_runtime/run_m4_hardware.py \
  --config /home/chen-lab/yifan/franka-rl/deployment/franky_runtime/config/m4.z_line.approved.yaml \
  --execute
```
Type `start`; type `stop` or Ctrl-C for native smooth stop.


## Z-line first-tick orientation fault corrected

`z_line.20261012T151511Z` stopped at first tick with fault 8 / pose_error.
Position error was only 1.26 micrometres; orientation error 3.14158039 rad.
Native fixture overrides reference orientation using CurveFixtureParams; its
default identity differed from the measured near-180-degree flange orientation.
Python had anchored origin only. Fixed measured-start Z-line setup to anchor
both origin and orientation in `make_curve_fixture`, preserving xyzw order.
No gain/guard/dependency changes or native rebuild; same approved YAML/command.
Successful smooth stop, no dropped samples; no valid active tracking interval
in this failed session. Added regression test for nonidentity measured start
and explicit fixed-orientation behavior: 23 passed, 1 native-module skip in
system Python. Native fixture constructed using recorded start in hardware
venv and config verified without FCI connection. Recommend gentle initial
+/-2 cm guidance; this is an operator test recommendation, not a Z-distance
guard. Position guard is distance from projected curve, workspace still bounds Z.


## Z-line speed fault audit (15:17 UTC)

`z_line.20261012T151711Z` passed the corrected first-tick orientation setup
but faulted at elapsed 2.144 s on measured linear speed 0.1001863 m/s, just
over the unchanged 0.100 m/s bound. Angular speed 0.04535 rad/s was below
0.5 rad/s. At fault, distance from projected line was 3.071 mm, within
30 mm; active Z displacement approximately 21.69 mm. Zero dropped samples.
Metrics JSON/CSV generated in session.

Smooth stop was NOT confirmed: native stop reached its 2.0-second deadline
with stop_finished=true, stop_at_rest=false; final J2 velocity 0.20058 rad/s
versus 0.02 rad/s rest threshold. Initial fault remains latched as speed fault9.
Logs do not establish whether operator force persisted during stopping; do
not attribute this uniquely to hand guidance or assume stop-at-rest success.
Before further tuning, clarify external guidance during the fault/stop and
verify robot is stationary. No speed, damping, stop or collision thresholds
changed by this audit. Suggested same-limit retry only with slower guidance
and release of guidance on stop; investigate native damping stop if motion
persists after release. Agent did not connect to or move hardware.


## Operator feedback and stop-phase diagnosis for Z-line 15:17

Operator reports hard locking after fault, possible continued force after lock,
and apparent large orientation change with little restoring feel. Separating
phases: active orientation error maximum was only 1.1154 degrees, not the
38.5242-degree end-of-stop error. During the two-second stop the flange moved
additional [-35.81,-4.43,-248.29] mm, with peak linear speed 0.34362 m/s.
This is substantial stop-phase motion; do not commission further guidance
by simply raising the speed guard. Operator force contribution remains unknown.

Native `src/motion/stop_motion.cpp` replaces task stiffness with joint damping
and optional Coriolis, ramps from prior command over 0.2 s, and calls
MotionFinished at max_duration even when not at rest. Current joint damping
[2,2,2,1,1,1,0.5] produces only ~0.40 Nm on J2 at 0.20 rad/s. Native runtime
stops enforcing ordinary task guards once stopping; latched original fault 9
also masks subsequent stop-timeout fault code18. Python reports smooth_stop_ok
false correctly. Locking after exit may be robot behavior, not proven by these
logs; native damping stop has no position-locking stiffness.

Active fixture stiffness was weak by design: at 3.07 mm normal displacement,
normal spring force is only ~0.307 N; Z has no spring. Rotational K40 gave
~0.78 Nm spring wrench at 1.12 degrees, plus damping. Joint task commands were
nonzero (peak J2/J4/J6 ~2.11/2.28/1.49 Nm), so the controller was not absent.
Next engineering priority: commission a stop suitable for hand guidance, with
adequate braking/pose retention and explicit deadline failure reporting, before
raising fixture gains or speed thresholds. Do not change firmware/libfranka.
No new config, motion or native build performed by this diagnosis.


## M4 stop v2 implementation (2026-10-12)

The user requested correction of the Z-line stopping failure. M4-only native
stop now captures measured encoder/FK pose and current joints when stopping
begins. It ignores subsequent publisher references and replaces the fixture's
free tangent with full-six-axis Cartesian hold using existing controller gains
(K800 translation/K40 rotation for this Z-line config), zero desired twist,
inherited Coriolis/joint-limit processing and additional configured joint damping.
A smoothstep ramp over the existing 0.2 seconds blends from actual prior
filtered tau_J_d into hold torque, then applies command bounds and the existing
1 Nm/tick slew; libfranka still filters at 100 Hz. No gravity added by client.
Regular guidance gains and immutable simulation bundle remain unchanged.

Normal MotionFinished requires all joints below 0.02 rad/s continuously for
100 ms after the ramp. Two-second deadline without sustained rest is stop
fault18 and a native control exception, not MotionFinished while moving.
Stop position/orientation containment uses 3 cm / 0.2 rad about captured stop
pose; joint and workspace bounds continue to apply. Model/nonfinite failure,
collision/robot errors or containment failure abort control explicitly.
A native abort may result in a stiff robot; it is not a compliant guidance
mode, and firmware behavior after FCI exits is not controlled by this code.
Pose hold under continued external force is not an assured collision-free stop.

Separate stop_fault preserves evidence even when original fault9 remains
latched. Python catches join exceptions, drains final telemetry and reports
stop_evidence/stop_fault without labeling an abort a smooth stop. Stop logs
include extra joint damping and identify raw torque as pre-transition hold
output. Standalone Franky TorqueStopMotion and existing velocity controller
are unchanged; only M4RuntimeMotion has this stop policy.

New isolated runtime: `~/yifan/venvs/franky-m4-stop-server10`, Franky
`2.0.1.dev64+gm4stop.libfranka.0.21.2`. All 21 native dependency files must
match baseline byte-for-byte; tested baseline environment is retained.
Z-line approved config now requires stop_behavior_version2 and the new
Franky version. `config/m4.stop-check.approved.yaml` is a five-second
measured-pose hold with the same Cartesian gains, no trajectory/homing or
hand guidance, for first stationary commissioning of the new stop.
Old circle/posture configs remain tied to their original runtime.

Native deterministic tests passed 18 cases/62 assertions, including former
fixture tangent restoring force, orientation restoring force, opposing braking
force, bounds/slew, low-speed dwell with reset, moving deadline and containment.
These tests do not prove physical braking or human-contact performance.
No robot access or motion performed during this fix. Verify stationary stop
first, then limited gentle Z guidance with release of hand force when stopping.

Final stop-v2 verification: 28 Python tests passed in the new runtime (no
skips), 18 native cases/62 assertions passed; installed dependency hashes
match all 21 baseline files. Wheel, SHA256SUMS, source/dependency receipt and
offline-validation.json are at
`~/yifan/deployment_bundles/2026-10-12/franky-m4-stop-v2/`.
Stationary commissioning command:
```bash
/home/chen-lab/yifan/venvs/franky-m4-stop-server10/bin/python \
 /home/chen-lab/yifan/franka-rl/deployment/franky_runtime/run_m4_hardware.py \
 --config /home/chen-lab/yifan/franka-rl/deployment/franky_runtime/config/m4.stop-check.approved.yaml \
 --execute
```
Type start, leave the arm untouched for this five-second check. Verify
smooth_stop_ok=true and stop_fault=0 before returning to guidance.


## Stop-v2 stationary hardware check passed (15:44 UTC)

`hold.20261012T154417Z` completed with smooth_stop_ok=true,
stop_behavior_version2, stop_fault0, native_fault0, zero dropped samples and
joined=true. Active measured-pose hold maximum position/orientation error
0.966 mm / 0.135 deg. Stop lasted 0.301 s (0.2-second ramp plus 100 ms
sustained rest); maximum drift from captured stop pose 0.267 mm / 0.0383 deg.
Final peak absolute joint velocity 0.004151 rad/s, below 0.02 threshold.
Metrics JSON/CSV saved in session. This validates stationary stopping, not
braking from a moving hand-guided state or resistance to continued forcing.

Next: same Z-line config in `franky-m4-stop-server10`, gentle short guidance,
release hand force when requesting stop; audit moving-stop displacement before
raising gains, speed or guidance range. No config or dependency changes by
this audit and no motion executed by the agent.


## Stop-v2 moving Z-line check passed; speed guard tripped (15:48 UTC)

`z_line.20261012T154821Z` faulted at 2.270 s on measured linear speed
0.1001625 m/s versus 0.100 m/s threshold. Angular speed 0.02977 rad/s
was below 0.5 rad/s. Active line error maximum 3.754 mm; orientation error
maximum 0.919 deg; active Z span 24.20 mm.

Corrected stop completed successfully from ~0.100 m/s: smooth_stop_ok=true,
stop_fault0, joined=true; original native fault9 preserved, zero drops.
Stop duration 0.301 s, maximum drift from captured stop pose 13.302 mm,
maximum orientation drift 0.0829 deg. Peak stop linear speed 0.10277 m/s,
final 0.0008524 m/s; final maximum joint speed 0.002199 rad/s. Compared
with prior damping-only failed stop's additional 248 mm downward motion
and 38.5-degree orientation error, this is a materially improved observed
stop in this one guided trial; not an exhaustive braking validation.

The 0.10 m/s guidance guard remains unchanged. For easier manual speed
control, a future user-authorized comparison could raise tangent damping
20 to 40 N*s/m (doubling resistance at a given Z velocity), before raising
allowed speed. No configuration/gain/limit/dependency changes made by this
audit; metrics saved in session. Agent did not access or move the robot.


## Z-line guidance damping and speed increased by user request

After 15:48 trial, operator reports barely touching the robot before speed
guard. User explicitly requested raising both damping and speed limit. Active
`config/m4.z_line.approved.yaml` now has tangent damping40 N*s/m (was20),
linear speed limit0.15 m/s (was0.10). Normal K100/D20, rotational K40/D13,
stop-v2 Cartesian hold gains, all joint/torque/workspace/orientation limits,
duration and other parameters unchanged. This is a two-variable guidance
comparison, not a controller identification ablation. Existing session configs
are preserved. Stop performance above ~0.10 m/s not hardware-validated yet.
Config/bundle/dependency and native fixture validation passed offline. No
robot access/motion, new wheel or dependency update. Same stop-v2 interpreter
and Z-line execution command. Evaluate drift/resistance and stop displacement
in next log; operator observation alone does not identify unforced drift.


## Z-line normal compliance audit (15:54 UTC)

Operator guided downward and sideways, reporting weak sideways impedance.
`z_line.20261012T155441Z` faulted at 9.705 s on distance from original Z
line 30.0067 mm, just over 30 mm guard. Original orientation error 2.3566 deg
was below 0.2 rad. K100 spring force magnitude ~3.001 N; including D20
normal velocity damping gives predicted normal force [0.3061,-3.4753,0] N.
These are controller-law predictions, not measured Cartesian forces.
Thus gentle hand force can plausibly overcome this soft normal spring.

First fault sample reference is overwritten by captured stop pose on this
tick; naive pose-minus-reference then yields zero fault error. Original
fixture geometry reconstructed from first active reference, and orientation
from prior active reference. `fixture-audit.json` records reconstructed
trigger and force predictions alongside m4-metrics.json/csv.
Stop-v2 succeeded: smooth_stop_ok=true, stop_fault0, no dropped samples;
stop duration0.688 s, maximum captured-stop pose drift12.399 mm and
orientation1.984 deg. No config changes or hardware access by audit.

Recommended next normal profile K500 N/m, D45 N*s/m, preserving tangent
D40 and linear speed0.15 m/s, rotational gains, stop and all limits.
At 1 cm offset spring force would be5 N versus current1 N. This is a
proposed comparison, not applied; stronger normal constraint does not
require widening the 3 cm guard.


## K500/D45 normal-constraint trial prepared

User authorized implementing the proposed next trial. Updated existing
`config/m4.z_line.approved.yaml` normal stiffness100->500 N/m and
normal damping20->45 N*s/m. Tangent damping40, linear speed limit0.15 m/s,
rotation gains, 30 mm line-distance guard, other bounds and stop-v2 unchanged.
Offline approved-config validation, actual native fixture/stop construction,
Franky version, immutable bundle and dependency checks passed. Receipt saved
to `config/m4.z_line.design-audit.json`. Existing executed configs remain
in session directories. No FCI access, motion, wheel build or dependency change.

Execute with `~/yifan/venvs/franky-m4-stop-server10/bin/python` and
`run_m4_hardware.py --config config/m4.z_line.approved.yaml --execute`.
Start gently, observe initial untouched behavior and then small sideways
displacements; stronger spring produces5 N at1 cm normal offset. Release
guidance force on stop. Audit normal deflection, orientation and stop behavior.


## Hand-guidance joint-speed guards relaxed after 16:00 fault

`z_line.20261012T160013Z` fault6 at8.828 s: J2 measured0.435958 rad/s
versus0.435 guard; Cartesian linear speed approximately0.096 m/s below0.15.
User explicitly requested relaxing velocity limits for hand guidance. Active
Z-line config now uses30% of existing model maxima (previous20%):
J1-J4 0.6525 rad/s, J5-J7 0.783 rad/s. Normal K500/D45, tangent D40,
Cartesian linear0.15/angular0.5, torque/joint-position/workspace/error guards
and stop-v2 settings unchanged. Native config/bundle/dependency validation
passed offline; updated design receipt. No robot access or motion by agent.

This session also failed stopping: stop_fault18, native timeout exception,
smooth_stop_ok=false; original fault6 retained, zero dropped samples.
Stop lasted2 s, maximum captured-pose drift27.818 mm. At deadline J6 speed
0.1051 rad/s, J4 -0.06033, J2 0.04849; Cartesian speed0.03398 m/s.
Final estimated external torque includes J2 -4.648/J4 6.392 Nm; this does
not alone prove continued hand force. Asked operator whether force persisted
during stopping; unresolved pending reply. Speed guard relaxation does not
resolve this stop failure. No stop damping/deadline changed by this trial
preparation. Metrics generated in session directory.


## Operator clarification: force continued during 16:00 stopping

Operator confirmed continued hand force during the two seconds after the
joint_velocity fault in z_line.20261012T160013Z. This resolves the missing
operator context: timeout occurred under continued forcing, not a confirmed
released-arm braking test. It does not establish that continued force was the
sole cause or validate stop performance at the raised joint limits.
Keep prepared30% joint speed guards and other current settings; no further
stop/gain/timeout changes from this clarification. Next trial should release
hand force when requesting stop or when a fault occurs, then audit sustained
rest and smooth_stop_ok. Stop-v2 requires100 ms continuously below0.02 rad/s
and explicitly aborts at2 s if that criterion is not met.


## Stronger K800/D57 normal-guidance trial prepared after16:08 run

`z_line.20261012T160859Z` fault8 at10.399 s: normal distance30.02185 mm,
orientation2.1988 deg. Original line geometry reconstructed because first
fault reference becomes captured stop pose. K500 predicted normal spring
magnitude15.0109 N at this displacement. Stop timed out at2 s with
stop_fault18, smooth_stop_ok=false, max stop drift24.819 mm, finalJ2
velocity0.03984 rad/s; no samples dropped. Operator confirmed force
continued during stopping; not a confirmed released-arm stop test.

User requested larger stiffness for stronger constraint. Active Z-line config
normal K500->800 N/m and D45->57 N*s/m. D scaled approximately with sqrt(K);
no claim of physical critical damping. Spring force8 N per1 cm normal error.
Tangent D40, Cartesian speed0.15 m/s, joint speed30% model maxima, 30mm
normal guard, torque/workspace/orientation bounds and native stop-v2 unchanged.
Existing full Cartesian stop K800/D56.5685 unchanged. No wheel/dependency
updates, FCI connection or agent motion. Offline approved-config/native
constructor/bundle/dependency checks passed; design receipt updated.
Same execution command with franky-m4-stop-server10. Release hand force
when stopping or upon fault to measure unforced settling in next run.


## K800/D57 guidance feel accepted; gentle repeat before circle

User reports robot behavior is now ideal in `z_line.20261012T161419Z`,
and plans gentle repeat followed by circular constraint. Run fault9 at24.948 s
on measured linear speed0.150863 m/s, just above0.15; angular0.01911 rad/s.
Active line error max26.510 mm; orientation max2.247 deg. Native stop-v2
succeeded: smooth_stop_ok=true, stop_fault0, joined=true, no dropped samples.
Stop lasted1.003 s, maximum captured-pose drift20.298 mm and orientation
1.195 deg, final linear speed0.0001043 m/s. Metrics JSON/CSV saved.

Record K800/D57 normal with tangent D40 as operator-preferred Z-line feel,
not a complete task acceptance since speed guard tripped. Configuration
unchanged: linear0.15 m/s, joint guards30% model maximum, all other limits
retained. Await gentle repeat audit before circular-constraint commissioning;
no circular configuration generated/activated by this audit. No robot access,
agent motion, gain/guard, wheel or dependency changes.


## Hand-guided YZ-circle fixture prepared

User requested circular constraint after accepting Z-line K800/D57 feel.
New `config/m4.yz_circle.approved.yaml`: radius0.05 m (10 cm diameter),
30-second hand guidance, normal K800/D57 for out-of-plane X and radial YZ
error, tangent D40, measured-start orientation K40/D13. Cartesian0.15 m/s,
joint speed30% model max, all other guards and installed stop-v2 unchanged.
Circle center is measured start minus[0,0,radius], so start is top of circle.
No automatic approach, homing, push or trajectory traversal.

Python measured-start fixture anchoring extended to YZCircle; native C++
fixture already supports it, so no wheel/libfranka/Ruckig changes. Live plan
checks circle plus30mm error margin inside workspace and65 orientation-
preserving sampled IK points/joint bounds with minimum Jacobian sigma.
Plan regenerated before prompt and after start; actual geometry saved to
fixture-plan.json. --preflight also checks and saves circle plan without motion.
These checks do not certify full swept-link collision clearance.

Offline design checks passed at prior recorded home-like start (minimum sigma
0.22053) and last recorded Z-line stop (0.17385), and native constructor and
bundle/dependency checks passed. Receipt config/m4.yz_circle.design-audit.json.
Python tests30 passed in stop-v2 venv, including start-on-circle orientation
and rejection of workspace overrun; no robot access/motion performed.

Run:
```bash
/home/chen-lab/yifan/venvs/franky-m4-stop-server10/bin/python \
 /home/chen-lab/yifan/franka-rl/deployment/franky_runtime/run_m4_hardware.py \
 --config /home/chen-lab/yifan/franka-rl/deployment/franky_runtime/config/m4.yz_circle.approved.yaml \
 --execute
```
Type start, begin with gentle small arcs in base-frame YZ. Release hand
force on stop/fault. Change fixture.radius for another size; live workspace
and IK checks rerun. Existing Z-line config retained unchanged. Output
folder kind is yz_circle under existing hardware_inventory/2026-10-12/m4.


## Circle radius changed to15 cm; recorded-pose workspace checks reject

User requested radius15 cm. Active m4.yz_circle.approved.yaml fixture.radius
updated0.05->0.15m (30cm diameter), approval records updated. All gains,
guards, workspace bounds, stop and measured-start placement unchanged.
At both recorded start and recorded stop from Z-line16:14 run, circle plus
30mm normal-error margin fails workspace check; design receipt replaced
with explicit rejection results. No FCI access or motion.

Current Y bounds +/-0.18m: radius0.15+margin0.03 leaves no allowance for
nonzero centerY, while measured-start placement uses actual startY. Current
Z lower0.17: top-anchored circle requires startZ>=0.50m to accommodate
30cm downward span plus3cm lower margin. Recorded stopZ0.4325m is too low.
Do not claim15cm geometry is ready or bypass live checks. Before running
this size, revise placement (e.g. exact centered Y with on-circle start) and
check full-circle IK at a sufficiently high start, or explicitly review
workspace changes. No workspace/joint guards automatically relaxed.


## Circle radius10 cm selected; higher recorded start passes

User proposed10cm instead of15cm. Active fixture.radius now0.10m (20cm
diameter), center10cm below measured starting pose. All gains, guards and
stop unchanged. Offline circle-plus30mm workspace margin and65 sampled
full-pose IK checks passed at recorded home-like startZ0.5901356m:
minimum Jacobian sigma0.18116, above0.02 guard. Native fixture constructor
and bundle/dependency verification passed; receipt updated.

At recorded lower stopZ0.4325133m, workspace check passes but full-circle
IK fails convergence (position3.061mm, orientation0.005765rad residual).
Do not claim radius10cm valid from any start. Use tested home-like start
and retain live workspace/IK checks; no automatic homing or repositioning.
No FCI access or robot motion executed. Same circle command with stop-v2
interpreter and m4.yz_circle.approved.yaml.


## 10 cm YZ circle completed: 20261012T163205Z

User executed `hardware_inventory/2026-10-12/m4/yz_circle.20261012T163205Z`.
Report: complete, native fault0, stop fault0, smooth stop true, joined true,
zero dropped samples. Active duration29.998s, 29,999 active samples;
30,504 total records. Same normal K800/D57, tangent D40, rotation K40/D13,
dynamic projector and stop-v2. Live 65-point IK passed, min sigma0.181159.

Encoder/model-FK geometric normal distance to 0.10m circle: RMS7.428mm,
maximum13.688mm (30mm guard). Maximum X out-of-plane9.140mm and radial
error11.879mm. Orientation RMS2.413deg, maximum4.018deg (0.2rad guard).
Unwrapped angular span408.301deg, net travel-367.092deg: more than one
full revolution with some reversal; this was manually guided, not timed
path tracking. Max linear speed0.074514m/s, angular0.098117rad/s, joint
speed0.163795rad/s. Minimum joint soft-bound margin0.099380rad.
Command success min0.96, mean0.994887 active; max active reference age
24.816ms. No velocity/error/communication guard triggered.

Stop settled in0.504s: maximum captured-pose drift12.804mm/1.147deg;
final max joint speed0.002367rad/s and linear speed0.001079m/s. Positive
first completed circle-fixture demonstration; keep current parameters for
repeatability testing rather than automatically increasing gains/limits.
All errors are encoder/model FK, not external metrology or calibrated
force/stiffness measurements. One run does not establish repeatability.

Offline audit outputs in the session folder: `m4-metrics.json`,
`m4-metrics.csv`, `fixture-audit.json`. Generic summarizer's circle-contour
field is null for fixture tasks; fixture-audit explicitly computes analytic
normal distance and unwrapped angular coverage from fixture-plan.json.
No FCI access or robot motion performed by the auditing agent.


## Repeated circle and residual-error investigation: 20261012T163526Z

Repeated same approved config/hash e1157ae8ad5e827641f5ae2237d24085532eb729b041bd1934d03a059abce0c5.
Completed, native/stop faults0, smooth stop true, joined true, dropped0.
Active circle normal error RMS5.770mm/max9.580mm (first run7.428/13.688mm),
orientation RMS2.199deg/max3.780deg, max speed0.064311m/s.
Angular span386.376deg/net-385.085deg. Stop completed in0.302s.

User reports hand released at end but residual offset did not correct.
Logs independently show motion nearly stationary over final5s: normal
error approximately4.79mm with mean linear speed below1mm/s. Last active
sample error4.789mm, commanded translational spring magnitude3.831N,
vector[-3.554,0.606,-1.294]N. Task torque still nonzero (J2~-1.27Nm).
Raw and slewed commands agree within1.4e-17Nm; this is not slew saturation.
No command-torque guard fault, nullspace and joint-limit terms zero at end.

Native active fixture law F=800*(project(p)-p)-57*v_normal-40*v_tangent;
torque J^T*wrench plus Coriolis (gravity remains firmware). It is a soft
spring, not a geometric lock. Under hand force,5mm offset gives4N and10mm
8N spring force. Damping affects moving response; cannot remove a static
error because velocity then nearly zero. No integral correction; Cartesian
friction feedforward remains disabled defaults in current hardware runner.
Static joint friction and gravity/load/model bias are plausible causes of
residual balance, but no measured release event or calibrated force sensor
establishes which dominates. Do not equate external torque estimates to a
calibrated hand-force measurement or claim missing gravity compensation.

Separate stop behavior: M4RuntimeMotion captures measured pose at30s and
switches from fixture to full6D pose hold there. It deliberately preserves
current off-curve position instead of snapping to its projection. This does
not explain the preceding5s plateau, but explains absence of correction
while stopping/after session complete. There is no tangential spring to
return to the start point while guiding.

Recommendation: preserve this tested stop behavior. For tighter active
normal guidance, compare a modest normal-stiffness increase with an explicit
hands-off interval before timed stop, recording the release time. Changing
damping alone will not solve static offset. Do not add uncalibrated friction
feedforward or integral action automatically. No controller/config/library
changes or hardware execution during this audit. Saved m4-metrics.json/.csv
and fixture-audit.json in run directory.


## Next circle comparison: normal K1200, normal D69.810, tangent D20

User requested higher impedance with lower tangent damping. Updated active
`config/m4.yz_circle.approved.yaml`: normal stiffness800->1200N/m,
normal damping57->69.810Ns/m (sqrt(1.5) scaling preserves previous nominal
ratio), tangent damping40->20Ns/m. Radius0.10m, duration30s, rotation
K40/D13, full6D stop-hold K800/rotation40 and associated damping, stop-v2,
all guards, stack and workspace unchanged. Fixture spring at5mm error now6N
versus4N; ideal static error under the same disturbance would shrink to2/3,
but friction/model effects make this a hypothesis, not a guarantee.
Tangential resistance at0.05m/s now1N versus2N; guide gently because the
same hand force can produce more tangential speed. Existing speed guards
remain active. Aim to release hand force at25s for5s active correction
before measured-pose capture at30s. No automatic correction after stop.

Approved config/schema and immutable transfer verified offline; native
fixture gain properties accepted; controller/stop/robot/guard dictionaries
checked identical against completed circle163526Z. No FCI access or motion.
Same execute command and stop-server10 interpreter. Existing session config
copies preserve the previous K800/D57/tangent40 experiments.


## Higher normal impedance / lower tangent damping completed: 20261012T164958Z

User-run circle radius0.10m, normal K1200/D69.81, tangent D20,
rotation and stop-hold gains unchanged; executed config SHA256
ac79addb84054c561f3030263a4dc0789df9a3ff2d90a9f244ec997a8a61628c.
Complete, native/stop faults0, joined, smooth stop true, drops0.
Active29.997s, angular span385.308deg/net-382.519deg.

Normal error RMS4.328mm/max9.116mm, versus prior repeat K800/D57/tangent40
5.770/9.580mm (RMS25.0% lower). Orientation RMS2.082deg/max3.416deg
versus2.199/3.780deg. Max Cartesian speed0.070241m/s, max joint speed
0.157668rad/s, minimum soft joint margin0.096642rad; min active command
success0.97, max reference age20.053ms. Raw/slew agree within7e-18Nm,
so no torque slew saturation. Max radial error7.515mm, X-plane error6.266mm.

Last active normal offset3.209mm versus4.789mm previously (~33% lower);
corresponding spring force3.851N vs3.831N, consistent with similar residual
load/friction and1.5x stiffness, but not proof of its physical source.
Operator release timestamp not recorded. Motion persisted after25s:
25-27s mean speed45.55mm/s;27-29s20.82mm/s;29-30s4.39mm/s.
Cannot call this a controlled5s hands-off static-settling comparison.
Last-second mean error3.451mm, down from6.282mm at25-27s.

Stop duration0.302s, max captured-pose drift0.565mm/0.0812deg,
final max joint speed0.005700rad/s. Improved run supports repeating this
profile, without further gain/guard changes. Differences in manual force,
trajectory/speed and two changed gain parameters prevent causal attribution
of RMS improvement to stiffness alone. Not external pose metrology.
Saved m4-metrics.json/.csv and fixture-audit.json in run folder. Config,
controller and dependencies unchanged by audit; no hardware execution.


## Compensation clarification: firmware motor friction versus client feedforward

Official FCI libfranka documentation states external torque commands receive
both gravity and motor-friction compensation on robot Control:
https://support.franka.de/docs/libfranka.html
Thus earlier statements that only gravity and Coriolis compensate this
stack were incomplete. Zero Franky friction coefficients disable only the
additional client friction feedforward, not robot-side motor-friction
compensation. Installed libfranka0.21.2 public Model/RobotState APIs expose
no calibrated friction coefficient table or friction-model query. Do not
assume full joint/transmission stiction is removed or infer its calibration
from tau_ext_hat_filtered. Residual-friction hypotheses remain unproven.

Franky acceleration feedforward is conditional on target_acceleration being
provided: unset means no Lambda*a_ref term, with no implicit acceleration
estimate from velocity references. Runner currently publishes pose/twist
only. Tracking trajectories could supply analytic acceleration in a future
explicitly implemented comparison. Hand-guided fixtures have no timed
acceleration reference. Additional Franky friction model is user-supplied
Coulomb*tanh(dq/epsilon)+viscous*dq, clipped per joint; zero at zero velocity
so not a static integral compensator. Any identification should estimate
residual behavior with firmware compensation active, rather than add a full
friction model and double-compensate. No runtime/config changes.


## Next work: matched controller comparisons

Operator reports all demo tasks successfully deployed and requests comparisons
and a brief report. Implementation plan:
`deployment/M4_CONTROLLER_COMPARISON_PLAN.md`.

First reuse current Cartesian controller for independent stiffness/damping
sweeps, add indexed multi-run metrics/reporting, then add native resolved-rate
DLS + explicit joint PD using Franky's existing joint impedance law. Local
Franky source audit found Jacobian and joint tracking APIs but no exposed
ready-made DLS-to-joint-PD controller; CartesianMotion outputs FCI Cartesian
commands, not that explicit baseline. Preserve tested dependencies below
Franky and captured-pose stop-v2; build an isolated additive comparison wheel.

Compare identical timed circle references (not hand-guided fixture geometry),
and identical feasible A/B stationary-EE posture targets. Separate kinematic
vs dynamic projector ablations; zero-posture circle has no projector effect.
Proposed gain candidates are not blanket hardware execution approvals.
Keep stiffness and damping sweeps separate, three repetitions of selected
profiles, phase-resolved errors, realized posture motion, timing/torques and
fault/stop outcomes. No controller/runtime/config changes or hardware execution
in this planning step.


## Comparison implementation delivered — 2026-10-12

See `deployment/franky_runtime/M4_COMPARISONS.md` for commands and current
contracts. Native optional DLS route added to M4RuntimeMotion, sharing the tested
watchdog/guards/ring log and captured-pose stop-v2. Shared joint PD algebra is
reused with JointImpedanceBase. Separate configured Cartesian stop gains ensure
active gain sweeps do not silently vary braking gains. No libfranka/Ruckig/
firmware/dependency changes: isolated franky-m4-comparison-server10, local
Franky2.0.1.dev65+gm4compare.libfranka.0.21.2, all21 bundled dependency hashes
identical to retained stop-server10. Wheel/receipts under deployment_bundles/
2026-10-12/franky-m4-comparison.

User requested each session automatically repeat3 times: runner now defaults3
(or experiment.repeats), one start,5s interval, fresh live checks, individual
trial folders. Fault, failed stop, logging loss, operator stop or repeat-start
drift aborts continuation; no auto recovery/homing. --repeats1 for commissioning.
Posture targets frozen within the session; paired sessions still require target
and start matching from recorded plans. Existing executable configs are not
silently replaced by new comparison candidates.

22 candidate profiles generated, plus local single-profile executable-copy
selector. Phase events tied to applied reference sequence, joint/DLS telemetry,
phase/contour/posture/torque/settling metrics and repeat-aware report aggregator
implemented. Existing historical runs exercised into comparison-baseline-report
with indexed JSON/CSV, Markdown and PNG/PDF figures; exploratory evidence remains
labelled as such. Tests42 Python and22 native cases/75 assertions passed.
No FCI access or hardware motion performed. DLS composed route needs first
hardware hold, then one circle trial before three-trial comparisons. Portable
DLS law at source/franka_motion/franka_motion/dls_tracking.py is available to the
simulation workstation; native/portable numerical parity tested.


Posture comparison refinement: all posture candidates reuse the exact recorded
successful150500Z posture-plan.json across profiles, with live FK/bounds/singular
checks and source SHA recorded. First start must be within0.15rad per joint of
its tested anchor and within existing pose guards. Returning posture sequences
check subsequent start drift relative to the shared intended return anchor;
other tasks use trial1 measured anchor. This is a validation/target selection,
not automatic homing. Small initial-state differences remain logged.

### Circle sweep and reduced posture comparison (2026-10-12)

All ten circle gain profiles have completed trials. Latest-representative position RMS selected Cartesian translation K1200/D69.282 (orientation K40/D13), 4.668 mm position and 1.860 deg orientation RMS; joint PD K200/D28.284, 1.107 mm and 0.111 deg. These are provisional single-trial findings with varying starts. The reduced posture experiment uses Cartesian kinematic, Cartesian dynamic and DLS joint-PD kinematic routes with the same saved A/B/return schedule. [Selection and commands](franky_runtime/M4_POSTURE_SELECTED.md). Full indexed circle report, faults and plots reside in hardware inventory `2026-10-12/m4/circle-comparison-report/`. No simulator, firmware, libfranka or Franky-wheel changes were needed for this preparation.

### Completed reduced posture comparison

[Posture audit](franky_runtime/M4_POSTURE_AUDIT.md): Cartesian kinematic: 3.00 mm / 0.60 deg RMS; Cartesian dynamic: 2.52 mm / 1.65 deg; DLS joint K100/D20: 1.47 mm / 0.25 deg across all active phases. All three completed with smooth stops; K200 failed the J7 torque guard and is excluded from full-task means. DLS K100 achieved A and return settling but not B; neither Cartesian case achieved the strict joint-norm settling criterion. Exact shared targets and nearly identical starts support a preliminary comparison, not statistical conclusions. Full report/log indices reside in hardware inventory `2026-10-12/m4/posture-comparison-report/`.

## Consolidated task report

The full 2026-10-12 path-tracking, posture and curve-fixture report, with equations, numerical tables and a 49-run inventory, is available at [M4_HARDWARE_RESULTS_REPORT.md](M4_HARDWARE_RESULTS_REPORT.md). It distinguishes completed experiments from faults and no-motion checks, and preserves the single-trial limitations.
