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
