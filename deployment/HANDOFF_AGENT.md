# Instructions for the receiving Franka deployment agent

Read `REAL_ROBOT_DEPLOYMENT_PLAN.md` and `README.md` before implementation.
Start with the inventory in section 11. Inspect installed ROS/libfranka sources
and compatibility before selecting APIs. Keep the ROS workspace separate from
the simulator environment. No Isaac installation is required for inference.

Run all hardware experiment components locally. Consume targets and experiment
schedules from YAML. Do not add workstation communication, interactive target
entry, live schedule overrides, network filesystems, or telemetry forwarding.

Default development to fake hardware or shadow mode. This handoff and its YAML
files do not authorize physical motion. Follow the lab's authorization and
physical-stop procedure before commissioning. Never recover a fault or resume
motion automatically. Keep faults terminal for the current suite.

Keep inference, allocation, locks, logging, disk access, and ROS publication out
of the 1 kHz controller callback. Validate observation order, joint order,
frames, units, timing, raw previous action, and default-offset action mapping
against the bundle. Do not substitute simulation termination limits for robot
limits or assume Panda/FR3 equivalence.

Preserve the received package as immutable. Develop in a separate local
workspace copied from `source/`; write generated logs outside the package.
Record revisions and suite/bundle hashes. Report C++ parity, fake-hardware,
shadow, and hardware results separately; numerical export parity alone is not
evidence that the policy can safely control this robot.

## 2026-09-25: hardware-control information required before further training

Pause further deployment-oriented training until the control contract below
is resolved. This is an information-gathering request, not motion authorization.

### Simulation findings

- Both saved Panda training configs enabled gravity: `(0, 0, -9.81)` and
  `disable_gravity=false`. Runtime reconstruction retained tiny authored link
  inertias and zero COM offsets. FR3 URDF link-inertia traces are about
  5,800–11,900 times larger; these are NOT whole-arm joint-space inertia ratios.
  Exact historical training-asset identity remains unproven. See
  `model_audit/2026-09-24_runtime/FINDINGS.md`.
- A separate FR3v2 bare-flange model now passes runtime mass, COM, full-inertia,
  joint-limit, flange-FK and payload-composition checks. See `FR3_SIMULATION.md`.
  It still assumes implicit PD 80/4, armature 0.001 kg m², 60 Hz physics and
  30 Hz policy updates. These are not identified hardware controller settings.
  No hardware-equivalent reference governor is implemented yet.
- New FR3 nominal and payload-only DR policies were trained from scratch.
  DR samples 0–1 kg point payload with flange-relative axial offset 0–5 cm.
  One evaluation seed, 512 paired episodes per scenario:

  | Scenario | Nominal success | Payload DR success |
  | --- | ---: | ---: |
  | Bare flange | 449/512 | 512/512 |
  | 0.5 kg at flange | 432/512 | 512/512 |
  | 1 kg at flange Z=5 cm | 406/512 | 512/512 |

  All failures were six-second timeouts; no configured unsafe terminations.
  Targets and initial joint states matched exactly. Nominal timeout endpoint
  error averaged 4.1–4.3 cm. DR mean success time was 0.357–0.358 s, but success
  means position error <3 cm for five policy samples, without a low-velocity
  or long-hold requirement. It is not a validated hardware settling time.
  Results: data-root `evaluation_suites/2026-09-24_23-30-50_fr3_nominal_vs_payload_dr`.
  Checkpoints: data-root `runs/logs/rsl_rl/fr3_reach/` runs
  `2026-09-24_21-27-32_nominal_seed42` and
  `2026-09-24_21-37-51_payload_dr_seed42`, both `model_999.pt`.
  Existing Panda bundles remain unchanged; these FR3 policies are not approved
  for hardware use. The suite is in `config/experiments/fr3_comparison.yaml`
  under the `franka_rl` Python package.

### Request to the real-time machine agent

Inspect existing evidence, installed sources/configs and version-matched
official documentation first. Report each item as verified, inferred or unknown,
with provenance. Do not guess inaccessible firmware internals or nominal gains.

1. **Platform and selected interface:** confirm robot model/revision, system
   image, libfranka, franky, ROS 2/franka_ros2/ros2_control versions and revisions.
   Identify the intended position/velocity/torque command API and controller
   mode, including the exact executable/controller and configuration to deploy.
   Recommend one path if undecided; explain unresolved choices.
2. **Complete command path:** specify how the 30 Hz policy output becomes a
   1 kHz command. Document default-offset mapping, raw-action clipping,
   interpolation/hold/filtering, rate limiting, processing order and state,
   start/reset initialization, timestamps and stale-command handling. Include
   defaults actually enabled in the installed API, not just available options.
   Supply reusable governor code or equations and offline input/output vectors
   so simulation can reproduce this path exactly.
3. **Low-level dynamics:** establish exposed impedance/stiffness/damping settings,
   units and configuration ownership; gravity/Coriolis compensation, torque
   saturation/rate limits and payload-model use. Distinguish known behavior
   from proprietary/unknown behavior. Do not equate position-interface behavior
   to our implicit PD or disable hardware compensation to imitate simulation.
4. **Observations and frames:** identify measured versus commanded q/dq,
   filtering, signal age, synchronization and inference latency/jitter. Confirm
   joint order, base/flange/EE transforms, FK source, units, previous-action
   semantics and normalization. Provide a versioned observation/action contract
   and offline parity examples against the FR3 flange convention.
5. **Limits and safety behavior:** return applicable position-dependent velocity,
   acceleration, jerk, effort and torque-rate limits and their sources. Separate
   manufacturer constraints, lab-approved operating limits, and sim thresholds.
   Document collisions/self-collision protections, workspace exclusions,
   watchdogs, disconnects, protective-stop behavior and terminal-fault handling.
   Do not invent acceptance thresholds or change robot safety settings.
6. **Actual installation and identification gaps:** confirm mounting/gravity
   direction, attached hardware, configured tool/load mass, COM and inertia,
   and any known calibration offsets. List uncertain friction, actuator response,
   delay and payload quantities with justified DR ranges where evidence exists.
   For missing measurements, propose a separately authorized conservative
   identification protocol, signals to log and acceptance checks; do not execute
   motions or alter load/controller settings under this request.

Return `deployment/hardware_control_audit/<date>/` with a concise findings and
blockers note, machine-readable control contract (YAML/JSON), version/config
evidence, frame/limit definitions and offline parity vectors. Reuse existing
inventory instead of duplicating it. Explicitly list which simulation changes
are required, which unknowns need lab decisions or authorized measurements,
and what is still needed before hardware acceptance. Keep large traces outside
Git. All eventual runtime components remain local to the real-time machine;
targets and schedules remain YAML-driven with no training-workstation link.

### Real-time host response — 2026-09-25

Read [hardware_control_audit/2026-09-25/FINDINGS.md](hardware_control_audit/2026-09-25/FINDINGS.md)
and its machine-readable control contract before resuming deployment-oriented
training. This source-based audit reports verified defaults, recommended choices
and explicit unknowns. It does not resolve the missing deployed governor,
identified internal gains or lab commissioning decisions.

Live follow-up: [LIVE_INSPECTION.md](hardware_control_audit/2026-09-25/LIVE_INSPECTION.md).
First deployment will preserve the robot's existing internal settings. The FCI
model now supplies corrected fr3v2.1 COM/mass data; numeric internal gains and
dynamic response remain unmeasured.

## Controller choices and shared reference governor — 2026-09-25

Retain all three options:

1. **First-deployment direction:** PPO joint-position targets → shared 1 kHz
   governor → robot's existing internal impedance controller; preserve defaults.
2. **Alternative:** PPO joint-position targets → shared governor → explicit
   host-side PD/impedance → FCI torques. The policy still outputs positions.
3. **Research alternative:** PPO outputs torques directly, requiring a new torque
   action contract and separate torque command processing/validation.

Do not substitute wide arbitrary PD randomization for measured controller-response
coverage. Read [REFERENCE_GOVERNOR_PLAN.md](REFERENCE_GOVERNOR_PLAN.md) for the next
implementation milestone, state/timing contract, feasibility and stopping design,
shared-core strategy and parity tests. The governor algorithm and deployment
thresholds remain to be validated; this is a plan, not a working controller.
Offline governor development and simulation validation may proceed. Freeze the
command path and resolve response-model assumptions before further deployment-
oriented training. Existing bundles and the no-motion authorization boundary remain.

### Governor implementation and shipment — 2026-09-25

The canonical core now lives in `franka-rl/deployment/reference_governor/`, not
franka_ros2. Read [its README](reference_governor/README.md) and the updated plan.
Version 0.1.0 ships as a standalone wheel/source archive plus CMake target. The
simulation workstation needs no ROS repo. `Franka-FR3v2-Governed-Reach-v0` is an
opt-in task targeting Isaac Sim 6.0.1 / Isaac Lab v3.0.0-beta2.patch1.

The implemented conservative quintic governor uses a cached bounded continuation
to rest; it is not time-optimal. Native, Python and adapter API-stub tests are local
evidence only. First run a small actual Isaac smoke test, validate the corrected
live-model asset and benchmark throughput before large PPO jobs. The supplied
simulation config is explicitly synthetic, not approved deployment settings.
Governor-aware simulation experiments can now proceed with these limitations;
hardware-equivalence claims still need identified response and reviewed limits.

Validated local release: `/home/chen-lab/yifan/governor_releases/0.1.0-validated/`.
Transfer its wheel/source archive, integration files, manifest and checksums out
of band. The installed wheel passes 32 tests; native CTest, 5,000-tick bounds/
allocation checks and downstream CMake consumption pass. No actual Isaac runtime
or hardware test was performed. Use this clean release rather than earlier
`0.1.0-rc1` / `0.1.0` build directories.

## Return request: identify hardware controller response before training — 2026-09-25

### New simulation evidence

The corrected live fr3v2.1 asset passed runtime mass/COM/inertia, limit, FK and
payload-composition validation. Actual governor smoke failed closed at 13 ms:
joint 4 measured velocity exceeded the synthetic 0.3 rad/s cap while its reference
was nearly stationary. A paired stationary-reference diagnostic reproduced the
crossing at 12.7 ms with gravity on; with gravity off, both arms remained exactly
stationary for 0.2 s. Initial states matched. At 13 ms, joint 4's model gravity
compensation term was about 18.95 Nm versus a PD estimate of 1.41 Nm.
This isolates gravity-induced motion in our uncompensated 80/4 surrogate; it
does not identify the real controller. See `FR3_SIMULATION.md` for artifacts and
reproduction. Do not disable physical gravity or hardware compensation, or relax
governor limits simply to pass a smoke test.

The transferred source lacked `franka_governor/__init__.py` due to the broad
ignore rule. A tracked simulation-only loader was added locally; reconcile it
with the validated RT release before shipping. The Isaac adapter now refuses
`command_valid=false` outputs before writing targets. Native tests and 33 Python
tests passed here; actual partial-reset/stopping/throughput checks were not
reached before the plant fault. Preserve these distinctions in release evidence.

### Requested work on the real-time machine

**This request authorizes offline preparation, not robot motion, control
activation, gain/load/safety-setting changes or automatic fault recovery.**
Pause deployment-oriented training until the response-model assumptions and
command path are resolved. Preserve current internal settings rather than
assuming they are factory defaults.

1. **Confirm the selected controller path.** Pin installed source/binary versions,
   position-interface mode, filtering, compensation evidence and actual enabled
   host processing. Distinguish verified behavior from unknown firmware internals.
   Do not infer active gains from URDF K/D metadata or example controllers.
2. **Complete and validate the offline integration prerequisites.** Use the shared
   governor in the selected ROS controller; check initialization from desired
   states, timing, command validity, watchdog, terminal faults and stopping.
   Remove automatic startup recovery from the selected path and test its absence.
   Reconcile shared-core/config hashes and numerical parity across machines.
   No hardware commissioning is authorized by passing these tests.
3. **Prepare a YAML identification suite and approval proposal.** Specify small,
   smooth, bounded joint-position references, initially one joint at a time at
   several reviewed configurations. Include warmup/hold segments, repetition,
   conservative start/end transitions and separate fitting/validation trials.
   Propose amplitudes, frequencies, speed/acceleration/jerk bounds, allowed
   workspace, attachment assumptions, stop criteria and watchdog budgets for
   lab review; do not invent approved values or execute the suite without
   separate explicit motion authorization and the lab's physical-stop procedure.
4. **Implement synchronized local logging.** Record actual post-governor commands,
   reference q/dq/ddq, robot desired q_d/dq_d/ddq_d, measured q/dq, available torque
   signals and model terms, governor status/interventions, policy/command sequence
   numbers, robot time and host monotonic receipt/processing/consumption times.
   Document signal frames, units, estimator/filtering uncertainty, and gravity/
   friction compensation conventions. Establish clock relationships before delay
   calculations. Buffer RT samples; keep allocation, locks and disk I/O outside
   the RT callback. Log faulted trials rather than silently discarding them.
5. **Identify effective response, not just two PD numbers.** Once approved data
   exist, estimate tracking delay, bandwidth, damping, steady-state error and
   cross-joint/configuration dependence. Test whether a PD-plus-compensation
   surrogate explains the observations. Position traces alone generally do not
   uniquely identify gains and unknown inertia; report assumptions, confidence
   and identifiability limits. Distinguish fitted equivalent gains from actual
   firmware settings. Do not claim a compensation law from an available model API.
6. **Validate and return a training contract.** Evaluate the fitted surrogate on
   held-out trajectories/configurations, reporting position/velocity errors,
   transient/settling behavior, timing and failure cases. Propose response-model
   parameters and evidence-supported DR ranges, separately from future payload
   scenario choices. Define acceptance checks for lab review. Do not replace
   missing measurements with broad arbitrary PD randomization.

### Return deliverables

Add a new versioned directory under `deployment/hardware_control_audit/` containing
the offline integration report, reviewed-or-pending YAML experiment suite,
signal schema and logging/analysis commands, version/config/model hashes, and a
concise remaining-blockers list. After separately authorized measurements, add
compact fitted-model parameters, held-out validation results and proposed DR
ranges with provenance. Keep raw traces outside Git and reference their location
and hashes. Clearly distinguish planned, offline-tested and hardware-measured
results. All experiment components remain local to the RT machine, YAML-driven,
with no training-workstation runtime connection. The objective is matched hardware
response, not forcing hardware to reproduce the simulator's assumed PD equation.

### RT-host offline response — 2026-09-25

The offline controller-response audit is returned in
[`hardware_control_audit/2026-09-25-response-identification/`](hardware_control_audit/2026-09-25-response-identification/README.md).
It confirms the joint-position/joint-impedance path and unavailable firmware gains,
provides a pending-only identification YAML, synchronized signal schema, analysis
tooling, reconciled governor hashes, offline validation evidence, and explicit
remaining blockers. No response-identification motion has been run and no fitted
hardware response or deployment DR range exists yet.


Coordinator follow-up: the same audit directory now includes
[`EXPERIMENT_COORDINATOR.md`](hardware_control_audit/2026-09-25-response-identification/EXPERIMENT_COORDINATOR.md).
The RT host has implemented one local YAML-driven coordinator for identification
and PPO experiments, coherent non-RT observation/status topics, isolated ONNX
inference, sticky terminal supervision, and FR3 flange-contract rejection of the
existing Panda bundles. Full fake ros2_control update/fault testing, an exported
FR3 bundle, reviewed runtime YAML, and authorized measurements remain open.

Coordinator preparation follow-up: the RT host now includes a direct encoded
Franka state/model configure/activate/update/deactivate harness, deterministic
approved controller-YAML generation, and a fail-closed deployment preflight with
immutable staging. The canonical governor now permits a sequence-0 armed hold
before explicit operator start and activates the action watchdog after accepting
sequence 1; state, timing, and tracking checks remain active during the hold.
Native governor tests (including C++/Python parity) and ROS package tests pass.
Portable matching governor release: `/home/chen-lab/yifan/governor_releases/0.1.0-armed-hold-20260925/`, manifest SHA-256 `fcc058d13ee5a14b547e60d3845bec12557bc2df1769b9aee81bb13eae9aa222`. See [`HARDWARE_IDENTIFICATION_PREP.md`](hardware_control_audit/2026-09-25-response-identification/HARDWARE_IDENTIFICATION_PREP.md)
for the remaining lab decisions and later authorized operator sequence. No
response-identification motion was performed.

### Direct Franky experiment coordinator — 2026-09-25

A direct Franky runtime is now implemented at
[`deployment/franky_runtime/`](franky_runtime/README.md) as an alternative to the
ROS 2 controller path. One YAML-driven coordinator runs both response-identification
references and FR3 PPO targets. It reuses the ROS coordinator's reference schedule,
24-value observation/action contract, immutable ONNX worker, approval gates,
sticky-fault lifecycle, and artifact layout.

The selected direct path explicitly uses `ControllerMode.JointImpedance` and sends
absolute `JointWaypointMotion` targets at 30 Hz. Franky's Ruckig generator produces
the 1 kHz position command with 5% velocity/acceleration/jerk factors; libfranka's
100 Hz command filter remains enabled and its separate rate limiter is disabled.
An independent 75 ms host watchdog submits `JointStopMotion` if target
replacement stalls. The runtime records Franky's actual
per-cycle command together with robot desired/measured state and torque signals,
so the identification experiment measures the path that later PPO execution uses.
It does not call impedance/load setters or automatic recovery.

Seven offline tests and an additional pinned-environment fake CLI run pass. This is
implementation evidence only: pending YAML remains non-executable, no Franky motion
or response-identification experiment has run, controller response is still
unknown, and a verified FR3 PPO bundle is still required. Because this path uses
Franky's own Ruckig target replacement, simulation should reproduce that command
path rather than assuming the standalone shared governor is active for these runs.

Franky hardware smoke follow-up: session `20260925193000` ran for 0.459 s and
logged 460 callbacks with maximum per-joint `|q-q_d|` below 0.000322 rad, then
failed closed with Ruckig synchronization error `-111` during repeated identical
home-target replacement. `JointStopMotion` completed and no samples were dropped.
The runtime now treats unchanged 30 Hz targets as watchdog keepalives and only
preempts Franky when the target vector changes. Eight offline tests and the exact
26-second fake reference schedule pass; no corrected hardware retry has run.

Franky response data follow-up: hardware session `20260926145004` completed the
0.005 rad, 0.25 Hz joint-1 schedule with 25,992 callbacks, no robot errors or
dropped samples, and a clean stop. In the four-cycle active window, measured over
generated-reference amplitude gain was 0.9964 and phase lag was 0.01951 rad
(about 12.4 ms); this is one small-signal point, not a gain/bandwidth model. The
trace and audit live under the RT host hardware inventory. A pending one-joint-at-
a-time seven-joint ±20 degree suite uses 0.03 Hz, 16 s quintic ramps, and two cycles
per joint; its conservative 0.1067 rad/s bound is within the 5% Franky cap. Expanded
swept-workspace review remains required before that suite is executable. Before the
full sweep, a separate pending pilot limits the same schedule to joints 1 and 2
(`reference.2joint-20deg.20260926150714.pending.yaml`, about 3.52 minutes) so the
frequency, ramp, and cycle count can be assessed from hardware data first. The two-joint
pilot session `20260926150714` subsequently completed with 211,325 callbacks, no
robot errors or dropped samples, and a clean stop. At 0.03 Hz the target-to-Franky-
command lag was about 46 ms on joint 1 and 83 ms on joint 2, while Franky-command-
to-measured lag was about 6-7 ms. A pending 20% dynamics pilot therefore uses
0.10 Hz and 4 s ramps for the same joints and amplitude; it remains subject to a
fresh workspace review and operator authorization. The 20% session
`20260926151901` then completed with 69,994 callbacks, no robot errors or dropped
samples, and a clean stop. At 0.10 Hz, measured-over-Franky-command gain was
0.99998/0.99995 and lag was 5.46/5.62 ms for joints 1/2; end-to-end target lag
was 45.75/76.50 ms. These are effective response points, not firmware gains. The next pending
experiment is a 20% dynamics grid for joints 1 and 2: 0.10 Hz at +/-2 and +/-10
degrees, 0.25 Hz at +/-10 degrees, 0.50 Hz at +/-4 degrees, and 1.00 Hz at
+/-1.5 degrees. It lasts about 4.9 minutes and retains the 1 kHz callback trace;
lower-rate training data will be filtered and derived offline. The grid faulted in trial 2 with `joint_motion_generator_acceleration_discontinuity`. At a 30 Hz target replacement, generated acceleration changed from about +3.0 to -3.53 rad/s^2 in one millisecond although the analytic sine bound was 0.318 rad/s^2. This confirms that repeated position-only `JointMotion` preemption is not a robust 30 Hz PPO interface. The config is retired against rerun. Franky joint-impedance tracking with explicit gains or a continuous 1 kHz position generator must replace this backend before more motion.

### Franky impedance-tracking migration — 2026-09-26

The fault-prone 30 Hz `JointMotion` preemption backend has been replaced offline by one long-lived `JointImpedanceTrackingMotion`. Policy/reference updates now publish `JointReference(q, dq=0)` into the same 1 kHz torque-control session. The watchdog and normal shutdown use `TorqueStopMotion`; the runtime contains no `JointMotion` or `JointStopMotion` construction. Logs add the actual controller `tau_command` and retain held q/dq references, robot desired/measured torque, state, and external torque.

The first explicit gain profile is Franka ROS 2's compliant `JointImpedanceExampleController`: K `[24,24,24,24,10,6,2]` Nm/rad and D `[2,2,2,1,1,1,0.5]` Nms/rad. Coriolis compensation, 1 Nm/ms torque slew, zero friction/feedforward, the pinned Franky 0.5 rad error clip, soft-limit repulsion, and torque-stop values are all config-validated and recorded. Alternative public profiles and provenance are in `deployment/franky_runtime/IMPEDANCE_PARAMETER_SURVEY.md`.

Ten offline tests pass and the hardware path was not opened. Runtime source SHA-256 is `a263da7724c9943361838b60d0fb037a2536779122ab7377d76e901700154a70`. A structurally valid but non-executable joints-1/2 ±2 degree smoke config is staged at `/home/chen-lab/franka_ros2_ws/hardware_inventory/2026-09-25/franky_reference/reference.2joint-impedance-tracker-smoke.20260926160714.pending.yaml`. It requires a fresh torque-control review and live pose within 0.02 rad of home.

Hardware impedance follow-up: sessions `20260926160714` and `20260926163104`
both completed the same J1/J2 +/-2 degree, 0.1 Hz reference with zero dropped
samples, no robot errors, and clean torque stops. The first used the Franka
compliant example gains; the second used Franky's default K `[50]*7` and D
`[14.1421]*7`. Franky's defaults improved measured/reference amplitude ratio
from 0.299 to 0.687 on J1 and from 0.054 to 0.510 on J2. RMS tracking errors fell
from 0.0241/0.0396 rad to 0.0149/0.0244 rad. Closed-loop phase was -34.4 degrees
on J1 and -41.4 degrees on J2 with the defaults. These gains are the provisional
minimum deployment profile, not a tight position servo. Isaac Lab must simulate
the explicit torque controller and should not replace its targets with achieved
joint positions. Full results and the profile decision are in
`deployment/franky_runtime/IMPEDANCE_PARAMETER_SURVEY.md`; the raw 1 kHz records
and detailed audits remain in the RT host hardware inventory.

The incremental J1/J2 K=100, D=20 smoke, session `20260926164144`, also
completed normally with 61,993 callbacks, no robot errors or drops, and a clean
stop. At 0.1 Hz, amplitude ratios improved to 0.885/0.816 and closed-loop phase
to -19.7/-24.0 degrees for J1/J2. RMS errors were 0.00872/0.01306 rad. Peak
command torque magnitude was 2.135 Nm and the largest one-millisecond torque
change was 0.138 Nm, well below the configured 1 Nm/ms slew limit. K=100/D=20
is therefore the selected tested J1/J2 profile for the first deployment pass.
This does not select J3-J7 gains; test those joints separately before freezing
the seven-joint Isaac/Franky actuator configuration.

A nominal-versus-gain-DR flange-path demo is planned in
`deployment/franky_runtime/GAIN_DR_EE_PATH_DEMO.md`. Hardware cases are defined
as multipliers 0.5/1/2 around a commissioned per-joint nominal gain vector, not
uniform gains on every joint. This gives J1/J2 K=50/100/200 while avoiding an
unsupported K=200 extrapolation on distal joints. The DR policy samples the
same multiplier log-uniformly per episode without observing it; the nominal
policy uses multiplier 1. All other training randomization and runtime settings
must match between policies. J1/J2 K=200 has a pending-only small smoke config;
J3-J7 and the complete seven-joint endpoint profiles must be commissioned before
PPO hardware evaluation.

The J1/J2 K=200, D=28.284 endpoint, session `20260926165740`, completed with no
errors or drops. Amplitude ratios were 0.954/0.908, phase -11.4/-13.2 degrees,
and RMS errors 0.00509/0.00733 rad. Peak command torque magnitude was 2.098 Nm;
the largest one-millisecond torque change was 0.265 Nm. K=200 is validated as
the J1/J2 stiff endpoint while K=100 remains the nominal center. A pending
sequential J3-J7 suite applies K=200 to all joints but reduces reference
amplitude from 2 degrees on J3/J4 to 1.5 degrees on J5/J6 and 1 degree on J7,
with correspondingly tighter tracking thresholds.

The all-joint K=200 endpoint was extended to sequential J3-J7 motion in session
`20260926170459`. It completed 157,995 callbacks with no errors or drops.
J3-J7 amplitude ratios were 0.887/0.934/0.907/0.927/0.831 and phase was between
-12.8 and -16.5 degrees. The largest command-torque magnitude was 1.850 Nm and
the largest one-millisecond change 0.252 Nm. Matching pending K=100 and K=50
suites preserve the exact schedule and complete the nominal/DR endpoint data.

The K=100 and K=50 J3-J7 suites, sessions `20260926171741` and
`20260926171742`, both completed with 157,994 callbacks, no errors or drops, and
clean stops. Across J3-J7, amplitude ratios were 0.763--0.844 at K=100 and
0.445--0.623 at K=50, versus 0.831--0.934 at K=200. Portable seven-joint metrics
are in `deployment/hardware_control_audit/2026-09-26-franky-impedance/`.
Hardware characterization now defines the K=50/100/200 controller family, but
production training remains blocked on the explicit torque/gravity model,
correlated gain DR, governor/direct-hold decision, continuous flange-path task,
and Isaac-to-hardware response validation listed in
`deployment/DEPLOYMENT_TRAINING_READINESS.md`.

## Current 50 Hz incremental PPO deployment handoff — 2026-09-27

This section supersedes earlier PPO descriptions in this file and in the Franky
runtime README that mention a 24-value observation, seven policy actions,
`q_default + scale * action`, 30 Hz PPO, repeated `JointMotion`, or the standalone
quintic reference governor. Those contracts describe retired or historical work.
Do not adapt the new checkpoints to those interfaces.

The present policies were trained against the explicit Franky-like impedance
task, not an implicit PhysX position drive:

- Isaac/torque-controller physics: 1 kHz;
- deterministic policy inference: 50 Hz (20 ms period);
- six normalized joint-reference increments controlling joints 1--6;
- joint 7 held at its measured episode/session start reference;
- one long-lived joint-impedance torque controller at K=100 Nm/rad and D=20
  Nms/rad for all seven joints;
- direct reference hold between policy samples, with no Ruckig planner, quintic
  governor, action low-pass filter, or other reference smoother;
- flange targets and all experiment schedules loaded locally from reviewed YAML;
- no runtime connection to the training workstation.

The policies are **offline integration candidates, not hardware-qualified
controllers**. Transferring a checkpoint or passing ONNX parity does not authorize
motion. Preserve the existing approval, operator-start, physical-stop, workspace,
watchdog, sticky-fault, torque-stop, and no-automatic-recovery requirements.
Targets are noninteractive YAML data; the explicit operator start/stop gate remains
a safety authorization step and is not a target-entry UI.

### Candidate checkpoints and evidence

The source workstation paths below identify the exact inputs to package. Transfer
immutable verified bundles, not raw paths. Check each SHA-256 after transfer.

| Label | Checkpoint | SHA-256 | Status |
| --- | --- | --- | --- |
| `position_nominal` | data root `runs/logs/rsl_rl/fr3_incremental_6d_impedance_reach/2026-09-26_19-55-38_fr3_incremental_6d_fine_nominal_v1/model_149.pt` | `7568e71a26e981eeefca50f808f2bfd526c42a900a48884071e0afa54318ca37` | Primary position-only integration candidate. Training ended near 5--6 mm mean position error and 100% training success. Circle evaluation was only a small smoke test, not hardware qualification. |
| `position_dr` | data root `runs/logs/rsl_rl/fr3_incremental_6d_impedance_reach/2026-09-26_23-42-59_fr3_incremental_6d_position_dr_v1_recovered/model_199.pt` | `a0f844c0fc702f79b6c3291c67eaf2294ce2750aaf4af9cc190089537d85420b` | DR comparison candidate. Training ended near 6.6 mm mean error and 99.8% tail success. It still needs a full deterministic evaluator run before it can be compared or selected. |
| `position_z_axis_nominal` | data root `runs/logs/rsl_rl/fr3_incremental_6d_position_z_axis_reach/2026-09-26_22-44-18_fr3_incremental_6d_position_z_axis_nominal_continued_v1/model_198.pt` | `829dc2fdbe8fa1f2283969957d39d8f5af4867ebb6527fcbf70f0bed77d61c12` | Orientation-capable integration candidate. Nominal deterministic evaluation recorded 125/128 sustained successes, 23.6 mm mean final position error and 0.0212 rad (1.22 deg) mean final z-axis error. It exceeded the 20% measured-velocity qualification envelope in 57.8% of episodes, with a worst ratio of 1.068; do not move hardware until this is resolved. |

Do **not** ship the current z-axis DR checkpoint as a good policy. Checkpoint
`73d0e66846c607e5c35d5ac7bf94262a24aec6b770b337d3ccc9e8100180678b`
(`.../2026-09-26_23-47-14_fr3_incremental_6d_position_z_axis_dr_v1/model_199.pt`)
achieved only 20/1024 nominal and 14/1024 randomized deterministic successes,
with about 0.20 m mean position error. It may be retained as a negative result.

These policies were produced from Git HEAD
`1ff4e8da59b08e057153ef8878662150b398928e` plus uncommitted task, evaluator,
scenario, distribution and z-axis files. The checkpoint parameter YAMLs do not
fully capture executable Python semantics. Before building the shipment, commit
or otherwise freeze and hash the complete working tree; recording HEAD alone is
insufficient.

### Robot, frame, and target contract

- Hardware joint order is exactly `fr3_joint1` through `fr3_joint7`. Isaac asset
  joints are historically named `panda_joint1` through `panda_joint7`; this is a
  name translation only, never a reorder.
- Robot/base frame: `fr3_link0`.
- Controlled/tracked body: bare-flange `fr3_flange`; no hand or tool-center offset.
- Training assumes identity `F_T_EE`, no attached end effector, and zero configured
  external load for the nominal case. Reject mismatches at preflight.
- Cartesian position targets are metres in the robot base frame. The sampled
  training box was x `[0.35, 0.60]`, y `[-0.20, 0.20]`, z `[0.20, 0.50]` m.
  Hardware suites require a separately reviewed subset and collision/workspace
  review; the training box is not automatic hardware authorization.
- Position-only policies ignore target orientation.
- The z-axis policy controls only the direction of the flange +z axis. Rotation
  about flange z is deliberately unconstrained. A standalone point-target suite
  may store a unit `z_axis_base: [x, y, z]`. The current path-catalog contract
  instead stores a normalized `target_orientation_xyzw` and derives its +z axis;
  arbitrary twist about that axis must not change the observation.
- Random-point and waypoint-path execution use the same policy. A path is a YAML
  sequence of static targets. The current path contract advances on its one-sample
  position threshold or waypoint timeout as specified below; this intentionally
  differs from random-point sustained-success evaluation. The policy has no phase,
  target velocity, or future waypoint input and is not a continuous trajectory
  generator.

### Hardware waypoint-path contract

The hardware implementation must reuse the version-1 path contract in
`source/franka_rl/franka_rl/config/paths.yaml` and the resolution rules in
`franka_rl.utils.paths`. Do not translate paths into ad-hoc target lists by hand.
Ship the catalog (or an exact reviewed derivative) in the deployment package,
record its SHA-256, select the path by name in the immutable experiment YAML, and
store the fully resolved waypoint list in run artifacts. Target selection remains
local and YAML-driven; there is no live waypoint entry or workstation control.

The path catalog root contains exactly:

```yaml
version: 1
paths:
  <path_name>: ...
```

Unknown root keys, path keys, types, non-finite values, invalid quaternion norms,
fewer than three waypoints, non-positive thresholds/timeouts, and circle waypoint
counts below three are errors. Two path types are supported.

An explicit waypoint path uses:

```yaml
type: waypoints
description: optional text
waypoints_m:
  - [x0, y0, z0]
  - [x1, y1, z1]
  - [x2, y2, z2]
target_orientation_xyzw: [qx, qy, qz, qw]
waypoint_timeout_s: 1.0
position_threshold_m: 0.01
```

A circle uses:

```yaml
type: circle
description: optional text
center_m: [cx, cy, cz]
orientation_rpy_deg: [roll, pitch, yaw]
radius_m: 0.075
waypoint_count: 24
phase_deg: 0.0
target_orientation_xyzw: [qx, qy, qz, qw]
waypoint_timeout_s: 1.0
position_threshold_m: 0.01
```

All positions are metres in `fr3_link0`. Circle RPY uses degrees and the XYZ
Euler convention whose matrix is `Rz(yaw) @ Ry(pitch) @ Rx(roll)`. For waypoint
index `i` in `[0, N-1]`:

```text
angle_i = phase + 2*pi*i/N
local_i = [radius*cos(angle_i), radius*sin(angle_i), 0]
waypoint_i = center + Rz*Ry*Rx*local_i
```

The stored quaternion uses xyzw order and is normalized by the loader. It is one
fixed target orientation shared by every waypoint. Position-only policies receive
only waypoint position; z-axis policies derive the target +z direction from the
quaternion and append the two-component error defined below. Full quaternion/twist
tracking is never added. The generated circle contains `N` distinct points and
does not append waypoint 0 after waypoint `N-1`; one evaluation traversal ends at
the final listed point rather than commanding an extra closing segment.

The current checked-in paths are:

- `circle_xy`: center `[0.475, 0, 0.35]` m, base XY plane, radius 0.075 m,
  24 waypoints, phase 0 degrees, 1.0 s timeout and 0.01 m threshold;
- `circle_yz`: center `[0.475, 0, 0.35]` m, local plane rotated by
  `[0, 90, 0]` degrees, radius 0.15 m, 24 waypoints, phase 180 degrees,
  1.0 s timeout and 0.01 m threshold.

Both currently use quaternion `[0, 1, 0, 0]`, so flange +z points along base -z.
These values are simulation/evaluation definitions, not automatic hardware
workspace approval. The reviewed hardware catalog may tighten geometry, timeout,
or threshold, but every change requires a new catalog hash and must not mutate the
policy observation/action contract.

Implement this exact per-path state machine:

1. At path start, command waypoint 0, set its elapsed count to zero, and retain
   the session's existing incremental reference and preceding action state.
2. At every 50 Hz policy step, compute flange position error for the active
   waypoint and update its minimum error.
3. Mark the waypoint `reached` on the first sample with position error
   `<= position_threshold_m`. This is a one-sample position test: it does not use
   the random-point evaluator's five-sample sustained-success criterion, measured
   velocity, or z-axis error.
4. If it was not reached, mark it `timed_out` after
   `ceil(waypoint_timeout_s / 0.020 s)` policy samples.
5. On either outcome, record the waypoint result and immediately select the next
   waypoint without resetting the environment, policy, held joint reference,
   preceding action, or joint 7 reference. Discard any stale inference result
   tagged for the preceding waypoint; never apply it to the new target.
6. Resolve every waypoint even if an earlier one timed out. After the last
   waypoint, the path succeeds only if every outcome was `reached`; completion
   with one or more timeouts is a path failure.
7. Independent robot, controller, limit, collision, watchdog, timing, inference,
   tracking-error and operator faults remain sticky and terminate the session
   immediately. A waypoint timeout is an experiment outcome, not permission to
   weaken or bypass those safeguards.

The simulation permits a maximum episode duration of
`waypoint_count * waypoint_timeout_s + 1 s` for completion bookkeeping. Hardware
uses monotonic timestamps and the same effective 50 Hz count semantics; do not
base timeout decisions on delayed logger or worker timestamps.

Version the Franky PPO YAML so `ppo.path` and the old `ppo.targets` mode are
explicitly mutually exclusive. A path selection must contain at least the catalog
path, trusted catalog SHA-256, path name, repetitions, and reviewed output/approval
fields. Validate the resolved waypoint positions against the reviewed workspace
before opening control. Do not infer path mode merely from a task or bundle name.

For each traversal, log:

- catalog path/hash/version, selected name and complete resolved `PathSpec`;
- resolved waypoint positions and normalized target quaternion;
- active waypoint index, target, transition reason and policy sequence;
- reached/timeout outcome, elapsed policy steps/seconds, final position error and
  minimum position error for every waypoint;
- for z-axis policies, final/minimum/integrated z-axis error as metrics even
  though z error is not an advancement gate;
- total reached/timeouts, path completion, path success/failure and any independent
  safety termination;
- the normal observation, deterministic action, integrated reference, inference
  timing and 1 kHz controller/state records needed for replay.

Return separate fake-backend and shadow-mode parity fixtures for `circle_xy` and
`circle_yz`. They must demonstrate identical resolved waypoints, transition steps,
stale-result rejection, persistent integrator state across waypoint changes,
per-waypoint artifacts and overall path outcome relative to the Isaac evaluator.

### Deterministic policy interface

Both actors are MLPs with two 64-unit ELU hidden layers and six outputs. Export
the deterministic actor including its final `tanh`; never sample its Gaussian
training distribution on hardware. Observation normalization is disabled.
All values are `float32`, SI units, and finite. Any size, order, unit, timing,
frame, or finite-value mismatch is terminal.

The 29-value position-only observation is concatenated in this exact order:

| Slice | Size | Expression |
| --- | ---: | --- |
| `[0:7]` | 7 | measured `q - q_default`, rad |
| `[7:14]` | 7 | measured `dq`, rad/s |
| `[14:17]` | 3 | `target_position_base - flange_position_base`, m |
| `[17:23]` | 6 | current held references for joints 1--6 normalized to their soft-limit midpoint/half-range |
| `[23:29]` | 6 | preceding bounded normalized increment action |

The 31-value z-axis observation appends:

| Slice | Size | Expression |
| --- | ---: | --- |
| `[29:31]` | 2 | minimal target-z alignment rotation about the current flange x/y axes, rad |

Use this deployment-home/default vector for `q_default`, in radians:

```text
[0, -0.7853981633974483, 0, -2.356194490192345,
 0, 1.5707963267948966, 0]
```

For z-axis error, express the target base-frame z axis in the current flange
frame as `z_tip = [x, y, z]`. Then compute:

```text
s = sqrt(x*x + y*y)
theta = atan2(s, clamp(z, -1, 1))
error_xy = [-y, x] * theta / max(s, 1e-8)
```

The antiparallel singular fallback is `[pi, 0]`. This is not Euler roll/pitch,
not full quaternion error, and not base-frame angular error. The runtime must add
the measured flange rotation to its state snapshot and parity fixtures; position
alone is insufficient.

At reset/session start:

- initialize the seven-joint held reference from measured `q`;
- initialize the preceding six-action vector to zeros;
- do not initialize the reference from `q_default`;
- keep joint 7's reference fixed at its measured start value for the session;
- start only inside reviewed pose/velocity tolerances.

### Action integrator and held reference

The deterministic actor output is a six-vector `a`. Validate it is finite and in
`[-1, 1]`; the ONNX `tanh` should already guarantee the bound. At each 50 Hz
policy tick, for joints 1--6:

```text
vmax = [0.435, 0.435, 0.435, 0.435, 0.522, 0.522] rad/s
dt = 0.020 s
dq_ref_max = vmax * dt
           = [0.0087, 0.0087, 0.0087, 0.0087, 0.01044, 0.01044] rad
q_ref_next = project_to_soft_limits(q_ref + a * dq_ref_max)
```

Hold `q_ref_next` for the next 20 one-millisecond controller steps and send zero
desired joint velocity. Projection applies only at the reviewed soft position
bounds in `LIMITS_AND_TIMING.md`. There is no additional velocity, acceleration,
jerk, interpolation, low-pass, Ruckig, or quintic stage. The reference-acceleration
term used during learning is a reward, not a runtime filter. The measured 20%
velocity envelope is a qualification metric and action-scale basis, not a hidden
online clamp. Any runtime intervention must be explicitly versioned, retrained or
shown equivalent, and recorded; silently smoothing these actions changes the MDP.

The preceding-action observation at tick `t` is the bounded action applied at
tick `t-1`. The reference observation is the corresponding currently held
reference before integrating action `t`.

### Low-level Franky controller equivalence

Use the existing long-lived `JointImpedanceTrackingMotion`, never repeated
`JointMotion` preemption. For the first policy integration profile:

```text
K = [100]*7 Nm/rad
D = [20]*7 Nms/rad
qdot_ref = [0]*7
position error clip = 0.5 rad
non-gravity torque slew = 1 Nm per 1 ms
torque filter cutoff = 100 Hz
joint-limit activation = 0.1 rad inside model limits
joint-limit K/D/cap = 4 Nm/rad, 1 Nms/rad, 5 Nm
Coriolis compensation = enabled
friction and constant feedforward = zero
```

The simulator explicitly adds gravity because PhysX applies physical gravity.
Franky/libfranka's torque-command convention supplies the hardware gravity
support outside the user torque command; do not add a second gravity term merely
because it appears explicitly in the simulator. Preserve the tested Franky
controller implementation and verify this convention with offline/controller
parity and logged hold behavior.

Gain DR uses one unobserved episode-level scalar `alpha`, log-uniform on
`[0.5, 2.0]`, with `K=100*alpha` and `D=2*sqrt(K)` applied to every joint. Other
position-DR variables were joint friction `[0,0.1]`, armature scale `[0.8,1.2]`,
link-inertia scale `[0.9,1.1]`, flange payload mass `[0,1]` kg with COM x/y
`[-0.03,0.03]` m and z `[0,0.10]` m, q noise `+/-0.002` rad, dq noise
`+/-0.02` rad/s, Cartesian-error noise `+/-0.002` m, 0--1 policy-step action
delay, and reset q offsets `+/-0.125` rad. The z-axis DR attempt additionally
used `+/-0.01` rad z-error noise. These ranges describe training, not permission
to change hardware gains or attach a payload.

### Required runtime and bundle work before transfer is usable

1. **Freeze and package the source state.** Commit/tag or create a checksummed
   source snapshot containing the current incremental and z-axis task code.
2. **Extend the bundle exporter.** The current
   `deployment/franka_policy_bundle/contract.py` still intentionally rejects FR3
   export. Add versioned FR3 contracts for the 29D position and 31D z-axis
   policies, embed the deterministic `tanh`, include checkpoint/source/config
   hashes, and generate recorded PyTorch/ONNX parity vectors. Do not weaken Panda
   validation or reuse its contract identifier.
3. **Verify the existing position runtime.** `franky_runtime` already implements
   the 50 Hz six-action incremental integrator and 29D observation. Its README has
   been reconciled on the receiving machine; validate the exact exported bundle
   contract against `ppo.pending.yaml` and the hardware implementation.
4. **Add the z-axis runtime variant.** Extend bundle-contract validation, YAML
   target schema, state snapshots/logging and observation assembly with flange
   rotation and the exact two-component formula above. Branch on a versioned
   contract ID; never guess observation size from an ONNX tensor.
5. **Keep suites YAML-driven.** Populate bundle path/hash, reviewed start pose,
   targets/waypoints, thresholds, timeouts, repetitions, output root and approval
   fields in immutable suite YAML. Do not add live target entry or workstation
   communication.
6. **Run offline gates.** Verify bundle manifest, ONNX parity, observation/action
   golden vectors, action-integrator parity including soft-limit projection,
   50 Hz scheduling, delayed/stale result rejection, fake-backend full suites,
   watchdog/timeout/sticky-fault paths, log completeness, and clean torque stop.
7. **Run shadow inference before motion.** At the reviewed home pose, compute and
   log observations/actions/references without applying policy references. Check
   frames, FK, q default, reference initialization, joint order, latency and
   finite/bound conditions against workstation fixtures.
8. **Requalify safety on every candidate.** Deterministically evaluate target and
   path suites for success, final/minimum error, z error where applicable,
   reference increments, measured velocity envelope, tracking error, joint-limit
   margin, torque, collision/workspace clearance and inference timing. Training
   success and self-collision reward are not safety proofs.

Do not select a hardware policy merely by nominal success. The current z-axis
nominal policy's measured-velocity result is an explicit blocker, while the
position DR policy lacks full deterministic evaluation. Return the immutable
bundle manifests, parity reports, fake/shadow results, reviewed suite YAMLs and a
remaining-blockers note before requesting separate authorization for motion.

## 2026-09-27 receiving-machine request: YZ-circle hardware demo

The selected hardware demo geometry is a **YZ circle**. For the first deployment,
use the 29D position-only policy; its target orientation is ignored. Do not wait
for or move hardware with the current 31D z-axis policy: that policy remains
blocked by its measured-velocity qualification result. Orientation-aware YZ
tracking is a later gate.

The current `circle_yz` radius of 0.15 m spans y `[-0.15,0.15]` and z
`[0.20,0.50]` m. Treat it as a simulation reference, not the first hardware
path. Produce and evaluate smaller immutable variants before selecting the
hardware catalog. Use 24 waypoints, `orientation_rpy_deg: [0,90,0]`, phase
180 degrees, threshold 0.01 m, and keep the first waypoint at
`[0.475,0,0.50]` m by choosing `center_z = 0.50 - radius`. At minimum compare:

| Candidate | Radius (m) | Center (m) | Y range (m) | Z range (m) |
| --- | ---: | --- | --- | --- |
| `circle_yz_r050` | 0.050 | `[0.475,0,0.450]` | [-0.050,0.050] | [0.400,0.500] |
| `circle_yz_r075` | 0.075 | `[0.475,0,0.425]` | [-0.075,0.075] | [0.350,0.500] |
| `circle_yz_r100` | 0.100 | `[0.475,0,0.400]` | [-0.100,0.100] | [0.300,0.500] |

The archived hardware home flange is approximately
`[0.30647,-0.00591,0.59069]` m, so the common first waypoint is still about
0.191 m away. Evaluate from the exact deployment home state. Sweep 1, 2, and 3 s
waypoint timeouts, or add a versioned lead-in path if the first-waypoint deadline
dominates. Do not change timeout semantics only on hardware.

The simulation workstation's next deliverable is:

1. Extend the exporter and produce immutable FR3 29D/6D bundles for
   `position_nominal` and `position_dr`, including deterministic tanh,
   source/checkpoint/config hashes and recorded PyTorch/ONNX parity vectors.
2. Run full deterministic point and YZ-path evaluation from the exact deployment
   home state for both policies, at K=50/100/200 and the existing nominal/DR
   evaluation scenarios. Model the same effective inference/action delay that
   will be measured by the hardware shadow runtime.
3. Evaluate the radius/timeout grid above. Report complete-path success,
   per-waypoint reached/timeout results, final/minimum error, action saturation,
   reference projection, measured-velocity envelope, tracking error, joint
   margins and torque metrics. Select the largest variant with adequate margins;
   do not select by success alone.
4. Add the selected variant to a versioned path catalog without mutating the
   original result, record its SHA-256, and generate simulator path-state-machine
   fixtures: resolved waypoints, transition policy steps, stale-result cases,
   persistent reference/previous-action state and final traversal outcome.
5. Return the bundles, manifests, parity/evaluation reports, selected catalog,
   fixtures and a concise remaining-blockers note. The receiving machine will
   then implement path execution, fake parity and read-only shadow inference
   before any separate request for motion authorization.

The original `circle_yz` at radius 0.15 m may remain a later expansion target.
The first hardware catalog must be chosen from simulation evidence and a separate
swept-workspace review.

## 2026-09-27 simulation-workstation implementation response

The requested 29D position-policy packaging and calibration infrastructure is
implemented. The transfer index is
deployment/config/fr3_position_policy_bundles_v1.yaml; its two final v2
bundles include deterministic tanh, versioned FR3 contract
fr3_incremental_position_29d_v1, checkpoint/config/source hashes, source
snapshots, and 1,024 PyTorch/CPU-ONNX parity vectors. Their manifest trust
anchors are:

- nominal model 149: 9c1a72d9b0430343572c5cfd3580fe1b52cb63f6092e844e73f39d11f3be12dd
- position-DR model 199: 34895a6f0e479e62a72f57678e820473d8f5a350255c1b68a9575700b6378788

The immutable YZ candidate grid is
deployment/path_catalogs/yz_circle_candidates_v1.yaml, SHA-256
b78b6fa5bbc788f471464c48edefee3d29c5ea0f4cab06c5a7f6d65cae067cae.
The evaluator and suite coordinator accept an explicit path_file, so the
catalog path/hash is preserved in artifacts. The nine-path coordinator input is
deployment/config/yz_circle_calibration_v1.yaml; exact-home K=50/100/200
nominal/DR scenarios are in
deployment/config/fr3_hardware_calibration_scenarios_v1.yaml. Point
qualification uses deployment/config/fr3_point_hardware_calibration_v1.yaml.
Artifacts now include measured-reference tracking error and soft-limit
reference projection in addition to the previously recorded path, velocity,
joint-margin, torque, and action-clipping metrics.

The measured effective shadow-runtime delay is still missing. The calibration
scenario file therefore declares a provisional zero-step delay. Replace this
with the measured value and rerun before treating results as final evidence.
No YZ radius is selected yet, and no motion is authorized. See
deployment/YZ_CIRCLE_CALIBRATION_STATUS.md for remaining blockers.

An end-to-end four-environment smoke run of circle_yz_r050_t1 at nominal K=100
validated the new artifact path. All four trials timed out only on the initial
0.191 m lead-in waypoint and then reached the other 23 waypoints; there were no
unsafe failures. This confirms that the 1 s first-waypoint deadline is a real
experimental factor, not a catalog or state-machine bug. It is pipeline evidence
only and does not select the hardware path.

## 2026-09-27 transfer-ready closure

The workstation phase is now wrapped for transfer. Use
deployment/FR3_REALTIME_TRANSFER_HANDOFF.md as the authoritative receiving
checklist. Transfer both complete v2 position-policy bundle archives (nominal
and DR), not standalone ONNX files. The document records archive, manifest, and
policy hashes and assigns all remaining fake/shadow/path/timing work to the
real-time machine. Raw PyTorch checkpoints stay on the training workstation.

The current 31D z-axis policy is explicitly excluded. Neither transferred
bundle is motion-authorized; receipt and parity verification do not constitute
approval to move the robot.
