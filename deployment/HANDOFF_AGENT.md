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
