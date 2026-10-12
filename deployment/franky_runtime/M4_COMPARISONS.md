# Running M4 controller comparisons

Implemented 2026-10-12. Initial DLS hold commissioning exposed a startup-period bug; corrected locally, awaiting hardware retry.

## Runtime and trial semantics

Use `/home/chen-lab/yifan/venvs/franky-m4-comparison-server10/bin/python`.
It contains local Franky `2.0.1.dev66+gm4comparefix.libfranka.0.21.2` with native
DLS support and separate stop gains. All 21 bundled native dependencies are
byte-identical to `franky-m4-stop-server10`; the tested environment is retained.
Wheel and receipts: `/home/chen-lab/yifan/deployment_bundles/2026-10-12/franky-m4-comparison-startup-fix/`.

The existing run_m4_hardware.py now defaults to **three automatic trials after
one terminal start**, for both controller routes. Each trial performs live
preflight/planning, a complete task, and native smooth stop. Five seconds separate
trials; `stop` or Ctrl-C ends the session. Any fault, unsuccessful stop, logging
loss or failed next preflight prevents continuation. No automatic recovery or
homing. Tracking entry/exit starts from the measured pose; posture A/B targets
and held pose are frozen after the first trial. Repeat starts must remain within
0.1rad per joint, 20mm Cartesian position and0.1rad orientation of trial1
(for returning posture sequences, the shared intended return anchor is used).
These tolerances are explicit in comparison profiles; they are checked in
addition to existing robot guards. Hand-guided fixtures may fail the next live
geometry check if the operator leaves the robot at an unsuitable pose.

`--repeats 1` overrides the default for commissioning. Each output session has
`trial-01`, `trial-02`, `trial-03`, each with executed config, start state,
servo.jsonl, phase-events.jsonl, report.json and native stop evidence. The parent
session-report.json indexes completed/failed trials and requested repeat count.

## Candidate profiles and commissioning

22 candidates are under `config/comparisons/`; profiles.json indexes them.
They preserve the successful October12 circle/posture task contracts and bounds.
Cartesian translation K factors0.5/1/1.5 and D factors0.75/1/1.25 are separate
sweeps. Joint K50/100/200 uses recorded nominal K100/D20 as the damping reference;
joint damping factors0.75/1/1.25 form the independent damping sweep. Orientation
K40/D13 is fixed for Cartesian sweeps. Stationary EE posture includes a separate
Cartesian dynamic-projector ablation; primary comparisons use the kinematic
projector. Normal/tangent hand-guidance stiffness profiles are not timed tracking
profiles. Stop gains remain K[800,800,800,40,40,40], D[56.5685x3,13x3] for all
comparison profiles and are independent of active gains.

DLS defaults: base-frame full-pose error gain2/s, rotation scale0.2m,
lambda0.01, SVD rank tolerance1e-6, posture-rate gain0.5/s for posture (zero
for circle/hold). The orthogonal secondary projector is built from the SVD
pseudoinverse, not the damped inverse. Reference velocity is scaled uniformly to
joint velocity bounds; integrated q_ref must remain within soft bounds and within
0.15rad of measured q. Violation faults19 and enters the existing stop. These
outer-loop candidates need commissioning; prior joint-PD tests alone do not
validate the composed DLS controller. Model Coriolis, firmware gravity/motor
friction, 1Nm/tick slew and100Hz FCI filter follow the tested pipeline.

First inspect the DLS hold with FCI preflight (no motion):

```bash
PY=/home/chen-lab/yifan/venvs/franky-m4-comparison-server10/bin/python
RT=/home/chen-lab/yifan/franka-rl/deployment/franky_runtime
"$PY" "$RT/run_m4_hardware.py" \
  --config "$RT/config/comparisons/joint.hold.k100.d1.pending.yaml" --preflight
```

Select an executable copy locally; this explicit command records the operator's
selection rather than modifying the immutable simulation bundle:

```bash
"$PY" "$RT/prepare_m4_comparisons.py" --approve-profile joint.hold.k100.d1
"$PY" "$RT/run_m4_hardware.py" \
  --config "$RT/config/comparisons/joint.hold.k100.d1.approved.yaml" \
  --execute --repeats 1
```

After inspecting the hold, commission joint.circle.k100.d1 with one trial, then
its three-trial session. Use the same profile-selection command for each desired
candidate, followed by --execute with no repeats override. Existing approved
copies are never overwritten by the selector. Screen one profile at a time;
this implementation does not automatically sweep gains or start a new profile
without a new operator start. Frozen feasible posture targets apply within a
session; across paired posture sessions the configs reuse the exact successful150500Z
posture-plan.json. The live planner verifies all target FK/bounds/singularities
and requires initial joints within0.15rad of its tested anchor; live initial pose
must also meet the pose-error guards. Each executed plan records the shared
source hash. Small start differences are recorded, not assumed identical.

## Analysis and report

For a three-trial session, or several matched profile sessions:

```bash
"$PY" "$RT/report_m4_comparisons.py" /absolute/path/to/session \
  --output-dir /absolute/path/to/new-report --plots-python /usr/bin/python3
```

Outputs: report.md, trial-index.json, trials.csv, paths-and-error.png/.pdf and profile-errors.png/.pdf.
Per-trial m4-metrics.json/.csv is regenerated with the current analyzer. Applied
reference sequence selects recorded phases; old sessions fall back to inferred
robot timing. Tracking reports isolate the track phase. Posture metrics include
realized excursion, goal error and settling attainment (joint error norm0.05rad,
position10mm, orientation0.05rad, sustained0.5s). Thresholds are stated, not fit
retrospectively to each result. Circle metrics include contour error and a
geometric phase-delay estimate; that estimate is not a complete system delay
identification. It is computed only while reference angular speed exceeds0.01rad/s.

Faulted trials remain indexed; means/spread use completed trials and treat trials,
not servo samples, as repetitions. Old unlabelled tuning runs stay separately
labelled exploratory. An exercised historical baseline report is at
`/home/chen-lab/franka_ros2_ws/hardware_inventory/2026-10-12/m4/comparison-baseline-report/report.md`.

## Verification and limits

42 Python tests passed, including native/portable DLS parity, fault persistence,
repeat cancellation, phase sequence alignment and aggregation with failures.
Native tests: 22 cases /75 assertions passed. Native constructor validation was
performed without a robot. The DLS helper including Python binding overhead
measured median11.174us/p9511.826us in an offline benchmark; this is not the full
hardware callback or a real-time deadline guarantee. Confirm actual timing and
physical behavior in the commissioning run. No new hardware result is claimed.

### DLS commissioning startup correction (2026-10-12)

The hold trial `hold.20261012T190454.670383Z/trial-01` faulted on callback 1 because libfranka supplies an initial zero elapsed period, which the DLS implementation incorrectly rejected. The fixed comparison wheel accepts zero only on callback 1 and integrates no reference displacement on that callback. Later zero, negative, nonfinite and excessive periods remain rejected. Gains, limits, stop behavior and all bundled native dependencies are unchanged. Existing comparison configs now require dev66; historical run configs remain unchanged. Smooth stop completed in the failed trial (stop fault 0, no dropped samples). Retry commissioning with `--repeats 1` before the three-repeat comparisons.

### Repeated command-success fault during Cartesian baseline

Sessions `tracking.20261012T191434.905431Z` and `tracking.20261012T191631.743364Z` reached 94% FCI command success against the 95% guard, at 28.179s in tracking and 59.817s in exit respectively. Both completed smooth stop without dropped logs. Native threads had FIFO priority 99. The retry completed the timed circle segment. Post-run host inspection found CPU governor `powersave` and no Ethernet interface errors/drops; these counters do not exclude command latency. Root cause remains unconfirmed. Test the `performance` CPU governor before further sweeps; preserve gain and guard settings. If recurring, collect NIC EEE/coalescing and callback computation/transmit timing evidence.

The investigation of the subsequent K0.5 communication reflex is recorded in [M4_COMMUNICATION_INVESTIGATION.md](M4_COMMUNICATION_INVESTIGATION.md). All CPUs were already on performance, but Ethernet EEE was enabled and active. Disable EEE as a separate controlled test before further comparisons; root cause is not yet established.

## Reduced posture comparison after circle sweep

The operator chose to omit the full posture gain sweep and use circle-selected gains. See [M4_POSTURE_SELECTED.md](M4_POSTURE_SELECTED.md) for the ten-profile circle table, exact source trials and three new pending configs. Cartesian translation K1200/D69.282 (orientation K40/D13) and joint K200/D28.284 were selected provisionally by lowest circle-segment position RMS among latest completed representatives. Run `cartesian.posture.selected.kinematic`, `cartesian.posture.selected.dynamic`, and `joint.posture.selected.kinematic`, each with one trial. Shared feasible A/B/return targets, all guards and separate stop gains are preserved. No posture hardware execution performed during preparation.

## Completed posture results

See [M4_POSTURE_AUDIT.md](M4_POSTURE_AUDIT.md). Cartesian kinematic/dynamic completed; selected joint K200 faulted at the J7 torque guard, while fallback K100/D20 completed. Use K100 for the completed posture comparison. Session completion did not imply all posture settling criteria passed. Historical selected configs and failed trial remain preserved.
