# Hardware controller comparisons after M4 commissioning

2026-10-12. Implementation plan, not a new executable hardware configuration.
All five demo types have now been deployed according to the operator. Existing
executed session configs and logs remain the evidence for each tested profile.

## Objective and available implementations

Compare timed circle tracking and stationary EE / changing arm posture using:

1. Existing native Cartesian impedance, with Cartesian stiffness/damping sweeps.
2. Native resolved-rate Jacobian DLS followed by explicit joint PD, with joint
   stiffness/damping sweeps and a kinematic secondary-motion projector.

The pinned Franky source provides `JointImpedanceTrackingMotion`, joint position
and velocity references, Coriolis compensation and model Jacobians. Searches of
the local motion sources and bindings did not find a public ready-made DLS
Cartesian-reference-to-joint-PD controller. `CartesianMotion` wraps Ruckig
Cartesian waypoint generation and outputs `franka::CartesianPose`; it is not
an explicit joint PD implementation. Upstream describes the trajectory and
impedance APIs here: https://github.com/TimSchneider42/franky . Do not upgrade
Franky or use firmware Cartesian motion merely to obtain a differently named
baseline. Confirm any newly proposed upstream DLS API against its actual source
before substituting it.

Local code to reuse:

- Franky `src/motion/joint_impedance_base.cpp` and
  `joint_impedance_tracking_motion.cpp`: joint PD torque implementation.
- Franky `src/motion/m4_runtime_motion.cpp`: native lifecycle, model feedback,
  buffered telemetry, guards and stop-v2 behavior.
- `deployment/franky_runtime/run_m4_hardware.py` and
  `franky_experiment/motion_demos/hardware.py`: coordinator, measured-start
  planning, pose/twist publication and posture schedule.
- `source/franka_motion/franka_motion/kinematics.py`: portable DLS feasibility
  solver and kinematic projector. Its iterative IK is an offline planner,
  not a continuous tracking controller; port the mathematics, not its Python
  iteration loop onto the hardware control thread.
- `deployment/franky_runtime/summarize_m4.py`: shared phase metrics.

## Controller definitions

Use base-frame flange pose, full orientation and measured encoder/model FK for
both routes. Let e be target-minus-measured position and base-frame SO(3) rotation
vector; J is the full 6x7 measured Jacobian, and V is spatial twist.

Existing Cartesian route:

```
w = Kx e + Dx (V_ref - J dq)
tau_client = J.T w + N_tau tau_posture + tau_limit + c(q,dq)
```

Keep the audited SO(3) implementation and configurable dynamic/kinematic torque
projectors. Do not change the current production controller to simplify tests.

Proposed DLS route:

```
Jbar = S J                    # explicit translational/rotational scaling
Vcmd = S (V_ref + Gx e)
Jbar_dls = Jbar.T solve(Jbar Jbar.T + lambda^2 I, I)
Nq = I - pinv(Jbar) Jbar       # orthogonal SVD kinematic projector
dq_ref = Jbar_dls Vcmd + Nq Gposture (q_posture - q)
q_ref_next = q_ref + dt dq_ref
tau_client = Kq (q_ref_next - q) + Dq (dq_ref - dq) + tau_limit + c(q,dq)
```

Initialize q_ref from actual q and desired velocity to zero. Maintain the
integrated joint reference explicitly; never silently reanchor it every tick.
Apply joint-reference position, velocity and tracking-error checks. If reference
lag exceeds its reviewed bound, stop rather than allow uncontrolled integration.
Use actual dt and reject anomalous callback periods as the current runner does.

S sets a documented characteristic length for rotation-versus-translation
weighting; choose it from an offline conditioning audit, then hold it fixed.
Lambda, S, Gx and Gposture are outer-loop parameters, not joint PD gains. Record
them and keep them fixed during each inner-gain sweep. Report singular values
with the scaling convention used. Current unscaled guard remains separately
available for continuity with previous results.

Use the SVD pseudoinverse for the secondary projector with an explicit rank
tolerance. `I-Jbar_dls Jbar` is a damped residual operator and generally leaks
secondary motion into the Cartesian task; do not label it an exact nullspace
projector. Verify J Nq numerically at nonsingular test configurations.

Gposture times posture error is a joint velocity in the DLS route. The existing
Cartesian posture controller projects a joint torque. These are different
secondary laws; do not assign equal numeric gains or claim identical posture
control simply because both use a kinematic projector.

Do not enable additional Cartesian shaping inside the joint PD controller: that
would create a hybrid controller rather than the intended baseline. All seven
joints participate. No automatic joint-7 hold inherited from the earlier PPO.

For both routes: firmware supplies gravity and motor-friction compensation;
Franky adds model Coriolis. Additional client friction, integral action,
acceleration feedforward and inverse-dynamics/Jdot compensation remain disabled
for the first matched comparison. A later acceleration experiment must supply
the target explicitly and form a separate comparison.

## Native implementation and coordinator changes

Add a native `M4DlsJointRuntimeMotion` in the maintained Franky fork. Reuse the
existing joint PD implementation and fixed-size Eigen model algebra. Keep DLS,
reference integration and torque calculation in C++ at 1kHz. Python publishes
pose/twist/posture at the existing 50Hz and drains logs; no Python servo callback.

Extract shared M4 feedback, limit evaluation, ring logging and stop handling only
where needed. JointImpedanceBase currently slews internally; provide a raw-torque
observation hook analogous to the Cartesian base so stopping can blend and slew
once. Avoid accidentally nesting two slew limiters or adding model terms twice.
Regression-test the existing Cartesian route after any shared refactor.

Keep stop-v2's captured measured Cartesian pose and full-six-axis hold for both
routes, using the same fixed stop gains, blend and rest criteria. Stop freezes
the DLS reference integrator and ignores subsequent target updates. Share the
tested pose-hold torque implementation explicitly; do not treat a joint-PD
reference freeze as an equivalent stop. Stop and active gains are distinct.

Add controller selection `cartesian_impedance` / `dls_joint_pd` to reviewed local
configuration, plus explicit outer-loop and joint-gain fields. Preserve existing
configuration behavior and immutable bundle manifests. Record trial/profile ID,
repeat index, task and all actual gains. Same CLI for both routes; one session
per operator start, no automatic motion batch.

Preserve libfranka0.21.2, firmware, Ruckig, native dependency hashes and runtime
dependencies. Package a locally built additive Franky wheel in an isolated
comparison environment after offline tests; retain the tested stop-server10
environment for rollback. New native capability requires a new wheel, not a
libfranka upgrade.

## Matched experiment design

Use the executed successful circle tracking config from
`tracking.20261012T142902Z` and repeated `143526Z` as the nominal reference:
Cartesian translation K800/D56.5685, rotation K40/D13, posture gains zero.
Do not substitute the hand-guided circle fixture K1200 for this timed path.
Freeze the exact circle catalog geometry, orientation, angular speed, entry,
settle and exit schedule from those saved configs/bundle. Publish identical
references and phase timestamps for both controllers. Keep start posture near
the same tested home and record the actual start, rather than assume equality.

| Comparison | Initial matrix | What stays fixed |
|---|---|---|
| Cartesian stiffness | 0.5x, 1x, 1.5x nominal translation K; D scaled by sqrt(K factor) | rotation K40/D13, zero posture, trajectory |
| Cartesian damping | 0.75x, 1x, 1.25x nominal translation D | nominal K, rotation and trajectory |
| Joint PD stiffness | Kq=50,100,200 Nm/rad on all 7 joints; Dq follows sqrt(K factor) from the exact previously tested K100 profile | DLS lambda/scaling/outer gains, trajectory |
| Joint PD damping | 0.75x, 1x, 1.25x reviewed nominal Dq | selected Kq and all outer parameters |
| Stationary EE / posture | each route at selected nominal gains; optional low/high profiles | same Cartesian hold, same feasible posture A/B/return and timing |

These are proposed candidates, not blanket hardware approvals. Derive numeric
Dq vectors from the saved successful joint-impedance session configs, not guessed
critical-damping formulas. Joint gain100 and Cartesian gain800 are not equivalent
units or equivalent Cartesian stiffness; report each complete controller.

First screen profiles once; then repeat selected low/nominal/high configurations
three times. Alternate controller order within repeat blocks to reduce ordering
bias. Avoid a full stiffness-by-damping-by-task grid unless the initial results
justify it. Preserve all partial/faulted trials in the trial index.

For posture, reuse `posture.20261012T150500Z`: 0.9rad seed excursion, 6s
transitions, 4s settling, return enabled, feasible A/B generated from actual
start. Compare exact target vectors as well as the seed: the seed excursion is
not the realized joint excursion. Use the same feasible targets in paired runs.
Measure realized posture movement as well as EE error so barely moving cannot
win solely through small EE drift.

Run Cartesian posture with the kinematic projector as the primary comparison
against DLS's kinematic secondary motion. Retain dynamic Cartesian posture as a
separate projector ablation. In zero-posture circle tracking the torque projector
has no effect, so do not claim a projector comparison from that task.

## Logging, metrics and brief report

Extend existing 1kHz buffered logs with q_ref/dq_ref, Cartesian outer correction,
nullspace velocity, lambda, scaling, rank, reference-limit events, joint errors,
raw torque decomposition and applied gains. Record publisher phase transitions
and robot-time alignment rather than relying only on inferred phase boundaries.
For optional hands-off tests add a nonblocking event marker, not a keyboard
action that changes the torque target.

Extend summarize_m4 and add an aggregator driven by a trial-index CSV/JSON:

- Timed tracking: position/orientation RMS, p95 and maximum, analytic circle
  contour error, phase lag estimated against the commanded path, lap completion.
  Separate approach, settle, steady path, exit and stop metrics.
- Posture: EE translation/orientation disturbance, A/B joint-reference error,
  realized joint excursion, target attainment and settling time. State the
  settling threshold/dwell in the report; do not choose it after seeing winners.
- Both: joint speed, minimum joint/workspace margins, torque RMS/max with explicit
  client-versus-measured conventions, slew activity, communication timing,
  completion/fault reason and stop outcome.
- Aggregate per-trial summaries, then mean/spread across repeats; do not treat
  1kHz samples as independent experimental repetitions. Faults are outcomes,
  never silently excluded to improve a mean.

Create a brief Markdown report plus standalone PNG/PDF figures: overlaid YZ
paths, time-aligned error, gains-versus-error table/plot, and posture movement
versus EE disturbance. Include exact configs, runtime/bundle hashes and links
to source sessions. Separate the earlier exploratory tuning from the matched
comparison dataset. Encoder-FK accuracy is not external Cartesian metrology;
command torque is not a calibrated hand-force or actuator-energy measurement.

## Implementation order and verification

1. Freeze successful path/posture reference configs and build a trial index for
   historical runs. Add phase logging and multi-run report aggregation first;
   Cartesian gain comparisons can proceed with the existing native controller.
2. Add pure portable resolved-rate DLS equations and fixed-size native helpers.
   Test pose-error signs, scaled solve, J Nq residual, full orientation, joint
   reference initialization/integration and singularity/bounds failures. Use
   existing Jacobian/FK fixtures for Python/C++ parity.
3. Add the native joint route with shared guards/logging and fixed stop-v2 hold;
   test compensation exactly once, slew exactly once, fault-to-stop transitions
   and invariance of the existing Cartesian tests. Benchmark callback duration
   offline, including SVD near the guard.
4. Build isolated comparison wheel and verify all below-Franky dependency hashes
   against the tested stack. Generate candidate configs and validate offline.
5. Commission the DLS route with matched-start hold, small Cartesian movement,
   then the established circle and posture task. User starts each run; inspect
   the first result before progressing to larger gain candidates.
6. Run the selected matched comparisons and produce the report automatically
   from the indexed session directories.

No controller implementation, environment install, FCI access or robot motion
is performed as part of writing this plan.


## Implementation update

`deployment/franky_runtime/M4_COMPARISONS.md` for commands and current
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
