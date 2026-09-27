# Hardware deployment plan for incremental FR3 policies

Status: nominal model 149 selected for staged commissioning; this document does
not authorize robot motion. The final selection in
`FR3_REALTIME_TRANSFER_HANDOFF.md` supersedes older candidate-grid guidance
retained below as history.

## 1. Frozen deployment contract

The source contract is Git revision
`612363a279c9970603c03bfcc779c524457ffd3c`. The repository was clean when
this plan was prepared. The hardware implementation must reproduce:

- FR3v2.1, bare `fr3_flange`, identity `F_T_EE`, zero configured load;
- 1 kHz Franky `JointImpedanceTrackingMotion`, K=100 Nm/rad and D=20 Nms/rad;
- deterministic PPO at 50 Hz;
- six bounded increments for joints 1--6 and a held joint-7 reference;
- a persistent joint-reference integrator initialized from measured start `q`;
- 29D position observation or versioned 31D position-plus-z-axis observation;
- direct 20 ms reference hold with zero desired velocity and no Ruckig,
  interpolation, low-pass action filter, or reference governor;
- local immutable bundle, path catalog, and suite YAML, with no connection to
  the training workstation during a run.

The exact formulas, waypoint state machine, and candidate checkpoint hashes are
in `HANDOFF_AGENT.md`. Limits are indexed by `LIMITS_AND_TIMING.md`.

## 2. Candidate policy decision

Use the policies as follows:

1. **Selected first motion:** `position_nominal`,
   checkpoint SHA-256
   `7568e71a26e981eeefca50f808f2bfd526c42a900a48884071e0afa54318ca37`.
2. **Not selected; provenance/comparison only:** `position_dr`,
   checkpoint SHA-256
   `a0f844c0fc702f79b6c3291c67eaf2294ce2750aaf4af9cc190089537d85420b`.
3. **Blocked from motion:** `position_z_axis_nominal`. Its deterministic
   evaluation exceeded the 20% measured-velocity qualification envelope in
   57.8% of episodes.
4. **Rejected candidate:** the current z-axis DR checkpoint. Its deterministic
   success rate is too low for deployment.

Both FR3 bundles are verified and read-only on the real-time machine. The
nominal v2 bundle manifest SHA-256 is
`9c1a72d9b0430343572c5cfd3580fe1b52cb63f6092e844e73f39d11f3be12dd`.
Older Panda handoff policies must not be loaded by the FR3 runtime.

## 3. Current receiving-machine implementation status

Already implemented:

- long-lived Franky torque-mode controller and torque stop;
- 1 kHz callback logging;
- 50 Hz PPO configuration validation;
- six-action incremental integration with soft-limit projection;
- measured-start reference initialization and joint-7 hold;
- 29D position observation assembly;
- bundle manifest trust anchor and worker-process isolation;
- stale point-target result rejection, sticky faults, operator start, watchdog,
  and fail-closed hardware configuration;
- 11 passing coordinator unit tests.

Still blocking any policy motion:

- the checked-in exporter still supports only the old Panda contract;
- the three candidate checkpoints and new immutable FR3 bundles have not been
  transferred to this machine;
- the runtime supports point-target suites only, not `ppo.path`;
- there is no standalone strict path-catalog loader in the deployment runtime;
- there is no path traversal state machine, path artifact summary, or
  simulator/runtime path parity fixture;
- there is no true hardware shadow mode that reads state and runs inference
  without opening a torque-control session;
- the 31D z-axis contract, flange rotation snapshot, and z-axis metric logging
  are absent;
- target/path workspace validation is absent;
- inference latency and the effective observation-to-command delay have not
  been qualified at 50 Hz;
- runtime documentation still contains historical 24D/7D/30 Hz text.

## 4. Phase A: produce and transfer immutable FR3 bundles

This work is performed on the simulation workstation.

1. Extend `deployment/franka_policy_bundle` with separate contract IDs:
   - `fr3_franky_incremental_6d_joint_impedance_v1`: 29 inputs, 6 outputs;
   - `fr3_franky_incremental_6d_position_z_axis_v1`: 31 inputs, 6 outputs.
2. Export the deterministic actor with the final tanh included. Reject Gaussian
   sampling, normalization, wrong tensor dimensions, non-finite values, and
   actions outside `[-1,1]`.
3. Put the exact action integrator parameters, soft bounds, joint/frame order,
   policy period, default pose, observation layout, controller profile, source
   revision, checkpoint hash, training configuration hashes, and ONNX/runtime
   versions in the manifest and policy contract.
4. Generate both seeded synthetic vectors and recorded rollout vectors. Include
   pre-integration reference, preceding action, measured state, target, flange
   pose, expected observation, actor action, and post-integration reference.
5. Run complete deterministic evaluation before transfer:
   - point and YZ-circle evaluation for `position_nominal`;
   - the same evaluation matrix for `position_dr`;
   - K=50/100/200, nominal and randomized physics, identical seeds;
   - reference increments, projection rate, measured velocity-envelope ratio,
     tracking error, torque, waypoint results, and inference-independent task
     metrics.
6. Package the exact revision `612363a...`, each bundle, evaluation reports,
   path catalog, and SHA-256 inventory. Transfer out of band and verify every
   hash on this computer.

Gate A passes only when the local worker verifies the new manifest and all
PyTorch-to-ONNX vectors without importing Isaac or PyTorch.

## 5. Phase B: complete the local runtime

### 5.1 Versioned bundle dispatch

Validate the explicit contract ID rather than guessing from tensor size or file
name. Keep the old Panda verifier separate. For the 29D contract, require the
five observation slices and six-action increment mapping exactly. Add the 31D
branch only after the position runtime passes shadow testing.

### 5.2 Point and path suite schema

Change PPO YAML to require exactly one of:

- `ppo.targets`: existing point-target suite; or
- `ppo.path`: `catalog_path`, `catalog_sha256`, `name`, and
  `repetitions`.

Copy or move the pure-Python `PathCatalog` implementation into a
simulator-independent deployment module and test it against the source utility.
Record the catalog bytes, trusted hash, version, normalized quaternion, complete
resolved `PathSpec`, and generated waypoints in every run directory.

Before opening control, validate every waypoint against an explicitly reviewed
hardware workspace. The training box is evidence about policy exposure, not
hardware approval.

### 5.3 Exact path state machine

Implement the 50 Hz state machine from `HANDOFF_AGENT.md`:

- retain integrated reference, previous action, and held J7 across waypoints;
- use one-sample position threshold for waypoint advancement;
- use exactly `ceil(timeout / 0.020)` samples;
- continue after waypoint timeout but keep independent safety faults terminal;
- tag inference requests with traversal and waypoint generations;
- discard every result from an earlier generation;
- complete only after all waypoints resolve and mark success only if all were
  reached.

Log per-waypoint final/minimum error, elapsed steps, transition reason and policy
sequence, plus the full traversal summary.

### 5.4 Shadow mode

Add `execution_context: shadow` as a distinct fail-closed mode:

- connect read-only, verify robot/inventory, and never call `Robot.move`;
- initialize the virtual held reference from measured `q`;
- run the exact 50 Hz observation, inference, integration, projection and path
  state machine;
- log the references that would have been sent;
- preserve the explicit operator start/stop gate;
- reject any code path that can construct or submit a motion in shadow mode.

Measure observation-to-inference and observation-to-reference latency. The
pending 15 ms inference deadline must be demonstrated under sustained local
logging load. Record p50/p95/p99/max and all missed periods. Also record whether
the current asynchronous worker produces a consistent one-policy-step delay;
the simulator evaluation must model the same delay before active deployment.

### 5.5 Z-axis runtime

Defer this until position-only commissioning succeeds. It requires:

- base-to-flange rotation in `Snapshot` and 1 kHz records;
- exact two-component tip-frame z-axis error including the antiparallel fallback;
- a separate 31D contract ID and observation golden vectors;
- final, minimum, and integrated z-axis error metrics;
- invariance tests for arbitrary target twist about z.

## 6. Phase C: offline acceptance

Required automated evidence:

1. Bundle verification and manifest/hash failure tests.
2. Golden 29D observation and six-action integration parity.
3. Soft-bound projection and held-J7 parity.
4. Catalog geometry/hash and invalid-input tests.
5. Fake-backend `circle_xy` and `circle_yz` traversals matching simulator
   waypoint indices, transition steps and outcomes.
6. Delayed-result tests across waypoint transitions.
7. Timeout, watchdog, callback loss, writer failure, non-finite state/action,
   wrong contract, and sticky-fault/clean-stop tests.
8. Complete artifact-schema verification.

The current coordinator baseline is 11 passing tests. Path and shadow additions
must extend that baseline; they may not replace existing point-suite coverage.

Gate C passes only with no FCI motion and reproducible artifacts from a clean
checkout.

## 7. Phase D: path and workspace qualification

The active catalog hash is
`39455c82dfec508cdac0d26befda5a701501006c3758d1c115adc868a1ff6c69`.
The earlier physical-clearance evidence used hash
`e88676a68e38d44c6c62e27bc9f91ae42301033d189796fda506437f595764b3`;
geometry is unchanged, but the operator must re-acknowledge the active hash and
2 s initial timeout.

Its envelopes are:

| Path | Cartesian envelope, m | First waypoint | Distance from archived home flange |
|---|---|---|---:|
| `circle_xy` | x [0.400,0.550], y [-0.075,0.075], z 0.350 | [0.550,0,0.350] | 0.342 m |
| `circle_yz` | x 0.475, y [-0.150,0.150], z [0.200,0.500] | [0.475,0,0.500] | 0.191 m |

The archived home flange was approximately
`[0.30647,-0.00591,0.59069]` m. The selected path therefore gives waypoint 0 a
2 s timeout and later waypoints 1 s. Fixed-delay simulation qualified the
nominal position policy on the original 0.15 m `circle_yz`; the operator has
cleared that geometry. Freeze and approve the active catalog hash rather than
editing the catalog on the robot. Hardware path YAMLs must set
`abort_on_timeout: true`; any missed waypoint causes a sticky fault and the
normal torque-stop sequence instead of advancing. The orientation-aware z-axis
policy remains blocked.

Workspace review must cover the complete flange path and intervening motion,
self-collision, table/base/fixture clearance, cable clearance, and the fact that
the flange is bare. Record the approved catalog hash in the suite.

## 8. Phase E: shadow qualification on the robot

Run with FCI available and no motion:

1. Reconfirm bare flange, zero load, identity transforms, robot/server versions,
   E-stop access, real-time scheduling and memory locking.
2. Run zero/recorded golden observations through the local worker.
3. Run at least ten minutes of 50 Hz shadow inference for each position policy.
4. Run complete shadow traversals for the reviewed point suite and path catalog.
5. Compare observations, actions, integrated references, waypoint transitions,
   stale-result decisions and summaries against workstation fixtures.
6. Inspect action saturation, soft-limit projection frequency, hypothetical
   tracking-reference motion, latency distribution and worker failures.

Gate E requires zero missed inference deadlines, zero non-finite/bound errors,
exact fixture parity, and signed review of the generated references. Shadow
success does not authorize motion.

## 9. Phase F: staged active commissioning

Use `position_nominal` first. Every stage gets a separate immutable YAML,
operator authorization, run directory, and post-run review.

1. **Controller hold:** initialize reference from measured `q`, apply no policy
   increment, and verify the already characterized K=100/D=20 hold and stop.
2. **Near-home point:** run one simulation-qualified target near the current
   flange pose and inside the policy training box. Use one repetition.
3. **Small point suite:** add several reviewed points with modest displacement;
   return to the validated start procedure between suites.
4. **Initial path approach:** run the reviewed 2 s approach to waypoint 0 and
   stop for review before authorizing the full traversal.
5. **Qualified YZ circle:** run the selected 0.15 m `circle_yz` once, review it,
   then repeat only under another authorized YAML. Use the 29D nominal position
   policy; defer orientation control.
6. **DR comparison:** only after its full deterministic and shadow gates pass,
   repeat the identical point/path suites from equivalent measured start states.

At every stage inspect robot errors, command success, callback timing, inference
age, projection count, joint margins, measured velocity-envelope ratio, tracking
error, torque and torque slew, Cartesian clearance, waypoint outcomes and stop
completion. Any independent fault terminates the suite without recovery or
automatic resume.

## 10. Go/no-go decisions

First active point motion requires all of A--E plus:

- immutable `position_nominal` FR3 bundle verified locally;
- exact target and workspace approval;
- exact start pose achieved by a separately validated procedure;
- shadow references reviewed;
- operator at the E-stop;
- no unresolved controller, timing, frame, load, or transform mismatch.

First path motion additionally requires path-state-machine parity, exact-home
simulation evaluation, approved catalog hash and full swept-workspace review.

Z-axis motion additionally requires a replacement or retrained policy whose
deterministic evaluation satisfies the velocity qualification, plus the complete
31D runtime and rotation parity. The current z-axis candidates are no-go.

## 11. Deliverables back to the simulation workstation

Return, without live telemetry during motion:

- receiving source revision and clean/dirty state;
- bundle, manifest, suite and catalog hashes;
- bundle verification and golden-vector reports;
- fake and shadow parity fixtures;
- latency and scheduler qualification;
- resolved suite configuration and preflight inventory;
- 50 Hz observations/actions/references and 1 kHz controller/state logs;
- per-point and per-waypoint summaries;
- velocity, tracking, torque, projection, clearance and fault findings;
- explicit remaining blockers and the decision for the next gate.
