# Shared reference governor plan — 2026-09-25

Status: standalone governor 0.1.0 is implemented in this repo; see
[implementation and shipping instructions](reference_governor/README.md).
Native/Python tests and Isaac API-stub conformance run on the RT host. Isaac
runtime, ROS adapter, controller response and hardware commissioning are unverified.
The three controller choices below remain unchanged. Existing bundles are immutable.

## Controller options to retain

| Option | Policy output | Low-level controller | First-demo assessment |
| --- | --- | --- | --- |
| A: internal impedance | Joint-position targets at 30 Hz | Robot's existing internal joint-impedance controller, fed smooth 1 kHz position references | Selected first-deployment direction; preserves the user's default-settings decision. Identify its observable response. |
| B: explicit external PD | Joint-position targets at 30 Hz | Custom host-side 1 kHz impedance/PD law, sending torques through FCI | Retained alternative if explicit gains/control law become necessary. Requires separate controller, compensation and torque-limit validation. |
| C: learned torque policy | Joint torques from PPO | Policy plus torque command processing; robot retains underlying servo/protection/compensation | Retained research alternative. New training/action contract and timing/torque qualification; not the quickest first-demo path. |

Option B is not a torque-output PPO: its actor still selects positions. Shared
position-reference generation can serve A and B; C requires a different torque
command governor. Changing between these paths is a versioned design decision.

For A, preserve existing settings: no setJointImpedance, factory reset,
setDefaultBehavior or collision-setting changes. Numeric internal gains remain
unknown. Broad arbitrary PD randomization is not evidence of controller coverage.
Use a measured response surrogate and supported uncertainty ranges when available.
The selected path does not authorize motion or establish current factory defaults.

Evidence: [live inspection](hardware_control_audit/2026-09-25/LIVE_INSPECTION.md)
and [control audit](hardware_control_audit/2026-09-25/FINDINGS.md).

## Scope and ownership

The governor converts discontinuous policy position targets into continuous
position references subject to reviewed position/velocity/acceleration/jerk
constraints. It does not replace the impedance controller, estimate its gains,
certify collision clearance, or implement the entire experiment coordinator.

The canonical implementation is `franka-rl/deployment/reference_governor/`:
C++17 header core, pybind11 bindings, batched Python API, CMake installation,
wheel/sdist packaging and an opt-in Isaac action term. It has no ROS/libfranka
runtime dependency. Both machines consume this same version; the simulation
machine does not need access to franka_ros2.

The ROS repo will contain only the deployment adapter/controller linking
`franka_governor::core`. The shared C++ code must not be copied and edited there.
Ship with `scripts/deployment/build_governor_release.py`; its manifest pins source,
artifacts and config provenance. The wheel targets the matching Python/platform;
the source archive supports rebuilding elsewhere.

The opt-in `Franka-FR3v2-Governed-Reach-v0` task uses the package. Existing tasks
remain unchanged. A GPU equivalent is deferred until a measured throughput need
and trajectory/fault parity validation; the CPU binding is the correctness baseline.

## Contract to freeze before training

Inputs per reference tick:

- Coherent measured q/dq and robot desired q_d/dq_d/ddq_d, joint order 1–7.
- Robot timestamp, local monotonic receipt time and session identifier.
- Latest policy message: raw float32 action[7], monotonic sequence, observation
  sequence/time, inference completion time, and session identifier.
- Immutable model/config hashes and approved operating bounds.

Outputs: float64 q_ref/dq_ref/ddq_ref, accepted policy sequence, mapped target,
projected target, intervention reason/magnitude, lifecycle status and fault code.
Only q_ref is sent through option A's position interface. Log reference derivatives
for verification; do not substitute them for measured velocity observations.

Preserve `q_policy = q_default + 0.5 * raw_action` and the existing q_default.
Float32 mapping follows existing parity fixtures, then converts to float64 for
trajectory calculations. Validate dimensions and finiteness before arithmetic.
No raw-action clipping is added. Position target projection is separate and
visible; it does not change the stored raw action.

The actor's previous-action observation is its preceding valid raw output in the
policy sequence, not the clipped target, governed reference or measured motion.
Publish the action and sequence atomically. The first version permits one worker
and no overlapping inferences. Missed deadlines or unconsumed outputs have explicit
outcomes; do not let silent mailbox overwrites redefine previous-action semantics.
Simulation reproduces the same publication/consumption ordering. Terminal faults
end the episode/session; reset clears all policy/reference history together.

## Timing and update order

Reference timestep is exactly 1 ms in the mathematical core. Policy period is
exactly 1/30 s; a phase accumulator/rational schedule avoids treating 33 ms as
1/30 s. With zero simulated inference latency and next-tick consumption, reference
indices 0,34,67,100,... illustrate the 34/33/33 tick pattern. Real worker latency
changes arrival ticks and must be represented/logged, not hidden by this example.

At each RT tick:

1. Validate session, robot state, mode, timestamps and measured/reference tracking
   against configured readiness/freshness/error criteria.
2. Accept a new policy message only if sequence and source-state age are valid.
   Hold the most recent valid *target* between policy messages; continue advancing
   the reference every millisecond. This is not a zero-order hold of q_ref.
3. Map raw action to target; project target into approved joint-position bounds
   with margins. Record all intervention. Excess intervention ends the trial
   according to configured magnitude/duration limits, never silently forever.
4. Calculate admissible velocity envelopes from the archived robot model and
   tighter lab caps. Check both reference feasibility and measured-state guards.
5. Advance the jerk-constrained reference generator toward that target or toward
   a feasible stopping state, preserving position/velocity/acceleration continuity.
6. Verify the generated reference and derivatives before publishing q_ref. On
   infeasibility, enter the defined fault/stop path rather than clipping q_ref
   after generation and thereby breaking derivative constraints.

Keep the installed adapter's position filter/rate-limit switches off when the
shared governor owns these operations. Do not add a second filter implicitly.
No extra low-pass filter is planned initially; adding one changes the contract
and requires new parity and closed-loop evaluation. Firmware behavior remains
part of the response surrogate.

The deployment adapter measures actual cycle gaps. A missed/late cycle cannot be
repaired by issuing a burst of catch-up writes with fictitious 1 ms timing. Timing
thresholds and the transition to stopping/fault are required config fields. Test
this behavior with injected delays before deciding any allowed missed-cycle budget.

## Reference state and trajectory feasibility

Maintain explicit q_ref, dq_ref and ddq_ref as governor state. On activation,
initialize from coherent robot desired q_d/dq_d/ddq_d after readiness checks;
measured q/dq must also be within approved bounds/tracking tolerance. Do not jump
to training q_default or zero derivatives while the robot is moving. First-demo
activation requires an approved near-stationary state. This stateful design is a
proposal, distinct from the stock wrapper's optional limiter, which reads robot
desired state afresh each cycle.

During RUNNING, do not silently overwrite governor state with measured or desired
state: that can create reference discontinuities. Monitor discrepancy with robot
q_d/dq_d and measured q/dq; excessive mismatch invokes the fault policy. Any future
resynchronization needs a separately specified continuous transition and fixtures.
Simulation must expose equivalent reference/controller state for this check.

Algorithm selected and implemented: `bounded_quintic_v1`. Each accepted short
quintic segment preserves current q/dq/ddq and ends at rest. Bezier control-point
hulls bound position and its first three derivatives over the full continuous
segment. Position-dependent velocity bounds use the worst envelope over its
position hull. At most twelve endpoint candidates are checked; a failed replan
retains the previous validated continuation. Persistent rejection triggers stopping.

The cached segment is also the stopping continuation. This design replaces the
open minimum-distance braking choice in the initial plan: time to rest is bounded
by the configured horizon, and remaining reference displacement by twice the
configured segment excursion. It may be slower than another feasible planner.
It is not collision protection or a model of internal motor response. A deploying
lab must explicitly approve the resulting stopping distance and tracking margins.

Exact equations, lifecycle semantics, assumptions and tests are in the package
README. The compiled core is available; the full hardware adapter remains open.

## Bounds and configuration

Use the robot-returned fr3v2.1 joint limits and velocity-envelope parameters from
[the captured model](hardware_control_audit/2026-09-25/robot_returned_model.urdf).
The offline description parameters matched these fields. Preserve manufacturer
values, numerical tolerances and tighter lab operating limits as separate data.

Mandatory deployment config has no guessed numeric defaults for:

- Joint position margins and lab velocity/acceleration/jerk caps.
- Measured/desired/reference mismatch bounds and activation tolerances.
- Observation age, policy age, inference and RT timing budgets.
- Intervention magnitude/duration budgets and validated stopping constraints.
- Approved workspace/exclusions, run timeout and settling/hold criteria.

Missing values make an active configuration invalid. Synthetic offline test
values are explicitly labeled fixtures, never proposed commissioning thresholds.
Torque caps/rates belong to option B/C's torque layer; option A cannot guarantee
internal torque bounds by inspecting q_ref alone.

Joint bounds do not establish self-collision or obstacle clearance. The initial
suite needs an approved restricted region/start state and separate collision
checks. A non-RT workspace monitor is not a certified instantaneous stop system;
its latency and braking clearance must be considered explicitly.

## Lifecycle and failures

Implemented core states: DISARMED → RUNNING → STOPPING → TERMINAL, plus FAULT.
Initialization/readiness checks occur in explicit reset; ROS lifecycle readiness
remains the future adapter's responsibility.
FAULT is latched for the session; no message, restart or reset resumes that suite.
A new authorized session starts from DISARMED with new identifiers and cleared
history. Reaching success still requires a continuous transition to hold/stop,
not an abrupt controller switch. Moving to the trial start pose is a separate
validated trajectory, not setting the policy's default vector in one update.

Distinguish two failure classes:

- Policy stale/invalid while robot state and control channel remain healthy:
  latch the trial failure, reject later policy outputs, follow the already
  feasible bounded stopping continuation, then hold/finish according to the
  reviewed lifecycle. Do not keep chasing the last distant target indefinitely.
- Robot fault, state loss, network failure or loss of reference feasibility:
  do not assume a controlled stop remains deliverable. Latch fault, cease normal
  policy operation, use the reviewed hardware-stop/error path and require operator
  handling. Never invent current state or extrapolate indefinitely.

Integration must remove/bypass the stock startup automaticErrorRecovery+retry
path and review recovery services. No automatic recovery or reactivation occurs.
A smooth stop is still robot control and requires authorization when exercised
on hardware. Independent physical stopping remains a lab commissioning gate.

## Simulation integration and response model

First update FR3 link masses/COMs to the live model and rerun model checks.
Apply the exact shared governor before the simulated actuator/controller.
Expose governed target, reference derivatives and intervention/fault telemetry.
Keep raw previous action unchanged and preserve the existing bundle math.

The implemented reference adapter uses 1/3000 s physics steps and decimation 100,
with a 1 ms governor tick every three substeps; a 60 Hz simulator that jumps directly between policy targets
is not a deployment-equivalent test. Establish timestep convergence. A cheaper
training approximation may follow only after comparison with that baseline and
must be explicitly versioned with its observed discrepancies.

Option A uses an identified closed-loop response surrogate with compensation;
PD 80/4 remains a provisional comparison model until evidence supports it.
Do not use wide arbitrary PD DR to declare equivalence. Evaluate the frozen FR3
policies with the new governor before deciding on fine-tuning/retraining.
Include velocity and sustained-hold success criteria approved for the demo;
current five-sample position-only success is not a settling-time metric.

## Implementation status and remaining acceptance

Implemented: canonical core, complete-config loader, terminal state handling,
CMake and Python packaging, native/Python trajectory/fault tests, batched partial
reset, and opt-in adapter inspected against Isaac Lab v3.0.0-beta2.patch1 with
Isaac Sim 6.0.1. The adapter rejects backend-internal decimation and preserves
30 Hz policy timing using the 3 kHz physics clock. Its queued action is consumed
on the next governor step after submission; this explicit tick delay is shared
with the core and is not a measured inference delay.

The CPU path is for initial small-batch correctness validation. API-stub tests
are not simulator execution. No GPU implementation, ROS policy plugin, live
control qualification or identified firmware-response surrogate is included.

The original staged acceptance requirements remain the checklist below:

## Milestones and acceptance evidence

1. **Offline design spike:** choose the trajectory algorithm; implement pure core,
   config validator and state machine. Document state recurrence, feasibility and
   stop behavior. No ROS/hardware dependencies. Record unresolved assumptions.
2. **Shared fixtures:** zero/hold, steps both signs, online retargeting, reversals,
   near-bound stopping, nonzero initial velocity/acceleration, infeasible inputs,
   asymmetric limits, long holds and deterministic replay. Check q/dq/ddq and
   finite-difference jerk over complete trajectories, not just one-step outputs.
3. **Temporal/fault fixtures:** stale observations/actions, duplicate/out-of-order
   sequences, future timestamps, NaN/Inf, policy overruns, missed RT cycles, resets,
   session mismatch, excessive intervention, robot-state loss and terminal latching.
4. **Cross-language/platform parity:** native C++ and simulator adapter consume
   identical timestamped inputs and reproduce lifecycle events and references
   within declared tolerances. GPU implementation, if added, receives the same
   checks. Store small golden vectors; large traces stay outside Git.
5. **Integration without motion:** ROS fake hardware verifies interface ownership,
   first command continuity, no allocations/blocking work in the governor update,
   publication/consumption order, stop/fault handling and local YAML scheduling.
   Read-only shadow can validate observations but cannot establish actuation response.
6. **Response identification and simulation evaluation:** separately approved
   bounded hardware test preserves internal settings; fit/validate response on
   held-out data. Evaluate original policies and governor interventions on corrected
   simulation before new deployment-oriented training.
7. **Commissioning:** approved limits/procedure/physical stop, timing qualification,
   new FR3 export/C++ actor parity, and an explicitly authorized narrow first demo.

Benchmark bounded core execution on this RT host outside the hardware loop, then
measure integrated timing under authorized testing. A fast offline benchmark is
not proof of 1 kHz FCI reliability. Record code/config/model/policy hashes with
all results. This plan supplies no new motion authorization.
