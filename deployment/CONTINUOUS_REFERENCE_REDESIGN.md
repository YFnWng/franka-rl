> **Status (2026-09-27): deferred.** The next experiment delegates continuous
> velocity trajectory generation to Franky's native `JointVelocityMotion` and
> Ruckig preemption. This custom 1 kHz governor design remains an alternative
> only if the native route cannot meet the measured deployment contract.

# Continuous reference redesign for path tracking

## Decision

The current 50 Hz incremental policy and held position reference completed the
hardware circle, but it is not the reference contract for the next training
generation. It is retained only as the frozen first-demo baseline.

For continuous path tracking, each policy action should specify a desired joint
reference velocity over the next policy interval. A 1 kHz stateful governor
must integrate a continuous position, velocity, and acceleration reference and
send both q_ref and dq_ref to the Franky impedance controller.

The existing bounded_quintic_v1 governor cannot be inserted unchanged. Every
accepted segment ends at zero velocity and acceleration, which is useful for
point-to-point stopping but recreates a start/stop pulse at every policy update.

## Hardware evidence

Both successful runs completed 24/24 circle_yz waypoints without robot errors,
sample loss, timeout, or stop failure.

| Profile | WP0 J5/J6 peak velocity | WP0 J5/J6 p99 acceleration | WP0 J5/J6 peak tracking error |
| --- | --- | --- | --- |
| K=200/D=28.284, session 2026092705 | 0.967 / 1.070 rad/s | 60.9 / 63.6 rad/s2 | 0.072 / 0.078 rad |
| K=100/D=20, session 2026092706 | 0.436 / 0.496 rad/s | 7.09 / 7.26 rad/s2 | 0.095 / 0.108 rad |

K=200 tracks the step reference more tightly but converts its edges into large
accelerations. K=100 is visibly smoother but accepts more lag. The command
contract, rather than controller gain alone, causes this tradeoff.

## Version-2 command contract

Policy rate remains 50 Hz and the impedance/physics rate remains 1 kHz.

For controlled joints 1-6, interpret actor output as

    v_target[k] = action[k] * v_max

where v_max remains [0.435, 0.435, 0.435, 0.435, 0.522, 0.522] rad/s.
This is numerically equivalent to the present maximum 20 ms increment but no
longer asks the low-level controller to settle at every integrated waypoint.
Joint 7 remains held by a separately continuous stationary reference.

The governor state is q_ref, dq_ref, and ddq_ref. At every 1 ms tick it advances
that state toward the held v_target subject to:

- soft position bounds and position-dependent braking envelopes;
- reviewed velocity, acceleration, and jerk bounds;
- continuity of q_ref, dq_ref, and ddq_ref;
- measured tracking, timing, watchdog, and projection faults;
- a bounded jerk-limited transition to zero velocity on stale policy input or
  normal stop.

The impedance law must consume the generated velocity:

    tau = K * (q_ref - q) + D * (dq_ref - dq) + coriolis + limit terms

The current runtime instead sends a new q_ref every 20 ms with dq_ref=0. That
zero desired velocity damps motion toward every intermediate waypoint and then
repeats at the next step.

## Timing and observations

Preserve the measured one-policy-step action delay before the governor input.
Do not add another implicit policy step. The observation keeps measured q and
dq, Cartesian target error, normalized q_ref, and previous raw action. Add
dq_ref and ddq_ref only if experiments show the partially observed governor
state harms learning; if added, version the observation and export contract.

The raw previous-action field remains the actor's preceding output. It is not
the governed velocity or a projected command.

## Implementation work

1. Add a continuous_velocity_governor_v2 core shared by Isaac Lab and hardware.
   Reuse the existing lifecycle, timestamp, position-envelope, and fault code,
   but replace rest-to-rest quintic segments with continuous velocity tracking.
2. Add an Isaac Lab action term that advances the governor at every 1 ms physics
   step and passes q_ref/dq_ref to the explicit impedance actuator.
3. Update the Franky runtime callback to advance the same core at 1 kHz and call
   JointReference(q=q_ref, dq=dq_ref). Policy threads publish only v_target.
4. Log v_target, q_ref, dq_ref, ddq_ref, intervention, and governor state at
   1 kHz. Never fill derivative columns with zero placeholders.
5. Train and evaluate nominal K=100 first, then repeat K=50/100/200 robustness
   evaluation with the same fixed delay and circle_yz path.
6. Commission with a small reference replay before PPO motion, followed by the
   existing initial-approach and one-circle sequence.

## Acceptance checks

- q_ref is continuous and has no 50 Hz jumps.
- dq_ref and ddq_ref remain within their reviewed bounds and do not reset to
  zero at ordinary policy boundaries.
- Simulation and hardware use identical update order, delay, integrator, and
  limit values, pinned by source/config hashes.
- The nominal policy completes 24/24 circle waypoints.
- Initial J5/J6 acceleration is materially below the K=100 staircase baseline
  while tracking error remains within its reviewed limit.
- There are no robot errors, dropped samples, watchdog events, projection
  persistence, or stop failures.
