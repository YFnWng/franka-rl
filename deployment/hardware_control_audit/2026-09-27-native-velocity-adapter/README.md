# Native velocity adapter hardware audit — 2026-09-27

## Decision

Do not rerun the nominal incremental-position policy through the native Franky
joint-velocity adapter. Session 2026092708 demonstrated that dividing the
policy's position increment by 20 ms preserves the numerical action scale but
does not preserve the trained actuator dynamics. Relaxing tracking error would
allow the mismatch and visible jitter to continue.

The adapter remains useful as an implementation experiment. A future hardware
run requires a policy trained for the native velocity/Ruckig route, or a small
open-loop velocity characterization before such training. The current nominal
position policy should use the already completed K=100 explicit-impedance route
if a repeat of the first demo is needed.

## Session outcome

- Config: `nominal.circle_yz.velocity-adapter.motion.2026092708.approved.yaml`
- Terminal state: `fault: tracking_error`
- Motion duration: 3.574 s
- Path progress: five waypoints reached; fault while approaching waypoint 6
- Exact crossing: J4 `abs(q - q_virtual) = 0.0820609 rad`, limit `0.08 rad`
- Smooth stop: successful
- Robot/current and last-motion errors: empty throughout
- Dropped samples: zero
- Control command success rate after startup: 1.0
- Control periods: every non-initial callback was exactly 1 ms

## Command and response evidence

The 174 policy targets reached the intended scale:

| Signal | J1 | J2 | J3 | J4 | J5 | J6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Max requested `abs(dq_ref)` rad/s | 0.4027 | 0.4170 | 0.3399 | 0.3255 | 0.4648 | 0.4965 |
| Max generated `abs(dq_d)` rad/s | 0.2563 | 0.4170 | 0.3374 | 0.2122 | 0.4538 | 0.4965 |
| Max generated `abs(ddq_d)` rad/s2 | 3.0 | 1.5 | 2.0 | 2.5 | 3.0 | 4.0 |
| Max `abs(q-q_virtual)` rad | 0.0374 | 0.0644 | 0.0447 | 0.0821 | 0.0736 | 0.0747 |

The generated acceleration stayed at the configured 20% FR3 limits, so the
tracking fault was not caused by Ruckig violating its configured acceleration
bounds. The actor nevertheless produced rapid velocity-target changes: over
3.57 s, J1--J6 changed sign 21, 16, 31, 26, 16, and 10 times. The largest
20 ms target changes corresponded to 10.68, 13.47, 11.08, 16.32, 3.86, and
5.82 rad/s2 before Ruckig smoothing. Measured acceleration p99 was higher than
the K=100 baseline on J1--J4, consistent with the reported jitter.

The cause is the actuator-contract mismatch. Training assumed each action
advanced a held position reference tracked by the explicit K=100/D=20 torque
impedance loop. Native velocity execution instead treats that increment as a
velocity target and uses Franky's Ruckig motion generator plus the firmware
joint-impedance controller. The virtual reference continues integrating even
while Ruckig and the physical arm lag it, and the actor was not trained on that
state transition.
