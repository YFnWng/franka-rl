# Velocity-impedance gain comparison

Sessions 2026092715 and 2026092717 ran the same model-149 velocity policy and
the same `circle_yz` schedule. The only intended controller change was from
K=100 Nm/rad, D=20 Nms/rad to K=200 Nm/rad, D=40 Nms/rad on all seven joints.
Both sessions completed 22.574 seconds, recorded 22,575 samples without drops,
reported no robot or last-motion errors, and stopped cleanly.

| Metric | K100/D20 | K200/D40 |
|---|---:|---:|
| Waypoints reached / 24 | 7 | 6 |
| Mean final waypoint error | 23.39 mm | 24.34 mm |
| Mean geometric circle error | 24.25 mm | 24.02 mm |
| Geometric circle error p95 | 56.03 mm | 51.71 mm |
| Maximum `abs(q-q_ref)` | 0.0362 rad | 0.0228 rad |
| Maximum measured velocity | 0.417 rad/s | 1.172 rad/s |
| Estimated acceleration p99 | 3.05 rad/s^2 | 27.83 rad/s^2 |
| Maximum commanded torque | 6.20 Nm | 11.03 Nm |

K200/D40 reduced maximum joint-reference lag by about 37%. That improvement did
not translate into better strict path completion: it reached one fewer waypoint,
and its mean final waypoint error was slightly larger. Mean geometric circle
error was essentially unchanged; only its p95 improved modestly.

The cost was a much sharper initial response. K200/D40 reached 1.172 rad/s on
joint 6 and an estimated 102.2 rad/s^2 peak acceleration. Its p99 estimated
acceleration was about nine times the K100/D20 value, and maximum commanded
torque rose from 6.20 to 11.03 Nm. The largest K200/D40 transients occurred on
joint 6 during the initial approach, around 1.0--1.2 seconds into the run.
Acceleration here is a finite-difference estimate from measured velocity and
host timestamps, but the distribution-level increase is too large to dismiss
as a single timestamp artifact.

Use K100/D20 as the nominal hardware-demo profile. Keep K200/D40 only as a
recorded comparison case and do not increase gains further. The remaining path
error is dominated by policy/path behavior and waypoint timing rather than a
lack of controller stiffness.

Machine-readable metrics and artifact hashes are in `comparison.json`.
