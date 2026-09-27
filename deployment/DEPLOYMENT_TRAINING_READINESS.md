# Deployment-level PPO readiness

## Current decision

The selected first hardware-demo policy is the transferred 29D/6D
position-only nominal model 149. The command contract is a 50 Hz,
six-joint bounded position-reference increment feeding the 1 kHz
`JointImpedanceTrackingMotion` at K=100/D=20, with joint 7 held.

The selected task is the existing `circle_yz` path in
`source/franka_rl/franka_rl/config/paths.yaml`: a 0.15 m radius vertical circle
with a 2 s first-waypoint timeout and 1 s subsequent timeout. The policy is
selected, but the exact hardware-motion YAML is not yet authorized.

## Candidate status

| Candidate | Status |
|---|---|
| `position_nominal` | **Selected:** bundle/shadow passed and final fixed-delay K=100 nominal/randomized path runs were 16/16. |
| `position_dr` | Not selected: bundle/shadow passed, but final strict path success was 0/16 in all six scenarios. |
| `position_z_axis_nominal` | Blocked by measured-velocity qualification and excluded from this deployment. |
| z-axis DR | Rejected because deterministic success was poor. |

The accepted bundles are read-only under
`/home/chen-lab/yifan/deployment_bundles/2026-09-27/`. Archive, manifest,
policy, parity, and fake-runtime evidence is in
`hardware_control_audit/2026-09-27-bundle-acceptance/`.

## Implemented on the real-time machine

- Long-lived Franky joint-impedance torque controller and torque stop.
- 50 Hz PPO timing with a separate verified ONNX worker.
- Six-action incremental reference, measured-start initialization, soft-limit
  projection, preceding-action state, and held joint 7.
- Strict SHA-anchored point/path exclusivity, resolved YZ geometry, reviewed
  workspace bounds, per-policy-step waypoint transitions, repeated traversal,
  and stale-result rejection.
- Read-only FCI shadow backend with no motion or keepalive operation.
- Encoder-based EE feedback: measured joint `q` is passed to pinned libfranka
  `Model.pose(Frame.Flange, ...)` to compute `fr3_link0 -> fr3_flange`.
  `O_T_EE` is audit-only and does not drive observations or path results.
- Explicit EE-feedback position, quaternion, source/frames, and
  FK-versus-reported discrepancy in events and 1 kHz hardware logs.
- Shadow latency/reset/held-joint analyzer and fail-closed configuration schema.
- Nineteen passing offline coordinator tests.
- Both transferred policies pass bounded end-to-end fake runs with clean stops.
- Both policies pass real-FCI, no-motion shadow runs with zero drops and clean
  stops; the measured hardware-matched delay is one 50 Hz policy step.
- The operator confirmed physical clearance for the exact checked-in
  `circle_yz` geometry; changes to the path require a new review.

## Remaining blockers

1. Synchronize and verify the nominal bundle, updated path catalog hash, and
   final calibration evidence on the real-time machine.
2. Re-acknowledge the unchanged cleared geometry with the 2 s initial timeout.
3. Obtain separate authorization for the exact immutable hardware-motion YAML.
4. Defer the 31D z-axis runtime until a replacement policy clears its simulation
   velocity gate.

## Fixed references

- Transfer and receiving checklist: `FR3_REALTIME_TRANSFER_HANDOFF.md`.
- Policy and path contract: `HANDOFF_AGENT.md`.
- Limits and timing: `LIMITS_AND_TIMING.md`.
- Hardware execution plan: `HARDWARE_DEPLOYMENT_PLAN.md`.
- Runtime and shadow procedure: `franky_runtime/README.md`.
- Receiving evidence:
  `hardware_control_audit/2026-09-27-bundle-acceptance/`.
- Real-FCI shadow qualification and delay:
  `hardware_control_audit/2026-09-27-shadow-qualification/`.
- `circle_yz` physical clearance:
  `hardware_control_audit/2026-09-27-circle-yz-clearance/`.
