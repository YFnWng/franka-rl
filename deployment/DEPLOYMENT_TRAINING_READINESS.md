# Deployment-level PPO readiness

## Current decision

The simulation contract and hardware controller model are now aligned at the
interface level: 1 kHz Franky-like impedance control, 50 Hz six-joint reference
increments, K=100/D=20 nominal gains, joint 7 held, and direct reference hold.
Training has produced position-only nominal and DR candidates plus z-axis
candidates.

No PPO policy is authorized for motion. Follow
`HARDWARE_DEPLOYMENT_PLAN.md` for implementation and commissioning gates.

The selected path demo is a reduced-size YZ circle using the 29D position-only
policy first. The simulation workstation must select its radius and timeout from
the evaluation grid appended to `HANDOFF_AGENT.md`.

## Candidate status

| Candidate | Status |
|---|---|
| `position_nominal` | Primary position-only bundle/shadow and possible first-motion candidate after export, deterministic evaluation, workspace review and shadow qualification. |
| `position_dr` | Comparison candidate; full deterministic evaluation is still required before selection. |
| `position_z_axis_nominal` | Blocked: exceeded the 20% measured-velocity qualification envelope in 57.8% of evaluated episodes. |
| z-axis DR | Rejected as a deployment candidate because deterministic success was poor. |

The candidate checkpoint paths and SHA-256 values are recorded in
`HANDOFF_AGENT.md`. None of the new immutable FR3 bundles is present on this
machine yet. The existing
`~/yifan/deployment_handoffs/franka_rt_handoff_1` package is an older Panda
24D/7D/30 Hz handoff and is incompatible with this contract.

## Implemented on the real-time machine

- Long-lived Franky joint-impedance torque controller and torque stop.
- 1 kHz state/controller logging and terminal-fault supervision.
- 50 Hz PPO timing validation.
- Six-action bounded incremental-reference integration.
- Measured-start reference initialization, soft-limit projection and held J7.
- 29D position observation assembly.
- Manifest trust anchor, isolated ONNX worker, stale point-result rejection,
  operator gate and watchdog.
- Eleven passing Franky coordinator tests.

## Remaining blockers

1. Export and transfer versioned 29D/6D FR3 bundles with deterministic tanh,
   source/checkpoint hashes and PyTorch/ONNX/worker parity.
2. Run the full deterministic evaluator for the nominal and DR position policies,
   including both path definitions and K=50/100/200.
3. Add mutually exclusive `ppo.targets` and `ppo.path` schemas, strict catalog
   loading, the exact waypoint state machine and complete path artifacts.
4. Add a true read-only hardware shadow mode and qualify inference latency and
   effective action delay at 50 Hz.
5. Add workspace validation and complete fake/simulator/shadow parity fixtures.
6. Reconcile remaining legacy runtime documentation and then commission in the
   staged order in `HARDWARE_DEPLOYMENT_PLAN.md`.
7. Defer the 31D z-axis runtime until a replacement policy clears its simulation
   velocity gate.

## Fixed references

- Policy and path contract: `HANDOFF_AGENT.md`.
- Limits and timing semantics: `LIMITS_AND_TIMING.md`.
- Hardware execution plan: `HARDWARE_DEPLOYMENT_PLAN.md`.
- Controller response evidence:
  `hardware_control_audit/2026-09-26-franky-impedance/comparison.json`.
- Pending fail-closed hardware template:
  `franky_runtime/config/ppo.pending.yaml`.
