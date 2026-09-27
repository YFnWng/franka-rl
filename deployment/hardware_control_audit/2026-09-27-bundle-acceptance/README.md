# FR3 bundle receiving-machine acceptance

Both transferred 29D/6D position-policy archives match the SHA-256 values in
`FR3_REALTIME_TRANSFER_HANDOFF.md`. They were extracted into new directories
under `/home/chen-lab/yifan/deployment_bundles/2026-09-27/` and made read-only.
Each portable verifier passed all 1,024 vectors with CPUExecutionProvider. Exact
hashes and numerical results are in `acceptance.json`.

The finalized local coordinator loaded each real bundle, created the isolated
ONNX worker through its venv interpreter, assembled 29-value observations, and
ran ten fake-backend reference updates at 50 Hz. Both bounded runs stopped for
the requested test-time limit, dropped no callback samples, and completed their
smooth stop. This validates loading and runtime wiring; it is not policy
performance or hardware evidence.

The hardware EE feedback contract now uses measured joint encoders with pinned
libfranka forward kinematics, `Model.pose(Frame.Flange, q, identity, identity)`.
This encoder-FK `fr3_link0 -> fr3_flange` pose drives policy observations,
waypoint transitions, and path errors. The robot-reported pose is audit-only;
logs contain both and their position/orientation difference.

The runtime now implements strict SHA-anchored YAML path selection, deterministic
per-policy-step waypoint timeouts, repeated traversal, persistent incremental
reference/action state, stale-result rejection at transitions, and per-waypoint
outcomes. It also has a read-only `shadow` backend that can read FCI state and
run inference but exposes no path that submits or keeps alive a robot motion.
Shadow events contain the complete 29-value observation, its measured inputs,
raw action, mapped held reference, inference latency, and state-to-reference
consumption delay in nanoseconds and 50 Hz policy steps.

Offline validation is 19/19 passing. It includes the exact transferred YZ
catalog hash and geometry, catalog/workspace rejection, fake path traversal,
transition stale-result handling, and a shadow test whose backend raises if the
coordinator attempts motion.

Both real-FCI, no-motion shadow sessions subsequently passed. Their measured
state-to-reference latency maps to exactly one integer policy-step delay in the
simulator. The combined evidence is in
`../2026-09-27-shadow-qualification/qualification.json`.

Remaining gates are final simulation calibration and path selection with the
fixed one-step delay, swept-workspace review for the selected path, and separate
motion authorization. The current YZ candidate catalog remains calibration
input and has not been approved for robot motion.
