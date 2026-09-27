# FR3 YZ-circle calibration status

Implemented on the simulation workstation:

- versioned FR3 29D/6D policy contract fr3_incremental_position_29d_v1;
- deterministic ONNX export including the final tanh;
- immutable nominal and DR candidate bundles with checkpoint/config/source hashes
  and 1,024 PyTorch/ONNX parity vectors;
- strict Franky-runtime checks for dimensions, inference transform, joint order,
  reference integration, soft limits, controller gains, torque slew, and default
  pose;
- immutable 3-radius by 3-timeout YZ catalog;
- exact-home K=50/100/200 nominal and DR evaluation scenarios;
- YAML-driven point and full YZ-grid coordinators;
- per-episode/path metrics for waypoint outcomes, final/minimum position error,
  action clipping, reference projection, measured velocity, held-reference
  tracking error, joint margin, and torque.

Current blockers:

1. The receiving machine has not yet returned the measured effective shadow
   inference/action delay. Calibration scenarios therefore use a clearly marked
   provisional zero-step delay and are not final qualification evidence.
2. The full point and 108-job path matrix has not yet been run. No circle is
   selected, and the original packaged path catalog remains unchanged.
3. A swept-workspace review and fake/shadow runtime parity still belong on the
   receiving machine before any motion authorization.
4. The 31D z-axis policy remains blocked and is not a first-demo candidate.

Pipeline smoke evidence:

- artifact directory:
  /media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data/evaluations/2026-09-27_yz_circle_pipeline_smoke
- nominal policy, K=100, radius 0.050 m, 1 s timeout, four exact-home episodes;
- all four traversals timed out only on waypoint 0, then reached the remaining
  23 waypoints; there were no unsafe failures;
- waypoint-0 minimum error was 0.0562 m, confirming the receiving-machine
  warning that the 0.191 m lead-in dominates a 1 s deadline;
- worst measured-velocity ratio was 1.014, worst joint margin 0.655 rad, worst
  tracking-error norm 0.154 rad, worst torque-vector norm 30.105 Nm, and
  reference projection/action clipping were zero.

This smoke result validates the pipeline but is not enough to select a path.

Final bundle trust anchors and locations are recorded in
config/fr3_position_policy_bundles_v1.yaml. Calibration inputs are
config/fr3_point_hardware_calibration_v1.yaml,
config/yz_circle_calibration_v1.yaml,
config/fr3_hardware_calibration_scenarios_v1.yaml, and
path_catalogs/yz_circle_candidates_v1.yaml.
