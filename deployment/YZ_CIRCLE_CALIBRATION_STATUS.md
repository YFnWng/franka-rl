# FR3 `circle_yz` calibration status

The deployment path is the existing `circle_yz` entry in
`source/franka_rl/franka_rl/config/paths.yaml`:

- center `[0.475, 0.0, 0.35]` m;
- YZ-plane radius `0.15` m;
- y range `[-0.15, 0.15]` m and z range `[0.20, 0.50]` m;
- 24 waypoints, phase 180 degrees;
- 1.0 s per-waypoint timeout and 0.01 m position threshold;
- catalog SHA-256
  `e88676a68e38d44c6c62e27bc9f91ae42301033d189796fda506437f595764b3`.

`config/yz_circle_calibration_v1.yaml` now selects only this path. The earlier
3-radius by 3-timeout candidate catalog is retained as historical calibration
input but is not part of the requested run.

## Available robustness matrix

The checked-in optional matrix has 12 jobs:

- one path: `circle_yz`;
- two policies: `position_nominal` and `position_dr`;
- six hardware scenarios: K=50/100/200 under nominal and DR evaluation
  conditions;
- one seed: 123;
- 128 episodes per job, or 1,536 episodes total.

All six scenarios use the measured fixed one-policy-step delay,
`action_delay_range: [1, 1]`. This matrix is available for broader robustness
comparison; it is not automatically a prerequisite for the first demo. The
separate point-calibration suite is not part of this path-only work.

## Hardware shadow closure

- nominal session 2026092701 and DR session 2026092702 both completed;
- zero dropped samples, clean stops, and exact held-joint-7 behavior;
- state-to-reference median delay was 1.043 steps nominal and 1.008 steps DR;
- evidence: `hardware_control_audit/2026-09-27-shadow-qualification/`.

## Remaining gates

1. Audit the existing successful `circle_yz` artifacts for the exact transferred
   checkpoint hashes, catalog/path hash, K=100/D=20 controller, exact deployment
   home, fixed one-policy-step delay, and the required path/safety metrics.
2. If fixed delay 1 is absent or unrecorded, run only the two K=100 nominal-
   condition confirmation jobs: one for `position_nominal` and one for
   `position_dr`. The 12-job matrix is optional robustness evidence.
3. Select the deployment policy from the accepted evaluation evidence.
4. Prepare and separately authorize an immutable hardware-motion YAML.

Fake and real-FCI no-motion shadow runtime parity have passed. The operator
confirmed physical clearance for the exact `circle_yz` geometry; evidence is in
`hardware_control_audit/2026-09-27-circle-yz-clearance/`. The path is selected,
but no PPO policy and no robot motion are authorized yet. The 31D z-axis policy
remains excluded from the first demo.

The executable calibration inputs are
`config/yz_circle_calibration_v1.yaml`,
`config/fr3_hardware_calibration_scenarios_v1.yaml`, and
`../source/franka_rl/franka_rl/config/paths.yaml`.
