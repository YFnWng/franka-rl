# FR3 `circle_yz` calibration status

The deployment path is the existing `circle_yz` entry in
`source/franka_rl/franka_rl/config/paths.yaml`:

- center `[0.475, 0.0, 0.35]` m;
- YZ-plane radius `0.15` m;
- y range `[-0.15, 0.15]` m and z range `[0.20, 0.50]` m;
- 24 waypoints, phase 180 degrees;
- 2.0 s first-waypoint timeout, 1.0 s later-waypoint timeout, and 0.01 m
  position threshold;
- catalog SHA-256
  `39455c82dfec508cdac0d26befda5a701501006c3758d1c115adc868a1ff6c69`.

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
- 16 episodes per job, or 192 episodes total.

All six scenarios use the measured fixed one-policy-step delay,
`action_delay_range: [1, 1]`. This matrix is available for broader robustness
comparison; it is not automatically a prerequisite for the first demo. The
separate point-calibration suite is not part of this path-only work.

## Hardware shadow closure

- nominal session 2026092701 and DR session 2026092702 both completed;
- zero dropped samples, clean stops, and exact held-joint-7 behavior;
- state-to-reference median delay was 1.043 steps nominal and 1.008 steps DR;
- evidence: `hardware_control_audit/2026-09-27-shadow-qualification/`.

## Final calibration result and selection

The fixed-delay 12-job run is complete at
`evaluation_suites/2026-09-27_11-51-48_fr3_circle_yz_calibration_v1/`.

- nominal policy: 16/16 success at nominal K=50/100/200 and randomized
  K=100/200; 11/16 at randomized K=50;
- DR policy: 0/16 strict path successes in all six scenarios, with persistent
  late-circle waypoint errors just above the 0.01 m threshold;
- all jobs: zero unsafe failures, zero action clipping, and zero reference
  projection;
- nominal K=100 worst measured/20%-envelope velocity ratio: 1.0196 nominal and
  1.0462 randomized.

The selected first-motion candidate is nominal model 149 at K=100/D=20. The
20% velocity envelope is not a hard runtime limit, so the recorded small
exceedance is accepted as non-blocking for a staged trial. It remains a required
logged metric. The DR model remains available for provenance but is not selected.

## Remaining real-time gates

1. Synchronize the active path hash and nominal selection to the real-time
   machine.
2. Re-acknowledge the unchanged cleared geometry with the new 2.0 s initial
   waypoint timing and catalog hash.
3. Prepare and separately authorize one immutable nominal-policy motion YAML.
4. Execute only the staged controller-hold/near-home/path sequence with the
   existing fail-closed safety and logging contract.

Fake and real-FCI no-motion shadow runtime parity have passed. The operator
confirmed physical clearance for the `circle_yz` geometry; evidence is in
`hardware_control_audit/2026-09-27-circle-yz-clearance/`. The path is selected,
and nominal PPO is selected, but robot motion still requires authorization of
the exact YAML. The 31D z-axis policy remains excluded from the first demo.

The executable calibration inputs are
`config/yz_circle_calibration_v1.yaml`,
`config/fr3_hardware_calibration_scenarios_v1.yaml`, and
`../source/franka_rl/franka_rl/config/paths.yaml`.
