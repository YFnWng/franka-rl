# FR3 real-time transfer handoff

Date: 2026-09-27

The bundle transfer and real-time fake/shadow acceptance phases are complete
for the first position-only hardware demo. The simulation workstation now owns
the fixed-delay calibration rerun; the real-time machine still owns physical
workspace review and final motion readiness.

## Transfer these two complete bundles

Transfer both archives. Do not transfer only policy.onnx: the manifest,
contract, test vectors, verification code, schemas, and provenance are part of
the trusted deployment unit.

### Position nominal

- archive:
  /media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data/deployment_bundles/transfer_2026-09-27/fr3_incremental_6d_position_nominal_model149_v2.tar.gz
- archive SHA-256:
  979677ba20d755873c4c779751b6779b405a54fdad1c50ba56bef0050bb1f92d
- directory name after extraction:
  fr3_incremental_6d_position_nominal_model149_v2
- manifest SHA-256:
  9c1a72d9b0430343572c5cfd3580fe1b52cb63f6092e844e73f39d11f3be12dd
- policy.onnx SHA-256:
  b60b3af8e8dcca1650789a406f67d114cd094a0ed1d2fa36154702917ead605a
- PyTorch/CPU-ONNX maximum absolute error: 1.8477439880371094e-06

### Position domain-randomized

- archive:
  /media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data/deployment_bundles/transfer_2026-09-27/fr3_incremental_6d_position_dr_model199_v2.tar.gz
- archive SHA-256:
  b21eba9c3321f3c25f501c2c4f7c98b2c13e31297293437b03f7721f3d633949
- directory name after extraction:
  fr3_incremental_6d_position_dr_model199_v2
- manifest SHA-256:
  34895a6f0e479e62a72f57678e820473d8f5a350255c1b68a9575700b6378788
- policy.onnx SHA-256:
  0f600d9662bdfa662814bbd4e2fb16f2b3efe4ec0f3b237d207b3aa827ed6972
- PyTorch/CPU-ONNX maximum absolute error: 2.294778823852539e-06

Both use contract ID fr3_incremental_position_29d_v1: 29 float32
observations, six deterministic tanh outputs, and a 50 Hz incremental held
position reference. Their policy_contract.yaml files are intentionally
identical.

Do not deploy or transfer the current 31D position-plus-z-axis actor as a
hardware candidate. It remains blocked by measured-velocity qualification.

## Repository files to synchronize

Synchronize the current franka-rl source tree, including:

- deployment/HANDOFF_AGENT.md
- deployment/YZ_CIRCLE_CALIBRATION_STATUS.md
- deployment/config/fr3_position_policy_bundles_v1.yaml
- deployment/config/fr3_hardware_calibration_scenarios_v1.yaml
- deployment/config/fr3_point_hardware_calibration_v1.yaml
- deployment/config/yz_circle_calibration_v1.yaml
- deployment/path_catalogs/yz_circle_candidates_v1.yaml
- deployment/franka_policy_bundle/
- deployment/franky_runtime/

The candidate path catalog SHA-256 is
b78b6fa5bbc788f471464c48edefee3d29c5ea0f4cab06c5a7f6d65cae067cae.
It is calibration input, not yet an approved hardware path catalog.

## Receiving-machine acceptance

For each archive:

1. Verify the archive SHA-256 before extraction.
2. Extract into a new immutable directory.
3. Verify manifest.json against the manifest trust anchor above.
4. In the policy verification environment, run:

       python verify.py . --manifest-sha256 MANIFEST_SHA256

5. Confirm CPUExecutionProvider, all parity vectors passing, contract ID
   fr3_incremental_position_29d_v1, observation size 29, action size 6, and
   deterministic_tanh inference.
6. Point a copy of ppo.pending.yaml at one bundle and its manifest hash. Keep
   executable and motion_authorized false.

## Acceptance closure and remaining work

Bundle verification, YAML path-mode implementation, fake-backend parity, and
both real-FCI no-motion shadow runs have passed. The measured state-to-reference
latency maps to one 50 Hz policy step; evidence is in
`deployment/hardware_control_audit/2026-09-27-shadow-qualification/`.

1. On the simulation workstation, audit the existing successful `circle_yz`
   artifacts against the transferred policy hashes and the measured fixed
   one-step delay. If that evidence is complete, do not rerun it.
2. If delay 1 is absent or unrecorded, run only the two K=100 nominal-condition
   confirmation jobs, one per policy. The checked-in 12-job suite is optional.
3. Select between the nominal and DR policies using velocity, tracking,
   joint-margin, torque, action-clipping, and reference-projection margins.
4. The operator has confirmed physical clearance for the exact checked-in
   `circle_yz`; any geometry change requires a new review.
5. Prepare one immutable hardware-motion YAML and request separate motion
   authorization. Neither bundle nor any path is currently authorized for robot
   motion.

Raw PyTorch checkpoints are not required by the real-time runtime. Retain them
on the training workstation for provenance and future re-export.
