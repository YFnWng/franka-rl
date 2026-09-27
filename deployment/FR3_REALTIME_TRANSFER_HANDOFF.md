# FR3 real-time transfer handoff

Date: 2026-09-27

The simulation-workstation implementation phase is closed for the first
position-only hardware demo. The real-time machine owns the remaining fake,
shadow, timing, path-execution, and hardware-readiness work.

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

## Remaining real-time-machine work

1. Implement the YAML path-mode contract and fake-backend state-machine parity.
2. Run read-only shadow inference from the reviewed deployment home.
3. Measure effective state-to-action/application delay and report it in 50 Hz
   policy steps. The workstation calibration scenarios currently use a
   provisional zero-step delay and are not final qualification evidence.
4. Check observation ordering, FK/base/flange frames, measured-start reference
   initialization, held joint 7, stale-result rejection, inference latency,
   watchdogs, sticky faults, logging, and clean torque stop.
5. Run fake and shadow comparisons for both nominal and DR bundles.
6. Complete the swept-workspace review for the eventual YZ candidate.
7. Return delay/parity/shadow results. Only then rerun or finalize the
   radius/timeout calibration and create the selected immutable path catalog.
8. Request separate motion authorization. Neither bundle nor any path is
   currently authorized for robot motion.

Raw PyTorch checkpoints are not required by the real-time runtime. Retain them
on the training workstation for provenance and future re-export.
