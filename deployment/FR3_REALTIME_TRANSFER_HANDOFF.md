# FR3 real-time transfer handoff

Date: 2026-09-27

The bundle transfer, real-time fake/shadow acceptance, and fixed-delay
simulation calibration phases are complete. The nominal position policy is now
selected for the first active hardware trial. The real-time machine owns the
immutable motion YAML, final preflight, staged execution, and run artifacts.

## Transferred bundles and selected policy

Both archives were transferred and verified. Do not use a standalone
policy.onnx: the manifest,
contract, test vectors, verification code, schemas, and provenance are part of
the trusted deployment unit.

### Position nominal

**Selected for the first hardware trial.**

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

Retain for provenance and later comparison; do not select it for the first
hardware trial. It failed every strict `circle_yz` traversal in the final
fixed-delay calibration.

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
- source/franka_rl/franka_rl/config/paths.yaml
- deployment/path_catalogs/yz_circle_candidates_v1.yaml
- deployment/franka_policy_bundle/
- deployment/franky_runtime/

The active `paths.yaml` SHA-256 is
`39455c82dfec508cdac0d26befda5a701501006c3758d1c115adc868a1ff6c69`.
Select `circle_yz`: radius 0.15 m, 24 waypoints, 2.0 s for waypoint 0,
1.0 s thereafter, and a 0.01 m position threshold. The older candidate-grid
hash is historical input and must not be placed in the final motion YAML.

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

## Final simulation selection

The final 12-job calibration used exact-home reset, fixed one-step delay, and
K=50/100/200 under nominal and randomized conditions. The nominal policy passed
16/16 traversals at K=100 in both conditions. The DR policy passed 0/16 in all
six scenarios. Select nominal model 149 with K=100/D=20.

For selected-policy K=100, there were no unsafe failures, action clipping, or
reference projection. Worst measured/20%-envelope velocity ratios were 1.0196
nominal and 1.0462 randomized. The 20% envelope is a recorded qualification
metric and action-scale basis, not a hard runtime limit; the operator accepts
this small exceedance as non-blocking for the staged trial. Continue to log it,
and do not add an unmodelled clamp or smoother.

Transfer the calibration directory if it is not already present:

`evaluation_suites/2026-09-27_11-51-48_fr3_circle_yz_calibration_v1/`

Trust anchors include:

- `calibration_index.json`: `c3fed116667b39f812d487417a57e85d0e54f9e2623184235ca15f9789ec9290`
- `path_scenario_summary.csv`: `fa011d39c2fe1ee08ae434f1b544f13c1ae523e0447f24b51876339dca073dde`
- compiled `jobs.csv`: `d63351bc409f8deb62e29ddef11afc4cc84799acbe4a540be3ec01c3bdaa65f2`

## Remaining real-time-machine work

1. Synchronize the updated repository files and verify the hashes above.
2. Create one immutable motion YAML selecting the nominal bundle manifest,
   active path-catalog hash, `circle_yz`, K=100/D=20, 50 Hz policy execution,
   one effective delay step, exact reviewed workspace, and one traversal.
3. Re-acknowledge the updated catalog: its geometry is unchanged from the
   cleared path, but waypoint 0 now permits 2.0 s rather than 1.0 s.
4. Run the existing fail-closed preflight, controller hold, and near-home staged
   check. Preserve all 1 kHz and 50 Hz logs and stop on any existing Franka,
   watchdog, tracking, collision, timing, or non-finite fault.
5. Obtain explicit authorization for that exact immutable YAML before motion.

Raw PyTorch checkpoints are not required by the real-time runtime. No further
training-workstation service or communication is required during experiments.
