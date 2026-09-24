# Offline deployment bundles

The exporter reads a checkpoint and its sibling `params/env.yaml`, `agent.yaml`,
and optional `scenario.yaml`. It never starts Isaac Sim or controls hardware.
It currently supports the project's RSL-RL `MLPModel` ELU actor, without
normalization or action clipping. Unsupported contracts fail explicitly.

The standalone package lives in `deployment/franka_policy_bundle` because
importing the existing `franka_rl` package registers simulator tasks. The
exporter uses the installed training environment's PyTorch/RSL-RL; bundle
verification only needs `deployment/requirements.txt`.

## Return audit from the real-time host (2026-09-24)

The Panda-to-FR3v2 model audit and portable evidence are in
[model_audit/2026-09-24/FINDINGS.md](model_audit/2026-09-24/FINDINGS.md).
The simulation workstation should continue with
[SIMULATION_HANDOFF.md](model_audit/2026-09-24/SIMULATION_HANDOFF.md), starting
with effective runtime inertias and the bare-flange dynamics mismatch.

## Export on the training computer

```bash
export FRANKA_RL_DATA_ROOT=/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data
/home/chen-lab/isaac/.venv/bin/python /home/chen-lab/isaac/franka-rl/scripts/deployment/export_policy_bundle.py \
  --checkpoint "$FRANKA_RL_DATA_ROOT/runs/logs/rsl_rl/franka_reach/2026-09-24_00-48-20_dr_v2/model_999.pt" \
  --bundle-id dr_v2_model_999_release_1
```

Outputs go to `${FRANKA_RL_DATA_ROOT}/deployment_bundles/<bundle-id>`.
The root must already exist and be writable; media mounts and free space are
checked. Existing bundles are never overwritten. Failed exports preserve a
hidden staging directory for diagnosis. Choose a new ID after source changes.

Each bundle contains:

- `policy.onnx`: deterministic raw mean action; dynamic batch, float32 24 → 7.
- `policy_contract.yaml`: ordering, frames, units, timing, defaults and mapping.
- `manifest.json`: checkpoint hash, source revisions/dirty status, dependency
  versions, invocation and hashes of all bundled files.
- `test_vectors.npz`: zero, per-input sentinel and seeded random observations,
  physical input fields, native RSL-RL outputs and mapped position references.
- `test_vectors.json`: the 25 zero/sentinel cases for a C++ parity harness.
- `verification.json`: numerical evidence; hardware/C++ validation remain false.
- `provenance/`: exact saved training configurations including DR scenario.
- `schemas/`, `franka_policy_bundle/`, `verify.py`, `requirements.txt`: portable
  verifier source and its dependencies.

Add `--recorded-observations observations.npz` to append a float32 `[N,24]`
array stored under the key `observation`. Recorded observations are optional;
without them the report explicitly records zero recorded cases. The generated
physical fields reconstruct these observations; they are not recorded robot
trajectories. Numerical test vectors are not motion commands or workspace tests.

Observation corruption in saved DR configurations is retained as provenance;
the hardware contract uses clean measured inputs and raw previous actor output.
No noise, exploration, controller, reference governor, or hardware safety
thresholds are added by the exporter.

## Verify on the real-time computer

Transfer the entire directory out of band, preserving the printed manifest
SHA-256 separately as the trust anchor. In an isolated local Python environment:

```bash
python -m pip install -r /local/bundle/requirements.txt
python /local/bundle/verify.py /local/bundle --manifest-sha256 TRUSTED_HASH
```

The verifier needs no checkpoint, Isaac, ROS, PyTorch, or RSL-RL installation.
Without `--manifest-sha256`, it checks internal consistency only, not the
identity of the expected bundle. Hashes are not digital signatures.
All verification/inference runs locally; the training computer can be off.
The C++ runtime should run the JSON fixtures at absolute tolerance `1e-5`.
Python CPU ONNX parity does not establish C++ parity or hardware readiness.

## YAML target suites

`target_sets/shadow_smoke.yaml` is a lightweight example. Copy
`suites/shadow_smoke.template.yaml`, fill in concrete local paths and SHA-256
values, then validate with:

```bash
python scripts/deployment/verify_hardware_suite.py /local/suite.yaml
```

On the transferred bundle the equivalent is:

```bash
python -c 'from franka_policy_bundle.suite import main; main()' /local/suite.yaml
```

Run the latter from the bundle directory. Validation prints the resolved trial
plan without creating a controller. Paths resolve relative to the suite file.
The bundle manifest hash and target file hash are mandatory. Duplicate IDs,
wrong frames, out-of-policy-workspace targets, invalid dwell/timeout settings,
and motion requests in shadow mode are rejected. The hardware adapter must
also enforce the lab's approved workspace, which can be smaller than the policy
workspace. A suite validator is included; hardware suite execution belongs to
the real-time controller milestone.

## Tests

```bash
/home/chen-lab/isaac/.venv/bin/python -m pytest tests/deployment -q
```
