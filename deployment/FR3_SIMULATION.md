# FR3v2 bare-flange simulation

Task: `Franka-FR3v2-Reach-v0`. The original `Template-Franka-Rl-v0` remains
the Panda baseline. Frozen policy observations remain 24D, actions 7D,
q_target = q_default + 0.5 * raw_action, with raw previous action and a 30 Hz
policy period. Joint names `panda_joint1..7` are deliberate policy-interface
aliases for the corresponding FR3v2 joints, preserving paired-reset datasets.
They do not identify the physical model as a Panda. The tracked body is
`fr3_flange`; position targets remain in the base frame.

## Model

The builder uses the audited FR3v2 URDF in `model_audit/2026-09-24/` and collision
meshes from the official `frankarobotics/franka_description` repository at
revision `1ccde30d5a30a710f335c9f6545528447a04bc7c`. It records input/output hashes.
All eight link masses, COMs and full inertia tensors come from that URDF.
The USD uses principal-axis decompositions; runtime validation compares the
reconstructed full tensors to the URDF values. Static position, effort and
velocity limits use the FR3v2 description.

There are eight rigid bodies and seven revolute joints. No hand/finger bodies
or fictitious payload mass are added. Link 7's body origin is translated by
0.107 m along its local Z to the flange. Its COM, mesh vertices and joint
attachment coordinates are translated consistently; its inertia about COM is
unchanged. This is a change of rigid-body coordinates, not a change of mass or
kinematics. Zero-mass URDF sensor frames and ROS control declarations are omitted.

Collision meshes use PhysX convex-hull approximation, and rendering reuses those
meshes. Detailed visual meshes and the manufacturer's complete collision and
safety envelope are not reproduced. Self-collision is enabled. This first
suite does not measure contact forces or certify collision clearance.

For a controlled model comparison, implicit position gains remain 80/4,
armature remains 0.001 kg m², and simulation runs at 60 Hz. Those are explicit
simulation assumptions, not identified FR3 motor settings. There is no hardware
reference governor or position-dependent velocity envelope yet. Keep hardware
transfer and controller validation separate from these smoke results.

## Wrist payloads

Payload remains a scenario/DR variable, attached to the `fr3_flange` physical
body. FR3 offsets are measured from the flange origin in flange coordinates.
The existing Panda scenarios retain their nominal-hand-COM reference.
Mass, combined COM and full inertia are recomputed from the unloaded baseline
on every reset using the parallel-axis theorem; mass does not accumulate.

The payload is a point mass with no own rotational inertia or collision shape.
Replace/extend that model when the dimensions and inertia of a real attachment
are known. The unloaded flange still carries link 7's real mass (0.694841 kg).

`config/fr3_scenarios.yaml` provides three fixed evaluation cases and a separate
`fr3_payload_dr` scenario: 0–1 kg, flange-relative axial position 0–5 cm,
resampled at every reset. These are provisional simulation ranges, not measured
hardware distributions. No new training has been started.

## Build and validate

Use the shared Isaac environment. Generated assets live on the data volume;
neither mesh downloads nor generated USDs belong in Git.

```bash
export FRANKA_RL_DATA_ROOT=/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data
mkdir -p "$FRANKA_RL_DATA_ROOT/generated_assets"
/home/chen-lab/isaac/.venv/bin/python scripts/deployment/build_fr3v2_asset.py

/home/chen-lab/isaac/.venv/bin/python scripts/deployment/validate_fr3_model.py \
  --device cuda:0 --viz none \
  --output "$FRANKA_RL_DATA_ROOT/generated_assets/fr3_validation_new.json"
```

The builder refuses to overwrite the generated asset directory. To build a new
version, use `--output-dir` and set `FRANKA_RL_FR3_USD` to the resulting USD path.
Existing asset: `generated_assets/fr3v2_bare_flange_v1/fr3v2.usda`.
Runtime checks validate masses, COMs, tensors, limits, independent URDF FK, and
payload COM/inertia composition across two resets. `runtime_validation.json`
beside the asset records the completed first check.

## Small initial evaluation

Two frozen policies × three scenarios × one seed = six jobs, 32 episodes each:

1. Bare flange.
2. 0.5 kg at the flange origin.
3. 1 kg at flange-frame z = 5 cm.

Both policies/scenarios replay the same targets and initial joint samples.
The suite uses 16 parallel environments. It is a smoke comparison, not a
statistical robustness conclusion or hardware acceptance test.

From the repository root with the shared Isaac environment active:

```bash
/home/chen-lab/isaac/.venv/bin/python scripts/experiments/run_evaluation_suite.py \
  --suite source/franka_rl/franka_rl/config/experiments/fr3_smoke.yaml --fail-fast
```

Relative `scenario_file` paths now resolve relative to their suite YAML.
Evaluation artifacts include the model manifest, USD hash, flange reference
and controller assumptions. Existing Panda deployment exporters reject FR3
configs so they cannot silently label a newly trained FR3 policy as Panda.
