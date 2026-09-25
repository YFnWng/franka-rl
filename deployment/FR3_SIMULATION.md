# FR3v2 bare-flange simulation

## Live-model correction and governor smoke — 2026-09-25

The original asset and checkpoints remain unchanged. New asset:
`generated_assets/fr3v21_bare_flange_live_v1/fr3v2.usda` under the data root.
Its mass/COM/inertia source is the archived robot-returned fr3v2.1 URDF in
`hardware_control_audit/2026-09-25/`. The builder stores that URDF and its hash
beside the USD. Geometry deliberately retains pinned public fr3v2 collision
meshes; equivalence to live fr3v2.1 collision geometry is not established.
No firmware metadata is interpreted as measured actuator gains/friction.

Actual PhysX validation passed: masses, COMs, full inertias, static limits,
flange FK and payload composition across two resets. Evidence is beside the
asset in `runtime_validation.json`. The validator uses the archived source
URDF rather than hard-coding the earlier public description.

The standalone governor source initially lacked its Python `__init__.py`
because `**/__*` was ignored. A tracked simulation-host initializer now exports
the native API and strictly loads explicitly opted-in simulation fixtures.
It deliberately rejects deployment configs pending receipt of the RT-host
loader/review schema. This locally rebuilt package is not the claimed validated
release wheel; transfer/reconcile these changes before parity claims.

Actual two-environment smoke **failed closed at 13 ms**, physics tick 39:
environment 1, joint 4 measured velocity -0.310698 rad/s exceeded the synthetic
0.3 rad/s cap. Governor reason was TRACKING; its reference velocity was only
about -0.000212 rad/s. Invalid references are now rejected before writing targets.
This demonstrates plant/reference mismatch, not excessive reference speed.
Uncompensated gravity with the retained 80/4 surrogate is a plausible contributor,
not an identified cause. Do not loosen limits or claim hardware equivalence to
make this check pass. No PPO training was launched. Actual partial-reset,
stop-termination and throughput checks were not reached; API-stub/core tests
cover them separately. Failure details are in `governor_smoke.json` beside the
asset. Next resolve the compensated/identified plant response model.

### Stationary-reference diagnostic — 2026-09-25

`scripts/deployment/diagnose_fr3_hold.py` runs two isolated Isaac processes with
identical seeded joint resets, 3 kHz physics and the existing implicit PD 80/4.
Each holds its exact measured initial joint position, with zero velocity target.
No policy, governor, automatic episode reset or termination runs during sampling.
Only the world gravity vector differs. Two environments, 0.2 seconds, 601 samples
including the initial state; traces are buffered and saved as compressed NPZ,
with JSON summaries and logs. Original task configurations remain unchanged.

Actual result in `generated_assets/fr3v21_bare_flange_live_v1/hold_diagnostic_v3/`:

- Initial q/dq matched exactly across cases.
- Gravity off: zero motion in all joints throughout the recorded interval.
- Gravity on: environment 1 joint 4 crossed 0.3 rad/s at 12.667 ms; its speed
  reached about 1.50 rad/s within 0.2 s. The original governor fault was at the
  next 1 ms sampling boundary, 13 ms.
- At 13 ms, joint 4 required about 18.95 Nm gravity compensation, whereas
  instantaneous PD estimate was only 1.41 Nm (position error 0.00211 rad).
  The initial PD estimate was zero. This paired ablation isolates gravity-induced
  motion under the uncompensated surrogate as the trigger, without a moving
  policy reference. It is not hardware controller identification.
- `isaac_applied_torque` is an implicit-actuator estimate, not a motor-torque
  measurement. Raw PhysX actuation-force readback was zero despite the drive;
  do not interpret it as total applied drive torque. `gravity_compensation`
  is the model term g(q), logged but not applied as feedforward.

To repeat (use a new output directory):

```bash
export FRANKA_RL_DATA_ROOT=/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data
export FRANKA_RL_FR3_USD="$FRANKA_RL_DATA_ROOT/generated_assets/fr3v21_bare_flange_live_v1/fr3v2.usda"
direnv exec /home/chen-lab/isaac/franka-rl /home/chen-lab/isaac/.venv/bin/python \
  /home/chen-lab/isaac/franka-rl/scripts/deployment/diagnose_fr3_hold.py \
  --device cuda:0 --duration 0.2 \
  --output-dir "$FRANKA_RL_DATA_ROOT/generated_assets/fr3v21_bare_flange_live_v1/hold_diagnostic_repeat"
```

### Asset and governor reproduction

Run with the repository direnv environment active, from the data-volume runs
directory. Verify the data volume is mounted, writable and has free space.
Build output directories and report files must not already exist.

```bash
export FRANKA_RL_DATA_ROOT=/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data
export FRANKA_RL_FR3_USD="$FRANKA_RL_DATA_ROOT/generated_assets/fr3v21_bare_flange_live_v1/fr3v2.usda"
export FRANKA_RL_GOVERNOR_CONFIG=/home/chen-lab/isaac/franka-rl/deployment/reference_governor/configs/simulation.json
export FRANKA_RL_GOVERNOR_SIMULATION_FIXTURE=1

# To rebuild, choose a NEW output directory; existing audited assets are immutable.
direnv exec /home/chen-lab/isaac/franka-rl /home/chen-lab/isaac/.venv/bin/python \
  /home/chen-lab/isaac/franka-rl/scripts/deployment/build_fr3v2_asset.py \
  --urdf /home/chen-lab/isaac/franka-rl/deployment/hardware_control_audit/2026-09-25/robot_returned_model.urdf \
  --mesh-dir "$FRANKA_RL_DATA_ROOT/generated_assets/fr3v2_bare_flange_v1/meshes" \
  --output-dir "$FRANKA_RL_DATA_ROOT/generated_assets/fr3v21_bare_flange_live_v2"

direnv exec /home/chen-lab/isaac/franka-rl /home/chen-lab/isaac/.venv/bin/python \
  /home/chen-lab/isaac/franka-rl/scripts/deployment/smoke_fr3_governor.py \
  --num_envs 2 --steps 30 --device cuda:0 --viz none \
  --output "$FRANKA_RL_DATA_ROOT/generated_assets/fr3v21_bare_flange_live_v1/governor_smoke_repeat.json"
```

The smoke runner records success or failure, refuses report overwrites, and exits
nonzero even when Isaac's launcher suppresses an exception. It tests a small
joint target, not trained-policy reaching performance. Earlier model/controller
documentation below describes the original public-description asset.

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
