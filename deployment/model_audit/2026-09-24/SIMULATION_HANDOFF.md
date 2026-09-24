# Return handoff to the simulation workstation

Read [FINDINGS.md](FINDINGS.md) first. The real-time computer has completed
inference parity and an offline authored-model audit; it has no Isaac Sim/Lab.
Continue the simulation-side investigation here. Do not treat successful ONNX
parity or the matching arm geometry as evidence of closed-loop FR3 transfer.

## Priority 1: establish what the training simulator actually used

Recreate the original nominal and DR environments using the saved configs
beside this note and the original training software versions/checkpoints.
Keep baseline evidence separate from any corrected model or new training run.

Export a machine-readable JSON with:

- Isaac Sim, Isaac Lab, PhysX (if exposed), Python and training source versions;
  checkpoint/config hashes, source dirty status, seed and exact invocation.
- Resolved root USD URL, selected variants, meters/kilograms per unit and up
  axis; hashes for every resolved USD composition layer. Compare with
  report.json asset_sha256 and report any differences. A current matching URL
  alone does not establish historical training asset identity.
- Body/joint names and ordering, parent/child relationships and tool/base frames.
- For each body: effective mass, COM position/orientation, and full inertia
  tensor with explicitly documented frame and units. Include the hand/fingers.
- For each joint: effective position/velocity/effort limits, stiffness, damping,
  armature and friction; distinguish authored values from runtime overrides.
- Values after initialization/physics reset, then after nominal reset and DR
  randomization. For DR retain seed, environment index and sampled parameters
  so baseline and randomized properties can be distinguished.

Use runtime articulation/physics data, not only configuration dictionaries or
USD attributes. The authored Panda link inertia traces differ from the FR3
URDF by roughly 5,817–11,915 times; determine whether runtime data retains,
recomputes, rescales or overrides those values. Do not assume a factor-of-10,000
correction. Include a minimal reproducible extraction script with the result.
If the original runtime/asset is unavailable, state exactly what is reconstructed
and which historical facts remain unknown.

## Priority 2: evaluate an FR3v2 bare-flange variant

After the dynamics issue is understood, create a separate simulation variant
matching the physical FR3 Arm3Rv2 and bare flange. Use the included URDF as
reference; resolve its meshes from franka_description 1.3.0 if needed.

Preserve the 24-element observation order, raw previous actor output, 30 Hz
policy period and default-plus-0.5-action mapping. Verify the base-frame
position observation: this asset's panda_hand origin coincides with the flange,
while hand orientation is approximately Rz(-pi/4) relative to it. Do not add
an assumed fingertip offset. Keep any intentional contract changes versioned.

Evaluate the frozen nominal and DR policies before deciding whether to
fine-tune or retrain. Use matched targets/seeds, and model the intended command
governor, controller timing and actuator behavior. If the deployment controller
is not yet specified, record that limitation and the tested assumptions.
Account for FR3 limits, particularly joints 4, 5 and 6; clipping alone is not a
complete velocity/acceleration/jerk governor.

Report success and tracking error, maximum velocities/efforts, governor
intervention frequency/magnitude, limit violations, collisions and unstable
trajectories. Compare original Panda, FR3 bare-flange, nominal and DR cases.
Report metrics and failures without inventing hardware acceptance thresholds.

## Return deliverables

Return a new versioned directory containing:

1. Runtime model JSON, resolved asset/config hashes and extraction script.
2. Explanation of the inertia/COM discrepancy, with before/after evidence.
3. FR3 variant source/config changes and evaluation commands/results.
4. Recommendation supported by results: retain frozen policies, fine-tune,
   or retrain; list remaining controller/model uncertainties.
5. If exporting new policies, new bundle IDs, manifests and parity vectors.
   Preserve franka_rt_handoff_1 and its policy contracts unchanged.

No robot connection is required. Physical commissioning remains on the
real-time host after the separate lab procedure and motion authorization.

## Check the copied evidence

From this directory:

```bash
sha256sum -c SHA256SUMS
```

The snapshot, expanded URDF, configurations and numerical report are copied
verbatim from the real-time host. Absolute paths in them identify that host.
The complete source audit and original evidence remain in the ROS workspace.
