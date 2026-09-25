# Runtime training-model extraction

PhysX retains the tiny authored Panda link inertias and zero COM offsets in this reconstruction.
No automatic mass-property correction was observed at initialization or after either reset.

Both runs used 8 environments, seed 123, PhysX GPU, 60 Hz physics and 30 Hz policy timing.
Snapshots cover initialization before reset, first reset, and second reset. No policy rollout or hardware access occurred.

All 14 robot USD composition-layer hashes match the returned hardware audit. Resolved physics-manager settings match the saved nominal/DR configs exactly.
Task and scenario source have no differences from the training reference revision. Final config differences are instance count, command visualization, log path, and nominal seed (42 → 123).

| Body | Runtime mass kg | Runtime inertia trace kg m² | FR3 inertia trace kg m² | FR3/runtime |
|---|---:|---:|---:|---:|
| panda_link0 | 2.814203 | 4.98553879e-06 | 0.029 | 5816.8 |
| panda_link1 | 2.360000 | 4.21334948e-06 | 0.0502 | 11914.5 |
| panda_link2 | 2.379519 | 4.28793203e-06 | 0.0418 | 9748.3 |
| panda_link3 | 2.649882 | 4.22276878e-06 | 0.0295 | 6985.9 |
| panda_link4 | 2.694802 | 4.34193407e-06 | 0.026801 | 6172.6 |
| panda_link5 | 2.981282 | 7.11249538e-06 | 0.066299 | 9321.5 |
| panda_link6 | 1.128581 | 1.12998381e-06 | 0.0113 | 10000.1 |
| panda_link7 | 0.405291 | 2.32439504e-07 | 0.0027 | 11615.9 |

The runtime principal-axis orientation is identity and COM position zero for the nominal bodies, including hand/fingers.
Armature remains a separate joint property (nominal 0.001 kg m²). The ratios above compare link inertia traces, not whole-robot generalized inertia or closed-loop response.

DR leaves arm-link masses/inertias unchanged. Realized payloads and gain/effort scales for 16 environment-reset samples are in comparison.json; full per-joint friction and all tensor values are in each runtime_model.json.
Hand inertia/COM change with the added offset payload, as intended. This DR does not remove the 0.586441 kg gripper assembly or cover the FR3 arm-link mass properties.

## Frames and provenance

Inertias are at COM, expressed in the rigid-body-prim frame; raw nine-element tensors are column-major. The matrix field converts that layout explicitly.
COM poses give the principal-axes frame relative to the body, with xyzw quaternions. This follows the installed low-level PhysX tensor API, despite the Isaac wrapper world-frame docstring.
The saved training USD hashes and exact historical runtime environment are unavailable. These results establish current effective physics with matching source/config and audit assets, not byte-for-byte historical execution.
Reduced instance counts change DR sample sequences relative to training. Seeds, environment indices, actual sampled values, config/checkpoint hashes, invocation and current software revisions are recorded.
Policy delay sampling was not executed: this is a physics/reset extraction; its saved distribution remains in scenario metadata.

## Next step

Build a separate FR3v2 bare-flange model with documented inertials and COMs, then evaluate frozen policies with the intended controller/governor. Preserve the original baseline and bundles. Do not apply a guessed global inertia scale.
