# Offline Panda policy / FR3v2 model audit — 2026-09-24

**Result: the position-observation geometry is compatible to model precision;
direct policy transfer is not established.** Joint-limit and dynamics differences
must be resolved before hardware policy control. No hardware was accessed and
no received bundle was modified.

## Evidence and scope

This copy is the return handoff from the real-time host to the simulation
workstation. That host has no Isaac Sim/Lab installation. All evidence needed
to read these findings is included alongside this note; original absolute paths
inside report.json are provenance only, not paths to use on the workstation.

- [Complete numerical report](report.json), including 1,003 configurations,
  joint limits, inertials, transforms and original input/asset hashes.
- [Expanded FR3v2 URDF](fr3v2_no_hand_fake.urdf), franka_description 1.3.0,
  commit 1ccde30d5a30a710f335c9f6545528447a04bc7c, bare flange.
  It contains ROS package mesh references and fake ros2_control declarations;
  it is comparison evidence, not a ready-to-import Isaac asset.
- [Saved robot snapshot](robot_state.json), collected read-only on the real-time
  host at 2026-09-24 21:41:07 UTC. No fresh connection is needed for this task.
- [Policy contract](policy_contract.yaml), identical for nominal and DR bundles.
- Exact saved configurations: [nominal](nominal_env.yaml), [DR](dr_env.yaml),
  [DR scenario](dr_scenario.yaml). These are provenance copies, not new configs.
- Source handoff: franka_rt_handoff_1, manifest SHA-256
  fb0f65407e1bd6c9ea5104a20248938294f10c9f75903ef1daa9ab78c6596af6.
  Training source revision: d0bfa66331a4813bcff0a0b1d24762618603d012.
- Local copied evidence integrity: [SHA256SUMS](SHA256SUMS).

The audit used CPython 3.12.14, usd-core 26.8, numpy 2.5.3 and PyYAML 6.0.3.
It composed the current saved-URL asset with Mesh=Performance, Gripper=Default,
then computed joint-chain FK from USD local joint frames and URDF origins/axes.
Samples used seed 123 and the common joint-limit box with 0.01 rad margins.
No collision checks or simulator rollouts were performed. NVIDIA USD files are
not included in this repository; report.json records the downloaded layer hashes.
For an independent repeat, resolve the saved URL on the simulation workstation,
compare hashes first, then compare FK and effective runtime data as requested
in [SIMULATION_HANDOFF.md](SIMULATION_HANDOFF.md).

The training-time USD asset hash and effective PhysX mass/inertia tensors are
absent from the handoff. Current URL contents cannot prove historical identity.
This is an authored-model comparison, not simulation, collision qualification,
real-robot calibration, or closed-loop policy validation.

## Frames and kinematics

All seven revolute-joint axes and zero transforms agree to USD float precision.
At 1,003 configurations (default, saved robot pose, common-limit midpoint, and
1,000 seeded random samples), Panda hand versus FR3 flange position differs by
at most 3.194e-7 m. After the fixed hand rotation, maximum orientation difference
is approximately 2.61e-15 rad.

`flange_T_panda_hand` is approximately Rz(-pi/4), with negligible translation.
The tracked `panda_hand` origin is therefore at the flange for this asset; do not
add a fingertip/TCP offset to the current position observation. The contract uses
base-frame position error only. Its orientation differs by 45 degrees, relevant
if a future controller observes orientation or defines a different tool frame.
Preserve joint order 1–7, joint signs, and robot-base target coordinates.

FR3v2 FK at the saved q agrees with reported O_T_EE to 8.45e-8 m and 9.07e-8 rad
(after projecting rounded rotations onto SO(3)). Snapshot tool transforms are
identity. This checks consistency with controller-reported FK, not metrology.

The default joint vector is legal for FR3v2; minimum joint margin is 0.26702 rad
at joint 4. Its computed flange position is [0.389448, 0, 0.457824] m, within the
contract target box. DR reset offsets of +/-0.125 rad remain inside static joint
limits. These facts do not establish collision clearance or authorize moving
to that pose. The target box is not a guaranteed safe robot workspace.

## Joint and action limits

Angles below are radians, rounded. Enforce FR3 limits with deployment margins;
Panda limits and simulation termination conditions cannot serve as robot limits.

| Joint | Training USD range | FR3v2 description range |
| --- | --- | --- |
| 1 | [-2.89730, 2.89730] | [-2.90074, 2.90074] |
| 2 | [-1.76280, 1.76280] | [-1.83609, 1.83609] |
| 3 | [-2.89730, 2.89730] | [-2.90074, 2.90074] |
| 4 | [-3.07180, -0.06980] | [-3.07702, -0.11694] |
| 5 | [-2.89730, 2.89730] | [-2.87630, 2.87630] |
| 6 | [-0.01750, 3.75250] | [0.43982, 4.62163] |
| 7 | [-2.89730, 2.89730] | [-3.05083, 3.05083] |

Joint 6 is the largest range incompatibility; joints 4 and 5 also have tighter
FR3 boundaries. Training USD static velocity limits are 2.175 rad/s (1–4) and
2.61 (5–7); description limits are 2.62 (1–4), 5.26 (5,7), and 4.18 (6).
These are model fields, not a complete operational envelope: FR3 also has
position-dependent velocity limits and acceleration/jerk constraints.

The contract emits unbounded raw actor output, maps q_target = q_default +
0.5 * action, and has no action clip. Even the legal default does not bound
future commands. A separate FR3 command governor needs position margins,
velocity/acceleration/jerk handling, stale-input/fault behavior, and simulation
validation. Keep the contract's previous action as the raw actor output;
changing it to a limited command would change the policy observation.

## Dynamics and end effector

The selected Panda asset includes a hand (0.558330 kg) plus two fingers
(0.014055 kg each): total 0.586441 kg. Your physical robot has a bare flange,
and its saved configured EE/load masses are zero. Nominal training therefore
has a different distal load. DR adds 0–2 kg to the baseline hand, as confirmed
by the bundled scenarios.py (`total_mass = default_mass + payload_mass`);
this does not include removing the gripper. DR link mass/inertia scales are null.

| Link | Panda USD authored mass kg | FR3v2 URDF mass kg |
| --- | --- | --- |
| 0 | 2.814203 | 2.396600 |
| 1 | 2.360000 | 2.437700 |
| 2 | 2.379519 | 2.237500 |
| 3 | 2.649882 | 2.193082 |
| 4 | 2.694802 | 2.181537 |
| 5 | 2.981282 | 2.186881 |
| 6 | 1.128581 | 1.579681 |
| 7 | 0.405291 | 0.694841 |

An unresolved asset issue is especially significant: the composed USD has
zero authored link COMs and link diagonal inertias around 1e-6 kg m². FR3v2
URDF inertia traces are approximately 5,817–11,915 times larger. Stage units
are meters and kilograms; link 1 has unit scale. Principal-axis attributes
are unauthored with zero-quaternion fallback, so the report leaves the USD
body-frame inertia matrix unspecified and compares traces only.

These are suspicious authored values, not proof that PhysX used those exact
values during training. Export effective masses, COMs and inertia tensors from
the original initialized Isaac articulation, including any runtime overrides,
before diagnosing or correcting this discrepancy. Do not apply an inferred
scale factor to hardware or silently alter the received policy bundle.

Training uses implicit position actuators with stiffness 80, damping 4,
armature 0.001, effort limits 87 Nm (1–4) / 12 Nm (5–7), 60 Hz simulation and
30 Hz policy updates. Matching effort numbers does not establish equivalence
to a real torque/impedance controller running at 1 kHz. DR varies gains, effort,
friction and 0–1 policy-step delay, but does not prove coverage of FR3 dynamics.

## Next work

1. On the training machine, capture the resolved USD hashes and effective
   Isaac articulation masses, COMs, inertias, joint limits and actuator settings
   after initialization and randomization. Resolve the inertia discrepancy.
2. Build an FR3v2 bare-flange simulation variant with matching observation
   frames and action contract. Evaluate both frozen policies with the intended
   FR3 command governor and controller timing; measure limit interventions,
   tracking, stability and success before deciding whether to fine-tune/retrain.
3. Develop/test the ROS controller and governor with fake hardware and then
   read-only shadow observations. Hardware motion commissioning remains a
   separate step with the procedure and stop checks maintained in the ROS
   repository at docs/HARDWARE_INVENTORY.md.

Inference parity and this audit are complete within their stated offline scope.
Physical model equivalence, closed-loop transfer and motion qualification are
not established.
