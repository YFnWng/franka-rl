# Circle gain selection and three-case posture comparison

Lowest track-phase position RMS among latest completed, smooth-stopped trial per circle profile. Orientation, contour and torque checked separately. Single-trial provisional selection.

| Circle profile | Position RMS (mm) | Orientation RMS (deg) | Contour RMS (mm) |
|---|---:|---:|---:|
| cartesian.circle.k0.5.d1 | 13.853 | 1.823 | 7.450 |
| cartesian.circle.k1.5.d1 | 4.668 | 1.860 | 2.388 |
| cartesian.circle.k1.d0.75 | 6.980 | 1.863 | 3.569 |
| cartesian.circle.k1.d1 | 6.953 | 1.837 | 3.589 |
| cartesian.circle.k1.d1.25 | 6.974 | 1.861 | 3.585 |
| joint.circle.k100.d0.75 | 2.198 | 0.225 | 1.867 |
| joint.circle.k100.d1 | 2.080 | 0.202 | 1.717 |
| joint.circle.k100.d1.25 | 2.038 | 0.202 | 1.751 |
| joint.circle.k200.d1 | 1.107 | 0.111 | 0.941 |
| joint.circle.k50.d1 | 3.853 | 0.374 | 3.247 |

Selected Cartesian translation K=1200 N/m and D=69.282 N s/m; orientation K=40 N m/rad and D=13 N m s/rad. Selected joint PD K=200 N m/rad and D=28.284 N m s/rad on all seven joints.

All three posture configs reuse the exact saved `posture.20261012T150500Z/posture-plan.json`, 32s schedule (2s initial hold, A/B/return each 6s transition and 4s settle), limits and independent stop gains. Cartesian secondary posture K=10 and D=2.236 are retained; DLS posture rate gain=0.5/s is retained. These secondary gains are not inferred from the circle experiment.

The Cartesian pair differs only in projector and experiment labels/provenance; DLS uses a kinematic velocity projector followed by joint PD. These compare complete controller architectures, not just interchangeable projectors. Assess posture goal attainment and realized joint excursion together with EE drift.

## Run one trial per selected profile

Use the existing `run_profile` helper; it creates approved copies for these pending names.

```bash
run_profile cartesian.posture.selected.kinematic 1
run_profile cartesian.posture.selected.dynamic 1
run_profile joint.posture.selected.kinematic 1
```

The saved-plan live start/FK/limits checks remain required. No automatic homing or hardware execution performed during preparation. Keep performance governor and EEE disabled.

## Evidence and limitations

[Full circle report](/home/chen-lab/franka_ros2_ws/hardware_inventory/2026-10-12/m4/circle-comparison-report/report.md); [selection and exact source paths](/home/chen-lab/franka_ros2_ws/hardware_inventory/2026-10-12/m4/circle-comparison-report/gain-selection.json).

- Single-trial gain sweep; no confidence intervals or causal gain claims.
- Starting joint and EE states vary; no external pose metrology.
- Joint K100 nominal source predates EEE change; environmental settings were not fully matched.
- Posture-optimal gains not established by circle tracking.

Faulted/incomplete sessions remain in the full report, excluded from gain selection. Latest completed representatives give each gain profile one run; historical nominal repeats are shown in the full report but not weighted more in selection. Raw client torque is gravity-exclusive and aggregated over all active phases, not just the circle.

| Selected source | Max absolute raw client torque (Nm) | Slew-active fraction | Min active command success |
|---|---:|---:|---:|
| cartesian.circle.k1.5.d1 | 2.203 | 0.00000 | 0.970 |
| joint.circle.k200.d1 | 3.505 | 0.03477 | 0.990 |

## Completed posture results

See [M4_POSTURE_AUDIT.md](M4_POSTURE_AUDIT.md). Cartesian kinematic/dynamic completed; selected joint K200 faulted at the J7 torque guard, while fallback K100/D20 completed. Use K100 for the completed posture comparison. Session completion did not imply all posture settling criteria passed. Historical selected configs and failed trial remain preserved.
