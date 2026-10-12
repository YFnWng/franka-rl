# M4 posture hardware audit — 2026-10-12

Three completed trials and one torque-guard fault. All three completed trials and the faulted trial ended with successful smooth stops and zero dropped samples. Each gain/controller case has one trial; conclusions are exploratory.

| Controller | Position RMS / max (mm) | Orientation RMS / max (deg) | Max joint excursion norm (rad) | A / B / return settled |
|---|---:|---:|---:|---|
| Cartesian K1200/D69.282 + kinematic | 2.999 / 4.390 | 0.600 / 1.469 | 0.808 | no / no / no |
| Cartesian K1200/D69.282 + dynamic | 2.516 / 3.464 | 1.655 / 2.089 | 0.814 | no / no / no |
| DLS joint K100/D20 + kinematic | 1.467 / 4.281 | 0.249 / 1.478 | 0.983 | yes / no / yes |

Errors above cover all active phases, including initial hold and transitions, using applied reference versus encoder-model FK. Excursion is the seven-joint Euclidean norm from measured initial q, not a single-joint angle. Settling requires posture error norm <=0.05rad, position <=10mm, orientation <=0.05rad continuously for0.5s. Session completion means execution finished; it does not imply all settling goals passed.

## What the trials show

- Dynamic Cartesian projection reduced position RMS by about16% relative to kinematic projection, but orientation RMS was about2.76 times larger. It did not dominate the kinematic route. Realized excursions were similar (0.814 versus0.808rad).
- DLS K100/D20 had the lowest position and orientation RMS and largest realized excursion (0.983rad). A and return met the joint/pose settling criteria; B did not.
- Both Cartesian cases moved substantially, but none of their three settle phases met the combined criterion. The mean posture error norms over the final0.5s of A/B/return were0.180/0.221/0.151rad for kinematic and0.176/0.219/0.159rad for dynamic.
- DLS K100/D20 final0.5s posture errors were0.030/0.073/0.035rad. B remains above0.05rad despite EE errors there being small (0.436mm and0.063deg in that tail).
- DLS K200/D28.284 faulted at14.603s during B transition: reconstructed pre-stop J7 torque8.314Nm exceeded the8Nm guard, dominated by damping7.858Nm rather than position0.456Nm. Its partial-run RMS is excluded from the completed-controller comparison.

## Comparability and limitations

All four runs used the identical saved posture-plan hash `a7031d34d706ff13343b0cf823cb5a2cd6e4646a5ef646e69a4d7fadaf68e181`; maximum pairwise measured start difference was 0.000202rad per joint. Timing, limits and separate stop gains were retained. Cartesian rotation K40/D13 and secondary posture K10/D2.236 were fixed. DLS secondary posture-rate gain was0.5/s. DLS and Cartesian compare different control architectures and secondary objectives; the matched Cartesian pair is the direct projector ablation.

Completed-run minimum active command-success rates were99%,100%,99% for Cartesian kinematic, Cartesian dynamic and DLS K100 respectively. Max absolute unslewed client torques were2.476,2.189,3.468Nm; slew active fractions were0%,0.0031%,3.34%. These torque summaries cover all active phases and exclude firmware gravity/motor-friction compensation. There is no external EE metrology. One trial per case cannot establish statistical superiority or robustness.

## Report recommendation

Use the completed Cartesian pair and DLS K100/D20 as the three-case posture comparison, with K200 failure explicitly included as a transfer limitation from circle-selected gains. Preserve the current guard values. Existing data are enough for a preliminary demo report, but not a claim of full posture-target convergence. If time permits, repeat the same cases and give B a longer settle interval for a separate convergence test, recorded as a schedule change.

[Generated report and plots](/home/chen-lab/franka_ros2_ws/hardware_inventory/2026-10-12/m4/posture-comparison-report/report.md); [detailed phase/tail audit](/home/chen-lab/franka_ros2_ws/hardware_inventory/2026-10-12/m4/posture-comparison-report/posture-audit.json).

