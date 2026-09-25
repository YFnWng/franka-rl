# First FR3v2 smoke evaluation

Artifacts: `/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data/evaluation_suites/2026-09-24_19-00-24_fr3v2_bare_flange_smoke`

Two frozen Panda policies evaluated on the separate FR3v2 bare-flange task. One seed (123), 32 paired episodes per policy/scenario, 16 environments. Success: position error below 3 cm for five consecutive policy steps.

| Scenario | Nominal policy successes | DR policy successes |
|---|---:|---:|
| Bare flange | 13/32 (40.62%) | 30/32 (93.75%) |
| 0.5 kg at flange | 14/32 (43.75%) | 31/32 (96.88%) |
| 1 kg, z offset 5 cm | 15/32 (46.88%) | 31/32 (96.88%) |

All remaining episodes terminated on the inherited 5 rad/s joint-velocity criterion. There were no joint-position, non-finite, or timeout terminations. This criterion is a simulation comparison metric, not the FR3 hardware safety envelope; the URDF allows 5.26 rad/s on joints 5 and 7.

Target and initial joint position/velocity pairing match rates were all 1.0. This small sample suggests the DR policy transfers better under the tested simulation assumptions; it does not establish hardware readiness or broad robustness.

Runtime model validation passed for URDF masses, COMs, full inertias and static position/velocity limits. Default-pose flange FK maximum absolute coordinate error was 8.01e-8 m. Random payload COM and parallel-axis inertia composition passed across two resets. Eleven offline tests passed, including Panda isolation and flange-relative scenario selection.

Next: inspect the velocity-failure trajectories and define/test the intended reference governor and controller timing before deciding on FR3 retraining. Payload DR remains available via fr3_payload_dr; no training was launched.
