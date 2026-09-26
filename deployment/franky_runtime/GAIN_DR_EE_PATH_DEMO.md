# Gain-domain-randomization end-effector path demo

## Demonstration claim

Compare a nominal PPO policy with a gain-randomized PPO policy while the real
FR3 tracks the same collision-free flange path under three explicit Franky
joint-impedance controllers. The useful claim is robustness to actuator-gain
variation. It is not a general proof of sim-to-real robustness or recovery from
unknown firmware gains.

The robot currently has no attached end effector, so the tracked body and
reported path are `fr3_flange`.

## Controller family

First finish small sequential tests and freeze a per-joint nominal stiffness
vector `K_nom`. J1/J2 are currently selected as 100 Nm/rad; J3-J7 remain to be
commissioned. Define the three hardware cases as a common multiplier:

- compliant endpoint: `alpha = 0.5`
- nominal: `alpha = 1.0`
- stiff endpoint: `alpha = 2.0`
- `K = alpha * K_nom`, elementwise
- `D = 2 * sqrt(K)`, elementwise

This produces K=50/100/200 for J1/J2 while preserving suitable relative gains
for the distal joints. Do not set every joint uniformly to 50/100/200 without
hardware commissioning.

Keep Coriolis compensation, torque slew, error clipping, soft-limit repulsion,
30 Hz reference hold, and 1 kHz torque computation identical in simulation and
hardware. The current Franky runtime does not use the shared reference governor;
resolve that command-path decision before training, as detailed in
`../DEPLOYMENT_TRAINING_READINESS.md`.

## Policies

Train two policies with identical observations, action mapping, reward,
curriculum, robot-property randomization, seeds, and training budget:

1. **Nominal:** fixed `alpha = 1` and `D = 2*sqrt(K_nom)`.
2. **Gain DR:** sample one episode-level `alpha` log-uniformly from `[0.5, 2.0]`
   and use `D = 2*sqrt(alpha*K_nom)`.

Do not expose `alpha` to either policy for this robustness comparison. If other
mass, friction, latency, or observation-noise randomization is used, apply the
same distributions to both policies so gain DR remains the controlled
variable. A later experiment may randomize a damping multiplier independently;
the first comparison should keep K and D coupled to remain easy to interpret.

Both policies must simulate measured joint-state observations, the 30 Hz held
position reference, the explicit impedance torque law, torque-rate limiting,
and the same gravity-compensation convention as the Franky/libfranka runtime.
An ideal position actuator would invalidate the comparison.

## Hardware prerequisites

1. Complete the pending J1/J2 K=200 small smoke.
2. Select J3-J7 nominal gains using small one-joint-at-a-time motions.
3. Validate `alpha = 0.5, 1, 2` for the complete seven-joint profile with small
   reference motions before running PPO.
4. Validate the policy bundle and governor offline for all three gain cases.
5. Review each collision-free flange path and its start pose independently.

## Evaluation matrix

Use one held-out slow free-space path with fixed flange orientation for the
quantitative demo. A circle or figure-eight in a reviewed central workspace is
simple to inspect. Run each cell at least three times from the same start pose:

| Policy | alpha=0.5 | alpha=1 | alpha=2 |
|---|---:|---:|---:|
| Nominal | 3 runs | 3 runs | 3 runs |
| Gain DR | 3 runs | 3 runs | 3 runs |

Use identical path timing and policy seeds across matched cells. A second path
may be shown as a held-out qualitative check after the primary matrix.

Report:

- flange translational RMSE and maximum error
- orientation RMSE and maximum error when orientation is commanded
- completion rate and safety/governor interventions
- joint-reference tracking RMS and maximum error
- peak measured speed and commanded torque
- path smoothness, policy compute time, and observation/action age

Plot path overlays and error versus time for all six conditions, then summarize
the repeated-run distribution rather than only the best rollout.

## Interpretation

A successful result is a nominal policy that performs best near `alpha=1` and a
DR policy whose error and completion rate degrade less at `alpha=0.5` and
`alpha=2`, possibly with a small nominal-performance cost. If both policies are
unchanged across gains, the path is too easy to demonstrate gain robustness. If
both fail at an endpoint, narrow the training/deployment range or revise the
controller profile; do not widen safety thresholds to force completion.
