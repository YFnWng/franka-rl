# M4 — five non-RL FR3 demos

Transfer candidate, **not hardware approval**. All execution stays on the
real-time machine; no workstation connection, ONNX, PPO, interactive targets or
Python 1 kHz torque loop. `hardware_transfer.yaml` selects exactly five demos.

| Demo | Source / control contract |
|---|---|
| Circle tracking | `circle_yz_m2`, 0.2 rad/s, desired twist feedforward |
| Lissajous tracking | `lissajous_yz_small`, 0.2 rad/s, desired twist feedforward; current YAML amplitudes preserved |
| Static EE / varying posture | Home flange pose; feasible DLS-corrected posture A/B, 0.60 rad null seed, 6 s quintic transitions, 4 s settling; original M3 kinematic torque projection |
| Z impedance | Infinite base Z line through home; hand-guided tangent motion |
| Circle impedance | Closed base YZ circle through home; hand-guided tangent motion |

The trajectory catalog is authoritative for geometry, orientation, entry and
return times. Runtime must construct tracking entry from **measured** initial
FK and revalidate limits/IK; bundled fixtures use deployment home and are
numerical parity evidence, not blindly replayable hardware commands. Standard
tracking and curve impedance have reset-anchored posture K=2 Nm/rad,
D=1 Nm·s/rad and full-six-row dynamically consistent torque projection:

$$
N_\tau=I-J^T(JM^{-1}J^T)^+JM^{-1}.
$$

Circle/line fixtures retain their home-on-curve initialization check (1 mm).
Hold fixed full xyzw `[1,0,0,0]` orientation, not merely an axis-only constraint.
Normal stiffness 100 N/m, normal/tangent damping 20 N·s/m. No tangent spring,
no time-driven motion and no simulated 0.5 N force injection on hardware.
Projection and posture torque must run **natively at 1 kHz**, using measured
flange pose/twist and hardware model M/J. Circle-center projection is rejected;
independent measured workspace guards remain required for endpoint-free curves.

## Transfer contents and offline checks

The bundle contains a `franka-motion` wheel and source snapshot, YAML catalogs,
robot-returned URDF, five numerical fixtures, runtime adapter scaffold,
dependency versions, optional accepted simulation summaries and SHA-256 manifest.
Large traces remain at their source paths; summary metadata explicitly says when
traces are not included. Evidence from old/smaller YAML is not approval of the
current larger Lissajous path. No simulation results are invented by the fake mode.

On the real-time machine, use its separate Python environment, not system Python
or the Isaac environment. Install the local wheel with its NumPy/PyYAML
dependencies reviewed for that machine; `requirements-workstation.txt` records
workstation versions, not a forced hardware upgrade. Example:

```bash
python -m pip install /path/to/bundle/wheels/franka_motion-0.1.0-py3-none-any.whl
python /path/to/bundle/runtime/run_motion_suite.py \
  --bundle /path/to/bundle --output-dir /path/to/new-offline-artifacts
```

Default mode is fake. It validates shared contracts and reference publication
lifecycle without importing a robot SDK or connecting to FCI. It does not emulate
robot dynamics or synthesize measured feedback. `no_motion` is also offline,
not a shadow robot connection. Configuration edits invalidate the manifest;
create a new reviewed bundle rather than editing an immutable transfer.

## Receiving-agent implementation gates

`FrankyMotionBackend` deliberately fails closed. The repository contains no
audited native continuous Cartesian/curve torque adapter; simply finding
`CartesianImpedanceMotion` or `PostureTask` does not establish these semantics.
Do not replace it with repeated waypoint moves, guessed constructor signatures,
silent SDK upgrades or a Python 1 kHz callback. Before any movement:

1. Record installed Franky/libfranka versions and exact native Cartesian,
   posture and torque-stop code. Confirm robot/tool/payload model and TCP parity.
2. Implement native SO(3)-log Cartesian impedance with timestamped pose/twist
   updates, dynamic tracking posture projection, and the separate M3 kinematic
   posture law. Native feedback-dependent fixtures must use the portable
   `CurveVirtualFixture`/`dynamic_torque_projector` as parity references.
3. Match non-gravity torque processing: impedance + posture + model Coriolis →
   1 Nm/ms slew → 100 Hz first-order filter. Hardware FCI supplies gravity;
   **do not add gravity twice**. Audit any additional libfranka filtering,
   friction compensation, effort/slew constraints and armature/model mismatch.
4. Implement startup pose/reference matching, reference watchdog, joint/contact/
   velocity/workspace guards, circle projection rejection, singularity handling,
   fault latching and locally reviewed torque stop. Simulation fault-stop speed
   spikes were observed previously; do not assume that surrogate is safe hardware.
5. Native buffered logs: robot/monotonic time, published/applied sequence, phase,
   measured q/dq and encoder-FK pose/twist, desired pose/twist/posture, gains,
   model/payload, measured/commanded/external torque with compensation conventions,
   fault and stop events. Never label echoed references as feedback.
6. Run offline golden fixtures and no-motion audits, then obtain local approval
   and commission short hold/small step, weak posture/short transitions, reduced
   path amplitude and brief hand guidance. Inspect tool/cable/self/environment
   clearance. Do not directly execute the 0.60 rad posture candidate.

Only then create a new hardware-approved configuration and functional native
adapter. Hardware results and simulation results must remain separately labeled.
