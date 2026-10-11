# Industrial motion demos: M0–M1

M0 provides a portable NumPy/YAML package, `franka_motion`, with no Isaac Lab,
Torch, Franky or robot initialization on import. Existing PPO tasks are unchanged.
M0 is reference-only. M1 adds the separate non-RL joint simulator task and runner
described below. No hardware runner or Cartesian controller is implemented yet.

## Implemented contract

- Joint reference: seven positions in rad followed by seven velocities in rad/s.
- Cartesian reference: base-frame position (3), explicit `xyzw` quaternion (4),
  linear velocity (3) and angular velocity (3), totaling 13 physical values.
- Frames: `fr3_link0` base and `fr3_flange` TCP, identity flange-to-TCP transform.
- Measured samples have host/robot clocks and optional torque signals. Missing
  torque is `None`, never fabricated as zero.
- YAML catalogs define model provenance, separate model/command/abort intervals,
  initial posture, trajectory geometry/time laws, controller candidates and suites.
  Suites expand speed × twist-feedforward × repeat cells. Unknown/duplicate keys,
  malformed references and unapproved hardware suites are rejected.

Circle and Lissajous geometry have analytic derivatives. A quintic **speed** ramp
controls phase, including ramp displacement in the requested number of turns.
Approach/exit use quintic position segments with continuous position, velocity
and acceleration at joins; their jerk is finite but can jump at joins. Tracking
phase speed/acceleration/jerk joins are continuous. No derivative clipping is used.
Disabling twist feedforward leaves the geometric path and its physical derivatives
unchanged; only the commanded twist becomes zero.

`validate_time_law()` uses analytic componentwise extrema. `validate_trajectory()`
checks sampled analytic Cartesian derivative norms and a numerical box, explicitly
not a continuous-time certificate, reachability result or collision approval.
The initial preview pose is numerical input, not home FK. All paths hold a fixed
orientation; later robot runners must reconcile initial orientation before motion.

## Commands

M0 was installed editable into the existing Isaac virtual environment. To reproduce:

```bash
cd /home/chen-lab/isaac/franka-rl
direnv exec . /home/chen-lab/isaac/.venv/bin/python -m pip install \
  --no-deps --no-build-isolation -e source/franka_motion

direnv exec . /home/chen-lab/isaac/.venv/bin/python -m pytest tests/motion_control -q

direnv exec . /home/chen-lab/isaac/.venv/bin/python \
  scripts/motion_control/audit_motion_inputs.py

direnv exec . /home/chen-lab/isaac/.venv/bin/python \
  scripts/motion_control/preview_trajectory.py \
  --suite source/franka_motion/franka_motion/config/suites/demo_a_tracking.yaml \
  --output-dir /tmp/franka-motion-demo-a-preview
```

The preview writes `reference.npz`, `reference.png` and `validation.json` to a
**new** explicit directory; it refuses overwrite. Use another directory for reruns.
`--trial` selects an expanded ID, e.g. `circle__cell_001__repeat_000` (see YAML
experiment name and sweep order). `--no-plot` removes the optional Matplotlib need.
No GUI/Kit process starts. Files in `/tmp` are small disposable verification
artifacts; real experiment datasets will use the supported data volume.

## Validation and remaining gate

22 tests pass. An independently built/extracted wheel loads its five packaged
YAML catalogs/suites without simulator imports. Both USD/URDF hashes and URDF
joint limits pass the local audit; the archived runtime model validation is
successful. This is not a new dynamics test.

Controller gains, reference hold and Cartesian limits are **simulation candidates**.
The installed real-time Franky Cartesian/posture source, reference interpolation,
torque processing and stop semantics still need auditing. See the M0 request in
`deployment/HANDOFF_AGENT.md`. M1 adds joint servo/PTP simulation; M2 adds full-pose
Cartesian tracking and parity checks before hardware transfer.

## M1: joint servo and PTP

Task: `Franka-FR3v2-MotionControl-Joint-v0`. It uses `ManagerBasedEnv`, with a
minimal Gymnasium registration adapter. Its deliberate non-RL step interface is
`(observations, extras)`, not an RL five-tuple. Targets are physical `[q_ref, dq_ref]`
14D vectors; there are no rewards, command resampling or automatic resets.

The audited live FR3v2.1 asset is hash-checked before creation. Gravity and
self-collision are enabled, implicit K/D are zero, and armature remains the
documented 0.003 kg·m² simulation stabilizer. State observations include measured
joints, base-frame flange pose/twist, applied reference, contact proxy and fault.

The explicit joint controller uses vector K/D (nominal 100/20), 0.5 rad spring-error
clip, 1000 Nm/s non-gravity torque slew and 100 Hz first-order low-pass filtering.
Its order is:

```text
K * clip(q_ref-q) + D * (dq_ref-dq) + PhysX Coriolis
    → non-gravity torque slew → low-pass → add PhysX gravity → effort clamp
```

`joint_servo` in `controller_profiles.yaml` records these settings and measured
speed-abort thresholds. Final effort clamp uses the live model's 87/12 Nm limits;
it is a simulator safeguard, not a new claim about the Franky hardware pipeline.
Measured speed-abort thresholds are provisional half-URDF values, **not** the old
PPO 20% qualification envelope. Model gravity compensation is added once, outside
the non-gravity filter. Soft-limit avoidance/posture and Cartesian torque terms
are not part of this M1 controller; small joint trials stay well inside the limits.

Ruckig 0.17.3 generates offline, local, zero-velocity/zero-acceleration endpoint
PTP references. We use only its [state-to-state planning API](https://github.com/pantor/ruckig),
not Community intermediate-waypoint/cloud planning. Joint sweeps use dimensionless
`speed_scales`, not Cartesian angular speeds. The publisher holds each reference
at 50 Hz while feedback/torque runs at 1 kHz; this is a reference zero-order hold,
not a hidden position smoother. Delay is explicitly zero for M1.

To reproduce installation if Ruckig is absent:

```bash
cd /home/chen-lab/isaac/franka-rl
CMAKE_BUILD_PARALLEL_LEVEL=2 direnv exec . \
  /home/chen-lab/isaac/.venv/bin/python -m pip install \
  --build-constraint source/franka_motion/build-constraints.txt ruckig==0.17.3
```

The build constraint fixes Ruckig 0.17.3's incompatibility with newer
scikit-build-core. It affects the isolated build, not the runtime controller.

Run the finite YAML suite (stationary hold and two repeated short PTP moves):

```bash
cd /home/chen-lab/isaac/franka-rl
direnv exec . /home/chen-lab/isaac/.venv/bin/python -u \
  scripts/motion_control/run_sim_demo.py \
  --suite source/franka_motion/franka_motion/config/suites/joint_smoke.yaml \
  --num_envs 4 --verify-reset-parity --device cuda:0 --viz none
```

For a visible single trial:

```bash
direnv exec . /home/chen-lab/isaac/.venv/bin/python -u \
  scripts/motion_control/run_sim_demo.py \
  --trial ptp__cell_000__repeat_000 --num_envs 1 \
  --device cuda:0 --visualizer kit
```

By default artifacts go to a new timestamped `${FRANKA_RL_DATA_ROOT}/motion_control/`
directory. Missing/unwritable volume or insufficient disk space fails explicitly.
Only bounded verification outputs may use an explicit new `/tmp` directory; no
silent repository/home fallback is allowed. M1 supports one controller profile
per process; arbitrary multi-profile suite coordination is a later milestone.

Each trial writes buffered `trace.npz`, `summary.json` and a hash-bearing
`FINALIZED.json`. Telemetry is preallocated on GPU (256 MiB cap), with no per-step
file writes. Trace rows contain **pre-physics** feedback, held references and
individual torque components; final **post-physics** joints are separate fields.
`applied` means torque sent to PhysX, not a measured motor-torque signal. Runtime
mass/COM/inertia snapshots and source/config hashes accompany the records.
First-fault feedback, code, reference and contact proxy are retained; the first
fault tick is the Isaac detection counter, not a presumed post-physics sample
timestamp. Faulted trials do not reset: they enter a measured-position hold with
zero desired velocity and damping, then the runner finalizes/closes. This is a
simulation stop candidate, not the commissioned Franky torque-stop protocol.

Verify artifacts, clone/repeat parity and optional matched single/batched runs:

```bash
direnv exec . /home/chen-lab/isaac/.venv/bin/python \
  scripts/motion_control/check_joint_artifacts.py \
  --suite-dir /path/to/batched/suite \
  --compare-suite /path/to/matched/single/suite
```

### Workstation acceptance (2026-10-11)

27 offline tests pass, including NumPy/Torch torque-component parity, effort
bounds, gravity-only startup, selective filter reset and Ruckig endpoint/derivative
checks. Live PhysX tests pass with one and four environments:

| Check | Observed result |
|---|---:|
| Stationary hold maximum joint error | $4.37\times10^{-8}$ rad |
| Short PTP planned motion duration | 1.170 s (plus 2 s initial hold and 1 s settle) |
| Short PTP maximum terminal joint error | $4.70\times10^{-5}$ rad |
| Short PTP maximum measured joint speed | 0.0886 rad/s |
| Faults in nominal hold/PTP trials | 0 |
| Repeat maximum q/dq difference | 0 |
| Clone maximum q/dq difference | $2.28\times10^{-7}$ (respective rad/rad·s⁻¹ units) |
| Single/batched maximum q/dq difference | $2.05\times10^{-7}$ (respective units) |
| Invalid-reference rejection, first-fault retention and selective reset | Pass |

Numerical commissioning limits are 0.002 rad hold drift, 0.01 rad terminal error
and $10^{-6}$ componentwise parity tolerance; these are not hardware guarantees.
Tests do not prove all joint trajectories are collision-free: the contact proxy
is a guard, not full path-clearance validation. Cartesian/source-parity audit and
full-pose path checks remain M2/transfer gates. Existing PPO tasks are untouched.
