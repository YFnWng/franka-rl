# Industrial motion demos: M0–M3

M0 provides a portable NumPy/YAML package, `franka_motion`, with no Isaac Lab,
Torch, Franky or robot initialization on import. Existing PPO tasks are unchanged.
M0 is reference-only. M1 adds the separate non-RL joint simulator task and runner
described below. M2 adds Cartesian impedance tracking and full-pose path checks.
No hardware runner is implemented yet.

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
orientation on the track; M2 joins measured entry orientation through a quintic
SO(3) geodesic with analytic base-frame angular velocity.

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

22 M0 tests pass. An independently built/extracted M0 wheel loads its five packaged
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
mesh-clearance review remain transfer gates. M2 full-pose checks are below.
Existing PPO tasks are untouched.

## M2: full-pose Cartesian tracking

Task: `Franka-FR3v2-MotionControl-Cartesian-v0`, physical 13D pose/twist action,
1 kHz torque servo and 50 Hz zero-order-held reference publication. There is no
policy, implicit motor PD, operational-space inertia shaping, hidden IK servo,
or secondary posture torque. Targets, speeds, feedforward choices and repeats
come from YAML. Clones reset explicitly; faults never trigger automatic recovery.

For base-frame geometric Jacobian $J$, flange pose error
$e=[p_d-p;\operatorname{Log}(R_dR^T)^\vee]$ and flange twist $V$:

$$
F=K_x e+D_x(V_d-V),\qquad \tau_{\rm task}=J^T F.
$$

The candidate pipeline is task torque + PhysX Coriolis → 1000 Nm/s non-gravity
slew → 100 Hz first-order low-pass → PhysX gravity → model effort clamp. The
Cartesian orientation law/filter ordering is **not yet matched to installed
Franky**. `cartesian_soft` uses translation K/D=100/20 and rotation K/D=10/6.5;
`cartesian_medium` uses 150/24.5 and 15/8, with the units in the profile catalog.
On a fault the simulator applies a gravity-compensated velocity damper candidate,
not a hardware-qualified stop. The 1 N moving-link contact proxy is a guard, not
a collision-distance certificate.

Important frame correction: this Isaac checkout's `body_lin_vel_w` is a COM
velocity shorthand. M2 uses `body_link_vel_w` at the flange origin, matching
the controlled pose and link Jacobian. Every trial checks the archived URDF FK
and Jacobian against runtime Isaac feedback and cross-checks a small full-pose
DLS update against Isaac's official `DifferentialIKController`.

`circle_yz_m2` holds the home tool orientation (`xyzw=[1,0,0,0]`), uses a 5 cm
radius about `[0.475,0,0.35]`, and smoothly approaches/exits over 6 s each. This is
separate from the original M0 circle with its different orientation. Home FK is
approximately `[0.306891,0,0.590282]` m, outside the track preview box. The runner
explicitly checks/logs the home-to-box entry corridor; it never teleports to the
preview start. A 2 s measured-pose hold precedes the approach.
Return-to-start returns the **TCP pose**, not necessarily the redundant joint
posture. Simulation resets each new trial to the named joint posture; a future
hardware adapter must use a separately commissioned physical return, not an
instantaneous reset. M3 will address secondary posture objectives.

Before any circle torque is applied, sampled full-pose DLS checks the **whole**
approach/settle/circle/exit at 20 Hz, seeds each solve with the previous solution,
requires 0.1 mm/1 mrad residual tolerances, retains 0.02 rad inside the commanded
joint interval, and rejects a minimum mixed-unit Jacobian singular value below
0.02. This is a sampled kinematic guard, not a continuous feasibility or mesh
clearance certificate. Analytic position derivatives are checked at 200 Hz;
angular reference speed is checked too. Analytic smoothness does not mean the
held 50 Hz command itself has continuous derivatives.

### Run M2

The coordinator first runs a 10 s commissioning trial (stationary hold, smooth
2 mm reference step and +10% stiffness step) for each profile in a **separate
process**. It stops if commissioning fails, then runs tracking and compiles
finalized artifact hashes, errors, clone/repeat differences and figures.
Both suites below use new output directories; reruns never overwrite artifacts.

```bash
cd /home/chen-lab/isaac/franka-rl

# Optional fast preflight, without Kit:
direnv exec . /home/chen-lab/isaac/.venv/bin/python \
  scripts/motion_control/validate_motion_paths.py \
  --suite source/franka_motion/franka_motion/config/suites/demo_a_tracking.yaml

# Two speeds × pose-only/pose+twist × two repeats, soft gains.
direnv exec . /home/chen-lab/isaac/.venv/bin/python -u \
  scripts/motion_control/run_sim_suite.py \
  --suite source/franka_motion/franka_motion/config/suites/demo_a_tracking.yaml \
  --num_envs 2 --device cuda:0

# Moderate gain comparison, 0.4 rad/s, twist feedforward, two repeats per profile.
direnv exec . /home/chen-lab/isaac/.venv/bin/python -u \
  scripts/motion_control/run_sim_suite.py \
  --suite source/franka_motion/franka_motion/config/suites/demo_a_gain_sweep.yaml \
  --num_envs 2 --device cuda:0

# One visualized finite circle, no hardware connection.
direnv exec . /home/chen-lab/isaac/.venv/bin/python -u \
  scripts/motion_control/run_sim_demo.py \
  --task Franka-FR3v2-MotionControl-Cartesian-v0 \
  --suite source/franka_motion/franka_motion/config/suites/demo_a_tracking.yaml \
  --trial circle__cell_003__repeat_000 --num_envs 1 --device cuda:0 --viz kit
```

Default outputs: `${FRANKA_RL_DATA_ROOT}/motion_control/<timestamp>_<suite>/`.
Small explicit `/tmp` verification outputs are also supported. `--device cpu`
is useful for small-batch numerical checks; CPU/GPU dynamics are not asserted
bitwise identical. The standalone `run_sim_demo.py --commissioning-only` and
`--controller <name>` options support isolated checks.

Each trial writes bounded 1 kHz GPU/CPU telemetry to `trace.npz` after stopping,
including q/dq, pose/twist, reference, pose error, wrench, torque components,
first fault, terminal feedback and runtime mass/COM/inertia. Full-pose IK samples
are stored separately. `summary.json` records model/config/source hashes and
software versions; `FINALIZED.json` is written last. No per-step disk I/O.
Compilation produces `metrics.csv`, `compiled.json`, per-trial XYZ/error/joint
dynamics plots and `tracking_summary.png`. Tracking error compares feedback to
the **applied held target**; contouring is exact distance to the analytic 3D
circle. Optional Lissajous uses a labeled nearest-sampled-curve approximation.

Commissioning acceptance: hold error ≤2 mm and ≤0.01 rad; motion max position
error ≤3 cm; finite state and no configured faults. Repeat/clone tolerances are
physical numerical tolerances, not bitwise equality: q/dq ≤0.001 rad/rad·s⁻¹,
position ≤0.1 mm, orientation ≤0.001 rad, linear/angular twist ≤0.001 m·s⁻¹/rad·s⁻¹.
Repeats test reproducibility, not independent statistical confidence intervals.

### M2 performance diagnostics

The 1 kHz explicit torque loop runs 20 physics steps before each 50 Hz reference/
viewport update. For one or two robots, eager Torch GPU launch overhead can
dominate useful computation. Rendering adds another cost; lowering the physics
rate is **not** an equivalent optimization because it changes the servo.

`run_sim_demo.py --profile-steps 2` records 40 physics steps after a 20-reference-
step warmup, writing `step_profile.json` (Chrome trace) and `step_profile.txt`
inside the trial output. Supported profiling windows are 1..5 reference steps;
profiling is off by default. Cartesian summaries always include `performance`
with median/p90 wall time per reference step and simulation/wall-time ratio.
Median/p90 exclude warmup and the profiling window; rollout wall time includes
the profiler's overhead but excludes startup, path feasibility, and artifact
serialization. Run headless and visible tests serially, on the same device,
with identical configuration. Kit may already own CUPTI: in that case the
visible profile contains CPU-side operators but no GPU activity timing.

First-fault bookkeeping now uses fixed-shape `torch.where` updates instead of
boolean-indexed gathers/scatters, avoiding CUDA `nonzero` synchronization in
every physics step. Raw first-fault state/reference capture, latched fault bits,
sanitized stop posture, and selective reset semantics remain unchanged.

On 2026-10-11, matched serial RTX 4090 tests used two robots and the 10-second
Cartesian commissioning trajectory (hold, 2 mm step, +10% K step):

| Mode | Median reference-step wall time before | After | Rollout wall time before → after |
|---|---:|---:|---:|
| Headless | 94.16 ms | 83.26 ms | 49.04 → 43.34 s |
| Kit visible | 129.38 ms | 118.18 ms | 68.04 → 62.04 s |

The 40-physics-step headless profile reduced `aten::nonzero` calls from 1,032 to
zero and CUDA stream synchronizations from 1,654 to 622. Remaining synchronization
and many small kernel launches still limit throughput; this is a modest
optimization, **not real-time performance**. Both matched before/after traces
were exactly equal for q, dq, applied torque, pose and reference. All trials,
live invalid-reference/first-fault/selective-reset checks, and 43 offline tests
passed. This benchmark exercises the circle task's servo but is not a full-circle
performance measurement. Raw results live in `/tmp/motion-m2-perf-{before,after}-`
`{headless,visual}` (temporary diagnostic artifacts, not committed).

The current Franky source/API audit, mesh-clearance approval and physical
commissioning remain transfer gates. M3 fixture/posture control is not included.

### M2 verification — 2026-10-11

42 tests pass (35 motion-control tests plus 7 existing deployment-controller
regressions). A new standalone wheel builds and loads all four packaged suites
without importing Isaac. Live PhysX commissioning, first-fault/selective-reset
checks and the existing M1 joint hold/PTP/reset regression pass.

The complete soft-gain speed/feedforward matrix passes on two CPU clones,
with two repeats per cell. Both gain profiles additionally pass GPU commissioning
and two GPU circles at 0.4 rad/s, providing a same-backend gain comparison.
Each row below is rounded from its two repeat results; all recorded contact
proxies/fault counts are zero.

| Physics | Profile | Phase speed (rad/s) | Twist FF | Tracking RMS (mm) | Contouring RMS (mm) |
|---|---|---:|---|---:|---:|
| CPU | Soft | 0.2 | No | 1.941 | 0.097 |
| CPU | Soft | 0.2 | Yes | 0.158 | 0.120 |
| CPU | Soft | 0.4 | No | 3.778 | 0.320 |
| CPU | Soft | 0.4 | Yes | 0.515 | 0.440 |
| GPU | Soft | 0.4 | Yes | 0.515 | 0.440 |
| GPU | Medium | 0.4 | Yes | 0.349 | 0.291 |

Twist feedforward substantially reduces time-domain lag, but does not necessarily
reduce contouring error. Increasing **both K and D** in the moderate-gain profile
reduces these two errors for the verified GPU cell; this is not a stiffness-only
ablation or a claim that higher gains are always better. Largest measured joint
speed is about 0.216 rad/s and peak applied torque about 23.9 Nm, including the
approach/exit. Stationary-hold position error remains below 0.3 µm. Full-path IK
retains at least 0.446 rad beyond its already-inset joint bounds, with minimum
mixed-unit Jacobian singular value about 0.224. Runtime FK/Jacobian/official-DLS
cross-checks and actor-twist/$J\dot q$ parity pass.

CPU repeat traces match exactly within each cell; clone differences are not
bitwise zero (maximum position difference about 0.076 mm). GPU repeat/clone
differences also satisfy the stated physical tolerances. These are deterministic
numerical checks, not measurement noise estimates or hardware validation.

Disposable verification artifacts:

- `/tmp/franka-motion-m2-corrected-circles` — eight CPU trials.
- `/tmp/franka-motion-m2-soft-gpu-suite` and
  `/tmp/franka-motion-m2-medium-suite` — gated GPU gain cells.
- `/tmp/franka-motion-m2-final-compiled` — combined finalized metric tables and
  equal-scale XYZ, tracking/contouring, joint dynamics and summary figures.
- `/tmp/franka-motion-m2-joint-regression` — retained M1 regression evidence.

The earlier `/tmp/*m2*commission-v1` / `circles-v1` frame-debug runs are not final
M2 evidence. Reproduce durable artifacts on the data volume using the commands
above. The original Franky library-motion baseline and optional Lissajous live
comparison remain unmeasured; M2 establishes the conservative Cartesian
impedance circle candidate, not complete hardware controller parity.

## M3 — finite line fixture and redundant posture control

`Franka-FR3v2-MotionControl-FixturePosture-v0` is a separate non-RL task with
physical 20D actions `[p(3), quaternion_xyzw(4), linear/angular twist(6),
q_posture(7)]`. M1/M2 registrations and their 14D/13D contracts remain intact.
All references use the fixed robot base and bare flange. Physics/torque remain
1 kHz, references 50 Hz, with the same compensation, slew, filter, effort and
latched fault processing as M2. No hidden IK or motion planner runs in the servo.

For the fixture, `LineVirtualFixture` projects measured position onto a finite
segment along the unit axis $d$ through measured home FK $p_0$:

$$
s=d^T(p-p_0),\qquad p_d=p_0+\operatorname{clip}(s,s_{\min},s_{\max})d.
$$

Axial feedforward velocity is first-order filtered at 50 Hz (5 Hz cutoff),
limited to 0.03 m/s, and suppressed when outward at an endpoint. Orientation
is held and posture gains are zero. The bounded home entry corridor and fixture
endpoints are validated and recorded. The fixture is a **compliant constraint**:
the reference lies inside ±30 mm, but physical motion may overshoot the segment.
It is not a hard workspace fence or authorization for human contact.

Two separate trials inject 0.5 N for one second, followed by two seconds of
release: along base Z and transverse to base Z. Forces act at the **flange body's
center of mass**, not at a presumed tool tip; the live runner explicitly requires
identity base/world rotation. Permanent wrenches are cleared after every trial,
including faults. Forces are simulated physical disturbances, not perturbed pose
commands. Buffered traces include the applied force schedule and geometric errors.

Posture trials hold the same full Cartesian pose and add

$$
N_\tau=I_7-J^T(J^T)^\dagger,\qquad
\tau_n=N_\tau\left[K_n(q_n-q)-D_n\dot q\right].
$$

This is an **orthogonal kinematic torque projector**, using an SVD pseudoinverse
with relative cutoff $10^{-6}$; it is not dynamically consistent. Record task
drift rather than claiming zero acceleration coupling. Two targets are generated
from ±0.15 rad along a deterministically signed home Jacobian null direction,
then full-pose DLS-corrected and checked against joint limits. A three-second
quintic secondary-target transition precedes three seconds of settling. Normal
gains are $K_n=2$ Nm/rad and $D_n=1$ Nm·s/rad; the weak comparison uses
$K_n=0.5$ with the same damping and positive target. Intermediate secondary
targets are objectives, not guaranteed exact fixed-pose IK solutions.

Implementation:

- `franka_motion/virtual_fixture.py`: `LineVirtualFixture`, strict Demo B YAML loading.
- `franka_motion/kinematics.py`: `nullspace_projector`.
- `franka_motion/impedance.py`: `compute_posture_torque`.
- Motion `mdp/controllers.py`: batched projector and fixture reference, NumPy/Torch parity.
- Motion `mdp/actions.py`: independent 20D `CartesianPostureMotionAction`, projected torque before common processing.
- `experiments/fixture_posture_runner.py`: known-force lifecycle, feasible secondary targets, analysis and acceptance.
- `scripts/motion_control/run_sim_demo.py`: `--demo-b-suite` entry point.
- `scripts/motion_control/compile_cartesian_results.py`: finalized artifact verification, CSV/JSON and dedicated fixture/posture figures.

Run all five cases headlessly (CPU is useful for this small batch):

```bash
cd /home/chen-lab/isaac/franka-rl
direnv exec . /home/chen-lab/isaac/.venv/bin/python -u \
  scripts/motion_control/run_sim_demo.py \
  --task Franka-FR3v2-MotionControl-FixturePosture-v0 \
  --demo-b-suite source/franka_motion/franka_motion/config/suites/demo_b_fixture_posture.yaml \
  --num_envs 2 --device cpu --viz none --verify-reset-parity
```

For a visible single-robot trial, use `--num_envs 1 --viz kit`, omit
`--verify-reset-parity`, and add `--trial posture_b` or `--trial fixture_axial`.
Compile the printed output directory with the existing Cartesian compiler:

```bash
direnv exec . /home/chen-lab/isaac/.venv/bin/python \
  scripts/motion_control/compile_cartesian_results.py \
  --suite-dir /path/to/m3-output --output-dir /path/to/new-compiled-directory
```

Verified two-clone CPU results (2026-10-11):

| Trial | Relevant response | Maximum task drift |
|---|---|---|
| Axial force | 37.25 mm axial travel; 7.25 mm segment overshoot | 2.31 mm transverse; 0.138° orientation |
| Transverse force | 5.74 mm transverse displacement; 0.096 mm axial travel | 0.120° orientation |
| Posture A | 0.1499 rad joint-vector change | 0.170 mm position; 0.035° orientation |
| Posture B | 0.1499 rad joint-vector change | 0.155 mm position; 0.035° orientation |
| Weak posture B | 0.1352 rad joint-vector change; 0.0147 rad target residual | 0.109 mm position; 0.033° orientation |

The force response is compatible with directional compliance; it is not a
calibrated hardware stiffness measurement. Acceptance separately checks axial
travel ≥2 mm, lateral error ≤10 mm, segment overshoot ≤10 mm and orientation
error ≤0.02 rad; posture change ≥0.01 rad, position drift ≤5 mm and orientation
error ≤0.02 rad. All retain M2 finite-state/hold/physical fault gates. No faults or
contacts occurred. Hardware parity, mesh-clearance certification and physical
manual-guidance approval remain M4 gates.

48 offline tests pass (including finite-segment/filtered-twist parity, full-rank
and rank-deficient NumPy/Torch projectors, secondary-torque processing/reset, and
feasible corrected targets). All five CPU trials and GPU posture B pass with two
clones, plus invalid-reference/first-fault/selective-reset checks. Diagnostic
artifacts are `/tmp/motion-m3-final-cpu`, `/tmp/motion-m3-final-gpu-posture` and
`/tmp/motion-m3-final-combined`. These temporary outputs are not committed; use
the commands above to reproduce durable data-volume artifacts. The axial fixture
can coast after force release because axial damping is largely cancelled by
permitted-axis feedforward; these trials do not claim terminal settling or
hardware hand-guidance safety. Joint-limit telemetry is included; compilation
also reports minimum measured-state abort-bound margin.
