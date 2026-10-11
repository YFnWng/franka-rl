# FR3 Industrial Motion-Control Demos — Implementation Handoff

**Status:** Implementation plan; revised 2026-10-11 for simulation-first development in this repository (not completed experiments).

**Target:** Existing Franka Research 3 (FR3), Isaac Lab simulation workstation, and the working FCI/libfranka/franky stack on the real-time machine. Aim for 1–2 days of focused implementation; hardware commissioning follows the simulation gates.

**Purpose:** Prepare technical evidence for an industrial articulated-robot motion-control interview (kinematics/dynamics, trajectory planning, robot-level control, integration, and testing).
**Scope:** Classical motion planning/control, simulation verification, then transfer to the existing hardware runtime. No reinforcement learning, pick-and-place, perception, or autonomous obstacle avoidance. All hardware experiment execution stays on the real-time machine; there is no streaming control connection to the simulation workstation.

## 0. Mission and priorities

Build and verify the algorithms in Isaac Lab first, then execute two compact **real-hardware demonstrations** with measured results:

1. **Demo A — Smooth joint/Cartesian trajectory execution and continuous tracking:** generate and follow time-parameterized Cartesian circle/Lissajous references, compare an existing trajectory-control baseline with impedance-based continuous reference tracking, and quantify tracking/contouring performance. Also demonstrate jerk-limited joint-space PTP motion.
2. **Demo B — Cartesian virtual fixture and null-space posture control:** (B1) make the FR3 end-effector compliant along a virtual axis but resistant transverse to it; (B2) regulate a fixed end-effector pose while changing redundant arm posture through a secondary null-space objective. Quantify tracking and posture changes.

**Priority:** Finish Demo A, including plots and one clean video, before spending time on Demo B. If only one day is available, Demo A is the deliverable. Reuse existing, validated robot connectivity, logging, and stop procedures; minimize setup and new dependencies.

**Crucial distinction:** A demo that only invokes a high-level library API is less persuasive than one that **creates an analytic reference trajectory**, records actual robot motion, computes interpretable metrics, and can explain the controller/model choices. Distinguish what franky implements from original code.

**Development order:** Pure numerical algorithms and synthetic tests → manager-based Isaac Lab trials → reproducible simulation artifacts and transfer bundle → fake/no-motion checks on the real-time machine → locally authorized hardware trials. Implement the analytic paths, time parameterization, DLS checks, impedance/fixture/posture algorithms, logging, and evaluation on this workstation before transfer. Ruckig remains an identified library planner; Franky remains the hardware real-time controller implementation.

## 1. Existing references and API caveats

- [franky repository](https://github.com/TimSchneider42/franky)
- [PR #79 — Enhance impedance control](https://github.com/TimSchneider42/franky/pull/79) — **merged July 2026**: joint/Cartesian impedance tracking, online reference updates, null-space posture tasks, joint-limit repulsion, optional friction compensation and torque-stop support.
- [Cartesian impedance tracking example](https://github.com/TimSchneider42/franky/blob/master/examples/cartesian_impedance_tracker.py)
- [Joint impedance tracking example](https://github.com/TimSchneider42/franky/blob/master/examples/joint_impedance_tracker.py)
- [Waypoint example](https://github.com/TimSchneider42/franky/blob/master/examples/waypoints.py)
- [Trajectory control documentation](https://timschneider42.github.io/franky/guide/trajectory-control.html)
- [Impedance control documentation](https://timschneider42.github.io/franky/guide/impedance-control.html)
- [Real-time motions documentation](https://timschneider42.github.io/franky/guide/realtime-motions.html)
- [Robot, state, limits, and kinematics documentation](https://timschneider42.github.io/franky/guide/robot.html)
- [Ruckig trajectory generation](https://github.com/pantor/ruckig)

**Verify before coding:** Which installed `franky-control` wheel/commit, libfranka version, and FR3 robot-server firmware are running; ensure APIs used in examples match the installed version. Do not update the working hardware stack blindly just to get PR #79 features. Franky's Python tracking loop can update references online, but **Python code does not itself provide guaranteed 1 kHz hard-real-time execution**; leave torque-loop timing and all mandatory safety functions to the validated lower-level control path. Use a conservative, verified reference-update rate consistent with available interfaces and observed jitter.

**Repository compatibility evidence:** `deployment/franky_runtime/requirements.lock.txt` records `franky-control==2.0.1.dev58+gf88f0e9b.libfranka.0.21.2`; `deployment/franky_runtime/VELOCITY_IMPEDANCE_ROUTE.md` separately records a patched velocity-reference build. These are archived pins, not a probe of the machine's current installation. Request the actual selected wheel hash, source/patch revision, Cartesian controller parameters, null-space projector, reference/gain interpolation, and torque filter/stop behavior from the real-time agent. Current upstream documentation is an API reference, not proof of installed behavior. Existing PPO velocity integration is not the contract for the new Cartesian demos.

Selected documented interfaces:

```python
from franky import (
    Affine, Robot, CartesianMotion, CartesianWaypointMotion,
    JointMotion, JointWaypointMotion, CartesianImpedanceTracker,
    JointImpedanceTracker, PostureTask, ManipulabilityTask,
    Twist, TorqueStopMotion,
)
```

Do not assume the exact constructor signatures on the machine until checked against installed docs (`help()` or the installed package).

## 2. Operating constraints and safety gate (mandatory before any motion)

- Use **only the existing authorized FR3 workspace and operating procedures**, preferably with a lab colleague/spotter; E-stop accessible, clear swept volume, known tool and payload settings, no people or loose objects in reach.
- Confirm current robot health, FCI connectivity, and that the documented stopping procedure works. **Never automatically dismiss or reset an unexplained protective stop/fault.**
- Verify the start joint configuration, joint limits, Cartesian workspace, self-collision clearance, tool transform, and conservative force/velocity/acceleration/jerk/torque-rate limits. Prefer FR3 model-specific limits, not Panda defaults. Avoid singular configurations.
- Precompute and visualize every path and representative IK solutions **offline/in simulation**; check limits/clearance before sending motions. Start with small amplitudes and low relative dynamics, then increase only after measured validation. Numbers in upstream examples (stiffness, collision thresholds, etc.) are **not** validated safety settings for this robot; do not copy them uncritically.
- For physical contact/manual-guidance experiments, validate torque mode and low-gain compliance first; only then use deliberate, light manual interaction within lab safety policy. **Never intentionally oppose a fast-moving arm.**
- For a tracking/torque mode, implement exception-safe exit and a verified smooth-stop transition; franky documentation describes `TorqueStopMotion` and warns that direct preemption via `Robot.stop()` can raise `ControlException`. Confirm stopping behavior on installed version before using.
- Log stop cause, faults, timing overruns, joint and Cartesian limits. A trial is invalid if it violates the safety gate; do not optimize parameters past protective limits.
- Separate experimental plots from safety claims: measured short-duration tracking quality does not establish collision safety or stability guarantees.

## 3. Repository architecture and implementation inventory

### 3.1 Ownership and dependency boundaries

Use the project's external Isaac Lab layout: reusable source packages under `source/`, Gym task configuration and manager terms under `franka_rl/tasks/manager_based/`, thin executables under `scripts/`, and machine-specific control adapters under `deployment/`. Keep the current RL tasks and their trained contracts intact.

The shared package is **`franka_motion`**, placed alongside `franka_rl`. This small second package is intentional: the existing `franka_rl/__init__.py` imports task registration and therefore brings in Isaac Lab. A real-time installation must be able to import planners, experiment schemas, and metrics without importing Isaac Lab, Kit, Torch, PPO, or ROS. Package `franka_motion` independently with NumPy/PyYAML; expose Ruckig as an optional, pinned planner dependency. The simulation uses equivalent batched Torch kernels with numerical parity tests against the shared NumPy reference implementations.

| Layer | Location | Responsibility |
|---|---|---|
| Shared numerical algorithms | `source/franka_motion/franka_motion/` | Time laws, paths, reference/state schema, DLS, Cartesian impedance/posture equations, fixture projection, metrics, artifact writing. No robot connection or simulator imports. |
| Isaac Lab task | `source/franka_rl/franka_rl/tasks/manager_based/franka_motion_control/` | FR3 scene, state observations, events, action-manager terms, GPU torque control, disturbances, diagnostic sampling. |
| Simulation experiment orchestration | `source/franka_rl/franka_rl/experiments/` and `scripts/motion_control/` | YAML suite expansion, trial lifecycle, process isolation, headless/visible runs, compilation. |
| Hardware adapter | `deployment/franky_runtime/franky_experiment/motion_demos/` | Adapt shared references into the installed Franky trackers/motions and reuse verified robot health, watchdog, stop, and logging procedures. |
| Configuration | `source/franka_motion/franka_motion/config/` | One versioned catalog of workspace, paths, gains, timing, and named suites, transferred with the shared package. |
| Generated artifacts | `${FRANKA_RL_DATA_ROOT}/motion_control/` | Simulation/hardware traces, figures, videos, verification results, and transfer bundles. No run data in Git. |

### 3.2 Isaac Lab environment contract

- Create **`Fr3MotionControlEnvCfg(ManagerBasedEnvCfg)`**, with `InteractiveSceneCfg`, `ActionTermCfg`, `ObservationTermCfg`, and reset/startup `EventTermCfg`. Use a small `Fr3MotionControlEnv(ManagerBasedEnv)` subclass for trial status and the last post-physics sample. It retains the base environment's `step(action) -> (observations, extras)` interface. There are no rewards, PPO configs, random target commands, Gym episode time limits, or automatic resets during a trial. `ManagerBasedEnv` does not supply the RL command/termination managers; reference scheduling and completion belong to the experiment runner.
- Register four task IDs in the new task package: `Franka-FR3v2-MotionControl-Joint-v0`, `Franka-FR3v2-MotionControl-Cartesian-v0`, `Franka-FR3v2-MotionControl-VirtualFixture-v0`, and `Franka-FR3v2-MotionControl-Nullspace-v0`. Register the custom environment entry point and `env_cfg_entry_point`, with `disable_env_checker=True`, matching the project's batched environment convention. The existing `franka_rl.tasks` discovery loads these registrations.
- Build the scene from the **audited live FR3v2.1 bare-flange USD**, selected by `FRANKA_RL_FR3_USD`. Require an explicit asset selection or a checked catalog entry; do not silently use the original Panda/public-description asset. Record URDF/USD hashes, mass/COM/inertia, armature, contact configuration, and runtime validation provenance. Existing joints `panda_joint1..7` are deliberate aliases; body `fr3_flange` and base `fr3_link0` identify the actual frame contract.
- Reuse the existing FR3 asset-building/validation scripts, ground/light conventions, PhysX articulation APIs, and contact-sensor patterns. Extract/copy only reusable scene and controller pieces into the new configuration; do not inherit a reward-bearing reaching environment to obtain the robot. All seven joints, including joint 7, remain available for full-pose tracking and redundancy control.
- For the explicit torque path, leave gravity enabled and set implicit actuator stiffness/damping to zero. Apply explicit model compensation exactly once. Preserve and record the existing armature stabilizer initially; it is a simulator assumption, not an identified hardware motor parameter. Self-contact and unexpected scene contact are monitored.
- Start with **1 kHz physics/torque updates** (`sim.dt=0.001`) and **50 Hz reference publication** (`decimation=20`). These are distinct clocks, not a policy contract. Repeat selected cases at a higher reference rate only after the hardware API/timing audit; a faster simulated publisher cannot demonstrate real Python timing. Record torque/gain/reference interpolation and effective publication delay explicitly; do not inherit the PPO's one-step delay automatically.
- The runner publishes physical values: joint actions are `[q_ref(7), dq_ref(7)]` (14D); Cartesian actions are `[p_ref(3), quaternion_xyzw(4), linear_twist(3), angular_twist(3)]` (13D); posture actions add `q_posture(7)` (20D). Units are m, rad, s, N, and Nm. These actions have no normalized `[-1,1]` map or PPO action clipping. Validate configured physical limits before execution and log any rejected reference.
- `process_actions()` accepts a reference once per publication tick; `apply_actions()` evaluates feedback/torque at every physics tick. Match the pinned Franky's actual reference hold/interpolation behavior. Analytic reference evaluation at 1 kHz is a separate diagnostic profile unless the hardware uses that same update contract.
- Observation group `state` exposes measured `q,dq`, base-frame flange pose/twist, last applied reference, contact signal, and trial/fault status as named tensors. No policy observation normalization. Cache joint/body IDs, Jacobian indexing, and frame transforms during initialization; fixed-base Jacobian rows and columns must be verified against the existing DLS tutorial.
- Fault state is latched without resetting the plant. Substep checks run inside the action/controller path; failed environments enter a defined stop state. The environment captures post-physics diagnostics, and the runner closes the trial before an explicit reset. The measured-state bound and commanded-reference bound must be separately specified: a projected reference can still have physical tracking overshoot. Preserve the terminal sample and exact first fault reason/joint/value/threshold.

### 3.3 Proposed source tree

All entries below are **new planned files**, not implemented by this handoff edit. Tables in §3.4–3.7 specify every Python file, including package markers. Existing files to update are listed separately in §3.8.

```text
franka-rl/
├── source/franka_motion/
│   ├── pyproject.toml
│   └── franka_motion/
│       ├── __init__.py
│       ├── contracts.py
│       ├── config.py
│       ├── time_scaling.py
│       ├── trajectories.py
│       ├── joint_planning.py
│       ├── kinematics.py
│       ├── impedance.py
│       ├── virtual_fixture.py
│       ├── trial.py
│       ├── metrics.py
│       ├── artifacts.py
│       └── config/
│           ├── fr3_workspace.yaml
│           ├── trajectories.yaml
│           ├── controller_profiles.yaml
│           └── suites/
│               ├── simulation_smoke.yaml
│               ├── demo_a_tracking.yaml
│               └── demo_b_fixture_posture.yaml
├── source/franka_rl/franka_rl/
│   ├── tasks/manager_based/franka_motion_control/
│   │   ├── __init__.py
│   │   ├── motion_env.py
│   │   ├── motion_env_cfg.py
│   │   └── mdp/
│   │       ├── __init__.py
│   │       ├── actions_cfg.py
│   │       ├── actions.py
│   │       ├── controllers.py
│   │       ├── observations.py
│   │       └── events.py
│   └── experiments/
│       ├── motion_runner.py
│       └── motion_suite.py
├── scripts/motion_control/
│   ├── preview_trajectory.py
│   ├── audit_motion_inputs.py
│   ├── check_joint_artifacts.py
│   ├── run_sim_demo.py
│   ├── run_sim_suite.py
│   ├── analyze_motion.py
│   └── build_handoff.py
├── deployment/franky_runtime/
│   ├── run_motion_suite.py
│   └── franky_experiment/motion_demos/
│       ├── __init__.py
│       ├── backend.py
│       └── runner.py
├── tests/motion_control/
│   ├── conftest.py
│   ├── test_m0.py
│   ├── test_m1.py
│   ├── test_trajectories.py
│   ├── test_joint_planning.py
│   ├── test_impedance.py
│   ├── test_virtual_fixture.py
│   ├── test_metrics.py
│   └── test_trial_and_handoff.py
└── docs/industrial_motion_control.md
```

### 3.4 Shared package: exact classes/functions to implement

Paths in this table are relative to `source/franka_motion/franka_motion/`.

| New file | Planned definitions and responsibility |
|---|---|
| `__init__.py` | Lightweight version/public exports. No task registration or device initialization. |
| `contracts.py` | `JointReference`, `CartesianReference`, `RobotSample`, `TrialResult`; `validate_reference()`. Immutable physical reference/sample records, explicit frame/unit/quaternion metadata, optional torque fields and validity flags. These records are reference-level/offline objects; the GPU hot path uses preallocated tensors. |
| `config.py` | `WorkspaceConfig`, `ControllerProfile`, `TrajectorySpec`, `MotionSuiteConfig`; `load_motion_suite()`, `expand_trials()`, `validate_suite()`. Resolve YAML names and relative paths, validate matrices/rates/limits, expand geometry × speed × feedforward × gain × repeat cells, and distinguish simulation-only from commissioned hardware settings. |
| `time_scaling.py` | `QuinticSegment`, `SmoothPhaseSchedule`; `evaluate_quintic()`, `validate_time_law()`. Polynomial position/phase with analytic derivatives through jerk, bounded phase ramp/constant-rate/ramp-down, and continuous boundary conditions. Stretch duration or reject infeasible schedules, never silently clip derivatives. |
| `trajectories.py` | `CircleTrajectory`, `LissajousTrajectory`, `ApproachAndTrackTrajectory`; `sample_trajectory()`. Base-frame pose, twist, acceleration and jerk from geometric path plus phase schedule; smooth approach/settle/track/exit phases; shared sampled preview and exact reference reconstruction. |
| `joint_planning.py` | `RuckigJointPlanner`, `JointTrajectory`; `plan_joint_ptp()`, `plan_joint_waypoints()`, `validate_joint_trajectory()`. Seven-joint jerk-limited library planning with exact initial/terminal state, limits and planner/version metadata. Quintic comparison uses `QuinticSegment`; it is not called jerk-limited unless checked. Blended multi-waypoint planning is conditional on the pinned planner's supported API/license. |
| `kinematics.py` | `DampedLeastSquaresIK`; `pose_error_base()`, `tcp_jacobian()`, `solve_dls_step()`, `nullspace_projector()`, `joint_limit_margin()`, `validate_ik_path()`. Model-independent numerical functions receiving FK/Jacobian providers; continuous IK seed from previous solution, scaling of position/orientation, singularity diagnostics and seven-joint limit checks. |
| `impedance.py` | `JointImpedanceController`, `CartesianImpedanceController`, `TorqueCommandProcessor`; `compute_posture_torque()`. NumPy reference implementations of PD/wrench-to-torque mapping, recorded null-space projector, compensation, clipping, torque-rate/filter state and reset. Expose individual torque components for parity tests. Exact Franky source choices are configuration/versioned contracts; do not silently substitute Isaac Lab OSC. |
| `virtual_fixture.py` | `LineVirtualFixture`; `project_to_line()`, `transverse_error()`. Project measured TCP position onto a finite permitted segment; compute base-frame permitted-axis feedforward from measured velocity or an explicitly configured filtered estimate, clamp excursion/reference speeds, and hold validated orientation. |
| `trial.py` | `MotionTrial`, `TrialPhase`; `advance_trial()`. Shared approach → settle → track/interact → exit → complete/fault lifecycle, finite duration, configured stop reasons, and trial clock origin. No robot APIs, automatic error recovery, or interactive target prompts. |
| `metrics.py` | `MotionMetrics`; `align_reference()`, `tracking_metrics()`, `circle_contour_error()`, `sampled_curve_contour_error()`, `orientation_error()`, `fixture_metrics()`, `posture_metrics()`, `timing_metrics()`. Phase-aware summaries from measured state, exact analytic 3D circle distance, reference-rate/delay audit and explicit excluded samples. |
| `artifacts.py` | `MotionArtifactWriter`; `write_manifest()`, `write_trace()`, `write_summary()`, `verify_motion_bundle()`. Buffered CSV/NPZ plus JSON metadata, configuration/source hashes, schema validation, finite-value handling, immutable completion marker and transferred replay-fixture verification. |

`source/franka_motion/pyproject.toml` declares this package, its lightweight dependencies, packaged YAML resources, and an optional Ruckig extra. Default import must work on the real-time machine without installing `franka_rl`.

### 3.5 Isaac Lab task and experiment files

Task paths are relative to `source/franka_rl/franka_rl/tasks/manager_based/franka_motion_control/`.

| New file | Planned definitions and responsibility |
|---|---|
| `__init__.py` | Register the four task IDs in §3.2, each resolving the new config and `Fr3MotionControlEnv`. No RL-agent entry point. |
| `motion_env.py` | `Fr3MotionControlEnv`. Maintain trial status, expose `capture_state()`, `begin_trial()`, `end_trial()`, and capture the post-physics terminal state; explicit reset only. Preserve `ManagerBasedEnv` stepping/renderer behavior. |
| `motion_env_cfg.py` | `Fr3MotionSceneCfg`, `MotionStateObservationsCfg`, `MotionEventsCfg`, `Fr3MotionControlEnvCfg`, `Fr3JointMotionEnvCfg`, `Fr3CartesianMotionEnvCfg`, `Fr3VirtualFixtureEnvCfg`, `Fr3NullspaceEnvCfg`; `make_fr3_robot_cfg()`. Audited articulation, physical gains/limits, contact sensing, selected TCP, reference rate and separate reference/measured safety intervals. |
| `mdp/__init__.py` | Explicit local exports of the new manager terms. Add a lazy export stub only if the chosen registration actually uses that mechanism; avoid the earlier missing-export failure. |
| `mdp/actions_cfg.py` | `JointMotionActionCfg`, `CartesianMotionActionCfg`, `CartesianPostureActionCfg`. Physical 14D/13D/20D action layouts, pinned controller profile, reference timing, substep safety and trace settings. |
| `mdp/actions.py` | `JointMotionAction`, `CartesianMotionAction`, `CartesianPostureAction`. Resolve IDs once; publish validated targets; obtain batched measured pose/twist, Jacobian and dynamics; apply torque at 1 kHz; cache every torque component; selective reset of filter/reference/fault state; bounded GPU diagnostic buffer and configured stop transition. |
| `mdp/controllers.py` | `BatchedJointImpedanceController`, `BatchedCartesianImpedanceController`, `BatchedTorqueCommandProcessor`; `pose_error_base_torch()`, `tcp_jacobian_torch()`, `nullspace_projector_torch()`. GPU versions checked against `franka_motion` NumPy functions. Disable implicit actuator drives on this path and add simulation gravity compensation in the audited order. |
| `mdp/observations.py` | `measured_joint_state()`, `tcp_pose_base()`, `tcp_twist_base()`, `applied_motion_reference()`, `motion_trial_status()`. Frame-correct, named telemetry from feedback and applied controller state. |
| `mdp/events.py` | `reset_motion_state()`, `configure_payload()`, `apply_test_wrench()`, `clear_test_wrench()`. Deterministic initial conditions, audited payload composition, scheduled known force/torque disturbances for fixture tests; force point/frame explicitly recorded. |
| `franka_rl/experiments/motion_runner.py` (relative to `source/franka_rl/`) | `IsaacMotionRunner`, `IsaacKinematicsAdapter`; `run_trial()`, `validate_motion_path()`. Connect shared schedules to environment action tensors, preview IK/clearance samples, trial reset/startup hold, batched Torch reference evaluation from exported coefficients, sampling and artifact finalization. |
| `franka_rl/experiments/motion_suite.py` (relative to `source/franka_rl/`) | `MotionSuiteCoordinator`; `build_sim_command()`, `validate_trial_artifacts()`, `compile_motion_suite()`. Serial isolated Isaac processes per controller/task/profile where needed, repeat pairing, resumable immutable artifacts and aggregate metrics. Reuse coordinator lifecycle ideas rather than treating motion trials as PPO evaluations. |

The last two rows name their actual package paths explicitly; they are not files inside `mdp/` or the task directory. Baseline DLS uses the installed `DifferentialIKController` as a cross-check. The installed `OperationalSpaceController` and `run_osc.py` are references for Jacobian/frame/dynamics plumbing, not a claim that its inertial/null-space control matches Franky.

### 3.6 Executable scripts and hardware adapter files

All paths here are relative to the repository root. Entry points call package implementations; they do not contain separate copies of controller math.

| New file | Planned classes/functions and behavior |
|---|---|
| `scripts/motion_control/preview_trajectory.py` | `main()`. Load named YAML trial, evaluate analytic/planned references, show Cartesian volume/path and derivative plots, write validation metadata. Pure-math preview needs no Kit; optional robot/IK preview dispatches to the simulation runner. |
| `scripts/motion_control/audit_motion_inputs.py` | `main()`. Read-only USD/URDF hash and joint-limit checks, Isaac Lab revision check, installed distribution versions and archived runtime-validation evidence. Reports the unresolved hardware controller audit; does not import Franky or launch Kit. Implemented in M0. |
| `scripts/motion_control/check_joint_artifacts.py` | `validate_trial()`, `main()`. Offline finalization/hash checks, accepted-trace checks, clone/repeat and matched single/batched q/dq parity. Implemented in M1. |
| `scripts/motion_control/run_sim_demo.py` | `main()`. Resolve one YAML trial and task, use the current `add_launcher_args` / `setup_preset_cli` / `launch_simulation` flow, create Gym environment, run/finalize a trial and close cleanly. Headless/visual modes share algorithm/timing settings. |
| `scripts/motion_control/run_sim_suite.py` | `main()`. Expand a named YAML suite and invoke `MotionSuiteCoordinator`, support resume/fail-fast and emit compiled index. All target choices come from YAML. |
| `scripts/motion_control/analyze_motion.py` | `main()`, `plot_tracking()`, `plot_joint_dynamics()`, `plot_fixture_posture()`, `plot_suite_comparison()`. Offline analysis of simulation or hardware artifacts; matched units/phase masks, figures and concise Markdown summary. No simulator launch. |
| `scripts/motion_control/build_handoff.py` | `main()`, `build_motion_handoff()`. Package the lightweight shared wheel, selected YAML suites/profiles, hardware adapter, dependency/version pins, numerical reference fixtures, simulation acceptance artifacts and SHA-256 manifest. No ONNX or PPO models. |
| `deployment/franky_runtime/run_motion_suite.py` | `main()`. Load transferred YAML suite locally on the real-time machine; fake/no-motion/hardware mode is configuration controlled; invoke `FrankyMotionRunner`. No workstation communication or interactive waypoint selection. |
| `deployment/franky_runtime/franky_experiment/motion_demos/__init__.py` | Lightweight adapter exports; importing them does not open FCI. |
| `deployment/franky_runtime/franky_experiment/motion_demos/backend.py` | `FrankyMotionBackend`, `FakeMotionBackend`; `audit_motion_capabilities()`, `start_joint_tracking()`, `start_cartesian_tracking()`, `set_reference()`, `set_posture()`, `execute_library_motion()`, `stop()`. Reuse the verified runtime's state/FK/logging/safety helpers by composition; adapt to the pinned joint/Cartesian trackers and `PostureTask`, not `policy_worker.py`. Keep mandatory torque loops in Franky C++. |
| `deployment/franky_runtime/franky_experiment/motion_demos/runner.py` | `FrankyMotionRunner`; `run_suite()`, `run_trial()`. Apply the shared trial/schedule lifecycle, publish timestamped references, buffer telemetry, enforce local watchdog and reviewed stop procedures, and run Ruckig/waypoint library baselines. Fake tests execute without connecting to a robot. |

The hardware adapter and its fake tests are developed here. Actual wheel/binding probes, shadow runs and motion commissioning remain work for the real-time machine after transfer. A missing Cartesian API should produce an explicit unsupported result; no silent library upgrade or fallback controller.

### 3.7 Configuration, verification, and documentation files

Config paths are relative to `source/franka_motion/franka_motion/config/`.

| New file | Contents |
|---|---|
| `fr3_workspace.yaml` | Live-model/URDF provenance, joint aliases, base/flange/TCP transform, named initial postures, nominal payload, commanded/measured limits, clearance checks and hardware review status. |
| `trajectories.yaml` | Named joint PTP/waypoints, circle/Lissajous center/plane/radius/amplitudes, constant tool orientation, phase ramp durations, approach/exit time laws, finite fixture segment and posture targets. Reuse the concept of `paths.yaml` without changing its existing waypoint-success contract. |
| `controller_profiles.yaml` | Explicit joint/Cartesian K/D units, secondary gains/projector type, reference-update rate/delay/hold, compensation, error clipping, torque-rate/filter/stop settings. Separate simulation candidates from hardware-reviewed profiles. |
| `suites/simulation_smoke.yaml` | Deterministic hold, small joint PTP, circle, feedforward comparison, fixture wrench test and two-posture trial. Start with one environment; verify multi-environment parity separately. |
| `suites/demo_a_tracking.yaml` | Slow/medium speed × pose-only/pose+twist × repeats; separate Ruckig/DLS/library baseline cells; matching initial state and trajectory phase. |
| `suites/demo_b_fixture_posture.yaml` | Known simulated axial/transverse forces, finite fixture duration, posture A/B and weak-secondary baseline; hardware manual interaction explicitly review gated. |

| New test/document file | Planned verification/functions |
|---|---|
| `tests/motion_control/test_trajectories.py` | `test_analytic_derivatives()`, `test_entry_exit_continuity()`, `test_rotated_plane()`, `test_reference_reconstruction()`. Finite-difference cross-checks, zero start/end speed, correctly transformed twist/jerk and timing. |
| `tests/motion_control/test_joint_planning.py` | `test_ruckig_bounds_and_endpoints()`, `test_ik_path_seed_continuity()`, `test_tcp_jacobian()`. Limits through jerk, feasible full-pose IK continuity, TCP offset cross-product convention. |
| `tests/motion_control/test_impedance.py` | `test_numpy_torch_controller_parity()`, `test_pose_error_sign()`, `test_nullspace_projection()`, `test_filter_slew_and_reset()`. Full-rank/rank-deficient Jacobians, torque decomposition, projector leakage and deterministic selective reset; Franky golden fixtures when available. |
| `tests/motion_control/test_virtual_fixture.py` | `test_line_projection()`, `test_segment_endpoints()`, `test_fixture_twist_and_frames()`. Axial/transverse separation, excursion limits and velocity filtering. |
| `tests/motion_control/test_metrics.py` | `test_circle_error_3d()`, `test_tracking_vs_contouring()`, `test_timestamp_alignment()`, `test_phase_masks()`. Known phase lag, off-plane error, quaternion sign equivalence, dropout/invalid-sample accounting. |
| `tests/motion_control/test_trial_and_handoff.py` | `test_yaml_resolution()`, `test_trial_fault_latch()`, `test_fake_backend_lifecycle()`, `test_bundle_hash_and_fixture_parity()`. Failures retain terminal state, no implicit restart, hardware package imports without Isaac Lab, and identical shared reference evaluation after transfer. |
| `docs/industrial_motion_control.md` | Final setup/run/analysis instructions, timing/frame/controller contract, measured findings and library-vs-original contribution table. Filled from completed experiments, not from proposed results. |

### 3.8 Existing files to reuse or minimally update

- Reuse `scripts/deployment/build_fr3v2_asset.py`, `validate_fr3_model.py`, and the model audit. Record `FRANKA_RL_FR3_USD` instead of rebuilding silently.
- Reuse `franka_rl/.envrc`, the shared `/home/chen-lab/isaac/.venv`, and `direnv exec /home/chen-lab/isaac/franka-rl ...` for all Isaac processes, including jobs launched from the data volume. Install only the shared package and any pinned planner dependency into this environment.
- Reuse the physical torque law/filter implementation in `franka_rl/tasks/manager_based/franka_rl/mdp/impedance_controller.py` as an audited reference for the joint route. Do not reuse the PPO action map/integrator, policy evaluator's success termination, or random point/waypoint command scheduling for continuous paths.
- Reuse verified pieces of `deployment/franky_runtime/franky_experiment/backend.py` and `core.py`: encoder-derived flange feedback, version checks, buffers and stopping procedures. Preserve old policy routes. Resolve the absence of `franky_experiment/__init__.py` deliberately; its current namespace-package layout does not justify altering policy imports.
- Update `source/franka_rl/setup.py` only as required to depend on/install `franka_motion` and package the new task. The new shared wheel owns the nested motion YAML resources; do not rely on the existing `config/*.yaml` glob to include them.
- Update `.gitignore` narrowly so required new `__init__.py` files are tracked despite `**/__*`; keep generated data excluded. Document the new demo in the existing root `README.md` and summarize transfer requirements in `deployment/HANDOFF_AGENT.md` once simulation verification passes.

### 3.9 Logging, artifacts, and timing

**Logging minimum fields:** monotonic host timestamp and robot/simulation time, reference publication timestamp/sequence, active applied-reference sequence, desired pose/twist, measured pose/twist, measured `q,dq`, reference joint state if applicable, phase, gains, payload, fault/stop event, and loop/tick timings. Log requested/slew-limited/filtered/applied torque separately in simulation; identify the available hardware commanded/measured/external signals and their compensation content. Pose feedback is from measured encoders/FK, not echoed commands. Store TCP and flange pose distinctly when a nonidentity tool transform is used.

**Sampling:** preallocate GPU buffers for simulation substeps and flush after a trial or in bounded batches. No filesystem writes, CPU conversions, or per-environment Python loops inside torque stepping. On hardware, retain the established native callback/ring-buffer logging path; if its callback crosses into Python, record the limitation and measured timing rather than labeling it hard real time. Keep latency-sensitive caches/temp state on the ext4 Isaac volume, and artifacts under the mounted/writable data root.

```text
${FRANKA_RL_DATA_ROOT}/motion_control/
├── simulation/<suite-id>/<trial-id>/
├── hardware/<suite-id>/<trial-id>/
├── comparisons/<comparison-id>/
└── handoff_bundles/<bundle-id>/
```

Each trial writes `manifest.json`, resolved configuration, `references.csv`, measured/controller `trace.npz` (and hardware trace CSV as available), `events.json`, `summary.json`, and `completed.json` only after validation. The suite writes a trial index, comparison table, figures, and excluded/faulted-trial counts. Trace schemas include sample rate and interpolation rules; absent signals remain absent, not fabricated zeros.

Simulation references use simulated elapsed time. Hardware references use the recorded monotonic elapsed time and actual update timestamps; pose/twist belong to the same reference instant. Log scheduled and applied references separately so both planner error and controller error can be analyzed. Report raw timestamp-aligned error first; any estimated time-shift correction is an additional explicitly labeled diagnostic. Exclude approach/exit only through named phase masks, never by an undocumented cut.

### 3.10 Frames, controller equations, and equivalence limits

**Markdown math convention:** Use `$...$` for inline equations and `$$` on separate lines for display equations, with blank lines around display blocks. This is supported by VS Code's Markdown math preview; leave `markdown.math.enabled` enabled. Equations below use KaTeX-compatible commands, not `\(...\)` or `\[...\]` delimiters.

All Cartesian vectors, orientation errors, twists and Jacobians are expressed in `fr3_link0`. The controlled origin is the configured TCP, initially the flange. **This pinned Isaac Lab checkout uses `xyzw`, as does the shared contract.** Franky-facing quaternion arguments must also be explicitly verified/labeled `xyzw`; do not apply a legacy Isaac `wxyz` conversion. Use the same TCP offset for feedback, target and Jacobian. If the base-frame vector from flange to TCP is $\mathbf r$, the shifted translational Jacobian is

$$
J_{v,\mathrm{TCP}}=J_{v,\mathrm{flange}}-[\mathbf r]_\times J_\omega.
$$

For a desired rotation $R_d$ and measured rotation $R$, define the base-frame error and measured twist as

$$
\mathbf e=
\begin{bmatrix}
\mathbf p_d-\mathbf p\\
\operatorname{Log}(R_dR^T)^\vee
\end{bmatrix},
\qquad
\mathbf V=J\dot{\mathbf q}.
$$

The planned Cartesian spring-damper and joint PD reference implementations are

$$
\mathbf F_d=K_x\mathbf e+D_x(\mathbf V_d-\mathbf V),
\qquad
\boldsymbol\tau_{\mathrm{task}}=J^T\mathbf F_d,
$$

$$
\boldsymbol\tau_{\mathrm{joint}}
=K_q(\mathbf q_d-\mathbf q)+D_q(\dot{\mathbf q}_d-\dot{\mathbf q}).
$$

For an unweighted kinematic pseudoinverse, the conceptual torque projector and posture task are

$$
N_\tau=I_7-J^T(J^T)^\dagger,
\qquad
\boldsymbol\tau_{\mathrm{posture}}
=N_\tau\left[K_n(\mathbf q_n-\mathbf q)-D_n\dot{\mathbf q}\right].
$$

An orthogonal kinematic projector removes the component along $J^T$; it does not guarantee decoupled task acceleration under a nonidentity inertia matrix. Damping a pseudoinverse also introduces task leakage. A dynamically consistent alternative must be labeled as a separate controller, not silently substituted for the Franky projector. Audit the selected Franky source for orientation-error definition/sign/scaling, pseudoinverse threshold/weighting, posture damping, error clips and gain/reference interpolation. If Franky uses a different orientation-error vector than the logarithm above, implement that exact vector as the hardware-matching profile and test it explicitly; the logarithmic error remains a geometric metric.

With physical simulation gravity enabled, the dynamics and torque processing are

$$
M(\mathbf q)\ddot{\mathbf q}+\mathbf c(\mathbf q,\dot{\mathbf q})+\mathbf g(\mathbf q)
=\boldsymbol\tau_{\mathrm{motor}}+J^T\mathbf F_{\mathrm{ext}},
$$

$$
\boldsymbol\tau_{\mathrm{sim}}
=\mathcal S\!\left(\boldsymbol\tau_{\mathrm{task/joint}}
+\boldsymbol\tau_{\mathrm{posture}}+\mathbf c
+\boldsymbol\tau_{\mathrm{limit}}\right)+\mathbf g.
$$

Here $\mathcal S$ denotes the explicitly recorded error/torque constraints, torque-rate limit and low-pass filter in the selected controller order; error clipping is applied before constructing task/joint torque. The existing joint simulation adds gravity after its torque filter. The hardware comparison must follow the actual robot/Franky compensation split: account for robot-side gravity without adding it twice. A library baseline's internal robot position controller is not reproduced merely by applying the same Ruckig reference to our explicit PD servo. Label that simulation cell as a reference/planner baseline until its plant/controller is identified.

Isaac Lab OSC can be useful as a separately labeled comparator. Passing these planned tests establishes algebra/reference parity and simulated behavior; it does not establish real-plant dynamic equivalence. The previous Axis-4 high-gain discrepancy makes this distinction especially important.

## 4. Demo A — Industrial trajectory planning and continuous tracking

### A.1 Learning objectives / interview claims

- Understand joint-space PTP vs Cartesian motion planning; path geometry vs time parameterization.
- Explain velocity, acceleration, and **jerk** limits and why industrial motion generators use them.
- Generate analytical Cartesian trajectory with continuous position/velocity references; track it through an impedance controller.
- Separate **time-synchronized tracking error** from **geometric contouring error**.
- Examine how speed, reference feedforward, and controller stiffness/damping affect tracking performance.

### A.2 Experiment A0: simple joint-space PTP (sanity check)

In simulation, start from a checked posture, generate joint PTP references using the pinned Ruckig wrapper, and track them through the explicit joint servo. Verify joint limits and derivatives first. On hardware, run the same reference through `JointImpedanceTracker` for a matched-reference controller test; additionally execute `JointMotion` / `JointWaypointMotion` at conservative `relative_dynamics_factor` as a separately labeled built-in baseline.

Expected output: a short video plus `q(t)`, `dq(t)`, estimated `ddq(t)`, and (if defensible at sample rate) smoothed estimated jerk; note numerical differentiation amplifies noise. Check joint position error at arrival, velocity/acceleration bounds, and smooth transitions.

Use franky's Ruckig-backed jerk-limited trajectory generation as a **library baseline**, not original planner implementation. Default waypoint velocity is zero, so intermediate waypoints **stop by default**; explicitly configure nonzero waypoint velocities for continuous blending if using that comparison.

### A.3 Experiment A1: analytic continuous reference (primary deliverable)

For a fixed safe plane and constant orientation, define a small circle:

$$
\begin{aligned}
\mathbf p_d(t)&=\mathbf p_c+rR_P
\begin{bmatrix}\cos\theta(t)\\\sin\theta(t)\\0\end{bmatrix},\\
\dot{\mathbf p}_d(t)&=r\dot\theta(t)R_P
\begin{bmatrix}-\sin\theta(t)\\\cos\theta(t)\\0\end{bmatrix},\\
\ddot{\mathbf p}_d(t)&=rR_P\left(
\ddot\theta(t)\begin{bmatrix}-\sin\theta(t)\\\cos\theta(t)\\0\end{bmatrix}
-\dot\theta(t)^2\begin{bmatrix}\cos\theta(t)\\\sin\theta(t)\\0\end{bmatrix}
\right).
\end{aligned}
$$

$R_P$ rotates the local XY circle plane into the robot base frame; it is independent of the constant desired tool orientation. For the constant-speed segment, $\theta(t)=\omega t+\theta_0$. During entry/exit, use the full derivatives of the phase schedule, not constant-speed formulas. Define $\mathbf u=[\cos\theta,\sin\theta,0]^T$ and $\mathbf v=[-\sin\theta,\cos\theta,0]^T$; analytic jerk is

$$
\dddot{\mathbf p}_d
=rR_P\left[(\dddot\theta-\dot\theta^3)\mathbf v
-3\dot\theta\ddot\theta\mathbf u\right].
$$

For a Lissajous option:

$$
\mathbf p_d(t)=\mathbf p_c+R_P
\begin{bmatrix}a\sin(\omega_x t+\phi)\\b\sin(\omega_y t)\\0\end{bmatrix}.
$$

`LissajousTrajectory` must also provide analytic velocity/acceleration/jerk and apply the same smooth time-law chain rule during transitions.

Use a smooth **entry/exit transition** rather than starting instantly on the circle at nonzero tangent speed; e.g., a bounded-ramp angular phase with separate approach/settle and ramp-up/slow-down phases. Check speed, acceleration and jerk throughout the complete concatenated reference. Hold a validated constant tool orientation. In simulation, publish pose/twist through `CartesianMotionAction`; the pose-only cell supplies zero desired twist while retaining the identical pose schedule. On hardware, use `CartesianImpedanceTracker.set_target(Affine(...), Twist(...))`, where twist is expressed in the robot base frame. Record actual publication/application timestamps. This base-frame twist contract is documented in the [Franky impedance guide](https://timschneider42.github.io/franky/guide/impedance-control.html).

Before dynamics trials, use seven-joint DLS to check **full-pose** feasibility of the entire path from the planned initial posture. The earlier position-only reachability result is insufficient for fixed-orientation circles. Continuous solution seeds, singularity diagnostics, joint margins and representative collision checks are part of the offline/Isaac gate. Unreachable or discontinuous paths are rejected rather than tuned through larger gains.

**Experiment matrix (minimum):**

| Factor | Settings | Comparison purpose |
|---|---|---|
| Geometry | Circle; optional Lissajous | Simple path then coupled multi-axis motion |
| Speed | Slow; medium **after validation** | Check bandwidth-induced tracking degradation |
| Reference input | Pose-only vs pose + analytic twist | Test reference-velocity feedforward |
| Trajectory mode | Built-in waypoint/Ruckig vs analytic impedance tracking | Qualitative architecture comparison; do not attribute all differences to one factor |
| Gains (optional) | Two prevalidated stiffness/damping settings | Show response/smoothness tradeoff |

For a rigorous feedforward ablation, **hold geometric reference, controller, gains, velocity, duration, and initial state fixed**; vary only whether desired twist is supplied. Built-in waypoint control vs impedance tracking is a useful systems baseline but **not** a controlled single-variable test because control laws/path timing differ.

### A.4 Error definitions and plots

For measured end-effector position $\mathbf p_k$ and a timestamp-aligned desired position $\mathbf p_d(t_k)$:

$$
e_{\rm track}(t_k)=\|\mathbf p_d(t_k)-\mathbf p_k\|,\qquad
\mathrm{RMSE}_{\rm track}=\sqrt{\frac{1}{N}\sum_{k=1}^{N} e_{\rm track}(t_k)^2}.
$$

For a circle with base-frame unit normal $\mathbf n=R_P[0,0,1]^T$, let $\mathbf a=\mathbf p_k-\mathbf p_c$, $h=\mathbf n^T\mathbf a$, and $\rho=\|\mathbf a-h\mathbf n\|$. Its exact 3-D contour error is

$$
e_{\rm circle}(t_k)=\sqrt{h^2+(\rho-r)^2}.
$$

For a general sampled curve, approximate closest-point distance by dense reference sampling or a segment projection search:

$$
e_{\rm contour}(t_k)=\min_{s\in[0,T]}\|\mathbf p_k-\mathbf p_d(s)\|.
$$

Track error contains timing/phase lag; contour error ignores phase and reflects path-shape accuracy. Also report max track error, orientation error on $SO(3)$, measured speed and overshoot, timing jitter, and rejected/faulted runs. For velocity/acceleration plots, state smoothing/differentiation assumptions. The orientation metric is $e_R=\|\operatorname{Log}(R_dR^T)^\vee\|$ in radians; use the same definition on simulation and hardware feedback.

**Required figures:** (1) desired vs measured XYZ trajectory in Cartesian space; (2) tracking and contouring errors versus time; (3) summary bar/table for speed × feedforward trials. Optional: joint velocity/acceleration/jerk-limit figures.

**Acceptance gate (process, not invented numeric success):** safely execute ≥2 repetitions of a conservative circle with logs; plot finite, correctly timestamped measures; document the observed max/RMSE and any failures. No promise of a target mm accuracy before experimentation.

### A.5 Implementation notes

- First implement the math and plotting with **synthetic data/offline**; validate smooth entry/exit and units.
- Use current pose as center/orientation only if the whole path is inside a validated workspace; otherwise choose a checked center and go there with a safe approach motion.
- Do not compare a sampled, stop-at-every-waypoint polyline as if it were a continuously tracked geometric circle. For a fair geometry comparison, command nonzero waypoint velocities and evaluate the **actual planned geometry** where available.
- For higher-frequency tracking, if Python tick jitter is significant, move reference generation to a lower-jitter validated control path or reduce update demands; report any limitation rather than hiding it.

## 5. Demo B — Virtual fixture + null-space posture control

### B.1 Virtual fixture (B1)

**Task:** With the end-effector initially at a safe $\mathbf p_0$, define a robot-base-frame unit direction $\mathbf d$ (e.g., vertical). First inject known bounded axial and transverse wrenches in Isaac Lab, then consider deliberate light manual displacement on hardware under the reviewed operating procedure. Permit motion along $\mathbf d$ and resist transverse translation and rotational deviations through Cartesian impedance.

Project measured position onto the desired line:

$$
s=(\mathbf p-\mathbf p_0)^T\mathbf d,\qquad
\mathbf p_{\rm ref}=\mathbf p_0+s\mathbf d,\qquad
\mathbf P_\perp=I_3-\mathbf d\mathbf d^T.
$$

The lateral displacement $\mathbf e_\perp=\mathbf P_\perp(\mathbf p-\mathbf p_0)$ is driven toward zero. Implement projection onto a finite segment with configured $s_{\min},s_{\max}$; unrestricted line motion must not leave the approved workspace. For ideal unbounded projection, a useful permitted-axis reference velocity is $\dot{\mathbf p}_{\rm ref}=\mathbf d\mathbf d^T\dot{\mathbf p}$, with a configured filter and bounds. At a segment endpoint, suppress outward reference motion. Record the filter/rate because sampled projection need not produce ideal zero axial stiffness. Feed pose and the bounded twist into the Cartesian action/Franky tracker, and hold initial orientation. **Choose conservatively tuned stiffness and damping**, and do not infer wrench safety from projection alone.

Reference implementation: [`cartesian_impedance_tracker.py`](https://github.com/TimSchneider42/franky/blob/master/examples/cartesian_impedance_tracker.py), whose current code updates the projected pose online.

**Quantify:** permitted-axis travel ($s$), off-axis error $\|\mathbf e_\perp\|$, orientation deviation, and response to known axial/transverse simulation wrenches and gentle, comparable hardware pushes. If no external force sensor is calibrated, avoid claiming a precisely measured hardware stiffness; describe the behavior qualitatively and show geometric errors. Reference perturbations and physical external-wrench tests must be reported separately.

**Safe alternative if manual guidance is not approved:** Demonstrate an automatically commanded small line trajectory with deliberately injected, simulated/no-contact off-axis pose-reference perturbations and measured Cartesian correction; do not imply physical human-contact validation.

### B.2 Null-space posture control (B2)

**Task:** Keep a fixed Cartesian end-effector pose while biasing FR3's redundant joints toward two different safe postures. Optional stretch task: bias away from a joint limit or favor manipulability.

Kinematic intuition (full-rank 6-DoF task, 7 joints):

$$
\dot{\mathbf q}=J^{\dagger}\dot{\mathbf x}+(I-J^{\dagger}J)\mathbf z.
$$

The second term produces instantaneous first-order self-motion without changing end-effector velocity. For a singularity-robust diagnostic one may examine damped least-squares IK:

$$
\dot{\mathbf q}=J^T(JJ^T+\lambda^2 I)^{-1}\dot{\mathbf x}_d.
$$

**Important:** franky's `PostureTask` and `ManipulabilityTask` implement *torque-level secondary objectives*, not exactly the above velocity-level Moore–Penrose formula. Present the formula as conceptual kinematic background; inspect library behavior rather than claiming exact controller equivalence.

Use `CartesianImpedanceTracker(..., posture_task=PostureTask(...), ...)` as documented. Secondary task must not demand joint-limit violations or leave a safe workspace. With a torque-level task the posture equilibrium and Cartesian target interaction are gain-dependent; characterize measured pose drift.

In simulation, implement the explicit projected posture torque from §3.10 and vary two feasible posture targets under the same fixed Cartesian reference. Check projector rank/leakage numerically and measure task drift dynamically; seven joints do not guarantee that arbitrary posture targets are compatible with the chosen fixed pose. Begin with slow target transitions and a weak secondary objective. The installed Franky projector/gain semantics must be matched before transferring the comparison.

**Quantify:** end-effector pose position/orientation error (to fixed reference), joint position change $\|\Delta\mathbf q\|$, joint-limit margins, and optional manipulability metric $\sqrt{\det(JJ^T)}$ away from singularity (take care of scaling translational/rotational units; use a consistent definition). Show video of visible elbow/joint reconfiguration with measured end-effector displacement.

**Acceptance gate:** at least two distinct, safe posture settings with measured joint change and a documented end-effector pose-error trace. If the observed posture change is negligible, report that and first check task strength, null-space feasibility, and safety limits rather than forcing large gains.

### B.3 Optional B3 (only if time remains): anisotropic joint/cartesian compliance

`JointImpedanceTracker` can combine low joint stiffness with selected Cartesian stiffness; demonstrate directional compliance or one joint posture bias without introducing unsafe contact scenarios. **Not required** for the two-day plan.

## 6. Schedule, milestones, and fallback

### M0 implementation status — 2026-10-11

The standalone package, strict YAML catalogs/suite expansion, physical joint/Cartesian contracts, quintic boundary-value segments, smooth phase-speed ramps, circle/Lissajous paths and approach/settle/track/exit references are implemented. `test_m0.py` verifies analytic derivatives, joins, sweep expansion, reference bounds, schema rejection, hardware-approval gating and simulator-free imports. The offline preview and model audit commands are documented in [docs/industrial_motion_control.md](docs/industrial_motion_control.md).

Local evidence checks pass: the FR3v2.1 bare-flange live USD and robot-returned URDF match their recorded SHA-256 hashes; YAML model joint limits match the URDF; Isaac Lab revision is `ffff603eafc6b74264a5261cc0183d6a65390d78`. Installed distribution metadata reports Isaac Sim `6.0.1.0` and Isaac Lab `6.1.14`; the Git revision is the authoritative source pin. Archived runtime validation reports success, but M0 does **not** perform a new PhysX launch.

The current pinned Franky Cartesian/posture controller implementation is not available on this workstation. Its audit remains an explicit M0 evidence gap, requested in `deployment/HANDOFF_AGENT.md`; no controller equivalence or hardware readiness is claimed. All profiles remain `hardware_approved: false`. The preview start is an explicit numerical point, **not** the FK of the home joint posture. Full-pose IK, collision, dynamics and controller parity are subsequent milestones. The directory tree above is the complete planned architecture, not a claim that later-milestone files already exist; `demo_b_fixture_posture.yaml` is deferred to M3, and the current smoke catalog covers M0 Cartesian reference math only.

### M1 implementation status — 2026-10-11

The independent joint task `Franka-FR3v2-MotionControl-Joint-v0` is implemented, with exact configured resets, physical 14D references, a 1 kHz explicit joint torque loop, 50 Hz zero-order-held publication, gravity enabled, zero implicit K/D, runtime contact/fault monitoring, no automatic resets, bounded GPU telemetry and terminal/first-fault preservation. Local Ruckig 0.17.3 planning and `suites/joint_smoke.yaml` provide stationary hold and repeated short PTP trials. Only the joint task is registered so far; Cartesian/fixture/nullspace registrations are deferred to M2/M3.

27 offline tests pass. Live single/four-clone PhysX hold and PTP tests pass: maximum hold drift $4.37\times10^{-8}$ rad, PTP terminal error $4.70\times10^{-5}$ rad, peak measured joint speed 0.0886 rad/s, no nominal faults. Repeats match exactly; clone/single-batched q/dq differences remain below $10^{-6}$ componentwise. Invalid-reference rejection, first-fault retention and selective-reset plant/filter isolation pass. Small disposable verification artifacts are under `/tmp/franka-motion-m1-*`; reproduce durable evidence using the data-volume command in [docs/industrial_motion_control.md](docs/industrial_motion_control.md).

Joint torque processing is explicitly documented and compared algebraically between NumPy/Torch; its final effort clamp and fault-stop candidate are simulator safeguards, not claims of complete Franky parity. The archived joint controller informed the slew/filter/compensation order. The current hardware Cartesian/posture audit remains pending; no physical motion or new hardware adapter has been implemented. Multi-profile process coordination and Cartesian tracking remain later milestones.

### M2 implementation status — 2026-10-11

`Franka-FR3v2-MotionControl-Cartesian-v0` now supplies the physical 13D
base-frame flange pose/twist contract, explicit Cartesian log-error spring /
twist damper / Jacobian-transpose torque, compensation and bounded telemetry.
It shares the non-RL manager-based scene, explicit reset and terminal fault
semantics with M1. NumPy/Torch Cartesian torque parity and quaternion handling
are tested. The feedback twist uses the **flange actor origin**, not Isaac's
COM-velocity shorthand. No operational-space inertia shaping, IK motion servo,
or null-space/posture torque is silently added.

Actual new M2 files:

- `franka_motion/kinematics.py`: `UrdfKinematicsProvider`,
  `DampedLeastSquaresIK`, `pose_error_base()`, `validate_path_ik()` and SO(3)
  quaternion helpers; sampled full-pose feasibility with continuous seeds,
  interior joint margins and mixed-unit singularity diagnostics.
- `franka_motion/metrics.py`: exact `circle_contour_error()` and explicitly
  approximate `sampled_curve_contour_error()` for optional Lissajous analysis.
- `franka_rl/experiments/cartesian_motion_runner.py`:
  `IsaacCartesianMotionRunner.reset()`, `commissioning()`, `run_trial()`;
  runtime URDF/Isaac FK/Jacobian and official-DLS cross-check, measured entry
  pose, whole-path guards, 1 kHz buffered artifacts and tracking summaries.
- `scripts/motion_control/run_sim_suite.py`: serial isolated-profile
  commissioning → tracking → compilation with fail-fast logs/manifest. It is
  an immutable fresh-run coordinator, not yet a resumable hardware coordinator.
- `scripts/motion_control/validate_motion_paths.py`: `main()`; simulator-free
  reference derivative and complete full-pose DLS checks from named YAML suites,
  using archived URDF home FK rather than a numerical preview start.
- `scripts/motion_control/compile_cartesian_results.py`: `validate()`,
  `differences()`, `compile_results()`; finalized hashes, physical repeat/clone
  tolerances, metric table and path/error/dynamics/speed-feedforward figures.
- `tests/motion_control/test_m2.py`: FK finite differences, full-pose DLS,
  unreachable-target rejection, SO(3) approach derivatives, torque parity,
  partial reset, profile rejection and actor-origin twist regression coverage.

Existing controller/action/config files and `run_sim_demo.py` are extended;
M1 and PPO task IDs remain intact. `demo_a_tracking.yaml` is the two-speed,
pose-only/pose+twist, two-repeat matrix. `demo_a_gain_sweep.yaml` compares soft
and moderate gains at 0.4 rad/s with feedforward and two repeats. Each profile
must first pass a stationary hold, 2 mm smooth reference step and 10% gain step.
The new `circle_yz_m2` holds the home orientation, and a bounded approach/exit
connects measured home to the track; it never substitutes the numerical preview
point for home feedback. Analytic reference limits and 20 Hz full-pose DLS checks
cover the entire approach/settle/circle/exit.

See [docs/industrial_motion_control.md](docs/industrial_motion_control.md) for
commands, acceptance criteria and verified results. This remains a simulation
candidate: the real-time Franky Cartesian audit is still pending, runtime
contact sensing is not a mesh-clearance certificate, and all profiles retain
`hardware_approved: false`. M3 fixtures/posture and M4 hardware transfer remain
separate work; no physical motion is authorized.

M2 workstation acceptance is verified: 42 unit/regression tests pass; all eight
two-clone CPU circle trials and both two-repeat GPU gain cells pass without
configured faults/contact, with finalized artifact/physical parity checks.
At soft gains, tracking RMS changes from 1.94 to 0.158 mm with twist feedforward
at 0.2 rad/s, and from 3.78 to 0.515 mm at 0.4 rad/s. The matched GPU moderate
gain cell achieves 0.349 mm RMS. Full-path IK and runtime FK/Jacobian/official
DLS checks pass; M1 hold/PTP/reset checks remain passing. Final buffered evidence
and reviewed figures are in `/tmp/franka-motion-m2-final-compiled`; use the
documented data-volume commands for durable results. These are simulation-only
findings, not robot accuracy/safety guarantees.

The workstation phase comes first. The old schedule's hardware-first example run is replaced by the following ordered milestones; only M4 crosses to the real-time machine.

| Milestone | Work | Deliverable / stop condition |
|---|---|---|
| M0 — contracts and model (first) | Confirm local Isaac version/model provenance and obtain the pinned Franky Cartesian/secondary-controller audit; write shared schemas/YAML and pure reference math | Importable standalone `franka_motion`, consistent frames/quaternions/units, finite analytic paths with derivative checks. Unknown hardware behavior is documented rather than guessed. |
| M1 — joint servo and PTP | New manager-based non-RL scene/task, deterministic reset/hold, explicit joint torque control, Ruckig PTP and buffered telemetry | Stationary hold and short PTP without unintended startup motion; correct torque compensation/filtering, measured bounds and preserved terminal/fault samples; single/batched/reset parity. |
| M2 — Demo A in simulation (priority) | Full-pose DLS path checks, Cartesian impedance action, ramped circle, pose-only vs pose+twist, then speed/gain cells and optional Lissajous | At least two repeatable conservative circles per chosen cell, path/error/dynamics figures and compiled metrics. Nominal hold and controller parameter step tests pass before high-gain sweeps. |
| M3 — Demo B in simulation | Finite virtual fixture with known wrenches; fixed EE pose with two secondary posture targets | Axial/transverse response and posture/task-drift plots; known wrench frames, feasible configurations, no unexplained numerical instability or limit crossing. Finish M2 first if time is short. |
| M4 — transfer package | Build lightweight wheel + YAML + Franky adapter + numerical/golden reference fixtures + acceptance evidence + manifest | Fake backend and package-import tests pass without Isaac Lab; paired NumPy/Torch/reference tests pass; exact unsupported/missing hardware APIs are listed in agent handoff. No hardware control from the workstation. |
| M5 — local hardware commissioning | Real-time agent validates installed APIs, tool/payload/workspace, fake/no-motion checks and stop transition, then runs conservative Demo A and optionally B | Logged hardware reference/feedback, timing/fault accounting and videos; matching suite cells for analysis, explicitly recorded controller/model differences. All experiment execution runs locally on that machine. |

**Fallback if tracker/API upgrade is blocked:** Finish a convincing joint PTP + Cartesian waypoint comparison and tracking analysis with current installed franky, and note impedance extensions as planned. Never destabilize a working hardware environment to meet the schedule.

### 6.1 Simulation verification matrix

| Check | Evidence required before transfer |
|---|---|
| Numerical reference correctness | Analytic derivatives agree with independent numerical checks; correct frame transforms; smooth approach/ramp/exit; each configured velocity/acceleration/jerk bound checked. |
| Model and kinematics | Audited live model loaded; seven-joint FK/Jacobian/TCP conventions validated; full-pose IK feasible along the complete path; joint margins and self-contact/clearance checks recorded. |
| Controller startup | Hold the exact reset pose with zero desired twist; seed filter/controller state deliberately and record startup transients. No accidental PPO reference normalization or initialization prime. |
| Algebraic parity | NumPy vs batched Torch torque components, null-space projection, clipping, filter and selective reset; Franky reference fixtures from the selected source when obtainable. |
| Task behavior | Ruckig PTP, circle with/without twist, known fixture wrenches, two posture targets; measured errors/dynamics and exact stop reasons reported, including failures. |
| Trial semantics | Simulation time independent of wall-clock speed; no reset during tracking; phase masks reproducible; first unsafe/nonfinite event and terminal feedback retained. |
| Repeat and batch behavior | Identical deterministic trials repeat within declared numerical tolerance; single vs cloned-environment parity; no base/world-origin mixing or cross-environment disturbance leakage. |
| Transfer reproducibility | Wheel imports without simulator libraries; YAML/resource inclusion and hashes verified; fake backend completes the same shared schedule, preserves fault/stop behavior and generates schema-compatible artifacts. |

Use physically selected and documented acceptance thresholds per trial; do not invent an accuracy or stability guarantee from the existing RL results. If the configured speed qualification envelope is used, label it separately from model hard limits and local hardware abort thresholds. For force/posture tests, report both tracking results and known numerical/model limitations.

### 6.2 Commands and configuration policy

All executable scripts above are planned. After they exist, use the repository direnv wrapper for both visible and headless Isaac jobs. The entry point pattern is:

```bash
# Future usage after M0–M2 implementation; not an executable command today.
export FRANKA_RL_DATA_ROOT=/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data
export FRANKA_RL_FR3_USD="$FRANKA_RL_DATA_ROOT/generated_assets/fr3v21_bare_flange_live_v1/fr3v2.usda"
direnv exec /home/chen-lab/isaac/franka-rl \
  /home/chen-lab/isaac/.venv/bin/python -u \
  /home/chen-lab/isaac/franka-rl/scripts/motion_control/run_sim_suite.py \
  --suite /home/chen-lab/isaac/franka-rl/source/franka_motion/franka_motion/config/suites/demo_a_tracking.yaml
```

YAML owns geometry, targets, gains, timing, repeats and backend mode. CLI options only select the suite/runtime options such as device, visualization, output and resume. The real-time machine reads the transferred suite locally; no UI, stdin waypoint prompts, ONNX policy, or network publisher is required. A shared algorithm/package version and matching YAML hashes link its hardware results to the verified simulation experiments.

## 7. Deliverables and review checklist

1. Shared numerical package, manager-based simulation tasks and thin scripts with pinned environment/version notes, YAML configuration inputs, and no hard-coded unverified joint/tool limits.
2. Timestamped reference vs measured state logs for all trials; machine-readable metrics summary and notes on excluded/failed runs.
3. Demo A video (PTP + circle) and plots: path overlay, time-tracking error, geometric contour error, speed/feedforward comparison.
4. Demo B video (virtual fixture + null-space motion) and plots: transverse virtual-fixture error, fixed-pose error vs posture/joint trajectories.
5. Short generated `experiment_summary.md` in the comparison artifact directory, and checked-in `docs/industrial_motion_control.md`, explicitly distinguishing **library features used**, **original analytic planning/control/evaluation implemented**, tested parameter values, quantitative findings, failure modes, and safety limitations.
6. A 45–60 second interview pitch for each demo:
   - **A:** “I compared jerk-limited robot motions against continuous time-parameterized Cartesian tracking, implemented analytic position/twist reference generation, and measured both time-domain tracking and path contouring errors on a real FR3.”
   - **B:** “I implemented a Cartesian virtual fixture and a redundant-joint posture task using Jacobian/null-space concepts and impedance control, then quantified the end-effector vs joint-space tradeoff on hardware.”

7. Reproducible transfer bundle and real-time agent instructions containing the approved candidate suites, library/source hashes, reference fixtures, simulation results, remaining hardware checks and no assumed upgrades.

**Definition of done:** verified simulation algorithms and a reproducible transfer, followed by working, safe, repeatable hardware motions + correct reference/feedback logging + explanatory plots. A visually impressive clip **without** quantitative measurements does not satisfy the handoff. Keep simulation completion and hardware completion as separate recorded milestones.

## 8. Optional extensions (strictly after baseline)

- Extend the required DLS feasibility checker into a separately labeled differential-IK motion baseline; compare singularity behavior and joint-limit margins to Cartesian impedance/built-in motions. Feasibility checking already belongs to M2; **do not deploy a newly written joint-velocity controller on hardware without separate validation**.
- Add joint-space S-curve planning through Ruckig vs a naïve quintic trajectory; quantify peak acceleration/jerk and duration. Distinguish time-optimality under limits from shortest Cartesian path.
- Add online target changes/replanning in a safe empty workspace, measuring latency and acceleration discontinuity; use franky's same-control-mode preemption/reaction semantics.

## 9. Questions to answer before touching the hardware

- Which firmware/libfranka/franky build actually runs on the FR3? Does it expose `CartesianImpedanceTracker`, `PostureTask`, and `TorqueStopMotion`?
- What exact Cartesian orientation error, null-space projection, gains/damping convention, error clipping, reference/gain interpolation, torque-rate limit and filter order does that installed build use? Supply versioned source or numerical reference fixtures for the simulation comparison.
- Is a tool/fixture attached? What is its mass, TCP transform, and allowed workspace?
- Which safe standby and emergency-stop procedure is currently validated in the lab?
- Is physical hand guidance permitted by the lab? If not, B1 should use a no-contact alternative.
- What feedback telemetry rate is realistically available, and how will target and feedback timestamps be aligned?
- Which local hardware reference publication rate and application delay are validated for Cartesian tracking? Is the Python timing loop adequately measured, and does the native controller expose the applied reference sequence/timestamp?
- Which initial posture, gain profiles, measured-state limits and stop settings are approved for the transferred suite? Keep the runtime on the real-time machine and preserve existing verified policy routes.

These questions are validation gates, not reasons to expand scope. Start Demo A once they are resolved.

## M3 implementation update — 2026-10-11

Demo B is implemented in the separate manager-based
`Franka-FR3v2-MotionControl-FixturePosture-v0` task. Its physical 20D reference
adds seven bounded posture targets to M2's Cartesian pose/twist. No M1/M2 action
contract was replaced. `config/suites/demo_b_fixture_posture.yaml` selects known
axial/transverse force tests, two feasible secondary posture targets and a weak
secondary-gain comparison. `run_sim_demo.py --demo-b-suite ...` runs finite trials;
the Cartesian compiler now validates 20D traces and produces fixture/posture plots.

The portable implementations are `LineVirtualFixture` in `virtual_fixture.py`,
`nullspace_projector` in `kinematics.py`, and `compute_posture_torque` in
`impedance.py`. Isaac adapters are `CartesianPostureMotionAction`, batched projector/
fixture functions in `mdp/controllers.py`, and `IsaacFixturePostureRunner`.
The projector is the documented unweighted kinematic SVD torque projector, not
a claim of dynamic decoupling or Franky parity. Known 0.5 N forces act at flange
body CoM in world axes (identity base rotation checked), not the nominal TCP.
The finite ±30 mm reference segment is compliant; measured axial overshoot was
7.25 mm. Both corrected ±0.15 rad posture targets produced approximately 0.15 rad
joint-vector change while tip-position drift stayed below 0.18 mm. The weak
comparison produced 0.135 rad within the same duration. Full protocol, limits,
commands and results are in `docs/industrial_motion_control.md`.

M4 must audit installed Franky's projector/rank threshold, orientation error,
posture damping, gain updates and compensation/filter/stop semantics; review
physical force application/tool mass and independently approve manual interaction.
All profiles remain simulation candidates with `hardware_approved: false`.
