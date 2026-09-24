# Franka RL Real-Robot Deployment Plan

## 1. Purpose and handoff

This document is the handoff for an agent working on the real-time Franka
computer. It describes how to package the trained Isaac Lab policy, reproduce
its observation/action contract, build a ROS 2 real-time adapter, and commission
the controller without allowing an agent to initiate unreviewed robot motion.

The recommended deployment is:

- a custom ROS 2 C++ `ros2_control` controller on the real-time computer;
- deterministic policy inference at 30 Hz in a non-real-time worker on that
  same computer;
- a 1 kHz real-time callback that applies a continuous, safety-limited joint
  position reference;
- the experiment coordinator, target-suite loader, policy runtime, controller,
  telemetry writer, result compiler, and plots all run on the real-time
  computer;
- targets and trial order come from versioned YAML experiment suites, with no
  interactive target entry, UI, or command-line prompts during a run;
- there is no network or process communication with the RL training
  workstation while preparing or executing a hardware experiment;
- Isaac Sim is not required at runtime.

Do not begin hardware motion from this plan alone. A trained lab operator must
verify the exact robot model, software compatibility, safety setup, workspace,
and independent stop path before enabling FCI motion.

## 2. Current simulation provenance

The policy was developed in the external project:

```text
/home/chen-lab/isaac/franka-rl
```

The local simulation stack at the time of handoff is:

- Isaac Lab checkout: `/home/chen-lab/isaac/IsaacLab`
- Isaac Lab revision recorded in `IMPLEMENTATION_PLAN.md`
- Python environment: `/home/chen-lab/isaac/.venv`
- task: `Template-Franka-Rl-v0`
- nominal checkpoint:
  `${FRANKA_RL_DATA_ROOT}/runs/logs/rsl_rl/franka_reach/2026-09-23_01-15-04_no_success_termination/model_299.pt`
- domain-randomized checkpoint:
  `${FRANKA_RL_DATA_ROOT}/runs/logs/rsl_rl/franka_reach/2026-09-24_00-48-20_dr_v2/model_999.pt`

The exact checkpoint selected for deployment must be identified by SHA-256,
not only by filename. A deployment bundle must also record the Git revisions,
scenario configuration hash, exporter version, and model-format versions.

## 3. Fixed policy contract

The current manager-based environment defines a 30 Hz policy around a 60 Hz
physics simulation with decimation 2. The real adapter must reproduce the
following contract exactly.

### 3.1 Joint order and default state

Joint order:

```text
panda_joint1
panda_joint2
panda_joint3
panda_joint4
panda_joint5
panda_joint6
panda_joint7
```

Default joint position [rad]:

```text
[0.000, -0.569, 0.000, -2.810, 0.000, 3.037, 0.741]
```

The fingers are not controlled by the policy. Gripper control must remain a
separate subsystem and must not interrupt the arm controller.

### 3.2 Observation

The actor input is one float32 vector of length 24, concatenated in this order:

```text
index 0:7    q_measured - q_default                    [rad]
index 7:14   dq_measured                               [rad/s]
index 14:17  target_position_base - hand_position_base [m]
index 17:24  previous raw policy action                 [unitless]
```

There is no running observation normalization in the current PPO
configuration. The real implementation must not silently introduce one.

`previous_action` is the prior raw actor output at the policy boundary. It is
not the measured joint displacement and not the final reference after the
safety governor. Log all three values separately.

At episode/controller initialization, `previous_action` is zero. After a
hardware stop or policy restart, do not resume with stale recurrent or action
state.

### 3.3 Action

The actor output is a float32 vector of length 7. Inference must use the
deterministic actor mean; Gaussian exploration sampling is forbidden on the
real robot.

The simulated action mapping is:

```text
q_policy = q_default + 0.5 * actor_output
```

The current environment does not declare an explicit actor-output clip. The
deployment code must therefore audit the output distribution before choosing a
hardware clip. Any hardware safety projection must be visible in telemetry;
frequent intervention means the policy is operating outside its validated
contract and should cause an experiment abort rather than silent continuation.

### 3.4 Target and frame

Training targets are positions in the robot base frame:

```text
x: [0.35, 0.60] m
y: [-0.20, 0.20] m
z: [0.20, 0.50] m
```

The tracked simulator body is `panda_hand`. Do not assume that libfranka's
configured `O_T_EE` origin is identical. The hardware agent must establish and
test one of these mappings:

1. compute URDF forward kinematics from the base link to `panda_hand`; or
2. prove that the configured Franka end-effector transform matches
   `panda_hand`, including translation and orientation.

This must be checked at several static joint configurations against Isaac Lab
forward kinematics. A constant frame error directly becomes a reach error.

## 4. Self-contained real-time-computer architecture

```text
Real-time Franka computer
  immutable policy bundle + experiment-suite YAML + local run request
                        |
                        v
  local experiment coordinator / suite validator / artifact manager
                        |
              +---------+------------------------+
              |                                  |
              v                                  v
    30 Hz non-RT policy worker           1 kHz RT controller update
    - read latest state snapshot         - read robot state interfaces
    - build 24D observation              - publish lock-free snapshot
    - run deterministic ONNX actor       - read latest governed target
    - write raw action buffer            - advance smooth reference
                                         - write joint positions
              |                                  |
              +----------------+-----------------+
                               v
                    local telemetry ring/writer
                               |
                               v
                  local summaries, CSV, and plots

  Franka FCI <-------- dedicated robot Ethernet NIC
```

The Franka connection must use the real-time computer's dedicated robot NIC.
The training workstation must not publish ROS topics, mount a network
filesystem, provide targets, coordinate trials, receive live telemetry, or act
as a watchdog. Policy bundles and experiment definitions are transferred to
the real-time machine out of band before the experiment, verified locally, and
then treated as immutable. Results may be copied out only after the experiment
is fully stopped.

The hardware experiment must be able to run with the training workstation
powered off and physically disconnected. The only runtime network required by
this design is the dedicated FCI connection between the real-time computer and
the Franka controller.

No software interaction is required after a suite starts: there are no target
prompts, per-trial confirmations, ROS service calls, or UI controls. This does
not remove the requirement for an authorized operator to prepare the robot,
approve suite start under the lab procedure, remain present, and have access to
an independent physical stop.

The policy worker should normally be a non-real-time `std::jthread` owned by
the lifecycle controller. Exchange state and actions with
`realtime_tools::RealtimeBuffer` or an equivalent allocation-free mechanism.
Do not execute ONNX Runtime, publish ROS messages, write logs, allocate memory,
acquire a blocking mutex, or call TF inside the real-time `update()` method.

## 5. Deployment bundle

### 5.1 Source versus generated artifacts

Commit deployment source, schemas, lightweight fixtures, and documentation to
Git. Do not commit checkpoints, exported policies, hardware logs, or large
test-vector sets.

Generate bundles under:

```text
${FRANKA_RL_DATA_ROOT}/deployment_bundles/<bundle_id>/
```

Copy a verified, immutable bundle to a versioned directory on the real-time
computer. Never deploy a mutable `latest` symlink without resolving and logging
the concrete bundle ID.

### 5.2 Source layout and implementation status

The offline bundle milestone is implemented. See [deployment/README.md](deployment/README.md)
for export, portable verification, and YAML suite validation commands. Its
simulator-independent package lives in `deployment/franka_policy_bundle/`
so loading it does not invoke `franka_rl` task registration. Schemas, target
examples, and suite templates live alongside it. The hardware coordinator,
C++ parity harness, and controller remain subsequent milestones.

The original planned integration layout below is retained as a reference;
`contract.py`, observation/action math, `bundle.py`, and `suite.py` are currently
provided by the standalone package rather than the simulator package.

```text
franka-rl/
├── deployment/
│   ├── README.md
│   ├── schemas/
│   │   ├── policy_contract.schema.json
│   │   ├── target_set.schema.json
│   │   └── hardware_suite.schema.json
│   ├── target_sets/
│   │   ├── shadow_smoke.yaml
│   │   └── commissioning_near.yaml
│   ├── suites/
│   │   ├── shadow_smoke.yaml
│   │   └── commissioning_near.yaml
│   ├── policy_contract.template.yaml
│   └── run_request.template.yaml
├── scripts/deployment/
│   ├── export_policy_bundle.py
│   ├── verify_policy_bundle.py
│   ├── verify_hardware_suite.py
│   └── generate_policy_test_vectors.py
├── source/franka_rl/franka_rl/deployment/
│   ├── __init__.py
│   ├── contract.py
│   ├── observation.py
│   ├── action_mapping.py
│   ├── bundle.py
│   └── suite.py
└── tests/deployment/
    ├── test_contract.py
    ├── test_observation_order.py
    ├── test_action_mapping.py
    ├── test_hardware_suite.py
    └── test_export_equivalence.py
```

This code is platform-neutral. It must not import ROS, libfranka, Isaac Sim, or
Isaac Lab in its core observation/action math. Isaac-specific extraction and
ROS-specific execution should be adapters around the same tested contract.

### 5.3 Generated bundle contents

```text
<bundle_id>/
├── policy.onnx
├── policy_contract.yaml
├── manifest.json
├── test_vectors.npz
├── verification.json
└── README.md
```

Required manifest data:

- bundle schema version and bundle ID;
- creation timestamp in UTC;
- checkpoint path for provenance and checkpoint SHA-256;
- ONNX SHA-256;
- Git revisions and dirty-worktree status;
- Python, PyTorch, RSL-RL, ONNX, and ONNX Runtime versions;
- task ID, policy frequency, observation/action dimensions and dtype;
- nominal versus DR training scenario and its configuration hash;
- expected input/output tensor names;
- test-vector hash and verification tolerances;
- frame names and units;
- exporter command line.

Required `policy_contract.yaml` fields:

```yaml
schema_version: 1
robot_model: panda                 # confirm on hardware; do not infer
joint_names: [panda_joint1, panda_joint2, panda_joint3,
              panda_joint4, panda_joint5, panda_joint6, panda_joint7]
default_joint_position_rad: [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
policy_period_s: 0.03333333333333333
observation:
  dtype: float32
  size: 24
  terms:
    - {name: joint_position_relative, start: 0, stop: 7, unit: rad}
    - {name: joint_velocity, start: 7, stop: 14, unit: rad_s}
    - {name: hand_position_error_base, start: 14, stop: 17, unit: m}
    - {name: previous_action, start: 17, stop: 24, unit: unitless}
action:
  dtype: float32
  size: 7
  inference: deterministic_mean
  mapping: default_plus_scaled
  scale_rad: 0.5
frames:
  base: panda_link0
  tracked_body: panda_hand
target_workspace_m:
  x: [0.35, 0.60]
  y: [-0.20, 0.20]
  z: [0.20, 0.50]
```

The robot model and link prefixes must be explicit bundle compatibility data.
If the lab robot is an FR3 rather than the older Panda/FER used in simulation,
stop and resolve model/kinematic compatibility before loading the model.

### 5.4 Bundle verification gates

A bundle is releasable only when:

1. its schema validates;
2. all hashes reproduce;
3. ONNX and the checkpoint actor produce equivalent deterministic outputs on
   randomized and recorded observations;
4. output tensor shape is exactly `[batch, 7]` and all values are finite;
5. maximum absolute ONNX-versus-PyTorch error is within the recorded tolerance
   (start with `1e-5` for float32 CPU inference and tighten if practical);
6. C++ ONNX Runtime reproduces the same test vectors on the real-time computer;
7. observation-term order is tested with sentinel values, not just zeros;
8. the bundle has no dependency on Isaac Sim or the training checkpoint at
   runtime.

### 5.5 Declarative target sets and experiment suites

Targets must be explicit, reviewable data. Do not generate targets on the
hardware machine at run time and do not accept target changes through a UI,
terminal prompt, command-line override, ROS topic, or service after a suite is
loaded.

A target-set file contains only named poses in the policy's validated frame:

```yaml
schema_version: 1
name: commissioning_near_v1
frame_id: panda_link0
targets:
  - id: near_center
    position_m: [0.45, 0.00, 0.35]
  - id: near_left
    position_m: [0.45, 0.05, 0.35]
```

A suite references immutable policy and target inputs and defines the complete
trial schedule:

```yaml
schema_version: 1
name: dr_policy_commissioning_v1
mode: shadow                       # shadow or active
policy_bundle:
  path: ../bundles/dr_v2_model_999
  sha256: <bundle-manifest-sha256>
target_set:
  path: ../target_sets/commissioning_near.yaml
  sha256: <target-set-sha256>
execution:
  repetitions: 3
  order: sequential                # or seeded_shuffle with an explicit seed
  seed: 123
  move_to_start_before_each_trial: true
  trial_timeout_s: 6.0
  inter_trial_hold_s: 2.0
success:
  position_threshold_m: 0.03
  consecutive_policy_steps: 5
failure_policy:
  abort_suite_on_fault: true
  continue_after_timeout: false
artifacts:
  root: /var/local/franka_rl/experiments
```

Use relative paths inside a versioned deployment directory where possible.
Before controller activation, the local coordinator must schema-validate both
files, verify hashes, reject duplicate IDs and invalid frames/units, check every
target against the approved workspace, and expand target order and repetitions
into an immutable trial plan. It must write the source files, resolved plan,
content hashes, and validation result into the run artifacts.

A local `run_request.yaml` selects exactly one concrete suite and deployment
directory. The system launch unit reads that file; it must not resolve a
mutable `latest` alias. The suite content hash forms part of the run ID. After
validation and the pre-run operator authorization required by the lab
procedure, all trial transitions are automatic and no target or scheduling
input is accepted until the suite reaches `COMPLETE` or `ABORTED`.

## 6. ROS 2 workspace design

Use a separate workspace on the real-time computer. Do not mix its environment
with the Isaac Lab Python environment.

```text
franka_rl_ros2_ws/src/
├── franka_rl_controller/
│   ├── include/franka_rl_controller/
│   ├── src/
│   ├── config/
│   ├── test/
│   └── franka_rl_controller.xml
├── franka_rl_experiments/
│   ├── include/franka_rl_experiments/
│   ├── src/
│   ├── schemas/
│   ├── config/targets/
│   ├── config/suites/
│   └── test/
├── franka_rl_bringup/
│   ├── launch/
│   └── config/
├── franka_rl_interfaces/
│   └── msg/
└── franka_rl_bundle_cpp/
    ├── include/
    ├── src/
    └── test/
```

Reuse `franka_ros2`'s supported hardware plugin and robot description. Do not
fork or reimplement the FCI hardware interface.

### 6.1 Controller interfaces

The controller should claim seven joint position command interfaces and read
seven position and velocity state interfaces. It also needs a validated hand
position in the base frame. Resolve that position outside the real-time loop
from a preloaded kinematic model, or use the Franka semantic robot state only
after proving its EE frame matches the bundle contract.

### 6.2 Controller state machine

Implement explicit states:

```text
DISARMED -> MOVE_TO_START -> SHADOW -> RAMP_IN -> RUNNING
    ^              |           |          |          |
    +--------------+-----------+----------+----------+
                           STOPPING / FAULT
```

Rules:

- `MOVE_TO_START` uses a conventional, manufacturer-supported trajectory, not
  the learned policy.
- `SHADOW` builds observations and actions but never claims or writes moving
  commands.
- `RAMP_IN` starts with the commanded position equal to the current desired
  position and smoothly blends toward the governed policy reference.
- `RUNNING` requires fresh robot state, fresh policy output, a locally
  validated current trial, valid frames, and all safety checks.
- `STOPPING` produces a controlled stop/hold compatible with the active command
  mode.
- `FAULT` never calls automatic recovery and resumes motion autonomously.
  Recovery requires an operator-reviewed transition.

The local experiment coordinator has a separate suite state machine:

```text
LOAD -> VALIDATE -> PREPARE -> TRIAL -> EVALUATE -> INTER_TRIAL
                 |             ^                         |
                 |             +-------------------------+
                 +---------------------------> COMPLETE / ABORTED
```

It advances only through the pre-resolved trial plan. A timeout follows the
suite's declared failure policy; a controller fault always aborts the suite.
There is no state that waits for a target, confirmation, or command from the
training workstation.

### 6.3 Reference governor

The 30 Hz output is a target, not a valid 1 kHz trajectory. The governor must:

1. map raw action to `q_policy`;
2. reject non-finite inputs;
3. enforce conservative model-specific joint-position margins;
4. generate continuous position, velocity, and acceleration across policy
   updates;
5. enforce model-specific velocity, acceleration, and jerk limits;
6. expose raw target, limited target, commanded target, and intervention flags;
7. hold or stop on a stale policy output;
8. avoid resetting intermediate velocity to zero at every 30 Hz update.

Use a continuously updated online trajectory generator or a verified equivalent
reference filter. Its exact implementation and parameters are part of the
deployment contract. Mirror the same governor in Isaac evaluation before real
motion; retrain if it materially changes success, time-to-success, action rate,
or safety margins.

Do not rely solely on undocumented middleware defaults. libfranka offers
low-pass filtering and rate limiting, while the exact `franka_ros2` defaults
depend on the selected compatible version.

### 6.4 Watchdogs

At minimum monitor:

- robot-state age;
- policy-output age and missed 30 Hz deadlines;
- local experiment-coordinator liveness, suite state, and current-trial
  generation/lease age;
- inference duration and exception status;
- non-finite observations/actions;
- joint position, velocity, acceleration, and commanded derivative margins;
- target workspace validity;
- action-to-command governor intervention magnitude;
- Franka contact, collision, reflex, and robot mode;
- controller lifecycle state.

Timeouts and limits must be conservative, configurable, versioned, and
reviewed by the lab operator. Do not invent hardware thresholds from the
simulation termination values.

## 7. Telemetry and evidence

Never perform disk I/O or ROS logging from the 1 kHz callback. Write fixed-size
samples into a bounded ring buffer and drain them from a non-real-time thread.

Record synchronized monotonic timestamps and at least:

- measured `q`, `dq`;
- configured/derived hand position and target position;
- complete 24D observation;
- previous action and current raw actor output;
- `q_policy`, governed target, and 1 kHz command;
- safety intervention masks and magnitudes;
- controller state and watchdog ages;
- inference duration;
- FCI robot mode/contact/collision/reflex signals;
- bundle ID, configuration hashes, robot model, firmware/system version,
  libfranka/franka_ros2/ROS versions, suite/target/bundle hashes, and the
  suite-derived run ID.

Write logs, summaries, and plots in batches to a local POSIX filesystem on the
real-time computer. Never write runtime artifacts to a network filesystem.
After the controller is stopped and the suite is complete or aborted, results
may be copied out of band to the experiment volume for analysis or archival.

## 8. Milestones and acceptance gates

### Milestone 0 — hardware and compatibility inventory

Collect without commanding motion:

- robot model: Panda/FER or FR3;
- robot system/firmware and server version;
- end effector and hand configuration;
- ROS distribution and Ubuntu version;
- `franka_ros2` and libfranka versions/commits;
- real-time kernel, permissions, CPU isolation, and NIC layout;
- available independent stop device and lab operating procedure;
- known-good official example controller and launch command;
- network latency/jitter/packet-loss test results;
- joint names, prefix, base frame, and EE transform reported by the stack.

Gate: versions match Franka's official compatibility matrix, the official
communication test passes, and a manufacturer example can be run by a trained
operator. The agent must not run that example autonomously.

### Milestone 1 — deployment bundle

Implement export, contract schema, target-set and suite schemas, reviewed
YAML examples, test vectors, hashes, and Python/ONNX equivalence verification
on the simulation computer.

Gate: a clean process can verify the policy bundle and fully resolve each suite
using only the versioned deployment directory and verification programs.

### Milestone 2 — C++ offline runtime

Implement contract loading, suite loading, local experiment coordination,
observation assembly, ONNX inference, action mapping, governor, state machine,
watchdogs, and result compilation with synthetic inputs.

Gate: C++ passes the Python-generated test vectors and deterministic timing
tests; complete suites run without UI or command-line interaction; malformed
contracts/suites and stale/non-finite inputs fail closed.

### Milestone 3 — ROS 2 fake-hardware integration

Load the custom lifecycle controller through the official Franka bringup/fake
hardware path. Test activation/deactivation, controller switching, suite
loading, repeated targets, timeouts, automatic trial transitions,
policy-worker failure, and clean shutdown with the training workstation
disconnected.

Gate: no allocations or blocking operations are detected in the real-time
path; no command discontinuity occurs on activation, trial transition, or
stop; the resolved plan and local artifacts reproduce the requested suite.

### Milestone 4 — real-hardware read-only and shadow validation

Read real state and compare frame/FK calculations. Execute a reviewed
`shadow` suite from local YAML through the complete policy path; do not claim
moving command interfaces.

Gate: frame agreement is documented, inference and data age fit inside the
budget, outputs remain finite and inside the audited distribution, all local
artifacts and plots are complete, and the operator approves them.

### Milestone 5 — conservative commissioning

Under the lab's operating procedure, an operator moves to the validated start
pose and authorizes a pre-reviewed local YAML suite. Begin with explicit targets
near the current hand position, conservative dynamics, an empty workspace,
physical stop access, and one short suite. Do not enter or alter targets
interactively during commissioning.

Gate: commanded and measured trajectories remain continuous; watchdogs,
collision/reflex behavior, and stop transitions work; real behavior is compared
against the simulated deployment governor before expanding the envelope.

### Milestone 6 — simulation alignment and final evaluation

After hardware is stopped, copy the completed local artifacts out of band.
Measure real latency, tracking, and filtering on the training workstation,
reproduce them in Isaac Lab, update the domain-randomization distribution,
retrain if required, and compare nominal/DR policies on paired targets and
initial states. Transfer any replacement bundle and suite back out of band.

Gate: the final hardware bundle and suite are regenerated from reviewed inputs,
copied to the real-time computer, and all local provenance/equivalence tests
pass again with the training workstation disconnected.

## 9. Known sim-to-real risks

- Isaac Lab uses implicit simulated joint actuators with nominal stiffness 80
  and damping 4; the real internal joint impedance is not assumed equivalent.
- The policy may request a new absolute joint target every 33.3 ms; a direct
  step command is not a valid real trajectory.
- The simulation's `5 rad/s` failure threshold is not a hardware limit.
- The policy was rewarded for position, not hand orientation.
- The environment has no obstacle-avoidance objective and no perception.
- Simulation failure causes reset; hardware failure must cause a controlled
  stop and operator-reviewed recovery.
- The action governor introduces dynamics not present during the original
  training unless it is added to simulation.
- A Panda-trained policy must not be assumed compatible with an FR3 solely
  because both expose seven joints.

## 10. Agent instructions and skill audit

### 10.1 Instruction files found locally

`/home/chen-lab/isaac/AGENTS.md` is primarily for the separate continuum-robot
project. Its data hygiene and environment isolation rules are useful, but its
Warp/Cosserat architecture is unrelated. Do not copy it wholesale to the
real-time Franka workspace.

`/home/chen-lab/isaac/IsaacLab/AGENTS.md` is the official local Isaac Lab
contributor guide. It is useful for Isaac-side naming, typing, SI-unit
documentation, wrapped Python, tests, and pre-commit practices. It applies when
modifying the Isaac Lab checkout; this external project should not inherit its
changelog/release rules blindly.

The locally installed `i4h-workflows` skills target medical-imaging and
catheter-navigation workflows. They are not appropriate for this deployment.

### 10.2 Public skills and guidance

Useful authoritative resources:

- NVIDIA's official Isaac Sim agent skills:
  <https://docs.isaacsim.omniverse.nvidia.com/latest/development_tools/agent_skills.html>
- Isaac Sim's public `AGENTS.md` and skill index:
  <https://github.com/isaac-sim/IsaacSim/blob/main/AGENTS.md>
- Isaac Lab's `AGENTS.md`:
  <https://github.com/isaac-sim/IsaacLab/blob/develop/AGENTS.md>
- official `ros2_control` controller tutorials:
  <https://control.ros.org/>
- official Franka ROS 2 repository and examples:
  <https://github.com/frankarobotics/franka_ros2>
- official FCI/libfranka documentation:
  <https://frankarobotics.github.io/docs/>

The NVIDIA skills are useful on the simulation computer for Isaac Sim startup,
robot import, physics, ROS bridge, troubleshooting, and validation. They target
specific recent Isaac Sim/Kit versions, so pin a revision compatible with the
installed simulator rather than copying `main` unreviewed. They are not needed
in the real-time control callback or on a minimal Franka machine.

Potential third-party ROS skills for review, not automatic installation:

- `dbwls99706/ros2-engineering-skills`:
  <https://github.com/dbwls99706/ros2-engineering-skills>
- `wzyn20051216/ros-robotics-skill`:
  <https://github.com/wzyn20051216/ros-robotics-skill>

Before adopting a third-party skill, inspect every instruction, script, hook,
dependency, network action, and supported ROS distribution. Pin a commit and
test it in fake hardware. Do not grant a skill permission to launch, activate,
recover, or command a real robot.

No official Franka-specific agent skill or `AGENTS.md` was found in
`franka_ros2`. Official Franka documentation, examples, compatibility tables,
and installed source are the authority.

### 10.3 Recommended project-specific agent setup

Create a root `AGENTS.md` in the real-time workspace with these non-negotiable
rules:

- never initiate physical motion, activate FCI, unlock brakes, disable a
  safety rule, or recover a fault without an explicit operator authorization
  before local suite start; the suite itself must not prompt for permission;
- default all new functionality to fake hardware or `SHADOW` mode;
- use the exact installed ROS distribution, `franka_ros2`, and libfranka source
  as API authority;
- prohibit allocation, blocking locks, file/network I/O, logging, parameter
  lookup, TF lookup, and inference inside the 1 kHz callback;
- require lifecycle-safe activation/deactivation and continuous first/last
  commands;
- require unit annotations for physical quantities and explicit frame names;
- fail closed on stale, malformed, non-finite, mismatched, or unverified bundle
  data;
- never use simulation termination thresholds as hardware limits;
- require build, unit tests, fake hardware, and shadow validation before any
  hardware command-path change is marked complete;
- preserve logs and version/hash metadata for every commissioned run;
- prohibit runtime dependencies on, communication with, or network mounts from
  the training workstation;
- prohibit interactive target entry and runtime target/schedule overrides;
- require schema validation, content hashes, explicit frames/units, and a
  resolved immutable trial plan for every target set and suite;
- do not install packages or change the real-time kernel/network configuration
  without operator approval.

After the first fake-hardware controller is working, create a narrow custom
`franka-rl-deployment` skill containing only repeatable, validated procedures:

- inventory and compatibility check;
- bundle, target-set, and suite verification;
- suite dry run and resolved-plan audit;
- fake-hardware launch and lifecycle tests;
- shadow-mode data collection;
- telemetry audit;
- failure-mode diagnosis;
- operator handoff checklist.

The skill must reference canonical source and scripts rather than embedding a
second copy of controller code or version-sensitive launch commands.

## 11. Required first report from the real-time agent

Before implementing or installing anything, the real-time agent should return:

1. robot model and system/server version;
2. Ubuntu, kernel, ROS distribution, `franka_ros2`, and libfranka versions;
3. repository locations and Git revisions;
4. whether a known-good official controller currently works;
5. NIC/interface layout and communication-test result;
6. reported joint names and end-effector transforms;
7. available compiler, ONNX Runtime, `realtime_tools`, and test framework;
8. existing lab launch files, safety configuration, and operator procedure;
9. proposed workspace, local suite/target installation, and artifact/log
   locations;
10. proposed local launch/autostart mechanism and how it operates with the
    training workstation disconnected;
11. discrepancies between this plan and the installed system.

No hardware-control implementation should begin until this report is reviewed.

## 12. Primary references

- Franka FCI overview and 1 kHz control architecture:
  <https://frankarobotics.github.io/docs/doc/libfranka/docs/overview.html>
- Franka software compatibility and setup documentation:
  <https://frankarobotics.github.io/docs/>
- Franka ROS 2 packages and example controllers:
  <https://github.com/frankarobotics/franka_ros2>
- libfranka robot state and EE transforms:
  <https://frankarobotics.github.io/libfranka/latest/structfranka_1_1RobotState.html>
- ROS 2 control documentation:
  <https://control.ros.org/>
- Isaac Sim agent skills:
  <https://docs.isaacsim.omniverse.nvidia.com/latest/development_tools/agent_skills.html>

