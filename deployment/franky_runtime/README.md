# Franky deployment runtime

Pinned environments:

- Position/reference runtime: `/home/chen-lab/yifan/venvs/franky-server10` with Franky `2.0.1.dev58+gf88f0e9b.libfranka.0.21.2`.
- Explicit velocity-impedance runtime: `/home/chen-lab/yifan/venvs/franky-velocity-impedance-zdt-server10` with Franky `2.0.1.dev60+gf88f0e9bvelimpzdt.libfranka.0.21.2`.
- Both use Python 3.12 and libfranka 0.21.2 against robot FCI server 10.
- Repaired velocity-impedance wheel SHA-256: `8590e1241a8e305cf046c148968b7dc9055d2f58bcb944ddb5db33ebda70c00d`.

Use the second environment only for configurations whose runtime is
`franky_joint_velocity_impedance_tracking_v1`; the original environment remains
the validated position/reference runtime.

`read_only_probe.py` has no motion operations. Without `--connect`, it only
checks and describes the local environment. With `--connect`, it constructs
`franky.Robot`, reads one state and the published model limits, writes an exclusive
JSON artifact, and exits. The probe itself never calls motion, recovery, stop,
impedance-setting, load-setting, collision-setting, or Desk APIs. However, the
pinned Franky constructor internally calls `setCollisionBehavior` with its defaults:
20 Nm for every joint torque threshold and 30 N for every Cartesian force threshold.
Consequently the connected probe is not strictly configuration-free.

Offline check:

```bash
/home/chen-lab/yifan/venvs/franky-server10/bin/python -I \
  /home/chen-lab/yifan/franka-rl/deployment/franky_runtime/read_only_probe.py
```

Read-only FCI probe:

```bash
/home/chen-lab/yifan/venvs/franky-server10/bin/python -I \
  /home/chen-lab/yifan/franka-rl/deployment/franky_runtime/read_only_probe.py \
  --connect \
  --host 172.16.0.6 \
  --output /home/chen-lab/franka_ros2_ws/hardware_inventory/2026-09-25/franky_probe/state.json
```

The output is opened with exclusive creation, so an existing artifact is never
overwritten.

## Read-only compatibility result

Captured 2026-09-25 in
`/home/chen-lab/franka_ros2_ws/hardware_inventory/2026-09-25/franky_probe/state_server10_libfranka0212.json`.
Franky 2.0.1.dev58 with bundled libfranka 0.21.2 connected to FCI server 10,
read one state, and disconnected without starting a control loop. Its constructor
also set Franky's default 20 Nm / 30 N collision thresholds as described above. The robot reported
Idle, no current or last-motion errors, no active control, identity `F_T_EE` and
`EE_T_K`, and zero configured external load. The maximum measured joint speed was
0.001243 rad/s. The maximum absolute difference from the tested home pose was
0.013663 rad.

The FR3 limits reported by this libfranka build are:

- joint velocity [rad/s]: `[2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]`
- joint acceleration [rad/s^2]: `[15, 7.5, 10, 12.5, 15, 20, 20]`
- joint jerk [rad/s^3]: `[7500, 3750, 5000, 6250, 7500, 10000, 10000]`

A Franky relative dynamics factor of 0.05 therefore produces caps of:

- joint velocity [rad/s]: `[0.10875, 0.10875, 0.10875, 0.10875, 0.1305, 0.1305, 0.1305]`
- joint acceleration [rad/s^2]: `[0.75, 0.375, 0.5, 0.625, 0.75, 1.0, 1.0]`
- joint jerk [rad/s^3]: `[375, 187.5, 250, 312.5, 375, 500, 500]`

## Experiment coordinator

`run_coordinator.py` is the local, YAML-driven Franky replacement for the ROS 2
experiment coordinator. It supports the same two experiment sources:

- `reference`: deterministic one-joint, quintic-ramped sine targets for response
  identification.
- `ppo`: an immutable FR3 ONNX bundle evaluated in a separate Python process.

Both modes share supervision, logging, and lifecycle code. The coordinator
does not import ROS 2. PPO supports the 29-value position-only observation and six-action incremental
contract through an isolated worker. It accepts either reviewed point targets
or a SHA-anchored YAML path catalog. The 31-value z-axis contract remains
excluded from this first deployment; see `../HARDWARE_DEPLOYMENT_PLAN.md`.

### Command path

Reference identification runs at 30 Hz. PPO runs at 50 Hz and integrates six
bounded normalized increments into a persistent joints-1--6 reference; joint 7
is held at its measured session-start reference. One
`JointImpedanceTrackingMotion` remains alive for the complete session and
computes torque at 1 kHz. No `JointMotion` is created or preempted after
startup.

The pending PPO profile uses the hardware-tested nominal gains:

```text
K = [100]*7 Nm/rad
D = [20]*7 Nms/rad
```

The YAML also fixes Coriolis compensation, torque slew, gain interpolation time constant, zero friction/feedforward torque, Franky's expected 0.5 rad error clip, soft-limit repulsion, and torque-stop parameters. See `IMPEDANCE_PARAMETER_SURVEY.md` for alternatives and provenance.

At each 50 Hz PPO tick, action `a[0:6]` is required to be in `[-1,1]` and
maps componentwise to `q_ref += a * [0.0087, 0.0087, 0.0087, 0.0087, 0.01044, 0.01044]` before
soft-limit projection. The reference initializes from measured `q`; desired
velocity is zero. The 29D observation contains measured q/dq, flange position
error, normalized current reference, and the preceding increment action.

A parallel explicit velocity-reference impedance route is implemented for newly
trained velocity policies. Its C++ control loop integrates `q_ref` at 1 kHz and
uses both `q_ref` and `dq_ref` in the torque law; it does not reuse the native
Ruckig adapter tested in session `2026092708`. The separate contract, Franky
source patch, build boundary, and pending template are documented in
[`VELOCITY_IMPEDANCE_ROUTE.md`](VELOCITY_IMPEDANCE_ROUTE.md). Existing position
policy bundles fail its contract checks.

The flange position comes from forward kinematics of measured joint encoder `q`:
`robot.model.pose(Frame.Flange, q, identity, identity)`. This is the physical
`fr3_flange` pose in `fr3_link0`. The robot-reported `O_T_EE` is never policy or
path feedback; after conversion to the flange frame it is logged only as an audit
comparison. Every 1 kHz sample records the encoder-FK position, quaternion, source
and frames, along with the reported flange position and their position/orientation
difference. Raw measured joints remain in the same row, and
`encoder_fk_compute_ns` records the FK-call duration for timing qualification.

The pending PPO template uses a 50 ms host watchdog and a 15 ms inference
deadline. A missed update replaces the tracking controller with
`TorqueStopMotion`; normal completion and operator stop use the same stop.
No automatic error recovery is called.

### Path and shadow execution

Path mode selects exactly one named path from a SHA-256-anchored catalog; a
configuration cannot mix point targets and a path. The loader resolves circle
waypoints, verifies every point lies inside the reviewed Cartesian bounds, and
stores the resolved geometry in the run configuration. Waypoint reach/timeout is
counted in exact 50 Hz policy steps. Reference integration and previous-action
state persist across waypoint transitions, while a late result from the prior
waypoint is recorded and discarded. Hardware path configurations must set
`abort_on_timeout: true`; a missed waypoint then produces a sticky fault and the
normal torque-stop sequence instead of advancing to another target. Shadow and
fake parity runs may keep it false to exercise every transition.

Shadow mode constructs Franky and reads live FCI state but cannot call
`Robot.move`, submit a target, or keep a motion alive. It still inherits the
documented Franky-constructor collision-threshold write. Each inference logs the
full observation and measured inputs; each consumed result logs the action,
mapped reference, inference latency, and state-to-reference-consumption delay.
Use `config/ppo_path_shadow.pending.yaml` as the fail-closed starting point.
After a completed shadow run, validate dimensions, reset semantics, held joint 7,
action bounds, and latency with:

```bash
python3 deployment/franky_runtime/summarize_shadow.py   /absolute/path/to/session-ID   --output /absolute/path/to/shadow-summary.json
```

### Retired JointMotion backend

Sessions through `20260926151901` used repeated position-only `JointMotion` preemption. Session `20260926153859` demonstrated that this is unsafe as a general 30 Hz PPO interface: Franky's generated acceleration switched from about +3.0 to -3.53 rad/s^2 at one target replacement and triggered the FR3 acceleration-discontinuity reflex. The analytic sine bound was only 0.318 rad/s^2. That backend and its identification grid are retired.

### Safety and lifecycle

Execution requires `status: approved`, `executable: true`, approval provenance,
reviewed start/tracking tolerances, and a positive session ID. Hardware and fake
execution require `motion_authorized: true`. Shadow execution requires
`motion_authorized: false` and supports PPO only. The runtime checks the bare
flange/zero-load/identity-`F_T_EE` inventory assumptions, reviewed start pose and
velocity, soft position bounds, state-read and inference deadlines, callback
continuity, tracking error, robot errors, trial timeout, and suite timeout. Tracking error is computed from the time-coherent measured q and applied q_ref pair in the 1 kHz torque callback; RobotState.q_d is not the impedance reference in this control mode. A controller that differs from the bundle's training nominal requires a
separately approved JSON qualification whose SHA-256, policy-manifest hash,
stiffness vector, and damping vector all match the run configuration.

Faults are sticky. It never recovers or resumes automatically.

Hardware and shadow execution pause after preflight and require the operator to
type `start`. `--auto-start` and `--max-runtime-s` are accepted only for fake configs.
For hardware motion, completion, fault, signal, or operator stop submits
`TorqueStopMotion` and joins it before disconnecting. Shadow disconnects without
starting or stopping a motion. The E-stop remains the independent physical stop.

Constructing `franky.Robot` writes Franky's default collision behavior (20 Nm joint
threshold and 30 N Cartesian threshold). Every hardware config must explicitly
acknowledge these exact values. This is a property of the pinned Franky build, even
before the operator types `start`.

### Configuration and use

Start from, but do not execute, the pending templates:

- `config/reference.pending.yaml`
- `config/ppo.pending.yaml`
- `config/ppo_path_shadow.pending.yaml`

They intentionally contain `null` review decisions and `session_id: 0`, so they
fail closed. Copy a template to the reviewed experiment directory and fill every
pending value. For PPO, the bundle contract must name `fr3_joint1` through
`fr3_joint7`, base `fr3_link0`, tracked body `fr3_flange`, a 50 Hz period,
the exact 29D layout, and the six-action increment limits. Existing Panda,
24D/7D, and 30 Hz bundles are rejected. The accepted nominal and domain-randomized bundles are stored read-only under
`/home/chen-lab/yifan/deployment_bundles/2026-09-27/`; their trust anchors and
verification results are recorded in
`../hardware_control_audit/2026-09-27-bundle-acceptance/`. Receipt does not
authorize hardware motion.

Validate a fully populated file without opening FCI:

```bash
/home/chen-lab/yifan/venvs/franky-server10/bin/python \
  /home/chen-lab/yifan/franka-rl/deployment/franky_runtime/run_coordinator.py \
  --config /absolute/path/to/experiment.reviewed.yaml \
  --validate-only
```

After the lab separately authorizes that exact YAML and the physical-stop/workspace
review is complete, the hardware command is:

```bash
RUNTIME_PY=/home/chen-lab/yifan/venvs/franky-server10/bin/python
# For franky_joint_velocity_impedance_tracking_v1 instead use:
# RUNTIME_PY=/home/chen-lab/yifan/venvs/franky-velocity-impedance-zdt-server10/bin/python

"$RUNTIME_PY" \
  /home/chen-lab/yifan/franka-rl/deployment/franky_runtime/run_coordinator.py \
  --config /absolute/path/to/experiment.approved.yaml \
  --execute
```

The command connects and performs preflight, prints `READY`, then waits for literal
`start` or `stop` on stdin. Do not use Python `-I` for this script because isolated
mode removes its adjacent `franky_experiment` package from the import path.

PPO inference stays in the previously verified worker environment configured by
`ppo.worker_python` (currently
`/home/chen-lab/yifan/venvs/franka-policy-verify-py312/bin/python3`). ONNX Runtime
is not installed into the Franky control environment.

Each run creates an exclusive `session-<id>` directory containing:

- `resolved_config.json` and `config.sha256`
- `events.jsonl`
- `samples.csv`
- `final_metadata.json`

The CSV writer is outside Franky's control thread and uses a bounded queue. The
supervisor compares callback robot time with synchronous state robot time so a
still-active but backlogged Franky Python callback queue also fails closed. Queue
overflow, writer failure, callback loss/lag, state timeout, inference timeout, or
any supervision violation terminates the session and remains visible in metadata.

### Offline verification

No FCI connection is made by these tests:

```bash
PYTHONPATH=/home/chen-lab/yifan/franka-rl/deployment/franky_runtime \
  python3 -m pytest -q \
  /home/chen-lab/yifan/franka-rl/deployment/franky_runtime/tests
```

The current suite has 33 tests. Coverage includes fail-closed pending
configuration, exact constructor collision contract, reference generation,
soft-bound rejection, full fake reference and PPO runs, strict catalog/hash and
workspace validation, exact transferred YZ geometry, path transitions and stale
results, read-only shadow behavior, complete artifacts, and absence of automatic
recovery calls. Offline tests do not qualify motion, thresholds, callback timing,
or controller response on hardware.

### Experimental native joint-velocity adapter

Runtime `franky_joint_velocity_preemption_v1` preserves the existing nominal
policy's 29D observation and virtual position-reference integrator, but changes
the actuator route. For each accepted 50 Hz policy result it computes

```text
q_virtual[k+1] = project(q_virtual[k] + action[k] * max_increment)
dq_target[k] = (q_virtual[k+1] - q_virtual[k]) / 0.02 s
```

and submits `dq_target` through a new asynchronous `JointVelocityMotion`.
Franky/Ruckig owns the 1 kHz jerk-limited transition between velocity targets;
the runtime does not implement another interpolation or motion-planning
governor. The 0.02 s division produces the bundle limits directly: 0.435 rad/s
for joints 1--4, 0.522 rad/s for joints 5--6, and zero for held joint 7.
Repeated virtual positions are still submitted because they mean a zero velocity
target. A stale command or normal completion uses `JointVelocityStopMotion`.

This route explicitly selects the firmware joint-impedance controller and does
not call `set_joint_impedance`; the K=100/D=20 host torque gains therefore do not
apply. It logs the virtual position as `q_ref`, the requested velocity as
`dq_ref`, and Franky's generated `q_d`, `dq_d`, and `ddq_d`. The first staged
configuration is
`/home/chen-lab/franka_ros2_ws/hardware_inventory/2026-09-27/franky_motion/nominal.circle_yz.velocity-adapter.motion.2026092707.pending.yaml`.
It is deliberately non-executable until the new actuator mismatch and 20%
Ruckig dynamics setting are reviewed.

Session 2026092707 faulted before starting a Franky motion because the velocity
backend initially used a constructor-time encoder sample while the policy's
virtual reference used the later preflight sample. Sub-milliradian encoder drift
on held J7 therefore appeared as a nonzero finite-difference velocity. The
runtime now calls `synchronize_reference` with the exact preflight `q` used to
initialize the policy integrator before any command is submitted. Session
2026092708 contains the identical policy and motion settings with this fix.

Session 2026092708 showed that the implementation-correct finite-difference
adapter is not dynamically compatible with the nominal position policy. It
reached five waypoints but visibly jittered and faulted when J4 lagged the
virtual reference by 0.082061 rad. Do not produce another approved hardware
config for this policy/adapter pair by relaxing tracking thresholds. See
`../hardware_control_audit/2026-09-27-native-velocity-adapter/`.
