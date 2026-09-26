# Franky deployment runtime

Pinned environment:

- Python: 3.12.14
- Environment: `/home/chen-lab/yifan/venvs/franky-server10`
- Franky: `2.0.1.dev58+gf88f0e9b.libfranka.0.21.2`
- Robot FCI server: 10

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

Both modes share one target, supervision, logging, and lifecycle path. The
coordinator does not import ROS 2. The policy worker is copied unchanged from the
ROS coordinator so manifest verification, 24-value observation ordering, and
stdio isolation remain the same.

### Command path

At 30 Hz, the coordinator either evaluates the reference schedule or consumes one
policy result. PPO actions map as

```text
q_target[i] = q_default[i] + scale[i] * raw_action[i]
```

There is no raw-action clipping. A non-finite action or mapped target outside the
5%-margin joint bounds terminates the session. The target is submitted as an
absolute `franky.JointMotion(return_when_finished=False)`. Franky generates its
1 kHz jerk-limited position command with Ruckig using the robot-level relative
dynamics factor (0.05 velocity, acceleration, and jerk). A same-signal asynchronous
move replaces the running joint-position motion seamlessly; the non-finishing mode
holds a reached target without cycling FCI. An independent host thread submits
`JointStopMotion` if no target is submitted for `command_watchdog_ms`, including
when the 30 Hz coordinator thread stalls. Target updates use `limit_rate=False`
and Franky's/libfranka's 100 Hz command filter. The backend explicitly selects Franky's default
`ControllerMode.JointImpedance`; the firmware's internal joint impedance controller
remains in use. No impedance or load setter and no automatic
error recovery is called by this runtime.

This path deliberately differs from the standalone shared governor. Its purpose is
to identify and then simulate the complete Franky `JointMotion` response. The
recorded `q_ref/dq_ref/ddq_ref` columns describe Franky's generated command;
`q_d/dq_d/ddq_d` are robot desired state, and `q/dq` are measured state.

### Constant-target keepalives

Franky motion replacement reinitializes a synchronized Ruckig trajectory from the
current desired state. Replanning an already reached, exactly identical seven-joint
target can fail with Ruckig `ErrorSynchronizationCalculation` (`-111`). The runtime
therefore submits a new `JointMotion` only when the target vector changes exactly.
At 30 Hz constant-target ticks it refreshes the independent command watchdog while
the active motion continues holding and emitting 1 kHz callbacks. `target_held`
events make these keepalives explicit. This preserves the requested waveform and
does not relax limits or change the controller.

The behavior was added after session `20260925193000` reached home with good
tracking and then faulted during its constant warmup after 0.459 s of callbacks.
That session stopped cleanly and must not be reused.

### Safety and lifecycle

Execution requires all of the following in the YAML: `status: approved`,
`executable: true`, `motion_authorized: true`, approval provenance, reviewed
start/tracking tolerances, and a positive session ID. The runtime checks the bare
flange/zero-load/identity-`F_T_EE` inventory assumptions, reviewed start pose and
velocity, soft position bounds, state-read and inference deadlines, callback
continuity, tracking error, robot errors, trial timeout, and suite timeout.
Faults are sticky. It never recovers or resumes automatically.

Hardware execution always pauses after preflight and requires the operator to type
`start`. `--auto-start` and `--max-runtime-s` are rejected for hardware configs.
On completion, fault, signal, or operator stop, the coordinator submits
`JointStopMotion` and joins it before disconnecting. The E-stop remains the
independent physical stop.

Constructing `franky.Robot` writes Franky's default collision behavior (20 Nm joint
threshold and 30 N Cartesian threshold). Every hardware config must explicitly
acknowledge these exact values. This is a property of the pinned Franky build, even
before the operator types `start`.

### Configuration and use

Start from, but do not execute, the pending templates:

- `config/reference.pending.yaml`
- `config/ppo.pending.yaml`

They intentionally contain `null` review decisions and `session_id: 0`, so they
fail closed. Copy a template to the reviewed experiment directory and fill every
pending value. For PPO, the bundle contract must name `fr3_joint1` through
`fr3_joint7`, base `fr3_link0`, tracked body `fr3_flange`, a 30 Hz policy, and the
same default offset/action scale. Existing Panda bundles are rejected.

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
/home/chen-lab/yifan/venvs/franky-server10/bin/python \
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

Coverage includes fail-closed pending configuration, exact constructor collision
contract, reference generation, soft-bound rejection, the full fake reference
run, the full fake PPO/worker path, complete artifacts, and absence of automatic
recovery calls. Offline tests do not qualify motion, thresholds, callback timing,
or controller response on hardware.
