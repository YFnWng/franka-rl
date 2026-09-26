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

At 30 Hz, the coordinator evaluates the reference schedule or consumes one policy result and updates a `franky.JointReference`. One `JointImpedanceTrackingMotion` remains alive for the complete session and computes torque at 1 kHz. No `JointMotion` is created or preempted after startup.

The first reviewed profile uses Franka ROS 2's compliant example gains:

```text
K = [24, 24, 24, 24, 10, 6, 2] Nm/rad
D = [2, 2, 2, 1, 1, 1, 0.5] Nms/rad
```

The YAML also fixes Coriolis compensation, torque slew, gain interpolation time constant, zero friction/feedforward torque, Franky's expected 0.5 rad error clip, soft-limit repulsion, and torque-stop parameters. See `IMPEDANCE_PARAMETER_SURVEY.md` for alternatives and provenance.

PPO actions still map as `q_target = q_default + scale * raw_action`. The velocity reference is zero, matching a position-target PD actuator. The 1 kHz log records the held `q_ref/dq_ref`, controller `tau_command`, robot desired torque, measured torque, measured state, and external-torque estimate.

The independent 75 ms host watchdog remains active. A missed 30 Hz update replaces the tracking controller with `TorqueStopMotion`; normal completion and operator stop use the same torque-mode stop. No automatic error recovery is called.

### Retired JointMotion backend

Sessions through `20260926151901` used repeated position-only `JointMotion` preemption. Session `20260926153859` demonstrated that this is unsafe as a general 30 Hz PPO interface: Franky's generated acceleration switched from about +3.0 to -3.53 rad/s^2 at one target replacement and triggered the FR3 acceleration-discontinuity reflex. The analytic sine bound was only 0.318 rad/s^2. That backend and its identification grid are retired.

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
`TorqueStopMotion` and joins it before disconnecting. The E-stop remains the
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
