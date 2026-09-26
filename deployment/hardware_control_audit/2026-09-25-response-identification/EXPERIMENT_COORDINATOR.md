# Shared experiment coordinator

`franka_experiment_coordinator` is the local, non-real-time session owner for both
joint-controller identification and PPO policy experiments. Both modes publish the
same `franka_msgs/PolicyCommand`; the 1 kHz controller and canonical governor do
not change between experiments.

The coordinator never configures or activates a controller, recovers a robot fault,
changes gains/load/collision settings, or restarts a failed session. An approved
configuration, the process flag `--execute`, and a later explicit `~/start`
service call are all required before it publishes. A fault is sticky for the
process. Stop/fault/normal completion cease publication so the governor watchdog
can execute its bounded stop; an operator must then deactivate and disposition the
controller.

## Data path

The controller copies each coherent 1 kHz state into its fixed SPSC log record.
Its existing non-real-time writer thread writes CSV and publishes:

- `~/policy_observation` at no more than 30 Hz: measured q/dq, measured base-frame
  configured-EE position (the flange only while F_T_EE is identity), the raw action actually accepted by the governor, state and
  policy sequences, robot time/mode/errors, governor state, and source
  `CLOCK_MONOTONIC_RAW` timestamp.
- `~/governor_status` at 30 Hz and on status changes. It includes the configured
  session and governor-config SHA-256.

The coordinator rejects session/config mismatches, stale or nonmonotonic state,
non-MOVE mode, current robot errors, invalid governor commands, and any increase in
dropped log samples. It writes `resolved_config.json`, `events.jsonl`, and
`final_metadata.json` below the configured local artifact root.

Identification mode builds reviewed, one-joint smooth sine references with quintic
amplitude ramps, converts desired q to the raw action using the exact
`(q_target - default_position) / scale` contract, and publishes it through the
same governor. Conservative analytic bounds for the ramped waveform must fit the
reviewed per-joint velocity, acceleration, and jerk budgets. It only checks a
reviewed start pose; it does not move the robot to that pose.

For the current bare-flange deployment, PPO validation requires `fr3_link0`, `fr3_flange`, and a reviewed identity `F_T_EE`; this matches the live inventory.

PPO mode assembles the 24-element observation in this exact order:

1. measured q minus the bundle default q,
2. measured dq,
3. target position minus measured tracked-body position in the base frame,
4. the raw actor output last accepted by the governor.

Inference runs in a separate local subprocess. At startup that worker verifies the
immutable bundle and manifest trust anchor, then uses ONNX Runtime CPU with its
single-threaded bundle session. The coordinator rejects non-FR3 joint names,
robot/frame/order/period mismatches, default-offset or scale mismatches, and action
clipping.

## Python environments

The ROS coordinator uses the system ROS Python environment; do not put the ROS
node in the policy virtual environment. Identification mode needs no virtual
environment. PPO configuration names an executable `worker_python`; only that
subprocess uses the policy verification environment containing numpy, ONNX,
ONNX Runtime, PyYAML, and jsonschema. The known local environment is
`/home/chen-lab/yifan/venvs/franka-policy-verify-py312/bin/python3`.

The two existing local bundles are Panda bundles and are intentionally rejected by
the FR3 contract check. A separately exported and reviewed FR3 bundle is still
required.

## Configuration and offline checks

The installed pending templates are:

- `config/experiment_identification.pending.yaml`
- `config/experiment_ppo.pending.yaml`

They contain null fields and false approval gates, so they cannot run. Set
`execution_context` explicitly to `fake`, `shadow`, or `hardware`; each result
records that context. Copy one to
a local run/config directory, fill it from reviewed values, and keep the approved
copy immutable. The `governor_config_sha256` must equal the value configured in
the controller. `session_id` must also match.

Offline structural validation performs no ROS publication:

```bash
source /opt/ros/humble/setup.bash
source /home/chen-lab/franka_ros2_ws/install/setup.bash
ros2 run franka_policy_controller franka_experiment_coordinator \
  --config /absolute/path/to/experiment.yaml --validate-only
```

For a separately authorized session, running with `--execute` still waits for the
explicit start service. The controller lifecycle remains a separate operator
action. The start service additionally requires a current coherent controller
observation, exact session/config hashes, MOVE mode, and the reviewed q/dq start
bounds. Use one approved session per start configuration; the coordinator never
automatically moves between identification configurations.

## Current evidence and boundary

Offline tests cover strict approval rejection, identification mapping and ramp
constraints, observation order, stale state, log overflow, sticky faults, and
Panda-bundle rejection. The ROS plugin and coordinator tests pass, and the
isolated worker verified and evaluated the immutable Panda test bundle as a worker
transport test. No controller was activated and no physical motion occurred.

A direct configure/activate/update/deactivate harness now passes with encoded Franka state/model interfaces and injected non-MOVE, nonfinite-state, and tracking faults. Canonical core tests cover timing, command ordering, watchdog, stopping, and terminal latching. A full controller-manager fake-system lifecycle/fault run remains an additional commissioning check. See `HARDWARE_IDENTIFICATION_PREP.md` for config generation, preflight, and the operator sequence.
