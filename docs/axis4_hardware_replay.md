# Axis 4: hardware-conditioned simulation replay

## Purpose

This workflow compares Isaac Lab with the four completed FR3 hardware sessions without hand-copying their controller or path parameters. It is an experiment runner, not a hardware controller. The immutable hardware session and its deployed bundle determine the task, checkpoint, initial state, path, and gains.

## Inputs and validation

For every session, `HardwareSessionReplay` reads:

- `resolved_config.json` for route, gains, safety limits, action mapping, timing, and explicit waypoints;
- `final_metadata.json` for terminal state and waypoint outcomes;
- `config.sha256` for cross-artifact identity;
- the first command-valid finite `q,dq` row in `samples.csv`; and
- the local deployment-bundle manifest, whose SHA-256 and checkpoint SHA-256 must match the hardware record.

The loader rejects incomplete sessions, dropped samples, non-smooth stops, unknown controller modes, a non-bare-flange robot, nonzero payload, nonzero modeled friction compensation, nonuniform K/D/error clip, wrong timing, a mismatched task/checkpoint/path, or an evaluation other than one environment and one episode. Rejecting unsupported physics is preferable to producing an apparently paired result that is not paired.

## Replay contract

| Quantity | Replay behavior |
|---|---|
| Policy | Exact `.pt` checkpoint referenced by the deployed ONNX bundle |
| Initial state | First command-valid measured encoder `q,dq` |
| Target schedule | Exact recorded waypoint coordinates and orientation |
| Scheduler | Advance on 10 mm reach or timeout; first/later timeout from session |
| Policy timing | 50 Hz with one effective policy-step delay |
| Low-level timing | 1 kHz |
| Position route | Held `q_ref`, zero `dq_ref` |
| Velocity route | Held policy `dq_ref`, 1 kHz forward integration of `q_ref` |
| Controller | Recorded K and D independently, error clip, torque slew, limit torque, compensation, and 100 Hz command filter (recorded for velocity; audited Franky backend default for position) |
| Payload | Zero for the four packaged sessions |

Franky's `gains_time_constant_s` is recorded but does not affect these fixed-gain sessions because no gain transition occurs after the controller starts.

## Metrics

Each simulation job emits normal evaluator artifacts plus:

- `simulation_trace.csv`: 1 kHz joint/reference/torque/flange trace;
- `simulation_trace_summary.json`: distribution metrics; and
- exact hardware replay provenance in evaluator `manifest.json`.

The suite compiler writes `compiled/paired_replay.csv` and `.json`. Geometric circle error is the Euclidean distance to the ideal 3-D circle, combining plane-normal error and radial error. The headline value excludes waypoint 0 (the home-to-circle approach); an all-sample value is retained for startup-transient inspection. Acceleration is the absolute finite difference of measured joint velocity. Distribution metrics flatten joints and time consistently on both sides. Hardware `tau_command` is compared with the simulator's filtered controller command before the separately applied gravity term.

Signed deltas are `simulation - hardware`. A small geometric delta with a large acceleration or torque delta is not plant parity. Waypoint agreement, dynamics distributions, and fault agreement must all be inspected.

## Command

```bash
cd /home/chen-lab/isaac/franka-rl
export FRANKA_RL_DATA_ROOT=/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data

direnv exec . /home/chen-lab/isaac/.venv/bin/python -u \
  scripts/experiments/run_axis4_hardware_replay.py \
  --device cuda:0 \
  --viz none
```

The runner is sequential because each job launches an Isaac/Kit process. It resumes validated completed jobs by default. Use `--no-resume` to force reruns, `--fail-fast` during debugging, and repeated `--session-id` arguments to select sessions.
