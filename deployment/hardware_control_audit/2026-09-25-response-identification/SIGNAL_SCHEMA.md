# Synchronized response log schema

Raw traces belong outside Git under `/home/chen-lab/franka_ros2_ws/hardware_inventory/<date>/response_identification_<UTC>/`. Each session stores the exact approved YAML and hashes beside `samples.csv`, `events.jsonl`, and `metadata.json`. Faulted and rejected trials are retained.

The ros2_control update callback creates one fixed-size `RtSample` per actual cycle and attempts a nonblocking push to a preallocated single-producer/single-consumer ring. It performs no allocation, locking, formatting, ROS publication, or disk I/O. A non-RT writer drains the ring to an already-open local file. Any ring overflow is a terminal trial fault and is itself recorded in session metadata; samples must never be silently overwritten.

## Time and ordering

- `session_id`: new nonzero identifier for each activation; no reuse after a fault.
- `rt_sequence`: every ros2_control update, starting at zero.
- `policy_sequence`: accepted policy output; unchanged between policy updates.
- `observation_sequence`: observation that produced that policy output.
- `robot_time_s`: FCI robot time from the same `RobotState` sample.
- `host_update_entry_ns`: `CLOCK_MONOTONIC_RAW` on update entry.
- `host_state_received_ns`: host timestamp immediately after obtaining the state used in the callback. In the current ros2_control architecture this is an observation boundary, not a measured network-packet arrival time.
- `host_policy_complete_ns`: worker timestamp carried with the policy message.
- `host_command_consumed_ns`: timestamp when the RT callback accepts the policy message.
- `host_command_written_ns`: timestamp immediately after setting all seven command interfaces; it is not wire-arrival or firmware-consumption time.

Before computing delays, fit `host_state_received_ns = a * robot_time_s + b` per uninterrupted session using robust linear regression. Report residual distribution, drift `a`, sample count, and clock discontinuities. Never subtract robot and host timestamps directly without this relationship. Exact firmware command-consumption time is unavailable and remains part of identified delay.

## Per-cycle columns

All joint arrays use FR3 joints 1 through 7 in radians, radians/s, radians/s^2, radians/s^3, or Nm as stated.

| Group | Columns | Meaning |
| --- | --- | --- |
| Identity | `session_id`, `trial_id`, `rt_sequence`, `policy_sequence`, `state_sequence`, `observation_sequence` | Immutable ordering/provenance |
| Clocks | five time fields above plus `period_ns` | Actual update timing |
| Policy | `raw_action_j*`, `mapped_target_j*`, `projected_target_j*` | Actor output and visible target intervention |
| Governor | `q_ref_j*`, `dq_ref_j*`, `ddq_ref_j*`, `status`, `reason`, `command_valid`, `replanning_blocked`, `projection_rad` | Exact shared-core output |
| Robot desired | `q_d_j*`, `dq_d_j*`, `ddq_d_j*`, `tau_J_d_j*` | FCI desired motion/torque state |
| Robot measured | `q_j*`, `dq_j*`, `tau_J_j*`, `tau_ext_hat_filtered_j*` | Measured/estimated state |
| Model terms | `gravity_model_j*`, `coriolis_model_j*` | Host model estimates using the logged state/load |
| Robot status | `robot_mode`, `control_command_success_rate`, current/last error bitsets | Health and terminal evidence |
| Logging | `ring_fill`, `dropped_samples` | Logging integrity |

`q_ref` is the actual post-governor value offered to the command interfaces. Record it before the writes and set `host_command_written_ns` only after all seven writes succeed. `q_d` is the robot's desired state returned by FCI and must not be renamed as the post-governor command. `tau_J_d` excludes gravity according to libfranka documentation. `tau_J` is measured joint torque; `tau_ext_hat_filtered` is an estimate. `gravity_model` and `coriolis_model` are model outputs, not proof of firmware terms.

## Events

`events.jsonl` records activation/config validation, trial boundaries, policy receipt/rejection, governor state transitions, watchdog events, controller-manager transitions, robot errors, logger overflow, operator stop, and writer failures. Each event includes session/trial/sequence IDs and host monotonic time. A final metadata record marks the trace `complete`, `faulted`, or `writer_incomplete`; absence of that record is treated as incomplete.

`current_error_bits` and `last_motion_error_bits` are unsigned 64-bit masks. Bit 0
through bit 35 follow the declaration order in `franka_msgs/msg/Errors.msg`; bits
36 through 63 are zero. Archive the message/source hashes in `provenance.json` so
the mask remains decodable if the upstream message changes.


## Non-RT supervisor topics

The controller's non-RT writer derives two ROS messages from the same
`RtSample` that is serialized to CSV. No publication occurs in the 1 kHz
callback.

`PolicyObservation` is rate-limited to approximately 30 Hz using
`host_state_received_ns`. It carries session/trial/state/accepted-policy
sequences, q/dq, accepted raw previous action, robot/governor state, error bits,
dropped-sample count, and `tracked_position_base_m`. The tracked position is
live `O_T_EE`; for the current PPO contract it represents `fr3_flange` only
under the inventoried and reviewed identity `F_T_EE` assumption.

`GovernorStatus` is emitted at the same rate and immediately on
status/reason/validity transitions. It additionally carries the
`governor_config_sha256` configured in the controller. The coordinator requires
this digest and session ID to match its approved YAML.

Both topic timestamps use `CLOCK_MONOTONIC_RAW`, as do the command completion and
observation timestamps. DDS receipt time is not substituted for the source
sample time.
