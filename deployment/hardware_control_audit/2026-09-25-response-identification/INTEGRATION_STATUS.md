# Offline integration status and blockers

## Completed and tested offline

- The canonical shared governor core/config hashes are reconciled. Its native and Python suites pass. Reset now holds the coherent desired state with sequence 0 while waiting for explicit operator start; the action watchdog is armed only after sequence 1, while state/timing/tracking checks remain active.
- `franka_policy_controller` links `franka_governor::core`, claims seven position interfaces plus coherent Franka state/model interfaces, initializes from `q_d/dq_d/ddq_d`, validates `command_valid` before writes, and has no automatic activation, recovery, or restart.
- A direct controller configure/activate/update/deactivate harness uses realistic encoded `fr3/robot_state` and `fr3/robot_model` pointer interfaces. It passes coherent command writes and fail-closed non-MOVE, nonfinite-state, and tracking-fault cases without rewriting command interfaces.
- Joint-position startup propagates libfranka start failure without automatic recovery/retry. The focused hardware-wrapper tests pass.
- The 1 kHz callback writes fixed records to an SPSC ring. The non-RT writer owns CSV formatting, disk I/O, and coherent `PolicyObservation`/`GovernorStatus` ROS publication. Ring overflow or writer failure terminates the controller.
- A shared local `franka_experiment_coordinator` supports identification and PPO modes through the same `PolicyCommand` and governor path. It owns approval validation, session/trial state, 30 Hz sequencing, start-state checks, watchdogs, sticky terminal faults, and `events.jsonl`/final metadata.
- Identification mode generates reviewed one-joint sine references with quintic amplitude ramps and exact default-offset raw-action mapping. Conservative analytic velocity/acceleration/jerk bounds must fit reviewed per-joint budgets. It does not move to a start pose.
- The coordinator now rejects action scales other than the native governor's fixed 0.5 rad mapping.
- `franka_experiment_prepare` deterministically generates a stock-`franka.launch.py` compatible controller YAML from approved experiment/governor inputs. It includes the state broadcasters, 1 kHz update rate, FIFO priority 98, exact core parameters, hashes, session, and log path.
- `franka_experiment_preflight` cross-checks all three approved inputs and verifies the RT kernel, unlimited memlock, RT priority, local libfranka 0.19.0 resolution, local writable artifact filesystem, free space, and unused runtime session path. It can stage immutable copies and a machine-readable report.
- The controller obtains the exact launch-generated URDF from `robot_state_publisher` with a bounded timeout, matching the official Franka controller pattern while avoiding hand-copied URDF.
- PPO inference uses a separate local subprocess and immutable bundle verification. PPO is restricted to `fr3_joint1..7`, `fr3_link0 -> fr3_flange`, and reviewed identity `F_T_EE`. Existing Panda bundles are rejected.
- A complete preparation and later authorized operator procedure is in `HARDWARE_IDENTIFICATION_PREP.md`. Identification needs system ROS Python 3.10 and no virtual environment.
- Pending controller and experiment templates remain deliberately non-executable.

## Still required before any motion

1. Lab reviewers must choose and approve one collision-free start pose per session, swept workspace/exclusions, physical-stop procedure, attachment/load/identity-`F_T_EE` assumptions, operating bounds, watchdog budgets, waveform set, session ID, and artifact root. The synthetic simulation governor config is not a hardware approval.
2. Run collision and swept-volume review for every start pose and identification reference. Joint limits alone are insufficient.
3. Generate the controller YAML from the approved inputs, run `franka_experiment_preflight --stage`, and require every check to pass in the exact deployment shell.
4. Separately authorize only the first low-amplitude hardware-identification session, with the operator at the E-stop. No motion authorization exists in this repository state.
5. A full controller-manager fake-system lifecycle/fault exercise remains desirable, especially explicit deactivation plus forced logger overflow/writer failure. The direct real-adapter harness and canonical core fault suites now cover the main command/state path but are not a physical or full-manager qualification.
6. Export, transfer, and verify an FR3 PPO bundle before any later policy session. This does not block response identification.
7. After authorized measurements, fit delay/bandwidth/damping/error/coupling models, validate on held-out trials/configurations, and return an evidence-based simulator response contract and DR ranges.

No response-identification motion has been performed. Passing these offline gates does not authorize controller activation or robot motion.
