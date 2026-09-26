# FR3 hardware-identification preparation and operation

Status: the software below is offline-tested. No hardware identification motion has been run. Commands in the **authorized hardware session** section connect to and activate control of the robot; do not run them until the lab separately authorizes that session and the operator is at the E-stop.

## What is already implemented

The governed joint-position controller now has a real adapter update harness using encoded Franka state/model interfaces. It verifies a coherent update and fail-closed behavior for non-MOVE mode, nonfinite state, and excessive tracking error. The canonical governor suite covers timing, sequence, mailbox, projection, watchdog, stopping, and terminal-state behavior. The controller does not recover, activate, restart, or move to a start pose by itself.

Two installed commands remove manual duplication between the reviewed files:

- `franka_experiment_prepare` generates a complete ROS controller YAML from the approved experiment YAML and reviewed governor JSON.
- `franka_experiment_preflight` verifies all hashes and values, the 1 kHz controller-manager settings, RT kernel/resource limits, libfranka runtime resolution, local writable storage, free space, and a fresh session ID. `--stage` copies the three inputs read-only under `ARTIFACT_ROOT/prepared/session-ID/` without creating the runtime directory.

Identification uses system ROS Python 3.10. It does not need a Python virtual environment. The Python 3.12 ONNX environment is only for later PPO worker inference.

## Inputs the lab must decide

Do not copy the synthetic `simulation.json` limits into a hardware approval. Starting from the pending templates, reviewers must supply:

1. One collision-free start configuration for this session. The coordinator checks it but never moves the arm to it.
2. Allowed swept workspace and forbidden regions for every one-joint waveform. Review self-collision, fixtures, table, cables, people, and the bare flange.
3. Governor margins, reference velocity/acceleration/jerk limits, measured/desired tracking bounds, projection bounds, and timing/watchdog budgets.
4. Identification joint, amplitude, frequency, ramp, holds, cycles, and fit/validation assignment. The coordinator rejects a waveform whose conservative derivative bounds exceed the reviewed limits.
5. A new positive session ID and a local artifact root with at least 2 GiB free.
6. Confirmation that the robot remains FR3 Arm3Rv2 revision 02.01, no end effector, zero external load, identity `F_T_EE`, existing internal impedance settings, and unchanged collision behavior.
7. Named approver, approval time, workspace review, physical stop procedure, and separate authorization for motion.

Use one session per start pose. First authorize only the smallest, lowest-frequency single-joint trial needed to prove the path. Expand the suite only after reviewing its trace.

## Prepare and validate without connecting to the robot

Set paths for a new review directory. The examples below use placeholders intentionally:

```bash
source /opt/ros/humble/setup.bash
source /home/chen-lab/franka_ros2_ws/install/setup.bash
export LD_LIBRARY_PATH=/home/chen-lab/local/libfranka-0.19.0/lib:${LD_LIBRARY_PATH:-}

REVIEW_DIR=/home/chen-lab/franka_ros2_ws/hardware_inventory/YYYY-MM-DD/identification_review
mkdir -p "$REVIEW_DIR"
cp /home/chen-lab/yifan/franka-rl/deployment/reference_governor/configs/deployment.template.json \
  "$REVIEW_DIR/governor.review.json"
cp /home/chen-lab/franka_ros2_ws/src/franka_ros2/franka_policy_controller/config/experiment_identification.pending.yaml \
  "$REVIEW_DIR/experiment.review.yaml"
```

Fill the governor JSON first. Keep `purpose: reviewed_operating_config`, replace every null in `core`, and fill `review_record.approved_by` and `review_record.approved_at`. The C++ governor performs the final numerical validation.

Compute its digest and put that exact lowercase value into `experiment.review.yaml`:

```bash
sha256sum "$REVIEW_DIR/governor.review.json"
```

Fill every null in the experiment YAML. Before motion authorization, leave `status: pending_lab_review`, `executable: false`, and `motion_authorized: false`, then run structural validation:

```bash
ros2 run franka_policy_controller franka_experiment_coordinator \
  --config "$REVIEW_DIR/experiment.review.yaml" --validate-only
```

After the lab has completed its review and separately authorizes the session, set `status: approved`, `executable: true`, and `motion_authorized: true`, and fill all four approval strings. Generate the controller YAML rather than copying values by hand:

```bash
ros2 run franka_policy_controller franka_experiment_prepare \
  --experiment "$REVIEW_DIR/experiment.review.yaml" \
  --governor "$REVIEW_DIR/governor.review.json" \
  --output "$REVIEW_DIR/controllers.yaml"
```

The generator is exclusive-create and marks the file read-only. Use a new filename if any approved input changes. Then run and stage the complete preflight:

```bash
ros2 run franka_policy_controller franka_experiment_preflight \
  --experiment "$REVIEW_DIR/experiment.review.yaml" \
  --controller "$REVIEW_DIR/controllers.yaml" \
  --governor "$REVIEW_DIR/governor.review.json" \
  --stage
```

Do not proceed unless it prints `"ready_for_hardware_identification": true`. Keep the staged path and hashes in the run record. Preflight intentionally rejects the stale system libfranka, a non-RT kernel, finite memlock, RT priority below 98, remote storage, an existing runtime session directory, a hash/value mismatch, or missing approval.

Re-run the offline package gate after any source change:

```bash
cd /home/chen-lab/franka_ros2_ws
source /opt/ros/humble/setup.bash
export CMAKE_PREFIX_PATH=/home/chen-lab/local/franka-governor-0.1.0:/home/chen-lab/local/libfranka-0.19.0:${CMAKE_PREFIX_PATH:-}
export LD_LIBRARY_PATH=/home/chen-lab/local/libfranka-0.19.0/lib:${LD_LIBRARY_PATH:-}
colcon build --packages-select franka_policy_controller --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select franka_policy_controller --event-handlers console_direct+
colcon test-result --verbose
```

## Final physical checks

Immediately before an authorized run, the operator must verify all of the following in person:

- E-stop is connected, tested according to the lab procedure, and directly reachable.
- Desk reports robot and safety status OK, FCI is enabled, and no self-test or maintenance condition is pending.
- Flange is bare; configured load and `F_T_EE` match the approval.
- The arm is at the approved q within the experiment tolerance and nearly stationary.
- The full reviewed swept volume is clear.
- Wired route to `172.16.0.6` uses `enp0s31f6` with source `172.16.0.2`; Wi-Fi and unrelated high-load work are stopped according to lab practice.
- The three approved files and staged SHA-256 values are unchanged.
- A fresh runtime directory `ARTIFACT_ROOT/session-ID` does not exist.

## Authorized hardware session

These commands are an operator procedure for a later authorized session. Use separate terminals with the same ROS and libfranka environment in each terminal.

1. Start the stock Franka bringup with the generated controllers file. This connects to FCI and starts state broadcasters, but does not spawn the governed motion controller:

```bash
ros2 launch franka_bringup franka.launch.py \
  robot_type:=fr3 robot_ip:=172.16.0.6 load_gripper:=false \
  controllers_yaml:="$REVIEW_DIR/controllers.yaml"
```

2. Confirm only state broadcasters are active:

```bash
ros2 control list_controllers --controller-manager /controller_manager
```

3. Start the coordinator. It exclusively creates `ARTIFACT_ROOT/session-ID`, writes its resolved config, and waits without publishing commands:

```bash
ros2 run franka_policy_controller franka_experiment_coordinator \
  --config "$REVIEW_DIR/experiment.review.yaml" --execute
```

4. Load and configure the governed controller in the inactive state. It reads the exact generated URDF from the stock `robot_state_publisher` with a bounded timeout:

```bash
ros2 run controller_manager spawner governed_joint_position_controller \
  --inactive --controller-manager /controller_manager
ros2 control list_controllers --controller-manager /controller_manager
```

5. With the operator on the E-stop and the arm at the approved start, activate the controller. Activation starts libfranka joint-position control and opens the 1 kHz log, so this step requires motion authorization:

```bash
ros2 control switch_controllers \
  --activate governed_joint_position_controller \
  --strict --controller-manager /controller_manager
```

6. Inspect fresh observation and governor status before starting the waveform:

```bash
ros2 topic echo /governed_joint_position_controller/policy_observation --once
ros2 topic echo /governed_joint_position_controller/governor_status --once
```

The session/config IDs must match, robot mode must be MOVE, errors and dropped samples must be zero, and `command_valid` must be true.

7. Start the approved suite explicitly:

```bash
ros2 service call /franka_experiment_coordinator/start std_srvs/srv/Trigger '{}'
```

For an ordinary operator stop, stop publication and allow the governor to execute its bounded watchdog stop, then deactivate the controller:

```bash
ros2 service call /franka_experiment_coordinator/stop std_srvs/srv/Trigger '{}'
ros2 control switch_controllers \
  --deactivate governed_joint_position_controller \
  --strict --controller-manager /controller_manager
```

Use the physical E-stop immediately for unsafe motion; do not wait for ROS services. Do not invoke automatic error recovery or reactivate after any terminal fault. Retain `samples.csv`, `events.jsonl`, `resolved_config.json`, and `final_metadata.json`, record their hashes, and require operator disposition before another session ID.

## After the first trace

Analyze and inspect the smallest trial before authorizing more trials. Verify cycle timing, clock ordering, command/state sequence continuity, no governor projection or replanning block, no log loss, bounded tracking error, and a return to the reviewed start. Only then proceed to fitting delay, bandwidth, damping, steady-state error, and cross-joint/configuration dependence. Fitted values are equivalent response parameters, not disclosed firmware gains.
