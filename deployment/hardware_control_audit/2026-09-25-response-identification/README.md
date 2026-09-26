# FR3 joint-position response identification audit — 2026-09-25

This directory answers the simulation workstation's controller-response request. Work here is offline-only: no robot motion, controller activation, settings change, or fault recovery was performed.

## Results

- [`CONTROLLER_PROPERTIES.md`](CONTROLLER_PROPERTIES.md) separates verified host/controller behavior from unavailable firmware internals.
- [`identification_suite.pending.yaml`](identification_suite.pending.yaml) proposes small one-joint smooth references, fitting/validation splits, watchdogs, and stop rules. It is deliberately non-executable: lab approval, start poses, workspace, and physical-stop procedure are null or false.
- [`SIGNAL_SCHEMA.md`](SIGNAL_SCHEMA.md) defines coherent signals, clocks, units, torque conventions, provenance, and fault-log retention.
- [`analyze_response.py`](analyze_response.py) fits the robot/host clock relationship and reports per-joint lag, tracking errors, and optional sine gain/phase. It labels the result as an effective response rather than firmware gains.
- [`INTEGRATION_STATUS.md`](INTEGRATION_STATUS.md) records the offline integration evidence and remaining work.
- [`EXPERIMENT_COORDINATOR.md`](EXPERIMENT_COORDINATOR.md) records the shared identification/PPO coordinator contract and environment split.
- [`HARDWARE_IDENTIFICATION_PREP.md`](HARDWARE_IDENTIFICATION_PREP.md) gives the offline approval/config/preflight workflow and the later separately authorized lifecycle procedure.
- `SHA256SUMS` pins all committed audit inputs. Raw traces stay outside Git.

The ROS adapter is implemented at `franka_ros2/franka_policy_controller`. It links the canonical governor, blocks invalid outputs before position writes, and owns the synchronized logger. Its fixed-size SPSC ring is allocation-free and nonblocking on the producer; CSV formatting is restricted to the non-RT consumer. The selected ROS joint-position startup path was also changed to propagate a start exception without automatic recovery/retry.

## Validation performed

```bash
cmake -S tools/response_identification -B /tmp/franka-response-log-build
cmake --build /tmp/franka-response-log-build
ctest --test-dir /tmp/franka-response-log-build --output-on-failure
# 1/1 passed

colcon test --packages-select franka_hardware \
  --ctest-args -R franka_hardware_robot_test --output-on-failure
colcon test-result --verbose
# 16 tests, 0 failures

# With LD_LIBRARY_PATH selecting the local libfranka:
colcon test --packages-select franka_policy_controller --return-code-on-test-failure
# plugin load, pending-config rejection, and 7 coordinator tests passed

GOVERNOR_NATIVE_TEST=/tmp/governor-audit-build/governor_native_test \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/chen-lab/yifan/venvs/franka-governor/bin/python -I -m pytest \
  deployment/reference_governor/tests -q
# 34 passed
```

The current governor release is outside Git at `/home/chen-lab/yifan/governor_releases/0.1.0-armed-hold-20260925`. Its manifest SHA-256 is `fcc058d13ee5a14b547e60d3845bec12557bc2df1769b9aee81bb13eae9aa222`. It includes the sequence-0 armed hold and sequence-1 watchdog transition required by the explicit operator start gate. The earlier reconciled release is superseded for deployment and training parity.

## Future trace analysis

```bash
python3 deployment/hardware_control_audit/2026-09-25-response-identification/analyze_response.py \
  /home/chen-lab/franka_ros2_ws/hardware_inventory/YYYY-MM-DD/response_identification_<UTC>/samples.csv \
  --frequency-hz 0.50 \
  --output /home/chen-lab/franka_ros2_ws/hardware_inventory/YYYY-MM-DD/response_identification_<UTC>/analysis.json
sha256sum /home/chen-lab/franka_ros2_ws/hardware_inventory/YYYY-MM-DD/response_identification_<UTC>/*
```

Only run analysis after an approved trace exists. Do not execute the pending YAML as motion instructions.
