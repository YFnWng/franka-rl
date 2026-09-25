# Franka real-time-machine handoff

Read `AGENTS.md`, then `REAL_ROBOT_DEPLOYMENT_PLAN.md`. All hardware experiment
components belong on this computer. No connection to the training workstation
is required. This package implements offline export/validation; it does not
contain a working hardware controller or experiment executor.

## Receiving the package

Transfer the archive and its `.sha256` sidecar out of band. Check the archive
hash before extraction. Use a local POSIX filesystem for source, dependencies,
and runtime logs. `handoff_manifest.json` records the exact source snapshot,
base Git revision, policy bundles, suites, and every delivered file hash.
`source_base_commit.tar` preserves the committed source; `source/` includes the
handoff additions, whose working-tree status is recorded explicitly.

Install `requirements.txt` in a separate local Python environment. The tested
producer used Python 3.12; select a compatible environment on this computer.
This package contains no dependency wheels: dependency installation may require
a package index or a separately prepared wheel cache for the receiving platform.
After dependencies are provisioned, preflight requires no network.

Run from any directory, supplying the separately recorded manifest hash:

```bash
/path/to/local/venv/bin/python /path/to/handoff/preflight.py \
  --manifest-sha256 TRUSTED_MANIFEST_HASH
```

Preflight checks the inventory, both ONNX policies, their training contracts,
the target files, and identical trial schedules. It prints the resolved trials
without launching ROS or touching hardware. Source instructions and scripts
must be reviewed before execution; hashes establish identity, not code safety.

## Controller choices and governor planning

The current first-demo direction is **position-action PPO with the robot's
existing internal joint-impedance settings**, preceded by a shared reference
governor. Preserve two alternatives: position-action PPO with an explicit
host-side PD torque controller, and direct torque-output PPO. These are distinct
control contracts; choosing a torque interface does not necessarily mean the
policy itself outputs torques.

The [shared reference governor plan](REFERENCE_GOVERNOR_PLAN.md) specifies the
30 Hz → 1 kHz path, state initialization, timing/freshness, trajectory feasibility,
stop/fault handling, simulation reuse and offline validation milestones. The standalone core and Python/Isaac adapter are now implemented in
[reference_governor](reference_governor/README.md), with wheel/sdist shipment from
this repo; the simulation workstation does not need franka_ros2. Actual Isaac
runtime validation and the ROS controller remain open. No hardware motion is
authorized by this implementation. New training should use the frozen command
path and an evidence-based response model, not assumed broad PD coverage.

## Runtime inputs and next implementation

- `run_request.yaml` selects the nominal shadow suite and disables hardware.
- `comparison.yaml` lists nominal and DR suites with identical targets/order.
- `suites/` contains concrete local relative paths and verified hashes.
- `target_sets/` contains explicit positions in `panda_link0`.

These shadow suites request no movement, including no start-pose movement.
They test local scheduling and inference, not reaching performance. They do not
establish paired physical initial states. Active suites require the receiving
implementation's validated start-pose trajectory and measured start-state
tolerances. Define their targets after inspecting the actual robot workspace.
Resolve runtime artifact paths relative to the handoff root; the configured
`../hardware_runs` keeps generated output outside the immutable package.

First return the hardware/software inventory requested in section 11 of the
plan. Then implement the C++ ONNX/contract runtime and verify the bundled JSON
fixtures; add the governor, watchdogs, fake-hardware integration, and local YAML
suite coordinator before progressing through shadow and commissioning gates.
The run-request and comparison YAMLs are proposed executor inputs, not ROS
launch files. Implement their loaders explicitly; no UI or runtime prompts are
needed. An active request must never restart automatically after a fault/reboot.

## Evidence and limitations

Both bundles were regenerated from clean project revision
`d0bfa66331a4813bcff0a0b1d24762618603d012`. Each passed 1,024 synthetic observation
cases against the native deterministic RSL-RL actor at absolute tolerance 1e-5.
The maximum errors were approximately 4.77e-6 (nominal) and 3.82e-6 (DR).

No recorded rollout observation fixtures were supplied. C++ parity, robot-model
compatibility, frame calibration, timing, the reference governor, and hardware
commissioning remain unvalidated. The exporter does not implement low-level
motor control. The simulator actuator settings are not hardware gains.
