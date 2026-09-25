# Franka reference governor 0.1.0

Canonical source lives here in **franka-rl**, not franka_ros2. This package builds
without ROS, libfranka, Isaac or a robot connection. It provides one C++17 core,
Python bindings, batched instances, an opt-in Isaac Lab action adapter, and tests.
The future ROS controller links this same core; no ROS adapter is implemented here.
No hardware control or configuration is performed by this package.

## Install and ship

On the simulation machine's Python 3.12 environment (Isaac Sim 6.0.1 / Isaac Lab
v3.0.0-beta2.patch1), install the supplied CPython 3.12 Linux x86-64 wheel, or build
from the source distribution with a C++17 compiler and CMake:

```bash
python -m pip install /path/to/franka_reference_governor-0.1.0-cp312-cp312-linux_x86_64.whl
# Alternatively, from this repo:
python -m pip install ./deployment/reference_governor
```

This package declares only NumPy as a Python runtime dependency. The optional
`franka_governor.isaaclab` module uses the workstation's existing Isaac/Torch.
The wheel is native-platform-specific, not advertised as a manylinux wheel.
Use the source distribution on another architecture/Python or an incompatible
Linux runtime. NumPy need not be upgraded if the existing version satisfies the
requirement. Build dependencies are pinned in pyproject.toml.

From the repository root, build a versioned wheel/sdist/manifest outside Git:

```bash
python scripts/deployment/build_governor_release.py --output /local/new-release-directory
```

The build environment needs `build`, `scikit-build-core==0.11.6` and
`pybind11==3.0.1`. The script refuses an existing output directory, records hashes
of source and artifacts, and includes the opt-in task integration files. It does
not build or replace policy bundles. Keep a copy of its manifest with training
provenance and select the exact governor config by hash.

## Algorithm and guarantees within its model

Version `bounded_quintic_v1` uses seven independent quintic Bezier polynomials
with a common fixed horizon T = horizon_ticks * 0.001 seconds. Given current
reference q,v,a and endpoint e, control points for each joint are:

```text
B = [q, q+v*T/5, q+2*v*T/5+a*T*T/20, e, e, e]
V[k] = 5*(B[k+1]-B[k])/T
A[k] = 4*(V[k+1]-V[k])/T
J[k] = 3*(A[k+1]-A[k])/T
```

Every candidate must pass bounds on all position and derivative control points.
The convex-hull property bounds the **entire continuous curve**. Position bounds
include the required margin. Each position control point also stays within
max_segment_distance of its start position. Velocity checks use the tightest
manufacturer position-dependent envelope over the position hull, intersected
conservatively with operating velocity caps (including a 0.001 rad/s numerical
reserve). Acceleration/jerk caps cannot exceed libfranka 0.19's 9.999/4999.999
interface constants. No random parameters enter the core.

Targets are mapped in float32, then projected to the operating position box.
The first candidate endpoint advances at most max_segment_distance per joint
from the current reference toward the target. Up to 12 fixed-count attempts halve
that displacement. Candidate rejection retains the prior certified segment.
Persistent blocked replanning or excessive projection latches a stopping fault.
Replan at accepted policy updates or segment completion, not continuously every
tick. This is conservative and may move slowly or fail to find a candidate even
where another planner could succeed. It is not time-optimal and synchronizes
segment duration, not Cartesian path geometry.

Each accepted segment ends with zero reference velocity/acceleration. On stale
or invalid policy input, reject later policies and finish that short cached
segment, then hold its stationary endpoint in TERMINAL. This is the explicitly
chosen **bounded continuation stop**, not minimum-distance braking. Time to rest
is at most one configured horizon; remaining displacement is conservatively at
most twice max_segment_distance per joint, because both current point and endpoint
lie in the start-centered hull. A deployment must review that stopping distance
and actual robot tracking, as well as obstacle clearance.

No sampled check is substituted for the polynomial bound. de Casteljau evaluation
clamps each floating-point interpolation to its two endpoint values to prevent
roundoff escaping the certified convex hull; it does not post-clip a generated
trajectory to joint limits. Numerical test tolerances are distinct from operating
margins. The proof assumes finite arithmetic within those tolerances and valid
configuration/state; tests are not a hardware safety certification.

## State, timing and failures

The core has no heap allocation in reset/submit/step/stop. Construction validates
configuration and may throw. The Python wrapper allocates and is **not RT code**.
Each environment has independent state/session. `reset` takes desired q/dq/ddq,
checks measured state, and must be given a strictly increasing nonzero session ID.
It initializes a feasible stopping segment without replacing derivatives by zero.
The simulator bridge supplies zero desired acceleration only for simulator resets.
An infeasible initialization returns false, with command_valid false.

`submit` accepts one outstanding message, exact next sequence number, same session,
finite raw actions and ordered monotonic timestamps. `step` advances one reference
millisecond, validates fresh coherent feedback, consumes the queued message, and
replans from the resulting reference. Thus a newly consumed target affects the
following reference tick; the reference cannot jump on target acceptance.

While RUNNING, policy target is held between updates but reference keeps moving.
No extra low-pass filter is applied. Keep the inspected ROS wrapper's optional
position filters/limiters disabled when this governor owns the command path.
Raw previous action is the accepted actor output, never its projected/governed
replacement. Observation normalization and policy mapping are unchanged.

Invalid/stale policy, projection or blocked-replanning faults enter STOPPING.
State/session/clock/tracking failures enter FAULT with command_valid false:
**the caller must not send that output as a command** and must use its reviewed
hardware fault/stop path. No fictitious continuous stop is promised after losing
state or communication. TERMINAL retains a valid stationary reference while
feedback checks continue; it cannot resume. New sessions require explicit reset.
Timing uses monotonic integer nanoseconds; late cycles are rejected using the
required configured tolerance, never caught up through burst robot writes.

The caller owns mode checks, timestamp conversion, transport, observer coherence,
workspace/collision checks, physical stop and controller lifecycle. Set
Feedback.healthy=false on unusable robot mode/communication/state. The core has
joint guards and reference validation; it does not know obstacles, tool geometry,
internal torques or independent hardware stopping. Config metadata/review records
are not authorization to control a robot. No recovery or boot restart is included.

## Configuration

`configs/simulation.json` is a **synthetic simulation fixture**, not lab-approved
operating limits. Loading it requires `allow_simulation=True` explicitly.
Manufacturer position/velocity parameters come from the archived live fr3v2.1
model. Fixture margins, caps, horizon, tracking thresholds and watchdog budgets
are choices for software tests. Training with it is a governor-aware experiment,
not proof of deployment equivalence.

`configs/deployment.template.json` deliberately contains null required values and
is rejected. Fill all values and a review record before proposing a deployment.
The loader rejects missing, unexpected or nonfinite core fields. Archive config
bytes plus SHA-256. There is no fallback to permissive limits.

## Isaac Lab integration

The adapter was inspected against the official pinned source:
https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-beta2.patch1
It uses ActionTerm.process_actions per environment step, apply_actions per physics
step, per-environment reset, ProxyArray.torch and set_joint_position_target_index.
Native-physics internal decimation is rejected because it bypasses the required
per-physics hooks. See ISAAC_API.json for inspected file hashes.

The reference timeline is physics dt=1/3000 s, decimation=100: exactly 30 Hz PPO
and one 1 kHz governor tick every three physics steps. Policy updates not aligned
with a governor boundary are queued to the next boundary. That message is consumed
on the following core step, an explicit additional tick delay. Simulation uses
synthetic monotonic timestamps and zero inference compute latency. It does not
claim to reproduce measured hardware latency, which remains unknown. Do not add
an independent fixed/random-action-delay wrapper without reviewing raw previous
action, timestamps and fault semantics.

The CPU batch bridge moves data between GPU and host. It is a correctness baseline
for small environment counts; no GPU kernel or high-throughput PPO claim is made.
Benchmark before large training. A later GPU core must pass complete reference
trajectory and fault-event parity, not just action-mapping parity.

The repository registers the opt-in task `Franka-FR3v2-Governed-Reach-v0`. After
installing this wheel and updating the repo, select it using existing training/
evaluation scripts and set:

```bash
export FRANKA_RL_GOVERNOR_CONFIG="$(pwd)/deployment/reference_governor/configs/simulation.json"
export FRANKA_RL_GOVERNOR_SIMULATION_FIXTURE=1
# FRANKA_RL_FR3_USD must identify the corrected, independently validated asset.
# Use a small --num_envs for the first simulator smoke test.
```

Existing Panda/FR3 tasks and received policy bundles are unchanged. New task
termination `governor_fault` marks failed environments when the governor latches
STOPPING/TERMINAL/FAULT. Full stopping trajectories are tested separately in the
core; simulator auto-reset must never be copied into robot fault recovery.
This package does not correct the FR3 USD, identify firmware response, or replace
80/4 with known hardware gains. Those are still simulator/commissioning tasks.

**Isaac Sim has not been run on this host.** API-stub tests verify method names,
scheduling and partial-reset behavior; real startup, reset, device transfers,
reward/termination interaction and throughput remain workstation validation gates.

## Native ROS-side consumption

```bash
cmake -S deployment/reference_governor -B /tmp/governor-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/local/governor-0.1.0
cmake --build /tmp/governor-build
ctest --test-dir /tmp/governor-build --output-on-failure
cmake --install /tmp/governor-build
```

A downstream CMake project uses `find_package(franka_governor 0.1 CONFIG REQUIRED)`
and `target_link_libraries(my_controller PRIVATE franka_governor::core)`.
A ROS controller must use the core directly, not call Python. It still needs
validated mailbox/observation collection, inference, command_valid handling and
terminal-fault integration, including removal of automatic startup recovery.

## Tests

```bash
GOVERNOR_NATIVE_TEST=/tmp/governor-build/governor_native_test \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -I -m pytest deployment/reference_governor/tests -q
```

Native tests cover 5,000 ticks, derivative/position bounds, stale stopping and
zero core allocations. Python tests add boundaries, reversals, clock/state/action
faults, config rejection, nonzero-velocity reset, selective batch reset, float32
mapping, deterministic replay, C++ trace parity and stubbed Isaac action hooks.
Use the native executable for parity; omitting its environment variable omits
that particular comparison. Hardware and closed-loop simulator acceptance are
reported separately.
