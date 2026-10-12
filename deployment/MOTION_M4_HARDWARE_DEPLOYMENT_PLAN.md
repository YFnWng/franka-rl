# M4 hardware deployment plan

Date: 2026-10-11 (America/New_York)

## Decision and current gate

Bundle `/home/chen-lab/yifan/2026-10-11_21-02-42_m4` is accepted as an
immutable **transfer candidate**, not as a hardware executable. Its manifest
has `hardware_approved: false` and `hardware_adapter_ready: false`. The local
receiver verified every manifest hash and all five numerical fixtures. The
included runner remains fake-only and `FrankyMotionBackend` deliberately fails
closed.

M4 will use direct Franky/libfranka FCI torque control on the real-time machine.
`franka_ros2` is not in the real-time command path. Python may publish 50 Hz
tracking references and coordinate sessions; all feedback-dependent control,
projection, safety checks, torque processing and logging that must run every
servo tick will execute in native C++ at 1 kHz.

Do not edit the transfer bundle. Hardware commissioning configurations and all
result artifacts live under
`/home/chen-lab/franka_ros2_ws/hardware_inventory/2026-10-11/m4/` and reference
bundle files by SHA-256.

## Pinned receiving stack

- Robot: FR3 Arm3Rv2_02.01, bare flange, zero configured external payload
- Robot Control/System image: 5.9.2 / 5.9.2, FCI server 10
- Kernel: `5.15.133-rt69 PREEMPT_RT`
- Limits: memlock soft/hard unlimited; real-time priority limit 99
- Franky base: `2.0.1.dev60+gf88f0e9bvelimpzdt.libfranka.0.21.2`
- Franky wheel SHA-256:
  `8590e1241a8e305cf046c148968b7dc9055d2f58bcb944ddb5db33ebda70c00d`
- M4 `franka-motion` wheel SHA-256:
  `b3798c283a42ebf8f85b636a2eff47bd1b1e3edbf5f6bbd2b69d6a5fbaafca4f`
- M4 manifest SHA-256:
  `26d718217de37b132bccdfb7ecd147dc47554e48f21637c19c5f174380ed7d09`

Build the native extension from the exact source corresponding to the pinned
Franky base. Package it as a new wheel version and install it in a new isolated
`franka-motion-m4-server10` environment. Preserve the working joint-policy and
velocity-impedance environments unchanged.

## Native adapter architecture

The exact installed Franky source and exact libfranka 0.21.2 source have now
been audited. The result is recorded in
`deployment/hardware_control_audit/2026-10-11-m4-franky-controller-parity/README.md`.
Franky's Cartesian task law, desired-twist damping, model Coriolis and M3
kinematic posture projector can be reused. Its posture projector does not match
the M4 dynamic projector, and its callback slew limiter followed by libfranka's
100 Hz filter is not the same state machine as the simulation processor. It also
does not implement the measured-state line/circle fixtures. Reuse it as the
native base and implement these differences explicitly.

Implement three native motion types with shared safety, telemetry and stop code:

1. **Cartesian tracking motion**
   - Timestamped pose and desired-twist references published at 50 Hz.
   - Measured flange pose from the hardware model and measured twist `J*dq`.
   - Full six-dimensional SO(3)-log impedance at 1 kHz.
   - Reset-anchored K=2, D=1 posture torque through
     `I - J.T @ pinv(J M^-1 J.T) @ J M^-1`.
   - Startup reference initialized from the same measured state used by preflight.
   - Applied-reference sequence and watchdog visible to the logger.

2. **Static-pose/posture motion**
   - Hold the measured home flange pose and full xyzw orientation.
   - Generate quintic posture references natively from explicit targets.
   - Preserve the M3 kinematic torque projector
     `I - J.T @ pinv(J.T)` for this demo only.
   - Separate weak and reduced-offset commissioning targets; do not begin with
     the transferred 0.60 rad candidate.

3. **Curve-fixture motion**
   - At every 1 kHz tick project measured flange position onto either the
     infinite base-Z line or nearest point on the base-YZ circle.
   - Reject the circle-center singularity and any non-finite/ill-conditioned
     projection.
   - Apply normal stiffness and normal/tangent damping, no tangent spring, no
     time-driven translation, and no synthetic force.
   - Hold full orientation `[1, 0, 0, 0]` xyzw and use the dynamically
     consistent six-row posture projector.

For every route, construct the non-gravity command as

```text
tau_raw = tau_cartesian_or_fixture + tau_posture + coriolis + tau_limit
tau_previous = robot_state.tau_J_d
tau_slew = tau_previous + clip(tau_raw - tau_previous, +/-1 Nm)
tau_cmd = low_pass(tau_slew, tau_previous, 100 Hz)
```

This is the existing Franky plus libfranka cascade, not two independent
controller states: Franky performs the first line against the last applied
desired torque, and libfranka performs the second against that same value.
Simulation must use this recurrence for parity.

Do not add gravity: the FCI torque interface expects commands without gravity
and the robot supplies it. Keep additional friction feedforward disabled unless
it is separately identified and reviewed.

## Required guards and telemetry

The native controller must latch faults and transition through the reviewed
`TorqueStopMotion`; Python exceptions or process exit are not the primary stop.
It must cover:

- stale/missing 50 Hz reference and invalid applied sequence;
- callback gap, non-finite state/model/reference/torque and logging overflow;
- measured joint position and velocity, joint reference, effort and torque slew;
- measured flange workspace, pose/orientation error and Cartesian speed;
- external joint torque/contact threshold;
- Jacobian conditioning and dynamic-projector solve failure;
- circle-center projection and line/circle initialization mismatch;
- robot errors, FCI command-success degradation and stop timeout.

Finalize numerical thresholds in a local reviewed commissioning configuration.
The transfer's `measured_abort` and Cartesian bounds are simulation candidates;
they cannot silently become hardware limits. Reuse the previously validated
FR3 soft joint interval and collision behavior as inputs to that review.

Use a native lock-free/bounded telemetry path. Record robot and monotonic time,
callback/reference/applied sequence, phase, q/dq, measured encoder-FK pose and
twist, desired pose/twist/posture, J/M/C metadata, raw/slew/filtered torque,
measured/desired/external torque, gains, payload/TCP, all guard values, faults,
and stop events. Echoed references must never be labeled measured feedback.

## Verification and commissioning sequence

### Gate 0 — receipt (complete)

- Immutable manifest verified.
- Five offline tasks regenerated with exact golden parity at 1e-9 tolerance.
- No FCI connection or hardware execution occurred.

### Gate 1 — source and native unit audit

- Recover/pin the exact Franky source and document every reused controller path.
- Add native tests for SO(3) sign/frame, target twist, both projectors, line and
  circle projection, center rejection, 1 Nm/ms slew, 100 Hz filter, watchdog,
  guards, buffered logging and torque stop.
- Run the five portable fixtures through the compiled native implementation and
  compare per-tick references, projection and torque components.
- Build a new wheel and record `ldd`, package metadata and SHA-256.

### Gate 2 — offline runtime and read-only FCI audit

- Install the M4 wheel and `franka-motion` wheel in the isolated environment.
- Run fake lifecycle and native fixture parity from that environment.
- With FCI open, perform read-only state/model inspection: robot identity,
  flange/TCP, payload, FK, J, M, Coriolis, start velocity, errors and workspace.
- Confirm the measured start is used to construct entry trajectories. Do not
  replay fixture-home commands.

### Gate 3 — torque-controller commissioning

Run one reviewed configuration per step and inspect the full log before the next:

1. Measured-pose hold, no posture torque, reduced Cartesian gains.
2. Five-millimeter Cartesian translation and return with desired twist.
3. Reset-anchored weak posture hold, then small posture offsets. Increase the
   offset gradually; the 0.60 rad A/B sequence is the final posture candidate.
4. Brief Z-line hand guidance at reduced normal stiffness, then nominal
   100 N/m if the first run is clean.
5. Brief small-radius circle guidance; explicitly test clean rejection behavior
   away from the valid curve initialization.

Every step requires clean FCI status, zero dropped native samples, bounded timing,
no unexpected guard, smooth stop, and measured feedback consistent with the
logged command convention.

### Gate 4 — tracking commissioning

Use new local configurations rather than altering the transfer:

1. 20 mm YZ circle at reduced angular speed.
2. 50 mm `circle_yz_small` at reduced then nominal speed.
3. Full 150 mm `circle_yz_m2` at 0.2 rad/s, after confirming the previously
   cleared workspace is still unobstructed.
4. Reduced-amplitude Lissajous, then the transferred 80 mm amplitudes at
   0.2 rad/s. Its current larger YAML has no inherited hardware approval.

Entry always starts from measured FK, holds full orientation, and returns by the
cataloged exit trajectory. Review peak pose error, orientation error, velocity,
torque, external torque, singular value, callback timing and stop transients.

### Gate 5 — final five-demo suite

Create one new immutable, checksummed hardware bundle containing only the exact
native wheel, Python wheel, reviewed configs, robot/model snapshot and accepted
commissioning evidence. Set hardware approval only in that derived bundle. Run
demos individually in this order: circle tracking, Lissajous tracking, static
posture, Z impedance, circle impedance. Do not use a single unattended command
to chain all five.

## Immediate next implementation step

Continue Gate 1 from the completed source audit. Extend the pinned Franky C++
base with the M4 dynamic projector and measured-state curve fixture. Preserve
the existing hardware torque processing (`max_delta_tau=1`, libfranka
`cutoff_frequency=100`, `limit_rate=false`) and change the simulation processor
to reproduce that cascade before re-running the M4 evaluations. Add the
watchdog/guards, applied-reference telemetry and native buffered logger, and run
the five portable fixtures through the compiled path. The current transfer
bundle itself needs no edits. The source implementation is staged on fork
`YFnWng/franky`, branch `m4-dynamic-projector-fixtures`; its detailed file/API/
test plan is `docs/m4-dynamic-projector-fixtures.md` in that checkout.

## Gate 1 native-library result — 2026-10-11

The dynamic-projector and curve-fixture portion of Gate 1 is complete at Franky
commit `057a0646` on `m4-dynamic-projector-fixtures`. C++ unit tests pass against
the pinned libfranka 0.21.2/Ruckig 0.17.3 stack. Python 3.12 binding construction
and the transferred Z-line/YZ-circle numerical fixtures pass from the installed
repaired wheel. Wheel SHA-256:

```text
a06c85249f705aa1ff9a8540dfe1b611098b2c77c85834886a3622d485a54e79
```

The runtime guard/logger work and compiled-path tracking/posture fixture replay
remain open. Gate 2 and all physical commissioning remain blocked until those
items and a local reviewed configuration are complete.


## Hardware runtime and dependency preservation — 2026-10-12

The receiver added native `M4RuntimeMotion` watchdog/guard/stop dispatch and a
bounded 1 kHz telemetry ring, plus a hardware adapter and single-task runner
`deployment/franky_runtime/run_m4_hardware.py`. It supports pose/twist tracking,
measured-pose hold, native quintic posture references, and line/circle fixtures.
The first local candidate is `config/m4.hold.pending.yaml`: 10-second hold with
zero posture gains. It is pending local review, not automatically authorized.

The new wheel `2.0.1.dev63+gm4baseline.libfranka.0.21.2` bundles byte-identical
copies of all 21 native dependencies from the tested RL velocity runtime,
including libfranka and Ruckig. The earlier dev62 dependency rebuild is
superseded for commissioning. Installed environment:
`~/yifan/venvs/franky-m4-baseline-server10`. Artifact and dependency/source hash
receipt: `~/yifan/deployment_bundles/2026-10-12/franky-m4-baseline/`.

Native tests (13 cases/41 assertions), installed binding/adapter tests (9),
portable five-fixture replay and native curve helper parity passed offline.
There has been no FCI connection or hardware motion. Full compiled tracking/
posture torque replay and physical timing/stop commissioning remain unvalidated;
the simulation torque-processing correction requested above also remains open.
Detailed commands, telemetry semantics and guard provenance are in
`deployment/franky_runtime/M4_HARDWARE_PREP.md`.
