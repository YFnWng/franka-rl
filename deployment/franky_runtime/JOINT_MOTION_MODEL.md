# Source model for Franky `JointMotion`

This audit targets the installed deployment build:

- Franky commit `f88f0e9b8f598d7c537b4e727e4411e67c95e017`
- Franky package `2.0.1.dev58+gf88f0e9b.libfranka.0.21.2`
- libfranka `0.21.2`, commit `9f9304ec0ac897eff3219a67f612b959948535e2`
- Ruckig `0.17.3`
- robot FCI server version 10

## Source-visible command path

`JointMotion` is a one-waypoint `JointWaypointMotion`. At the start of a control
loop it initializes Ruckig from the robot's desired state. On preemption it uses
the last Franky joint-position command for position continuity and the robot's
reported `dq_d` and `ddq_d` for desired velocity and acceleration. A normal
position target has target velocity zero and target acceleration zero.

Ruckig runs at 1 kHz with seven degrees of freedom and time synchronization. At
each tick its new position, velocity, and acceleration are passed back as the
next initial state. If a tick is longer than 1 ms, Franky extrapolates with a
constant-acceleration model before the next update.

The effective Ruckig limits are the componentwise product of the robot, motion,
and waypoint relative-dynamics factors and the configured absolute limits. With
only the robot factor set to 0.05, the limits on this FR3 are:

- velocity [rad/s]: `[0.10875, 0.10875, 0.10875, 0.10875, 0.1305, 0.1305, 0.1305]`
- acceleration [rad/s^2]: `[0.75, 0.375, 0.5, 0.625, 0.75, 1.0, 1.0]`
- jerk [rad/s^3]: `[375, 187.5, 250, 312.5, 375, 500, 500]`

Franky's default `Robot.move` options then pass the Ruckig joint position to
libfranka with rate limiting disabled and the first-order low-pass filter enabled
at 100 Hz. At the fixed 1 ms step its scalar update is:

```text
alpha = 0.001 / (0.001 + 1 / (2*pi*100)) = 0.3858695
q_filtered[k] = alpha*q_ruckig[k] + (1-alpha)*q_d[k-1]
```

The filtered joint-position command is sent to the robot's firmware joint
impedance controller (`ControllerMode::kJointImpedance`). Franky does not expose
or implement that firmware controller's default gains.

Repeated asynchronous `JointMotion` calls do not create independent FCI control
loops. A new motion of the same signal type replaces the pending motion in the
existing 1 kHz loop at the next synchronized update. The new Ruckig problem still
plans toward zero terminal velocity and acceleration. Therefore a 30 Hz policy
that repeatedly sends `JointMotion` targets continually replans toward rest.
Upstream issue 76 reports jerky behavior and discontinuity errors for this exact
30 Hz usage pattern.

Constructing `franky.Robot` also sends collision thresholds of 20 Nm for each
joint and 30 N for each Cartesian axis. This is a configuration side effect and
must be part of the deployment contract.

## What can be reproduced in Isaac Lab

The following can be implemented deterministically from source:

1. 30 Hz target arrival and same-type latest-target replacement.
2. One 1 kHz Ruckig 0.17.3 trajectory generator per environment.
3. Initialization from the prior command plus desired velocity/acceleration.
4. Zero terminal velocity and acceleration for each `JointMotion` target.
5. The exact per-joint velocity, acceleration, and jerk caps.
6. The libfranka 100 Hz first-order command filter.
7. Optional tick-delay and target-arrival jitter randomization.

These reproduce the command generator from policy target to firmware desired
joint position. They do not reproduce the physical robot's closed-loop response.

## What source cannot provide

The firmware joint-impedance gains, gain scheduling, motor-current loop, friction
compensation, safety shaping, and effective command/measurement delay are not in
Franky or libfranka source. Mechanical friction, cable effects, payload/model
error, and joint coupling are also robot-specific. Isaac Lab's articulation drive
will not automatically match those behaviors even if it receives the exact same
Ruckig reference.

## Recommendation

Use a hybrid model:

1. Implement the source-exact Ruckig and libfranka filter as the shared policy
   action layer.
2. Run a small free-space identification experiment to fit only the remaining
   mapping from firmware desired state (`q_d`, `dq_d`, `ddq_d`) to measured
   (`q`, `dq`).
3. Fit delay and a per-joint second-order response first. Check residual joint
   coupling before adding a coupled model.
4. Domain-randomize the fitted delay, natural frequency/damping or equivalent PD
   response, friction, and payload around measured confidence ranges.

The experiment should log the raw 30 Hz target, Franky's 1 kHz callback command,
`q_d`, `dq_d`, `ddq_d`, measured `q`, measured `dq`, robot timestamp, host monotonic
time, errors, and command-success rate. Small single-joint and low-amplitude
multisine tests around the reviewed home pose are sufficient to distinguish the
source-known reference generator from the unknown firmware/plant response.

A source-only model is appropriate for debugging the action pipeline. The
identification experiment remains necessary for deployment-faithful training,
but it is now much smaller: it estimates the inner closed-loop response rather
than all of Franky's behavior.

## Primary sources

- Franky `JointMotion`: https://github.com/TimSchneider42/franky/blob/f88f0e9b8f598d7c537b4e727e4411e67c95e017/src/motion/joint_motion.cpp
- Franky joint waypoint generator: https://github.com/TimSchneider42/franky/blob/f88f0e9b8f598d7c537b4e727e4411e67c95e017/src/motion/joint_waypoint_motion.cpp
- Franky generic waypoint loop: https://github.com/TimSchneider42/franky/blob/f88f0e9b8f598d7c537b4e727e4411e67c95e017/include/franky/motion/waypoint_motion.hpp
- Franky dynamics scaling: https://github.com/TimSchneider42/franky/blob/f88f0e9b8f598d7c537b4e727e4411e67c95e017/include/franky/motion/position_waypoint_motion.hpp
- Franky preemption: https://github.com/TimSchneider42/franky/blob/f88f0e9b8f598d7c537b4e727e4411e67c95e017/src/motion/motion_generator.cpp
- Franky robot construction/control: https://github.com/TimSchneider42/franky/blob/f88f0e9b8f598d7c537b4e727e4411e67c95e017/src/robot.cpp
- libfranka filter path: https://github.com/frankarobotics/libfranka/blob/0.21.2/src/control_loop.cpp
- libfranka filter equation: https://github.com/frankarobotics/libfranka/blob/0.21.2/src/lowpass_filter.cpp
- 30 Hz `JointMotion` report: https://github.com/TimSchneider42/franky/issues/76
