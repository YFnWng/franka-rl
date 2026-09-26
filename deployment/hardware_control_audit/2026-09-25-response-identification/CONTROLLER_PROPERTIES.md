# Joint-position controller properties

Status: offline source audit complete; no motion was commanded. Hardware response identification is pending separate lab approval.

## Selected deployment path

The selected first-demo path is:

`PPO raw action (30 Hz) -> shared bounded_quintic_v1 governor (1 kHz) -> ros2_control position command -> franka_hardware -> libfranka active joint-position control -> robot joint-impedance controller`.

The ROS adapter calls `Robot::startJointPositionControl(ControllerMode::kJointImpedance)`. This establishes the mode, but it does not reveal the active firmware gains. The public APIs can set joint impedance, but expose no getter for the active stiffness/damping values. This deployment therefore preserves the current robot settings and identifies their observable closed-loop response.

## Verified properties

- Robot/controller: FR3 Arm3Rv2 revision 02.01, control/system image 5.9.2, FCI server version 10, bare flange and zero configured external load from the read-only inventory.
- Host stack: franka_ros2 v2.2.0 at `7ef0ab00964a0d2b5e19e9292bbcef9f32dda811`; libfranka 0.19.0 at `4df37dd2b527f41ef34b76d6d4d3805864b09993`.
- Control mode: joint-position motion generation with `kJointImpedance`.
- Host write path: `franka_hardware::Robot::writeOnceJointPositions` constructs `franka::JointPositions` and calls `ActiveControl::writeOnce`.
- ROS wrapper low-pass filter: disabled by default. It must stay disabled while the shared governor owns shaping.
- ROS wrapper position rate limiter: disabled by default. It must stay disabled while the shared governor owns shaping.
- libfranka callback-control defaults do not prove the options used by the active external-control-loop wrapper; the observed ROS wrapper settings above are the relevant host settings.
- Available state includes measured `q`, `dq`, `tau_J`; desired `q_d`, `dq_d`, `ddq_d`, `tau_J_d`; filtered external torque; robot time; mode; and command success rate.
- `tau_J_d` is documented as desired link-side joint torque without gravity. Model gravity and Coriolis values are separately computable estimates; their availability does not prove the firmware's internal compensation equation.
- The selected joint-position startup path now propagates a libfranka `ControlException` without calling `automaticErrorRecovery()` or retrying. Operator recovery remains explicit.

## Unknown or not identifiable from source/API

- Active numeric joint stiffness and damping.
- The exact firmware servo equation, filters, saturation order, friction compensation, gravity compensation, and gain scheduling.
- Whether active settings equal any historical factory default.
- Tracking delay, bandwidth, damping, settling, cross-joint coupling, and configuration dependence on this arm.

URDF K/D values, example-controller gains, and a simulator actuator's PD gains are not evidence for the active firmware gains. Position traces alone generally cannot uniquely identify both physical inertia and controller gains. Results from the proposed tests must therefore be reported as an effective closed-loop response model, with assumptions and confidence intervals, rather than as recovered firmware settings.

## Interpretation of the simulator failure

The gravity-on stationary-reference failure is consistent with an uncompensated simulator surrogate: the reported joint-4 gravity term was about 18.95 Nm while the provisional 80/4 PD term was about 1.41 Nm. Gravity-off stationarity supports that diagnosis. It does not identify the hardware gains or prove the hardware compensation law. Training should replace the provisional plant with an identified response surrogate after approved measurements; broad arbitrary PD randomization is not a substitute.
