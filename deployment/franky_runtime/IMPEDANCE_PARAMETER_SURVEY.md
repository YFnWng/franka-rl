# Joint-impedance parameter survey

This survey records public parameter sets considered for the FR3 Franky tracking runtime. Values are joint stiffness in Nm/rad and damping in Nms/rad, ordered joints 1 through 7.

| Source | Stiffness | Damping | Context |
|---|---|---|---|
| Franka ROS 2 `JointImpedanceExampleController` | `[24, 24, 24, 24, 10, 6, 2]` | `[2, 2, 2, 1, 1, 1, 0.5]` | Official compliant joint-impedance example; selected initial profile. |
| Franka ROS 2 follower / IK example | `[600, 600, 600, 600, 250, 150, 50]` | `[30, 30, 30, 30, 10, 10, 5]` | Official stiff trajectory-following profile; not selected for the first torque-mode test. |
| Franky `JointImpedanceTracker` default | `[50, 50, 50, 50, 50, 50, 50]` | critical damping (`2*sqrt(K)` in the Python tracker) | Upstream Franky default; uniform distal stiffness is substantially higher than Franka's compliant example. |
| OpenPI FR3 hybrid follower | `[40, 30, 50, 25, 35, 25, 10]` | `[4, 6, 5, 5, 3, 2, 1]` | Used together with Cartesian stiffness/damping, so it is not a standalone joint-only comparison. |
| aiofranka example | `[80, 80, 80, 80, 80, 80, 80]` | `[4, 4, 4, 4, 4, 4, 4]` | General joint-impedance example rather than a validated profile for this arm and task. |

Selected first profile: Franka's official compliant example. The runtime also fixes Coriolis compensation on, torque slew to 1 Nm per millisecond, zero constant torque and friction feedforward, a 0.5 rad Franky position-error clip, and explicit soft-limit repulsion. Every value is stored in the reviewed YAML and resolved run config.


## Hardware comparison at the home pose

Three J1/J2 tests used the same +/-2 degree, 0.1 Hz reference. All completed
without robot errors or dropped samples. Values below are measured/reference
sinusoidal amplitude ratios; phase is closed-loop phase, not pure transport
delay.

| Profile | J1 amplitude ratio / phase | J2 amplitude ratio / phase | J1/J2 RMS error (rad) |
|---|---:|---:|---:|
| Franka compliant example | `0.299 / -57.8 deg` | `0.054 / -49.0 deg` | `0.0241 / 0.0396` |
| Franky default | `0.687 / -34.4 deg` | `0.510 / -41.4 deg` | `0.0149 / 0.0244` |
| J1/J2 K=100, D=20 | `0.885 / -19.7 deg` | `0.816 / -24.0 deg` | `0.00872 / 0.01306` |
| J1/J2 K=200, D=28.284 | `0.954 / -11.4 deg` | `0.908 / -13.2 deg` | `0.00509 / 0.00733` |

K=100/D=20 remains the nominal J1/J2 profile and K=50/100/200 define the
0.5x/1x/2x demo cases. K=200 passed as the stiff endpoint with amplitude ratios
0.954/0.908, RMS errors 0.00509/0.00733 rad, and no torque-slew activation.
This selection applies only to J1/J2; J3-J7 require sequential commissioning
before a seven-joint profile is frozen. Simulation must use the explicit torque
law rather than treating joint targets as achieved positions.

Sources:

- https://github.com/frankarobotics/franka_ros2/blob/humble/franka_bringup/config/controllers.yaml
- https://github.com/frankarobotics/franka_follower_controllers
- https://github.com/TimSchneider42/franky/blob/master/franky/tracker.py
- https://github.com/Physical-Intelligence/openpi-basic-control/blob/main/docs/fr3.md
- https://github.com/Improbable-AI/aiofranka
- https://github.com/TimSchneider42/franky/issues/76
- https://github.com/TimSchneider42/franky/pull/79

The full seven-joint K=200 endpoint was subsequently commissioned with
sequential J3-J7 motions. Amplitude ratios for J3-J7 were
`[0.887, 0.934, 0.907, 0.927, 0.831]`; phase was between -12.8 and -16.5 degrees.
The run completed without errors, drops, or torque-slew activation. Matching
K=100 and K=50 J3-J7 suites are pending to complete the hardware endpoint data.

