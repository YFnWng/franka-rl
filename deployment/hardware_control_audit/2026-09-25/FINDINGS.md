# Hardware control audit — 2026-09-25

**Later live update:** [LIVE_INSPECTION.md](LIVE_INSPECTION.md) records successful
read-only FCI access, robot-returned fr3v2.1 mass/COM differences and the decision
to preserve default internal settings. It supersedes the earlier statements
below that no live model had been captured. Offline-test scope remains unchanged.

**Deployment-oriented training remains blocked on the controller contract.**
The installed stack supports joint-position commands into the robot's internal
joint-impedance controller. There is no deployed RL controller, 30→1000 Hz
reference governor, policy watchdog or identified hardware gain set. PD 80/4,
armature 0.001 and 60 Hz physics are simulation assumptions, not verified
hardware behavior. No robot connection, motion, configuration change, load
change, fault recovery or ROS launch was performed for this audit.

This answers the six requests in [HANDOFF_AGENT.md](../../HANDOFF_AGENT.md).
`verified` means inspected source/package metadata or identified existing
evidence; it does not mean live control-tested. `inferred/recommended` denotes
an engineering proposal. `unknown` is an explicit blocker, never a default.

Read [control_contract.json](control_contract.json), [limits.json](limits.json),
[evidence.json](evidence.json), and [offline reproduction](README.md).
The contract is an audit schema, not a launch file or approved policy bundle.
Null fields deliberately prevent interpreting it as deployable configuration.

## 1. Platform and command-interface decision

Verified from the existing operator inventory: FR3 Arm3Rv2, hardware rev 02.01,
Control/System image 5.9.2. The 2026-09-24 read-only snapshot reported FCI server
version 10. Robot identity/attachment was not rechecked today. Reuse the ROS
repository's `docs/HARDWARE_INVENTORY.md` and the previously returned
[robot state](../../model_audit/2026-09-24/robot_state.json).

Verified locally:

| Component | Installed / inspected version |
| --- | --- |
| Kernel | 5.15.133-rt69 |
| ROS | Humble |
| franka_ros2 | v2.2.0, 7ef0ab00964a0d2b5e19e9292bbcef9f32dda811 |
| franka_description | 1.3.0, 1ccde30d5a30a710f335c9f6545528447a04bc7c |
| libfranka used by ROS hardware plugin | local 0.19.0, 4df37dd2b527f41ef34b76d6d4d3805864b09993 |
| controller_manager / hardware_interface | Debian packages 2.54.2 |
| realtime_tools | 2.15.1 |
| ros2_control metapackage | 2.35.0; this old metapackage does not identify the upgraded component binaries |
| franky-control | 1.1.3, Python 3.10 user site; bundled libfranka .so.0.18.0 |

Full package versions, revisions, dirty status, source hashes and `ldd` output
are in evidence.json. A source revision for the franky wheel and ROS Debian
packages is unknown. franky's bundled .18.0 and hashed Poco libraries are
separate from the ROS plugin's local .19.0/Poco .80 chain. Do not assume that
installing libfranka .19 changes franky's bundled library. No franky connection
was tested. It is not selected for this deployment.

Recommended path: custom ROS 2 C++ ros2_control policy controller claiming
`fr3v2_joint1..7/position`, with a non-RT local inference worker and a 1 kHz
reference governor. The existing executable is
`controller_manager/ros2_control_node`, loading
`franka_hardware/FrankaHardwareInterface`. Position-interface startup uses
`startJointPositionControl(kJointImpedance)`; writes use
`ActiveControlBase::writeOnce(JointPositions)`.

The custom policy plugin and its exact deployable config/launch do not exist.
Stock `franka_bringup/launch/franka.launch.py` and `config/controllers.yaml`
provide infrastructure (1000 Hz, priority 98). `franka.config.yaml` is an
example with robot_type=fr3 and IP 172.16.0.3; it is not this robot's deployment
configuration (FR3v2, 172.16.0.6, no gripper). No example should be launched as
an RL policy. The joint-impedance *example* claims effort interfaces and uses
its own moving reference and PD gains; it is a different control path.

Position control retains firmware impedance behavior but needs identification.
A future custom torque impedance controller would expose more of the control
law, but would add torque-law, model and commissioning obligations. It is an
alternative requiring an explicit design decision, not a way to assume that
simulation PD already matches hardware.

Provenance: evidence.json entries ros_robot_cpp/hpp, ros_hardware,
ros_controllers, ros_launch, ros_launch_config, ros_impedance_example.

## 2. Exact existing command path versus missing policy path

Verified installed behavior:

1. On the first read after position-interface activation, the hardware adapter
   initializes its position command array from **measured q**.
2. `write()` checks command arrays for NaN/Inf and returns ERROR if found.
3. `writeOnceJointPositions()` creates JointPositions from that array.
4. Its low-pass switch is false; its position rate-limit switch is false.
   Thus finite commands pass through unchanged at this host layer.
5. ActiveControl forwards to Robot::Impl, which checks finite positions and
   assigns them to the motion-generator packet's `q_c`.

There is no host-level position clamp or 30 Hz interpolation in this path.
The optional filter cutoff member is 100 Hz but **the filter is off**. These
are private source members, not exposed launch settings. If deliberately
changed, processing order is filter then limiter. Filter history is robot
`q_d`; limiter history is robot `q_d,dq_d,ddq_d`, not measured q/dq and not an
independently integrated host trajectory. `Robot::control(callback)` is a
different API whose default 100 Hz filter must not be imported into this
ActiveControl path. Firmware-internal interpolation/filtering is not established
by the host's identity mapping.

If enabled, the stock limiter uses deprecated hard-coded velocity helpers.
libfranka 0.19 recommends robot-model-derived limits for system images >=5.9.0.
Its Robot constructor fetches a robot URDF and initializes those limits; this
URDF was not archived by the earlier state-only collector. Description 1.3.0
parameters are available offline, but equality with firmware-returned limits
must be verified before deployment. The fixture shows both formulas explicitly.

The current 30 Hz policy path is **absent**, so there are no actual policy
scheduler, hold/interpolation, staleness thresholds or inference jitter values
to report. Proposed ordering is recorded in control_contract.json: coherent
snapshot → local inference → validate shape/finite/freshness → default-offset
mapping → position-margin/reference governor at 1 kHz → existing adapter.
Retain raw previous action. Use timestamps/sequence numbers; 1000/30 is not an
integer, so fixed 33-cycle scheduling would change the policy frequency.
Initialize a future governor from desired q_d/dq_d/ddq_d after readiness checks,
not by jumping to training q_default. Reset raw previous action to zero. This
is a proposal, distinct from the adapter's measured-q initialization above.

[verify_vectors.py](verify_vectors.py) supplies reusable position-dependent
velocity and libfranka scalar limiter equations. [offline_reference.cpp](offline_reference.cpp)
provides the native oracle. They cover the default identity path, optional
100 Hz filtering, optional legacy filter+limiter, and description-based limiter.
**They are not a complete governor:** no approved position margins, feasibility
handling, workspace constraints, timestamps, watchdog, stopping trajectory or
lifecycle behavior. Numerical fixtures are not motion commands. A step held at
30 Hz must not be sent directly to FCI; implement and validate the missing
continuous reference path first. Do not install the offline fixtures in control.

Provenance: ros_robot_cpp/hpp, ros_hardware, lib_active_motion, lib_robot_impl,
lib_filter_cpp, lib_limits_cpp/h, lib_velocity_cpp, lib_robot_h.

## 3. Low-level dynamics and configuration ownership

Verified API: `Robot::setJointImpedance` / ROS `SetJointStiffness` sets seven
internal stiffness values in Nm/rad; the 0.19 header documents input range
[0,14250]. That API range is not a recommended operating gain range. No
independent damping setter or stiffness getter was found in this interface;
robot state does not report the current gain vector. Existing evidence does
not identify which gains the robot currently uses.

Official ROS documentation describes damping as derived internally from
stiffness. Exact values, gain scheduling, damping law, internal torque
saturation and servo bandwidth remain unknown for this firmware. Do not set
hardware gains to 80/4 by analogy to the simulator. Example YAML K=[24,24,24,
24,10,6,2], D=[2,2,2,1,1,1,0.5] belongs only to the external effort example;
it neither reveals nor configures the internal position-controller gains.
[Official service description](https://support.franka.de/docs/franka_ros2.html).

The versioned API documents torque commands without gravity/friction; these
components are compensated by the robot. Preserve compensation. The host
position path sends positions, not a host-computed gravity/Coriolis torque.
The exact internal joint-impedance equation and Coriolis treatment are unknown;
a public `model.coriolis()` function does not prove a particular firmware law.
[libfranka 0.19 API](https://frankarobotics.github.io/libfranka/0.19.0/classfranka_1_1Robot.html).

In the external effort path, the ROS wrapper enables torque-rate limiting by
default using tau_J_d and libfranka kMaxTorqueRate; no host absolute torque
saturation was found in that method. This limiter is not the position path's
internal torque controller. Do not infer its saturation behavior from the ROS
wrapper. Keep configured EE/load parameters correct because robot model-based
behavior depends on attachment parameters; their exact proprietary use is not
fully observable. setLoad is a non-RT configuration operation, not a DR knob
for hardware. Nothing was changed.

Required simulation change: treat gravity as physical and model retained
controller compensation, rather than setting gravity to zero or disabling
hardware compensation. PD+gravity feedforward can be an explicitly labeled
surrogate, not an identified internal controller. Current armature and friction
assumptions are likewise not measurements.

## 4. Observations, frames and timing

Verified: hardware read copies RobotState.q → position and dq → velocity,
tau_J → effort. q/dq are documented measured quantities; q_d/dq_d are desired
signals and must not replace policy inputs. The adapter adds no q/dq filter.
Firmware estimator filtering and signal ages are unknown. Use one RobotState
packet for q,dq,transforms and its monotonic robot timestamp; do not assemble
policy inputs from separately published joint-state/TF topics.

The versioned audit contract binds policy aliases panda_joint1..7 to physical
fr3v2_joint1..7, same signs/order. This preserves the new simulation's intentional
aliases without misidentifying the robot. Its 24D float32 math is q-q_default,
dq, target_base-flange_base, previous raw actor output, no normalization; map
q_target=q_default+0.5*raw_action with no raw clip. These are proposed FR3 bundle
binding requirements; the new FR3 checkpoints have not been exported or checked
on this host. Previous Panda ONNX parity does not cover those checkpoints.

Frames: O=fr3v2_link0, F=fr3v2_link8=simulation fr3_flange. Compute
`O_T_F = O_T_EE * inverse(F_T_EE)` using column-major RobotState matrices.
Snapshot F_T_EE is identity, so flange position is O_T_EE[12:15]. Always check
current transforms rather than hard-coding this assumption. Robot-reported FK
and description FK agreed at the old snapshot; it is not external calibration.
If using URDF FK later, version and validate it separately. Panda hand's fixed
45° orientation offset is not an extra FR3 flange position offset.

The fixture contains 24 input sentinels, a zero case and the saved bare-flange
snapshot; C++/Python observation and action math match exactly in float32.
Limiter/filter math matches within 1e-12. This tests math only, not policy
outputs, real timing or an end-to-end ROS controller.

Unknown: sensor/filter delay, packet age, inference latency/jitter, command
acceptance delay, scheduling distribution and motor response. Log robot time
and local monotonic receive/inference-start/inference-end/RT-consumption times,
plus sequence numbers. Establish a clock relation before subtracting robot and
host times; ROS wall time alone is unsuitable for freshness. Prior ping and
fake-hardware FIFO results are not FCI timing measurements.

## 5. Limits and faults

[limits.json](limits.json) records description position/velocity/effort fields
and libfranka constants, with units. Library joint acceleration is 9.999 rad/s²,
jerk 4999.999 rad/s³ and torque slew 999.999 Nm/s (nominal 10, 5000, 1000 less
0.001 numerical margin), all seven joints. Torque change per 1 ms is 0.999999
Nm. Description effort values are 87 Nm for joints 1–4 and 12 Nm for 5–7.
These are manufacturer/model/interface data, not lab-approved limits or
measured servo saturation. Current official specifications corroborate the FR3
position ranges; local versioned sources determine this audit's precise numbers.
[Official specification](https://frankarobotics.github.io/docs/robot_specifications.html).

For each joint, using description parameters vmax, offset, deceleration d:

```text
v_upper(q) = min(vmax, max(0, -offset + sqrt(max(0, 2*d*(qmax-q))))) - 0.001
v_lower(q) = max(-vmax, min(0,  offset - sqrt(max(0, 2*d*(q-qmin))))) + 0.001
```

This is the inspected libfranka model-based formula with local description
inputs. It can yield a negative upper bound/positive lower bound at extreme
positions due to the epsilon; do not force it to a symmetric velocity cap or
claim it resolves infeasible states. Original legacy helper outputs are also
in vectors.json. Fetch/archive the actual robot model before selecting final
limits. A rate limiter does not independently enforce all position/collision
constraints.

Verified library error surfaces include joint limits, self-collision avoidance,
joint/cartesian reflex, motion discontinuities, torque discontinuity and
communication-constraints violations. Robot collision thresholds are settable
but current values are unknown. A single snapshot of zero collision flags does
not establish protection coverage. Self-collision protection does not define
lab obstacle/workspace clearance. Exact disconnect/protective-stop trajectories
and timing have not been verified. Network/control exceptions are possible;
no policy-age watchdog exists in the inspected adapter.

**Conflict:** stock initialization catches ControlException, invokes
`automaticErrorRecovery()`, then retries starting control. This contradicts the
suite's terminal-fault/no-automatic-recovery requirement. Correct and test this
before commissioning; do not rely on the unmodified wrapper. Service/action
recovery paths also require run-time access/lifecycle review. The wrapper uses
mutexes in read/write paths; RT contention and service activity need review,
not merely a configured 1000 Hz rate. No source was patched by this audit.

Lab-approved position margins, speeds, acceleration/jerk limits, workspace
exclusions, collision settings, timeout budgets and stop acceptance thresholds
remain unknown. Simulation success/unsafe conditions are separate. In particular,
<3 cm for five policy samples without a velocity/hold test is not a hardware
settling or safe-stop criterion.

## 6. Installation gaps and proposed identification

Verified prior snapshot: bare-flange configuration with zero m_ee, m_load,
m_total, COMs and inertias; identity tool/stiffness transforms. Attachment is
operator-reported and must be reconfirmed for a later session. Physical mounting
orientation, base tilt, actual world-to-base transform, joint calibration offsets
and friction are unknown. O_ddP_O≈[0,0,-9.81] is not independent mounting evidence:
the installed RobotState header explicitly describes that field as hard-coded.

No measured DR ranges for friction, gains, delay, armature or payload uncertainty
can be justified yet. The new simulation's 0–1 kg / axial 0–5 cm payload range is
a scenario choice, not uncertainty in today's bare flange. Future real payload
geometry, mass, COM and rotational inertia must be measured separately.

Proposed protocol, requiring separate lab approval before any control/motion:

1. Operator confirms mounting, attachment, active safety setup, independent stop,
   permitted workspace and exact controller/config. Archive a fresh state and
   robot-returned URDF using read-only methods; document units and transforms.
2. Agree numeric commissioning limits, stop criteria, timeout budget and signal
   logging rates before enabling control. Select controller mode and stiffness
   ownership; any deliberate setting change is reviewed and logged separately.
3. Validate watchdog/fault logic and smooth stopping offline/fake first. Then,
   only with separate authorization, qualify the live control loop and an
   operator-approved low-amplitude, low-speed identification trajectory in a
   clear region. No trajectory or amplitude is authorized by this document.
4. Log coherent q,dq,q_d,dq_d,ddq_d,tau_J,tau_J_d,external torque estimates,
   poses/tool/load transforms, model gravity/Coriolis (if used), raw actions,
   policy target, governed target, robot/host timestamps, sequence/drop counts,
   inference duration, limiter interventions, robot mode/errors and success rate.
   Use a preallocated RT buffer and non-RT writer; large traces stay outside Git.
5. Fit delay/response/friction surrogates only within the excited range and
   compare on held-out motions. Require finite consistent data, continuous
   commands, no unexpected contacts/faults, no limit violations, and timing,
   tracking, hold/settling and stop metrics within the **lab's preapproved**
   numeric thresholds. Stop on any failure; no automatic recovery/resume.

## Required next decisions and return to simulation

- Choose/implement the custom position controller, common governor and terminal
  fault lifecycle. Set lab margins and watchdog budgets from approved limits and
  timing evidence; do not invent values for training convenience.
- Identify or deliberately configure/document the internal stiffness and measure
  the retained compensated response. Internal damping/servo details may remain
  proprietary; validate a surrogate and uncertainty ranges experimentally.
- Use 1 ms command-path substeps for evaluating the governor. Compare physics
  timestep convergence and latency scheduling; 60 Hz implicit PD alone does not
  resolve a 1 kHz loop. Retain 30 Hz policy math and raw previous action.
- Add position-dependent velocity constraints, realistic actuation/compensation,
  and separate measured dynamic uncertainty from hypothetical payload scenarios.
- Export new versioned FR3 bundles with explicit FR3 frame/model metadata;
  validate Python/C++ actor parity before any deployment. Existing Panda bundles
  remain immutable. Validate governor trajectories and fault cases separately
  from the helper parity supplied here.
- Hardware acceptance still requires lab procedure/stop validation, current
  installation checks, resolved control contract, FCI timing qualification and
  separately authorized commissioning. No training-workstation connection is
  permitted during runtime; local YAML targets and schedules remain the plan.
