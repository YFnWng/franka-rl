# Live read-only inspection — 2026-09-25 16:04:40 UTC

FCI inspection succeeded with local libfranka 0.19.0, server version 10.
The user selected the robot's default controller settings for first deployment.
Interpret this as **preserve existing internal joint-impedance settings**: do not
call stiffness/damping setters, reset parameters or invoke example
`setDefaultBehavior()`. That helper actively writes gains and collision settings;
its name does not mean it reads or preserves the robot's current defaults.
Current settings have not been proven equal to factory-reset values.

The probe called only Robot construction, getRobotModel, readOnce, serverVersion
and the model-based upper/lower velocity getters. No control activation, motion,
recovery or setting changes occurred. Its source is in the ROS repo at
`tools/read_only_inventory/inspect_controller.cpp`. Full 500-sample raw trace
stays outside Git under the directory recorded in [live_inspection.json](live_inspection.json).

## Gain/response result

The installed libfranka API exposes a stiffness setter, but no internal
stiffness/damping getter. RobotState does not contain the current gains. Thus
numeric internal gains remain **unknown** even with FCI open. There is no basis
to substitute the simulation's 80/4 or example-code gains.

The returned URDF includes dynamics attributes K=7000, D=1, damping=0.003,
friction=0.2, mu_coulomb=0, mu_viscous=16 on the seven arm joints. These are
model metadata, **not a verified readback of active servo gains or identified
friction**. Similar fields occur in the public description templates; there is
no inspected getter tying them to setJointImpedance's current value. Preserve
these attributes as evidence without using them to tune a deployment surrogate.
[Official description template](https://github.com/frankarobotics/franka_ros/blob/develop/franka_description/robots/common/franka_arm.xacro).

500 distinct state samples span 499 ms of robot time with exactly 1 ms robot
timestamp increments. Robot mode was Idle throughout, all current and last
motion errors empty. Maximum joint-position peak-to-peak variation was
1.2364e-5 rad. This is a short passive state baseline, not a measurement of
closed-loop tracking, stiffness, damping, friction, delay or settling time.
No commanded input was applied and no position controller was activated.
Host read-call timing can include buffering; robot timestamp increments do not
qualify the host's 1 kHz control loop.

Dynamic response cannot be identified from these idle data. The next useful
experiment is a separately reviewed, bounded joint-position reference test
using the existing internal settings, with no calls to setDefaultBehavior,
setJointImpedance, setCollisionBehavior, or automaticErrorRecovery. Before
execution it needs a concrete smooth trajectory, lab-approved bounds/stop
criteria and an implemented stop/watchdog path. The earlier audit lists the
signals and validation requirements. Neither FCI availability nor a choice of
default gains defines that trajectory. No response test was executed here.

## New live model evidence for simulation

[robot_returned_model.urdf](robot_returned_model.urdf) identifies itself as
`fr3v2.1` (Desk previously reported Arm3Rv2 rev 02.01). Preserve both identifiers;
this is not evidence that the physical robot was replaced.

Compared numerically against the exact description 1.3.0 expanded URDF used
for FR3 simulation:

- All seven position/velocity/effort limits and position-based velocity
  parameters match exactly. This closes the earlier live velocity-model gap.
- Joint origins 1–8 and revolute axes match exactly.
- Link inertia tensor attributes match numerically, with unchanged inertial
  orientation, but COM offsets differ for links 2–7.
- Masses differ on links 3–7 by approximately 3.3–4.9 grams per link.
  Example: link 7 is 0.698686 kg versus 0.694841 kg in description 1.3.0.
- Link 2 COM is [-0.010630,-0.184739,-0.010479] m, previously
  [0.004541,-0.180691,-0.010479] m. This approximately 15.7 mm shift is not
  accounted for merely by the small mass changes.

Full field-by-field differences are in live_inspection.json. Update the
simulation's versioned mass/COM source to this robot-returned model and rerun
mass/COM/inertia/FK/payload checks. Preserve the simulator's deliberate link-7
origin-to-flange translation consistently when importing COM coordinates.
Retain earlier evaluation results under their original model provenance.
The unchanged inertia-at-COM tensor does not imply unchanged joint-space
inertia or gravity torques when COM changes.

Configured m_ee/m_load remain zero; tool/EE transforms remain identity in the
saved [live_state_snapshot.json](live_state_snapshot.json). This is consistent
with the previously reported bare flange. Physical mounting/attachment still
requires operator confirmation; no camera/metrology inspection occurred.

[control_contract.json](control_contract.json) is now audit revision v2,
recording the user's settings decision and the live model evidence. It remains
incomplete for deployment: numeric gain readback is unavailable, response is
unidentified, and the governor/controller/watchdog/commissioning gates remain.
