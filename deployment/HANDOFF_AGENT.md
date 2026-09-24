# Instructions for the receiving Franka deployment agent

Read `REAL_ROBOT_DEPLOYMENT_PLAN.md` and `README.md` before implementation.
Start with the inventory in section 11. Inspect installed ROS/libfranka sources
and compatibility before selecting APIs. Keep the ROS workspace separate from
the simulator environment. No Isaac installation is required for inference.

Run all hardware experiment components locally. Consume targets and experiment
schedules from YAML. Do not add workstation communication, interactive target
entry, live schedule overrides, network filesystems, or telemetry forwarding.

Default development to fake hardware or shadow mode. This handoff and its YAML
files do not authorize physical motion. Follow the lab's authorization and
physical-stop procedure before commissioning. Never recover a fault or resume
motion automatically. Keep faults terminal for the current suite.

Keep inference, allocation, locks, logging, disk access, and ROS publication out
of the 1 kHz controller callback. Validate observation order, joint order,
frames, units, timing, raw previous action, and default-offset action mapping
against the bundle. Do not substitute simulation termination limits for robot
limits or assume Panda/FR3 equivalence.

Preserve the received package as immutable. Develop in a separate local
workspace copied from `source/`; write generated logs outside the package.
Record revisions and suite/bundle hashes. Report C++ parity, fake-hardware,
shadow, and hardware results separately; numerical export parity alone is not
evidence that the policy can safely control this robot.
