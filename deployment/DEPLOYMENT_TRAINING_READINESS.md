# Deployment-level PPO training readiness

## Decision

Hardware controller characterization is sufficient to implement the training
actuator family: uniform K=50/100/200 Nm/rad with D=2*sqrt(K), K=100 nominal,
30 Hz held position references, and a 1 kHz Franky torque controller. The
current Isaac task is not yet deployment-faithful, so nominal/DR production
training should not start from the existing task unchanged.

## Available inputs

- validated FR3v2.1 bare-flange mass, COM, inertia, kinematics and joint limits;
- home/default joint vector and default-plus-0.5-action mapping;
- 24D observation order and 7D action contract;
- 30 Hz policy timing and Franky 1 kHz callback timing;
- explicit impedance equation and K=50/100/200, D=2*sqrt(K) cases;
- Coriolis compensation, zero friction/feedforward, 1 Nm/ms torque slew,
  0.5 rad error clip, joint-limit repulsion, watchdog and stop behavior;
- seven-joint 0.1 Hz hardware response targets in
  `hardware_control_audit/2026-09-26-franky-impedance/comparison.json`;
- policy bundle/export/parity infrastructure and Franky PPO worker.

## Blocking implementation gaps

1. **Actuator model.** The current FR3 task still sends joint-position targets to
   an implicit PhysX drive inherited from the Panda config. Implement the
   Franky torque law explicitly, including zero desired velocity, torque-rate
   limiting, error clipping, joint-limit torque and the libfranka gravity/
   Coriolis convention. Run physics at 3 kHz if needed, but update and hold the
   controller torque at 1 kHz to match hardware.
2. **Gain DR.** Current scenario gain randomization scales implicit stiffness
   and damping independently. Implement one episode-level common multiplier
   `alpha`, log-uniform on `[0.5, 2]`, with `K=alpha*100` and
   `D=2*sqrt(K)` for all seven joints. The nominal policy fixes `alpha=1`.
3. **Command path decision.** The current Franky runtime directly holds each
   30 Hz policy target; it does not run the shared reference governor. The
   governed Isaac task therefore models a different command path and its
   simulation fixture contains the old default pose. Either train/deploy with
   direct held targets, or integrate the same governor into Franky first and
   then use identical q/dq references in both systems.
4. **Path-tracking task.** The current task and Franky coordinator implement
   fixed Cartesian targets/waypoints, not a continuous time-indexed flange
   path. Add a common path generator and feed its current base-frame target to
   both observation builders. Decide whether target velocity/phase belongs in
   the observation contract; version the contract if the 24D observation
   changes.
5. **Simulation validation.** Re-run stationary hold with gravity compensation,
   then replay the calibration references for all three gains. Compare
   amplitude, phase, DC error and RMS error against `comparison.json` before
   PPO training. The old 80/4 uncompensated governor smoke is invalid for this
   controller.
6. **Training/export.** Train matched nominal and gain-DR policies with the same
   seeds, base physics DR and budget; evaluate the 2x3 gain matrix; export new
   FR3 bundles and run PyTorch/ONNX/worker parity. Existing Panda-era bundles
   are not candidates for this demo.

## Follow-up before moving PPO hardware

Training can begin after gaps 1-5 pass on the Isaac workstation. Before moving
PPO hardware, run a held-out simulation/hardware reference with more dynamic
content than the 0.1 Hz calibration, complete workspace/path review, validate
all bundle hashes and observation vectors, and perform shadow inference. Since
nothing is attached, the demo tracks `fr3_flange`, not a tool-center point.
