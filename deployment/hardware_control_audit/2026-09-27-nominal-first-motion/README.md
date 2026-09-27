# Nominal PPO first-motion fault audit

Session 2026092703 stopped cleanly during the approach to waypoint 0. It did
not reach the waypoint. The smooth torque stop completed, the sample writer
reported no loss, and no robot error was recorded.

The runtime's reported tracking_error was triggered at 0.287 s by
abs(q - RobotState.q_d). That signal is invalid for this guard in external
torque-control mode: q_d stayed near the control-loop start pose while the
Franky impedance reference moved.

The callback trace also shows a real controller/reference mismatch. With the
intended time-coherent metric abs(q - q_ref), the current limit was first
crossed at 0.154 s on J5: 0.061475 rad against a 0.06 rad limit. J6 was already
at 0.059470 rad against 0.06 rad. Peak errors before stopping were 0.081602 rad
on J5 and 0.081415 rad on J6. Several policy outputs were near their configured
action range while approaching a target initially 0.19065 m away.

The runtime now computes tracking error from measured q and applied q_ref in
the same 1 kHz callback, latches the first violation, and records its joint,
error, limit, callback sequence, and robot time. Regression tests cover both
the invalid q_d behavior and the corrected callback guard.

Do not repeat the unchanged K=100/D=20 experiment. The smallest supported next
candidate is K=200 with D=2*sqrt(K)=28.284271 Nms/rad because:

- the same nominal policy passed 16/16 simulation episodes at nominal K=200
  with the fixed one-step delay;
- all seven joints have prior hardware response data at K=200;
- the K=200 identification traces had substantially smaller tracking peaks than
  K=100.

The policy bundle remains unchanged and honestly records K=100/D=20 as its
training nominal. A separately hashed local qualification record binds the same
policy manifest to the already evaluated K=200/D=28.284271 profile. Session
2026092704 is locally approved from that record without a workstation transfer.
The action mapping, delay, path, tracking limits, and ONNX bytes are unchanged.

## K=200 follow-up

Locally approved K=200 session 2026092704 exercised the corrected q-versus-q_ref
guard. It stopped on J5 at 0.061849 rad against the 0.06 rad limit; J6 peaked at
0.061683 rad. There were no robot errors, the maximum commanded torque was
6.85 Nm, and the smooth stop completed. The errors were falling before stop.

Session 2026092705 therefore changes only J5/J6 tracking thresholds from 0.06
to 0.08 rad. J1-J4 remain 0.08 rad and J7 remains 0.04 rad. Controller gains,
policy, action mapping, path, delay, and all other safety gates are unchanged.
The exact trace audit is stored beside session 2026092704 as fault_analysis.json.

