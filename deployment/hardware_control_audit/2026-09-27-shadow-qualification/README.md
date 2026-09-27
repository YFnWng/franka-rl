# FR3 nominal and DR no-motion shadow qualification

Both transferred 29D/6D policies passed the real-FCI, read-only shadow runtime
at the reviewed home pose. The sessions completed the full 24-waypoint state
machine, dropped no samples, held joint 7 exactly, started with zero preceding
action, and stopped cleanly. The runtime source hash was identical in both
sessions.

The measured state-to-reference latency was 20.87 ms median for nominal and
20.16 ms median for DR. At the 50 Hz policy rate these are 1.043 and 1.008
policy steps. The observed combined range was 0.985--1.067 steps, so the
integer-valued simulator calibration contract is now a fixed one-policy-step
delay, `[1, 1]`. Training-time action-delay DR remains `[0, 1]`.

The largest inference time was 8.02 ms, below the 20 ms policy period. The
largest encoder-FK computation time was 0.319 ms. Encoder FK agreed with the
robot-reported flange position to within 9.17e-8 m and orientation to the
reported numerical precision. The maximum initial held-reference mismatch was
6.36e-6 rad, maximum held-joint-7 drift was zero, and both runs recorded zero
dropped samples.

Each run discarded 23 stale results, exactly one at each boundary between 24
waypoints. One final request was still in flight when path completion closed the
worker. These counts demonstrate the designed transition rejection behavior.

Every waypoint timed out because shadow mode deliberately leaves the arm at
home. This is runtime, observation, FK, timing, and state-machine evidence; it
contains no path-tracking performance result and authorizes no robot motion.
The full source summaries and their hashes are embedded in `qualification.json`
so this record can be transferred back to the simulation workstation without
the hardware inventory tree.
