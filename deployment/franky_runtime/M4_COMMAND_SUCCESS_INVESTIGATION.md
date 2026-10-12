# M4 command-success investigation — 2026-10-12

The investigation found and corrected a Python/native scheduling separation
issue. It has not yet established that this was the root cause of the delivery
faults; confirmation requires a new tracking run with the corrected adapter.
No robot motion or host/network tuning was performed during the investigation.

## Evidence

- Kernel 5.15.133-rt69 PREEMPT_RT, unlimited soft/hard memlock and rtprio 99.
- Route to 172.16.0.6 uses enp0s31f6 from 172.16.0.2. Link 1000 Mb/s,
  full duplex. NIC counters: zero RX/TX errors, drops and missed packets.
- Idle 20-packet ping: zero loss, min/avg/max 0.193/0.263/0.335 ms.
  This is not a control-load UDP deadline test.
- NIC IRQ thread irq/130-enp0s31 runs FIFO priority 50, IRQ affinity CPU0.
  irqbalance active; CPUs 0/6 are SMT siblings. No CPU isolation boot options.
- CPU governor powersave under active intel_pstate. This does not by itself
  establish low clock speed under load. Chrome/code activity was present.
- ethtool and cyclictest are unavailable; interrupt coalescing, EEE, driver
  tuning and scheduler latency were not measured. No guessed NIC changes.
- K400/K800 faults both occur at reported success 0.949999988; the configured
  strict <0.95 guard treats nominal 95% as slightly below threshold. Prior
  samples show success genuinely decreasing, so floating-point representation
  does not explain the preceding delivery failures. Guard unchanged.
- K800 maximum observed callback-entry interval ~2.294 ms. Existing telemetry
  timestamps callback entry, not callback completion/send; it cannot uniquely
  separate native compute latency, scheduling delay and network lateness.

## Confirmed priority leak and fix

libfranka `src/robot_impl.cpp` constructs Robot::Impl by calling
setCurrentThreadToHighestSchedulerPriority. `src/control_tools.cpp` selects
SCHED_FIFO maximum priority 99. Franky's robot.hpp spawns the native controller
using std::thread; Linux threads inherit their creator's scheduler. Previously
both native control and Python's publisher/1 kHz JSON serialization remained
FIFO99. On a shared core, Python could delay native processing and the NIC IRQ50.
Actual CPU sharing during the failed runs was not recorded.

Live no-motion evidence confirms Python OTHER/0 -> constructor FIFO/99 ->
restored OTHER/0. The adapter now preserves/restores the calling scheduler around
Robot construction, then promotes only while spawning the native control thread
and restores Python again immediately afterward. This retains native FIFO99 and
keeps publisher/logging at the caller's original priority. No Franky wheel,
libfranka, Ruckig, gains, torque processing or guard settings changed.

The runner saves `scheduling-evidence.json` in each new session, including actual
new native thread policies/priorities after motion starts. A separate local
no-robot inheritance check verifies child FIFO99 with restored parent OTHER/0:
`hardware_inventory/2026-10-12/m4/rt-thread-inheritance-check.json`.

17 adapter/metrics tests pass, including priority restoration on exceptions.
Circle no-motion preflight completed:
`hardware_inventory/2026-10-12/m4/tracking.20261012T141550Z/`.
An earlier hold-box preflight correctly rejected the current circle stop pose as
outside the +/-30 mm home hold box; no bounds were widened for that check.

## Next comparison

Retry the existing active K800 circle config with the corrected adapter. Inspect
new scheduling evidence: Python policy 0/priority 0, native control policy 1/
priority 99 for the usual normal-shell launch. Compare command-success rates
and host callback tails, retaining failed/partial trials. If faults persist,
collect runtime CPU/IRQ affinity and callback completion-time evidence before
making targeted CPU, NIC or logging changes. Keep watchdog and success guards
unchanged for this comparison. No claim that link counters or idle ping exclude
late control commands, and no motion automatically starts as part of diagnostics.


## Scheduler-fixed retry and float32 guard boundary

`tracking.20261012T141804Z` records Python OTHER/0 and actual new native
thread FIFO99: priority separation works. It completed circle tracking and
faulted during return at 62.544 s on command success 0.949999988. Smooth stop
succeeded, zero dropped samples. Full observed track RMS/max: position
6.883/8.867 mm, contour 3.451/6.436 mm, orientation 0.06670/0.07743 rad.
Host callback-entry max 4.692 ms (earlier during track); in the 100 ms before
fault max 1.939 ms. Entry timestamps do not reveal send deadlines or callback
completion time. The scheduling fix did not eliminate the fault; do not claim
confirmed communication root cause or completed full-session acceptance.

Confirmed robot wire `rbk_types.h` uses float for control_command_success_rate,
while libfranka widens it to double. Native threshold was a double 0.95, causing
nominal 95% to fault on strict comparison. Adapter now rounds only this threshold
to wire float32 representation (0.949999988079071), so reported 95% meets the
configured minimum and 94% still faults. New sessions save effective-limits.json;
the summary includes this evidence. No minimum-success intent relaxation,
watchdog change, gain change or native-wheel rebuild. Threshold boundary regression
covers 94/95/96%. This change has not been tested in another hardware motion.
Real delivery misses still exist; allowing equality will not cure their cause.
