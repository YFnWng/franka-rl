# M4 command-delivery investigation — 2026-10-12

## Findings

The most recent K0.5 Cartesian tracking run (`tracking.20261012T194210.983461Z`) triggered the runtime guard at 94% command success after 44.930 seconds. During the next 66 ms, success fell to 41%, and libfranka aborted with `communication_constraints_violation`. Smooth stop did not finish. Logged robot periods remained 1 ms and the maximum callback-entry gap was 2.153 ms. Receipt of state packets does not demonstrate timely delivery of outgoing commands.

Host inspection after that run:

- All 12 logical CPUs: `performance`, driver `intel_pstate`.
- Direct route to 172.16.0.6 on enp0s31f6; this establishes the interface, not the absence of a physical switch.
- Ethernet: 1000 Mb/s, full duplex.
- **EEE enabled and active**, transmit low-power-idle setting 17 us. This is a latency candidate, not a proven cause.
- RX interrupt coalescing `rx-usecs: 3`. This driver value must not be interpreted as a measured fixed three-microsecond delay; leave it unchanged for the first EEE experiment.
- No driver/interface RX/TX errors, CRC errors, missed packets, receive-buffer failures or transmit timeouts reported.
- Ethernet IRQ 130 affinity CPU0, IRQ thread FIFO50; control thread was FIFO99 in trial scheduling evidence. No CPU isolation configured on the kernel command line. These settings are recorded, not established causes.
- Kernel RT runtime budget 950000 us per 1000000 us; no evidence of throttling captured.

Source inspection:

- Franky built in Release with `-O3 -DNDEBUG`.
- M4 reference consumption uses the unsafe single-reader triple-buffer operation (no user-side mutex); servo log is a preallocated single-producer/single-consumer ring.
- JSON serialization and disk writes run in Python outside the native servo callback. No Python callback is registered by this runner.
- Normal Cartesian law and stop remain native. Zero posture gain skips nullspace/mass projector work in these circle profiles; no target acceleration is provided.
- Callback entry timestamps omit computation duration, wrapper work and socket transmission. They cannot exclude outgoing deadline misses.
- No `mlockall` call found in the audited Franky/libfranka source trees. Unlimited memlock permission alone does not prove memory is locked; actual process page-fault evidence is not captured.

## Controlled next test

Robot idle, before another run, disable only EEE:

```bash
sudo /tmp/m4-ethtool/sbin/ethtool --set-eee enp0s31f6 eee off
/tmp/m4-ethtool/sbin/ethtool --show-eee enp0s31f6
```

The binary was downloaded from Ubuntu's package repository and extracted under `/tmp`; no system package was installed. The setting is temporary and may reset on link/network reconfiguration or reboot. Previous setting can be restored with `eee on`. Retain CPU performance mode, all controller gains, watchdogs, rate/filter settings and the 95% command-success guard.

Capture the changed settings with:

```bash
python3 /home/chen-lab/yifan/franka-rl/deployment/franky_runtime/tools/collect_m4_host_diagnostics.py \
  --output /home/chen-lab/franka_ros2_ws/hardware_inventory/2026-10-12/m4/communication-investigation/host-after-eee-change.json
```

Recover any robot reflex using the existing operator procedure before retry; this investigation did not connect to FCI, recover errors or start motion. Retry one known profile with `run_profile cartesian.circle.k1.d1 1`. A successful retry supports further testing but does not prove EEE caused the failures.

If recurring, measure full callback computation/return durations, thread CPU migrations/page faults and NIC counter deltas during a run. Then compare a minimal native control test to M4. Official `communication_test` enters torque control and changes collision thresholds: it is not a passive network probe and has not been run automatically. Separate network scheduling from controller workload before changing gains or thresholds.

## Evidence

Read-only snapshot: `/home/chen-lab/franka_ros2_ws/hardware_inventory/2026-10-12/m4/communication-investigation/host-before-eee-change.json`.
Reusable collector: `tools/collect_m4_host_diagnostics.py`. It records read-only CPU, RT scheduler, process/thread scheduling and NIC data; it never accesses FCI or changes settings.

Official guidance: [FCI troubleshooting](https://frankarobotics.github.io/docs/troubleshooting.html), [FCI control/communication overview](https://frankarobotics.github.io/docs/doc/libfranka/docs/overview.html). These identify CPU/network timing and loop computation as potential causes. Hardware validation of the EEE hypothesis remains pending.

## First completed trial after operator disabled EEE

`tracking.20261012T195600.810579Z/trial-01` (`cartesian.circle.k1.d1`) completed with native fault 0, stop fault 0, successful smooth stop and no dropped samples. Minimum active command success after startup was 98%; maximum logged callback-entry gap was 1.80732 ms. Operator reported EEE disabled before this run. This supports retaining EEE disabled for subsequent single-trial comparisons, but one run does not establish causality or eliminate intermittent failures.
