"""Capture read-only host/NIC evidence; never connects to FCI or changes settings."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess


def capture(argv):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        return dict(command=argv, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=argv, error=str(error))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--interface', default='enp0s31f6')
    p.add_argument('--ethtool', default=shutil.which('ethtool') or '/tmp/m4-ethtool/sbin/ethtool')
    a = p.parse_args()
    files = [Path('/proc/cmdline'), Path('/proc/interrupts'),
             Path('/proc/sys/kernel/sched_rt_runtime_us'), Path('/proc/sys/kernel/sched_rt_period_us')]
    files += list(Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_governor'))
    files += [Path('/sys/class/net') / a.interface / name for name in
              ('speed', 'duplex', 'device/power/control')]
    result = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  interface=a.interface, files={}, commands=[])
    for path in files:
        try:
            result['files'][str(path)] = path.read_text()
        except OSError as error:
            result['files'][str(path)] = dict(error=str(error))
    commands = [['ip', '-s', 'link', 'show', a.interface],
                ['ps', '-eLo', 'pid,tid,cls,rtprio,psr,comm'],
                [a.ethtool, a.interface], [a.ethtool, '--show-eee', a.interface],
                [a.ethtool, '-c', a.interface], [a.ethtool, '-S', a.interface]]
    result['commands'] = [capture(cmd) for cmd in commands]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve previous snapshots instead of silently replacing evidence.
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(a.output)


if __name__ == '__main__':
    main()
