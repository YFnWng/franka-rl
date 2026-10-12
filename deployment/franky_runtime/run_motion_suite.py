"""Verify immutable transfer and run offline fake/no-motion contracts locally."""
import argparse
import sys
from pathlib import Path
sys.dont_write_bytecode=True  # Do not mutate the immutable bundle on import.
from franka_motion.transfer import verify_motion_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    verify_motion_bundle(args.bundle)
    from franky_experiment.motion_demos.runner import FrankyMotionRunner
    suite=args.bundle/'config/suites/hardware_transfer.yaml'
    report=FrankyMotionRunner(args.bundle,suite).run_suite(args.output_dir)
    print(f"{len(report['results'])} offline contracts checked; hardware_executed=False")


if __name__=='__main__':main()
