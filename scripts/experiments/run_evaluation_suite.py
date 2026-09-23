#!/usr/bin/env python3
"""Run and compile a subprocess-isolated evaluation suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from franka_rl.experiments import EvaluationSuiteConfig, EvaluationSuiteCoordinator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True, help="Evaluation suite YAML file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Suite output directory under FRANKA_RL_DATA_ROOT.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Run every job instead of reusing valid completed artifacts.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed job.",
    )
    args = parser.parse_args()

    config = EvaluationSuiteConfig.from_yaml(args.suite)
    coordinator = EvaluationSuiteCoordinator(
        config,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        fail_fast=args.fail_fast,
    )
    return 0 if coordinator.run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
