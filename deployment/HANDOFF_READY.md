# Transfer-ready handoff

Directory:
`/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data/deployment_handoffs/`

Transfer these two files together:

- `franka_rt_handoff_1.tar.gz`
- `franka_rt_handoff_1.sha256`

The unpacked directory is `franka_rt_handoff_1/`. Its manifest SHA-256 is:

```text
fb0f65407e1bd6c9ea5104a20248938294f10c9f75903ef1daa9ab78c6596af6
```

Verify the archive hash against the first line of the checksum sidecar before
extraction. After extraction, `sha256sum -c franka_rt_handoff_1.sha256` checks
both the archive and the unpacked manifest. Then provision the local verifier
environment with the bundled `requirements.txt` and run:

```bash
/path/to/venv/bin/python /path/to/franka_rt_handoff_1/preflight.py \
  --manifest-sha256 fb0f65407e1bd6c9ea5104a20248938294f10c9f75903ef1daa9ab78c6596af6
```

Ask the receiving agent to read `AGENTS.md`, `README.md`, and
`REAL_ROBOT_DEPLOYMENT_PLAN.md` inside the package. It should begin with the
hardware/software inventory and C++ offline parity milestone.

Included: clean-revision nominal and DR exports, hash-verified source snapshot,
base commit archive, two matched six-trial shadow suites, target YAML,
run request, comparison definition, portable preflight, schemas and tests.
The new packaging source is included as a checksummed working-tree snapshot;
the policy exports themselves were made before those additions from clean
revision `d0bfa66331a4813bcff0a0b1d24762618603d012`.

Validation: nine tests passed, lint/whitespace checks passed, both archive
checksums passed, and preflight passed after extraction into a different
directory under isolated Python. No hardware was accessed. The package does
not include dependency wheels, recorded rollout observations, or a hardware
executor. These limitations and receiving milestones are explicit in its README.

To build a subsequent handoff, choose a new ID in `deployment/handoff.yaml`
and run `scripts/deployment/build_handoff.py` with the training environment.
Existing handoffs are never overwritten.
