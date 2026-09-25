# Runtime model evidence

Read [FINDINGS.md](FINDINGS.md) and [comparison.json](comparison.json).
These small derived summaries are kept in the repository. Full tensors,
exact extractor copies, saved/reconstructed configurations, logs and checksums
are in:

```text
/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data/model_audits/2026-09-24_runtime/return_evidence_1/
```

The `nominal/` and `dr/` subdirectories each contain `runtime_model.json`.
Each report includes all 11 bodies and 9 joints for 8 environments at three
stages: initialized, first reset, second reset. SI units and frames are named
in the tensor keys and explained in the report. `comparison.json` additionally
contains inferred realized payload parameters and normalized gain/effort scales.
`SHA256SUMS` validates every file in the return directory. Transfer that directory
out of band if the hardware agent needs the full data.

The parent data directory contains earlier development captures, including
incomplete attempts. Use only `return_evidence_1` for the finalized handoff.

## Reproduce

Activate the shared Isaac environment using the project's usual direnv setup
(`/home/chen-lab/isaac/direnv/isaac-lab.envrc`), then run each scenario in a fresh
process. Choose new output directories; the extractor refuses overwrites.

```bash
/home/chen-lab/isaac/.venv/bin/python -u scripts/deployment/extract_training_physics.py \
  --scenario nominal --num_envs 8 --seed 123 --device cuda:0 --viz none \
  --output-dir /path/on/data-volume/new_audit/nominal_complete

/home/chen-lab/isaac/.venv/bin/python -u scripts/deployment/extract_training_physics.py \
  --scenario franka_dr_train_v2 --num_envs 8 --seed 123 --device cuda:0 --viz none \
  --output-dir /path/on/data-volume/new_audit/dr_complete

/home/chen-lab/isaac/.venv/bin/python scripts/deployment/summarize_training_physics.py \
  --captures /path/on/data-volume/new_audit \
  --hardware-report deployment/model_audit/2026-09-24/report.json \
  --output-dir /path/on/data-volume/new_audit/return_evidence_1
```

Create the audit parent directory first. Successful extractions print
`EXTRACTION_COMPLETE` and set `complete: true` in the JSON; a simulator process
exit status alone is insufficient because the launcher may catch exceptions.
The summarizer requires both completion markers and matching physics configs,
checks every tensor for finite values, and verifies all robot layer hashes
against the hardware audit.
