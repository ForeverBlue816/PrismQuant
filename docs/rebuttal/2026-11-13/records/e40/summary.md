> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

Paused by user on2026-11-12: only FlatQuant and DuQuant remain active; job1736074 was cancelled before starting. All prior outputs are retained.

E40 is restored on Meta-Llama-3.1-8B and submitted as one worker in five-GPU job1736074, with no Table1 gate; compute job1736074 replaces the unstarted1734028 after repairing Slurm request-option conflicts and is pending Priority as of2026-11-12 21:49 Singapore.
The worker collects shared four-site calibration statistics, measures actual deployment-map energy and Ky Fan bounds, then runs the six-step W4A4KV4 statistical-ablation PPL ladder.
Had/PQk8/PQkmax reuse audited seed0 checkpoints; only OffQ/PQ-k1-top1/PQ-k1-full run new GPTQ.
OffQ is an explicitly labelled in-pipeline reproduction with fixed algorithm choices, not unpublished official code.
The complete six-step ladder automatically produces20,000 paired sequence-bootstrap PPL intervals and differences from Hadamard.

- Pending: all new GPU measurements; current state is PENDING Priority.
- Deferred: original-paper Llama3-8B validation; it is not a prerequisite.

[Current submission recovery](../e38/raw/slurm_bundle_recovery_20261112.json).
