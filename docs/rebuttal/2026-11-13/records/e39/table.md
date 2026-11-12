> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

**E39 — Meta-Llama-3.1-8B, FlatQuant official, seed0, actual native method configuration**

| Model | Method | Actual configuration | Full141 PPL | Eight-task % | Tasks | Calib/quant seconds | Quantize+PPL GPU hours | Peak allocated GiB | Run status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Meta-Llama-3.1-8B | FlatQuant official | W4 RTN per-channel symmetric / A4 per-token symmetric / native asymmetric per-head K4V4, BF16 residual | pending | pending | pending | pending | pending | pending | submitted 1736252 (PENDING Priority in submission manifest) | pending |

Cost scope: pending; checkpoint serialization and total allocated wall time are separate fields in raw results. No E14/GPTQ/group128 compatibility is implied.

Paired PPL difference from the explicitly linked same-token FP16 control: pending / unpaired; [direct_fp16_8b_seed0_job1733513.json](../e38/raw/direct_fp16_8b_seed0_job1733513.json).

Submission record (scheduler states are observations at submission, not a live query): [parallel_8b_submissions_001.json](../e38/raw/parallel_8b_submissions_001.json).

Preserved pre-collector table: [table.md](history_before_collect8b/table.md).

Collector: [collect_rebuttal8b.py](../../entrypoints/e38/scripts/collect_rebuttal8b.py); [collection provenance](collection_8b.json).


Active scope (2026-11-12 23:22 Singapore): only FlatQuant and DuQuant; job1736252 is a two-GPU bundle, one method per GPU, pending Priority. No new FP16 or PrismQuant runs. [Submission](../e38/raw/baselines_only_submission_20261112.json).
