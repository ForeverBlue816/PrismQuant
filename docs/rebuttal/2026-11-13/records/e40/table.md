> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

Paused by user on2026-11-12: only FlatQuant and DuQuant remain active; job1736074 was cancelled before starting. All prior outputs are retained.

**E40 — Meta-Llama-3.1-8B, seed0: g128 deployment-map calibration energy; GPTQ W4 / asymmetric g128 A4 / E14 KV4 PPL ladder**

Submitted in shared five-GPU job **1736074**, one GPU for energy followed by the six-step PPL ladder; pending cluster priority. No Table1 or original-paper numerical prerequisite.

| Stage | Configurations | Status |
| --- | --- | --- |
| Per-layer/site energy and Ky Fan bound | Hadamard, OffQ, PQ k1/k2/k4/k8/kmax; common BF16 calibration statistics | Submitted, no new measurements |
| Full141 PPL and paired bootstrap | Had → OffQ → PQ k1 (OffQ token-selection rule) → PQ k1 (full moment) → k8 → kmax; W4A4KV4 | Submitted, no new measurements |

OffQ is **在本文管线内的复现**; [fixed implementation choices](../../manifest.json) are recorded. Had/PQk8/PQkmax reuse existing audited GPTQ checkpoints; only OffQ/PQ-k1-top1/PQ-k1-full require new GPTQ. Every row saves its checkpoint and NLL. Energy statistics are collected once and reused. Completed results will be published from `run_job1736074/` with 20,000 paired sequence resamples.

The two k1 controls share E14 qkv-only global R1 and per-layer down R4 pooling: max-norm token selection versus full second moments. Literal OffQ instead pools qkv+upgate for R1 and does not align down_proj; the full chain is not a pure one-factor attribution. Energy k2/k4 inherit frozen kmax directions, so the energy rank curve is not a strictly nested optimum under one covariance.
