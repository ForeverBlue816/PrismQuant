> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../../README.md) for the active scope and observed scheduler state.

Paused by user on2026-11-12: only FlatQuant and DuQuant remain active; job1736074 was cancelled before starting. All prior outputs are retained.

**E41 B6 — Meta-Llama-3.1-8B, seed0, W16(BF16)/A4 only at qkv and down inputs/KV16(BF16); six arms × g32,g512,full-width d**

| Group | Arm | Full141 PPL | Delta vs matched Had [95% paired CI] | Status | Source |
| --- | --- | --- | --- | --- | --- |
| 32 | hadamard_asym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| 32 | pq_constant_asym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| 32 | pq_signed_asym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| 32 | hadamard_sym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| 32 | pq_constant_sym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| 32 | pq_signed_sym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| 512 | hadamard_asym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| 512 | pq_constant_asym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| 512 | pq_signed_asym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| 512 | hadamard_sym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| 512 | pq_constant_sym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| 512 | pq_signed_sym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| d | hadamard_asym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| d | pq_constant_asym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| d | pq_signed_asym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| d | hadamard_sym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| d | pq_constant_sym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |
| d | pq_signed_sym | pending | pending | submitted 1736074 (PENDING Priority in submission manifest) | pending |

Full-width d means4096 at qkv and14336 at down; BF16 weights/KV are retained, without GPTQ/R2. Each completed group uses20000 paired sequence resamples for all six arms, matching the original g128 reporting convention.

| Group | Contrast | Point | 95% paired CI | Statistics source |
| --- | --- | --- | --- | --- |
| 32 | asym_signed_minus_constant | pending | pending | pending |
| 32 | sym_signed_minus_constant | pending | pending | pending |
| 32 | difference_of_contrasts | pending | pending | pending |
| 32 | operational_offset_fraction | pending | pending | pending |
| 512 | asym_signed_minus_constant | pending | pending | pending |
| 512 | sym_signed_minus_constant | pending | pending | pending |
| 512 | difference_of_contrasts | pending | pending | pending |
| 512 | operational_offset_fraction | pending | pending | pending |
| d | asym_signed_minus_constant | pending | pending | pending |
| d | sym_signed_minus_constant | pending | pending | pending |
| d | difference_of_contrasts | pending | pending | pending |
| d | operational_offset_fraction | pending | pending | pending |

Ratios are operational quantities, not causal attribution; near-zero/negative denominators and nonfinite draws are retained in the statistics files. Original g128 six-arm table and bootstrap are immutable and remain separate: [table.md](../table.md); [paired_bootstrap_001.json](../raw/paired_bootstrap_001.json).

Submission record (scheduler states are observations at submission, not a live query): [parallel_8b_submissions_001.json](../../e38/raw/parallel_8b_submissions_001.json).

Preserved pre-collector table: [table.md](history_before_collect8b/table.md).

Collector: [collect_rebuttal8b.py](../../../entrypoints/e38/scripts/collect_rebuttal8b.py); [collection provenance](collection_8b.json).
