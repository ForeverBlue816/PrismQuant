> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../../README.md) for the active scope and observed scheduler state.

All new E38 experiments use Meta-Llama-3.1-8B and the traced seed0 Had/PQk8 GPTQ checkpoints, with no Table1 prerequisite.
Exclusive A40 job1734010 is submitted for paired full141 reference/integer logits and PPL, five same-instance timing rows, and format/rotation/operator profiling.
The g128 prototype expands packed INT4 to INT8 GEMM and simulates KV4; the separate per-token symmetric path uses native packed INT4 compute and actual packed KV4 bytes.
Only E38 Had/PQk8 receive seeds1/2, and main-table eight-task evaluation follows checkpoint creation in background workers.
Historical3B8.718/60.03 used k=max/seed0/default symmetric GPTQ/BF16, unlike the manuscript k8/g128 asymmetric GPTQ/FP32/three-seed label, and is archived rather than treated as a matched Table1 result.

- Pending: all new8B integer/logit/timing numbers; submitted does not mean measured.
- Limitation: original publication's exact three-seed evidence was not found; new seed measurements will be explicitly distinguished.
- Limitation: Appendix D used random weights, so its timing does not identify a real Table1 checkpoint.
