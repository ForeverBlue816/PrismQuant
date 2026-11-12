> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../../README.md) for the active scope and observed scheduler state.

**E38 — Meta-Llama-3.1-8B; real GPTQ W4, PQ k=8/Hadamard, g128 asymmetric A4 + KV4 reference; deployment format variants below**

Base revision `e9a141a2091ea561b96483212645a2a05e6f99fc`; seed0 quantized release `fc812e6687165115feee00293c953a1b521d3bd0`. Actual GPTQ saved weights use per-output-channel symmetric quantization in a BF16 container. No Table1 quality gate. All3B rows excluded.

| Actual format/backend | Rotation | Full141 PPL | Max logit difference / argmax agreement | Prefill ms | Decode tokens/s | Peak GiB | Status |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| FP16 weights/activation/KV | none | pending | n/a | pending | pending | pending | Submitted1734010 |
| Native packed W4 / per-token symmetric A4 / packed per-head KV4 | Hadamard | pending | format variant, not g128 equivalence | pending | pending | pending | Submitted1734010 |
| Native packed W4 / per-token symmetric A4 / packed per-head KV4 | PQ k=8 | pending | format variant, not g128 equivalence | pending | pending | pending | Submitted1734010 |
| Packed W4 expanded to INT8 GEMM / g128 asymmetric A4 / simulated E14 KV4 | Hadamard | pending | pending | pending | pending | pending | Submitted1734010 |
| Packed W4 expanded to INT8 GEMM / g128 asymmetric A4 / simulated E14 KV4 | PQ k=8 | pending | pending | pending | pending | pending | Submitted1734010; first row executed |

Timing uses exclusive A40, batch1/prefix2048,10 warmups/20 repetitions,128 decode steps with the first8 discarded, fixed decode token100. Each row evaluates and times the same loaded model; all NLL and numerical discrepancies are retained. Native KV4 prefill consumes actual packed cache bytes after reconstruction, with unpack and transform costs included. Component profiling is separate and inclusive/non-additive; the Python g128 implementation is an arithmetic prototype, not a native INT4 deployment claim.

Eight-task integer evaluation is a background phase using the same packed checkpoints. E38 Had/PQk8 seeds1/2 generate actual separate E14/GPTQ checkpoints; fixed original E11 R4 factors and seed-dependent signs are declared explicitly. Table1 comparison is reported after measurements: Had7.21/65.37%, PQk8 6.99/65.36%; legacyseed0 logs give7.20638/65.36590% and6.98984/65.35919%, respectively.

Historical FP16 control: PPL6.23843/eight-task68.68%, prior A40 timing417.469ms and29.038tokens/s. Those prior accuracy/timing jobs are separate from the newly submitted five-row cohort. [Sources](../../../manifest.json).
