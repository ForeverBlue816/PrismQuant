# Meta-Llama-3.1-8B — measured deployment latency and throughput on A40, seed0

Actual trained checkpoints, full141 WikiText2 PPL, then timing on the same loaded object within each row. The shared workload is batch1, prefix2048, decode128 with the first8 tokens discarded, fixed continuation token100,10 warmups and20 measured repetitions. Last-position logits and real incremental cache are used. Fresh cache creation is inside prefill timing; decode prefix construction is outside its timer. The two allocations below are distinct physical GPUs; ratios are reported only within a matched allocation.

## g128 prototype versus FP16: job1736915, complete

[Full measured table](e38/g128_vs_fp16/job1736915/table.md) · [limitations and source identities](e38/g128_vs_fp16/job1736915/summary.md) · [machine-readable comparison](e38/g128_vs_fp16/job1736915/comparison.json).

| Backend | Full141 PPL | Prefill median ms | Decode median tokens/s | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| FP16 native paged cache | 6.238432 | 389.488 | 33.1000 | 17.4618 |
| PQ k8, asymmetric g128 integer prototype | 6.989246 | 13214.108 | 0.406281 | 7.3721 |

Both rows ran on the same exclusive A40, UUID `GPU-6e3338c3-8b6c-64b7-549b-90febc02999e`, with43/43 exclusivity checks per row and20 prefill plus20 decode samples. The g128 prototype is **33.9269× slower for prefill and81.4707× slower for decode** than this FP16 backend. This is an actual latency measurement, not an operation-count estimate, and it does not demonstrate acceleration.

The prototype stores symmetric per-channel GPTQ W4 but expands each group to INT8 GEMM, applies INT32 partial/FP32 affine correction, and simulates KV4 in a BF16 DynamicCache. It is **not a native packed INT4 g128 backend**. FP16 uses different attention/cache implementations and Torch/Transformers versions. These results describe the current complete implementations; they do not isolate the inherent latency cost of asymmetric g128 or support inferring latency from the previously reported1.6% multiply-add and3.1% storage overheads.

## FlatQuant and PrismQuant native deployment: job1736351

All five rows share the exclusive A40 UUID `GPU-c83e163a-aa72-7904-e707-3eb59a70b80b`, with43/43 exclusivity checks each and20+20 measured samples. All five rows are complete. [Full five-row table](native_8b/table.md) · [Protocol and speed/quality comparison](native_8b/summary.md) · [Source identities and GPU evidence](native_8b/collection.json).

| Backend | Full141 PPL | Prefill median ms | Decode median tokens/s | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| FlatQuant official packed INT4 | 6.952981 | 302.482 | 9.74138 | 6.2440 |
| FlatQuant, overwritten random allocation removed | 6.952981 | 294.382 | 10.40427 | 6.2440 |
| PQ k8 optimized packed INT4, per-token symmetric A4 | 9.995008 | 377.209 | 16.75236 | 7.7907 |
| FlatQuant official floating fake-quant | 7.020863 | 1070.645 | 5.55694 | 16.6105 |
| DuQuant official floating fake-quant | 8.198439 | 1163.505 | 4.84537 | 16.4019 |

PQ's per-token native variant decodes **1.7197× faster than the unmodified official FlatQuant backend**, but its prefill latency is **24.7046% higher** and its PPL is **9.9950 versus6.9530**. Against the cleanup control, decode is1.6101× faster and prefill latency is28.1359% higher. This is not an equal-quality overall speed advantage. The cleanup removes only a random allocation immediately overwritten by a saved matrix; it preserves measured full141 PPL.

PQ native uses per-token symmetric A4 and packed per-head KV4, not the Table1 g128/KIVI format. FlatQuant official native prefill consumes transformed unquantized KV, while decode reads packed KV; it repeats GQA KV heads to query-head count and retains official dequantization arithmetic. The table measures complete deployed backends and records their separate PPL values; it is not an isolated rotation-cost comparison.

The earlier same-A403B floating implementations are also measured: [FlatQuant raw](../2026-11-13-1115/flatquant_3b_timing.json), [DuQuant raw](../2026-11-13-1115/duquant_3b_timing.json). Those results concern Llama-3.2-3B and are not native packed-kernel performance.

[Manifest](manifest.json) records source/export SHA256 for every exported file. These are local A40 measurements, not RTX3090 numbers quoted from another repository. OffQ remains excluded. No model weights or benchmark sample text are exported.

Raw results for job1736351: [FlatQuant official packed INT4](e39/raw/packed_native_8b_seed0_official_ppl-timing_job1736351.json) · [FlatQuant cleanup control](e39/raw/packed_native_8b_seed0_remove-dead-random_ppl-timing_job1736351.json) · [PQ optimized per-token native INT4](e38/raw/stage_a_8b_nar_k8_seed0_fast_officialcomparison_job1736351.json) · [FlatQuant floating implementation](e39/raw/native_8b_seed0_ppl-timing_job1736351.json) · [DuQuant floating implementation](e42/raw/duquant_8b_seed0_ppl-timing_job1736351.json).
