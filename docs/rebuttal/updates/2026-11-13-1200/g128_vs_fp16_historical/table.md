# Meta-Llama-3.1-8B — historical E38 g128 asymmetric A4 versus FP16 (seed 0, A40, job 1734010)

Measured on 2026-11-11; assembled from existing raw records, with no new GPU execution. Quantized rows use **symmetric per-output GPTQ W4 / asymmetric g128 A4 / simulated KV4**. Their compute is **expanded INT8 GEMM**, not native INT4. FP16 uses FP16 W/A/KV.

Timing is median [IQR] over 20 repetitions after 10 warmups. Batch 1, prefill 2048, decode 120 measured steps after an untimed 2048-token prefix and 8 discarded steps. The two software stacks and cache-allocation policies differ; see [summary.md](summary.md).

| Method / execution | WT2 PPL | Prefill ms [IQR] | Decode tok/s [IQR] | Prefill speedup / slowdown vs FP16 | Decode speedup / slowdown vs FP16 | Peak allocated GiB prefill / decode | Peak reserved GiB prefill / decode | Raw record |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 | 6.23843 | 389.349 [0.373] | 33.061367 [0.001544] | 1.000000x / 1.000x | 1.000000x / 1.000x | 17.445979 / 17.461848 | 17.638672 / 17.656250 | [stage_a_8b_fp16_seed0_job1734010.json](../raw/stage_a_8b_fp16_seed0_job1734010.json) |
| Hadamard g128 INT8 prototype | 7.21191 | 13579.846 [19.993] | 0.340398 [0.007432] | 0.028671x / 34.878x | 0.010296x / 97.125x | 7.322946 / 7.322946 | 7.398438 / 7.398438 | [integer_8b_hadamard_seed0_equivalence-timing_job1734010.json](../raw/integer_8b_hadamard_seed0_equivalence-timing_job1734010.json) |
| PrismQuant k=8 g128 INT8 prototype | 6.98925 | 13225.179 [14.691] | 0.377550 [0.011211] | 0.029440x / 33.967x | 0.011420x / 87.568x | 7.371088 / 7.372065 | 7.464844 / 7.466797 | [integer_8b_nar_k8_seed0_equivalence-timing_job1734010.json](../raw/integer_8b_nar_k8_seed0_equivalence-timing_job1734010.json) |

A speedup below1 means slower than FP16; the adjacent slowdown is its reciprocal. All repetitions passed GPU exclusivity checks on GPU-02833fb5-df29-3b93-e608-b64499c0ae93. Peak memory includes model, rotations, cache and temporaries; decode peaks also include the untimed prefix. Different cache allocation policies prevent interpreting memory ratios as packed-KV savings.

## Meta-Llama-3.1-8B — matching reference versus g128 INT8 prototype quality (same checkpoint, seed 0)

| Method | Reference PPL | Prototype PPL | PPL delta | Relative PPL delta | Maximum absolute logit difference | Token argmax agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hadamard g128 INT8 prototype | 7.20973526 | 7.21190717 | +0.00217191 | +0.030125% | 27.06250 | 85.305505% |
| PrismQuant k=8 g128 INT8 prototype | 6.99041222 | 6.98924611 | -0.00116611 | -0.016682% | 28.00000 | 86.913024% |

Both comparisons cover 141 windows and 288,768 logit positions (288,627 scored next tokens). Small aggregate PPL differences do not imply logit equivalence. Full precision measurements,20 raw samples per phase, source hashes, software versions, checkpoint paths and ratio definitions are in [comparison.json](comparison.json).
