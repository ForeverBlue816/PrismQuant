> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

**E38 — Meta-Llama-3.1-8B, seed0; reference GPTQ W4 / asymmetric g128 A4 / E14 KV4, compared with packed-W4 storage expanded to INT8 GEMM and simulated KV4**

Job 1734010 completed both integer rows over the same frozen 141 × 2048 WikiText2 sequences, then measured each loaded model on its exclusive A40; all five attempted rows and raw records are in [table.md](table.md).

| Rotation | Reference PPL | Integer PPL | Integer − reference | Relative difference | Max absolute logit difference | Token argmax agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hadamard | 7.2097352581 | 7.2119071668 | +0.0021719087 | +0.03012% | 27.0625 | 85.30550% |
| PQ k=8 | 6.9904122155 | 6.9892461051 | −0.0011661104 | −0.01668% | 28.0000 | 86.91302% |

Corpus PPL is close, but this does not establish logit or prediction equivalence. The paired sequence-bootstrap 95% intervals for the PPL differences are [−0.01102, +0.01513] and [−0.01261, +0.01042], respectively; they quantify fixed-run sequence uncertainty, not calibration-seed uncertainty. Eight-task differences and sample flips remain unmeasured.

The source audit found identical B1 input IDs and labels, `use_cache=False`, evaluation/inference mode, shared rotation/KV hooks, and aligned token/vocabulary slices in both paths; no shift or masking mismatch was found. Argmax diagnostics include all 2048 positions per sequence while PPL scores 2047 next-token targets, a population difference of 1/2048 that cannot explain the observed disagreement.

There is a concrete arithmetic difference: the reference uses individually BF16-rounded activation QDQ and saved BF16-rounded weights, while the integer path accumulates integer products before FP32 scale/offset application and casts to BF16 at the output. All 224 packed matrices reconstruct the saved weights exactly after a BF16 cast, but their unrounded FP32 scale products differ from those saved weights by up to 0.0004394501. This confirms different arithmetic; the current evidence does not isolate how much each source contributes to the final logit discrepancy.

| Backend / rotation | Prefill median ms | Decode median tokens/s | Peak allocated GiB, prefill/decode |
| --- | ---: | ---: | ---: |
| FP16 | 389.349 | 33.061 | 17.446 / 17.462 |
| Expanded INT8 prototype / Hadamard | 13579.846 | 0.340 | 7.323 / 7.323 |
| Expanded INT8 prototype / PQ k=8 | 13225.179 | 0.378 | 7.371 / 7.372 |

These are B1, prefix2048, decode128 with the first eight decode steps excluded, ten warmups and twenty repetitions; PPL and timing use the same loaded model within each row. The prototype reduces allocated memory but does not provide a speedup. Its groupwise expansion and INT8 arithmetic, simulated KV4, and Python implementation prevent treating these numbers as native INT4 throughput or as an isolated format-overhead comparison. The raw records also contain fourteen real-weight operator measurements per rotation and separate component diagnostics.

Native per-token symmetric INT4 Had/PQk8 rows failed before the first packed layer on unsupported CUDA FP16 `nextafter`; only small candidate-scale vectors now use exact CPU FP16 neighbor arithmetic. Five targeted tests passed under the production PyTorch 2.4.1 environment, and retry job 1735190 runs both rows without repeating the three completed rows. No numerical difference blocks any experiment.
