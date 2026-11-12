> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../../README.md) for the active scope and observed scheduler state.

E38 direct accuracy: seed 0, full 141 × 2048 WikiText-2 sequences, all eight zero-shot tasks. PQ rows use group-128 asymmetric activations and original E14 KV simulation. No numerical execution thresholds were applied.

| Model / implementation | GPU | PPL ↓ | Eight-task avg (%) ↑ | Tasks completed |
| --- | --- | ---: | ---: | ---: |
| 3B FP16 | A40 | 7.81658 | 62.73 | 8/8 |
| 3B PQ k=max simulator | A40 | 8.71802 | 60.03 | 8/8 |
| 3B PQ k=max integer prototype | PRO6000 Server | 8.71940 | 60.50 | 8/8 |
| 8B FP16 | A40 | 6.23843 | 68.68 | 8/8 |

Integer minus simulator: PPL +0.00137719 (0.01580% absolute relative difference); average accuracy +0.4663 percentage points. These runs used different GPUs, so the difference is descriptive. The integer prototype stores signed 4-bit weights and expands them for INT8 GEMM; these results establish accuracy, not native INT4 speed.

| Task / metric | 3B FP16 | 3B PQ simulator | 3B PQ integer | 8B FP16 |
| --- | ---: | ---: | ---: | ---: |
| arc_easy / acc_norm | 72.10 | 71.55 | 70.71 | 82.58 |
| arc_challenge / acc_norm | 46.25 | 44.11 | 43.60 | 54.95 |
| boolq / acc | 74.16 | 68.84 | 70.37 | 83.06 |
| hellaswag / acc_norm | 74.16 | 71.08 | 71.14 | 79.30 |
| openbookqa / acc_norm | 41.00 | 38.80 | 39.80 | 45.40 |
| piqa / acc_norm | 78.02 | 74.76 | 75.30 | 81.18 |
| social_iqa / acc | 46.72 | 44.11 | 44.88 | 48.36 |
| winogrande / acc | 69.46 | 67.01 | 68.19 | 74.59 |

Eight-task averages are unweighted across the displayed metrics; task batch size is 8 for PQ and 2 for FP16. [Collected measurements and raw-file links](../../../manifest.json).

Earlier FP16 timing (separate job1733011, seed0/session1, A40):

| Model | Prefill median [IQR], ms | Decode median [IQR], tokens/s | Peak allocated GiB, prefill / decode |
| --- | ---: | ---: | ---: |
| 3B FP16 | 210.955 [0.576] | 42.911 [0.134] | 8.105 / 8.119 |
| 8B FP16 | 417.469 [0.889] | 29.038 [0.011] | 17.446 / 17.462 |

Timing uses B1, prefix2048, 10 warmups, 50 repetitions; decode measures120 steps after8 excluded steps. No new timing was requested or run. [Historical timing collection](../../../manifest.json).

Outstanding: complete Hadamard/k8/kmax accuracy comparisons across both model sizes, official E39 baselines, and wider per-item uncertainty. Multi-seed and OffQ experiments are excluded from the current scope.

Paired item uncertainty: integer minus simulator, 10000 task-stratified bootstrap resamples; all entries in percentage points.

| Task | Difference | 95% interval |
| --- | ---: | ---: |
| arc_easy | -0.8418 | [-1.9360, +0.2946] |
| arc_challenge | -0.5119 | [-2.3038, +1.3652] |
| boolq | +1.5291 | [+0.3058, +2.7523] |
| hellaswag | +0.0597 | [-0.3585, +0.4880] |
| openbookqa | +1.0000 | [-1.0000, +3.0000] |
| piqa | +0.5441 | [-0.7617, +1.8498] |
| social_iqa | +0.7677 | [-0.4094, +1.8936] |
| winogrande | +1.1839 | [-1.1050, +3.4728] |
| Eight-task mean | +0.4663 | [-0.0649, +1.0020] |

The mean interval includes zero. Intervals describe item sampling conditional on these two fixed runs; they do not include seed or hardware variability. Per-task intervals are nominal and are not adjusted for multiple comparisons. [Bootstrap data and provenance](../../../manifest.json).
