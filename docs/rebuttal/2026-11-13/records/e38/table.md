> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

**E38 — Meta-Llama-3.1-8B, seed0; GPTQ W4 / asymmetric g128 A4 / KV4 reference, with five explicitly labelled deployment formats**

| Actual format/backend | Rotation | Full141 PPL | Eight-task % | Tasks | Max logits / argmax | Prefill ms | Decode tok/s | Peak GiB P/D | Run status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 W/A/KV | fp16 | 6.23843 | pending | 3/8 | n/a | 388.445 [0.595] | 33.069 [0.006] | 17.446/17.462 | COMPLETE; tasks RUNNING | [stage_a_8b_fp16_seed0_job1735315.json](raw/stage_a_8b_fp16_seed0_job1735315.json); [stage_a_8b_fp16_seed0_job1736075.json](raw/stage_a_8b_fp16_seed0_job1736075.json) |
| Packed W4 / per-token symmetric A4 / native per-head KV4 | hadamard | 8.56296 | pending | pending | format variant | 643.051 [0.086] | 11.292 [0.378] | 7.954/7.958 | COMPLETE | [stage_a_8b_hadamard_seed0_job1735315.json](raw/stage_a_8b_hadamard_seed0_job1735315.json) |
| Packed W4 / per-token symmetric A4 / native per-head KV4 | nar_k8 | 9.99501 | pending | pending | format variant | 346.651 [0.056] | 18.719 [0.371] | 7.787/7.791 | COMPLETE | [stage_a_8b_nar_k8_seed0_fast_job1736296.json](raw/stage_a_8b_nar_k8_seed0_fast_job1736296.json) |
| W4 expanded INT8 GEMM / asymmetric g128 A4 / simulated E14 KV4 | hadamard | 7.21191 | pending | 0/8 | 27.06250 / 0.853055 | 13579.846 [19.993] | 0.340 [0.007] | 7.323/7.323 | COMPLETE; tasks RUNNING | [integer_8b_hadamard_seed0_equivalence-timing_job1734010.json](raw/integer_8b_hadamard_seed0_equivalence-timing_job1734010.json); [integer_8b_hadamard_seed0_eight-tasks_job1736075.json](raw/integer_8b_hadamard_seed0_eight-tasks_job1736075.json) |
| W4 expanded INT8 GEMM / asymmetric g128 A4 / simulated E14 KV4 | nar_k8 | 6.98925 | pending | 0/8 | 28.00000 / 0.869130 | 13225.179 [14.691] | 0.378 [0.011] | 7.371/7.372 | COMPLETE; tasks RUNNING | [integer_8b_nar_k8_seed0_equivalence-timing_job1734010.json](raw/integer_8b_nar_k8_seed0_equivalence-timing_job1734010.json); [integer_8b_nar_k8_seed0_eight-tasks_job1736075.json](raw/integer_8b_nar_k8_seed0_eight-tasks_job1736075.json) |

Timing: median [IQR], same loaded model after PPL, exclusive A40, B1/prefix2048, 10 warmups/20 repetitions, decode128/drop8/token100; component profiles are independent inclusive, non-additive diagnostics. Native per-head KV4 and per-token symmetric A4 differ from Table1 asymmetric g128/KIVI; the g128 backend expands W4 storage to INT8 arithmetic and is not native INT4. Cross-backend latency differences include implementation and KV differences and do not isolate pure format overhead.

| Row | Paired PPL delta [95% CI] | Comparison |
| --- | --- | --- |
| native/fp16 | pending / unpaired | variant minus matched-token FP16; different format |
| native/hadamard | +2.32453 [+2.19662, +2.45738] | variant minus matched-token FP16; different format |
| native/nar_k8 | +3.75658 [+3.56223, +3.95765] | variant minus matched-token FP16; different format |
| integer/hadamard | +0.00217 [-0.01102, +0.01513] | integer minus same-run reference |
| integer/nar_k8 | -0.00117 [-0.01261, +0.01042] | integer minus same-run reference |

Native Stage A component diagnostics (milliseconds per model call, separate instrumented pass; inclusive and non-additive):

| Rotation | Workload | Activation quant/pack ms | Linear ms | R4 rotation ms | KV update ms | Source |
| --- | --- | --- | --- | --- | --- | --- |
| fp16 | prefill | 0.000 | 248.999 | 0.000 | 63.000 | [stage_a_8b_fp16_seed0_job1735315.json](raw/stage_a_8b_fp16_seed0_job1735315.json) |
| fp16 | decode | 0.000 | 25.507 | 0.000 | 3.291 | [stage_a_8b_fp16_seed0_job1735315.json](raw/stage_a_8b_fp16_seed0_job1735315.json) |
| hadamard | prefill | 25.746 | 103.796 | 360.422 | 51.296 | [stage_a_8b_hadamard_seed0_job1735315.json](raw/stage_a_8b_hadamard_seed0_job1735315.json) |
| hadamard | decode | 13.801 | 20.753 | 20.939 | 12.762 | [stage_a_8b_hadamard_seed0_job1735315.json](raw/stage_a_8b_hadamard_seed0_job1735315.json) |
| nar_k8 | prefill | 25.756 | 104.502 | 87.601 | 50.892 | [stage_a_8b_nar_k8_seed0_fast_job1736296.json](raw/stage_a_8b_nar_k8_seed0_fast_job1736296.json) |
| nar_k8 | decode | 10.555 | 20.291 | 6.410 | 8.831 | [stage_a_8b_nar_k8_seed0_fast_job1736296.json](raw/stage_a_8b_nar_k8_seed0_fast_job1736296.json) |

The selected records identify original or optimized E14 R2/R4 implementations through rotation_backend; the observed Had/PQ latency gap is not an intrinsic rotation-cost comparison. The selected native-backend control rows were not all verified on the same physical GPU; GPU identities are retained in each raw record. KV update includes prefill unpack/inverse-Hadamard where applicable.

Seed1/2 checkpoint/NLL generation uses fresh E14 R1/R2 and GPTQ samples, fixed original E11 R4 factors with seed-specific signs; these are separate from seed0 timing.

| Seed | Rotation | PPL | Status | Source |
| --- | --- | --- | --- | --- |
| 1 | hadamard | pending | pending | pending |
| 1 | nar_k8 | pending | pending | pending |
| 2 | hadamard | pending | pending | pending |
| 2 | nar_k8 | pending | pending | pending |

Historical FP16 timing/accuracy comes from separate jobs and is not inserted into the new cohort: [fp16_collection_002.json](raw/fp16_collection_002.json).

Submission record (scheduler states are observations at submission, not a live query): [parallel_8b_submissions_001.json](raw/parallel_8b_submissions_001.json).

Preserved pre-collector table: [table.md](history_before_collect8b/table.md).

All observed attempts (no selection by score):

| Source | Status | Error |
| --- | --- | --- |
| [stage_a_8b_nar_k8_seed0_job1735315.json](raw/stage_a_8b_nar_k8_seed0_job1735315.json) | COMPLETE |  |
| [stage_a_8b_nar_k8_seed0_fast_job1736296.json](raw/stage_a_8b_nar_k8_seed0_fast_job1736296.json) | COMPLETE |  |
| [stage_a_8b_hadamard_seed0_job1735190.json](raw/stage_a_8b_hadamard_seed0_job1735190.json) | COMPLETE |  |
| [integer_8b_hadamard_seed0_equivalence-timing_job1734010.json](raw/integer_8b_hadamard_seed0_equivalence-timing_job1734010.json) | COMPLETE |  |
| [stage_a_8b_nar_k8_seed0_job1734010.json](raw/stage_a_8b_nar_k8_seed0_job1734010.json) | FAILED | RuntimeError('"nextafter_cuda" not implemented for \'Half\'') |
| [stage_a_8b_fp16_seed0_job1735315.json](raw/stage_a_8b_fp16_seed0_job1735315.json) | COMPLETE |  |
| [stage_a_8b_fp16_seed0_job1736075.json](raw/stage_a_8b_fp16_seed0_job1736075.json) | RUNNING |  |
| [stage_a_8b_hadamard_seed0_job1735315.json](raw/stage_a_8b_hadamard_seed0_job1735315.json) | COMPLETE |  |
| [stage_a_8b_nar_k8_seed0_original_job1736296.json](raw/stage_a_8b_nar_k8_seed0_original_job1736296.json) | COMPLETE |  |
| [integer_8b_hadamard_seed0_eight-tasks_job1736075.json](raw/integer_8b_hadamard_seed0_eight-tasks_job1736075.json) | RUNNING |  |
| [stage_a_8b_fp16_seed0_job1734010.json](raw/stage_a_8b_fp16_seed0_job1734010.json) | COMPLETE |  |
| [integer_8b_nar_k8_seed0_equivalence-timing_job1734010.json](raw/integer_8b_nar_k8_seed0_equivalence-timing_job1734010.json) | COMPLETE |  |
| [integer_8b_nar_k8_seed0_eight-tasks_job1736075.json](raw/integer_8b_nar_k8_seed0_eight-tasks_job1736075.json) | RUNNING |  |
| [stage_a_8b_nar_k8_seed0_job1735190.json](raw/stage_a_8b_nar_k8_seed0_job1735190.json) | COMPLETE |  |
| [stage_a_8b_hadamard_seed0_job1734010.json](raw/stage_a_8b_hadamard_seed0_job1734010.json) | FAILED | RuntimeError('"nextafter_cuda" not implemented for \'Half\'') |

Collector: [collect_rebuttal8b.py](../../entrypoints/e38/scripts/collect_rebuttal8b.py); [collection provenance](collection_8b.json).
