**Llama-3.2-3B — seed0; official FlatQuant and DuQuant INT4 quantization recipes, actual floating fake-quant execution**

| Method | Actual configuration | Full141 PPL | Eight-task % | Tasks | Prefill ms | Decode tokens/s | Peak GiB | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FlatQuant | W4 RTN / per-token symmetric A4 / asymmetric per-head KV4; BF16 floating fake-quant | pending | pending | 0/8 | pending | pending | pending | PPL RUNNING; tasks pending; timing pending |
| DuQuant | Official run.sh W4A4 / native QK/V quantizers; FP16 floating fake-quant | 10.34987 | pending | 1/8 | pending | pending | pending | PPL COMPLETE; tasks RUNNING; timing pending |

Timing uses the same loaded checkpoint after full141 PPL, B1/prefix2048/decode128/drop8,10 warmups/20 repeats, real KV-cache decode and an exclusive A40; full-vocabulary logits at the last token position only are computed for both methods. These numbers measure the official floating fake-quant implementations and are not native INT4 kernel throughput.

A same-physical-GPU timing pair is not yet complete.

Paired PPL difference FlatQuant − DuQuant [95% sequence-bootstrap CI]: pending / unpaired.

| Method | PPL source | Eight-task source | Timing source | Calibration source |
| --- | --- | --- | --- | --- |
| FlatQuant | [native_3b_seed0_train-ppl_job1736285.json](../../records/e39/raw/native_3b_seed0_train-ppl_job1736285.json) | pending | pending | [native_3b_seed0_train-ppl_job1736285.json](../../records/e39/raw/native_3b_seed0_train-ppl_job1736285.json) |
| DuQuant | [duquant_3b_seed0_quantize-ppl_retry1_job1736285.json](../../records/e42/raw/duquant_3b_seed0_quantize-ppl_retry1_job1736285.json) | [duquant_3b_seed0_eight-tasks_retry2_job1736285.json](../../records/e42/raw/duquant_3b_seed0_eight-tasks_retry2_job1736285.json) | pending | [duquant_3b_seed0_quantize-ppl_retry1_job1736285.json](../../records/e42/raw/duquant_3b_seed0_quantize-ppl_retry1_job1736285.json) |

No new FP16 control is run; saved checkpoints and NLLs are reused, and missing values remain pending. Native packed FlatQuant performance and optimized PrismQuant are separate backend results and must not be mixed into these floating-implementation timings.
