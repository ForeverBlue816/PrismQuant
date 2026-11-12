**E38 rotation optimization — Meta-Llama-3.1-8B, PrismQuant k=8, seed 0; GPTQ per-channel symmetric W4, per-token symmetric A4, native asymmetric per-head128 KV4; CUTLASS packed INT4 (Stage A, not Table 1 g128/KIVI)**

| Rotation execution | Full141 PPL | Prefill ms, median [IQR] | Decode tok/s, median [IQR] | Peak allocated GiB, prefill/decode | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Original E14 PyTorch rotation | 9.995008261 | 625.837 [0.289] | 13.765 [0.093] | 8.111 / 8.115 | COMPLETE |
| CUDA H128 + cached gathers / active reflectors | 9.995008261 | 346.651 [0.056] | 18.719 [0.371] | 7.787 / 7.791 | COMPLETE |

Prefill speedup: **1.805×**; decode speedup: **1.360×**. Peak allocated memory decreases by 332.469 MiB in each workload.

Same NVIDIA A40 UUID `GPU-08ae42ad-9fd5-758c-de1f-d715a5622efc`, job 1736296; 43/43 recorded exclusivity checks per path report EXCLUSIVE and no other GPU process. Rows ran sequentially. Each row measured full PPL and timing on its own single loaded model instance.

Timing protocol: batch 1, prefix 2048, 10 warmups, 20 repetitions, cached decode 128 with the first 8 discarded, fixed token 100, last-position logits only. Cache reset and decode-prefix construction occur outside the measured decode interval. IQR is q75−q25 with linear interpolation; decode rates are computed per repetition before summarizing. Peak memory is the PyTorch peak allocated measurement, including the loaded model.

| Quality comparison, fast − original | Result |
| --- | ---: |
| Aligned WikiText-2 windows / scored tokens | 141 / 288627 |
| Exactly equal recorded segment NLL | 141 / 141 |
| Maximum absolute segment NLL difference | 0 |
| PPL difference / relative difference | 0 / 0% |
| Actual-input rotation probe sites | 64: 32 R2 + 32 R4 |
| Probe elements / maximum FP32 absolute difference | 3702784 / 0 |
| Probe maximum relative L2 / changed elements after original output cast | 0 / 0 |

Probes cover the first 8 rows per site in the first actual PPL forward. These measurements do not constitute an exhaustive comparison of full-model logits or argmax predictions.

Independent component diagnostics below use CUDA events after the main timing pass. Values are inclusive and non-additive; they do not partition end-to-end latency.

| Execution | Workload | Act quant/pack ms | Linear ms | R4 ms | R2 ms | KV update ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original | prefill | 25.747 | 104.614 | 344.962 | not separately measured | 50.842 |
| original | decode | 10.360 | 20.144 | 15.309 | not separately measured | 9.216 |
| fast | prefill | 25.756 | 104.502 | 87.601 | 5.925 | 50.892 |
| fast | decode | 10.555 | 20.291 | 6.410 | 5.903 | 8.831 |

The fast path retains saved weights, compact-WY factors, reflector order, signs, permutations and all quantization settings. It replaces the block Hadamard implementation with CUDA H128 and precomputes fixed gather/active-index metadata. Both paths reuse all 32 packed layer exports. Table 1 instead uses asymmetric g128 activations and KIVI token-axis grouped K with residual32; these Stage A PPL/speed values cannot be presented as the Table 1 format.

Sources: [original raw](../raw/stage_a_8b_nar_k8_seed0_original_job1736296.json), [fast raw](../raw/stage_a_8b_nar_k8_seed0_fast_job1736296.json), [comparison and provenance](comparison.json), [summary](summary.md).
