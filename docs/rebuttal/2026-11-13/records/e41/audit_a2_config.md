> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

E41 configuration audit: Meta-Llama-3.1-8B, seed 0; original BF16 weights and K/V, with simulated A4 only at qkv/down inputs (g = 128).

All six completed E41 rows share this configuration. They are an activation-only mechanism comparison, not the submitted Table 1 W4A4KV4 configuration. No new model inference or GPU job was performed for this audit.

| Component | Actual E41 execution |
| --- | --- |
| Model | `unsloth/Meta-Llama-3.1-8B`, snapshot `e9a141a2091ea561b96483212645a2a05e6f99fc`; 32 decoder layers, hidden width 4096, intermediate width 14336 |
| Weights | Original BF16 model; no GPTQ, weight quantizer, checkpoint weight replacement or folded R1/R2/R4 weight transform |
| Quantized activation sites | Each layer's `input_layernorm` output shared by q/k/v projections, and the input to `mlp.down_proj`: two sites per layer, 64 hooks |
| Other activation sites | No E41 quantizer on gate/up, attention-output projection, embedding, output head or residual stream |
| K/V | No K/V quantizer or R2 rotation; original BF16 attention path; `use_cache=False` during PPL evaluation, so this run does not allocate or measure a quantized decode cache |
| Forward operation | `R^T Q(R x)` before the original BF16 linear; rotation and quantizer arithmetic use FP32; the inverse-rotated result is cast back to the incoming BF16 dtype |
| Hadamard | Original full-width random-sign Hadamard and its transpose; down width 14336 uses the supported factored full transform |
| PQ constant | Frozen E11 layer/site factors, k=max = 32 qkv directions and 112 down directions; compact-WY alignment, frozen permutation/signs and H128; target Walsh column 0 |
| PQ signed | Same factors and random-sign arrays; swap target coordinates 0 and 64 within every group, producing an equally flat half-positive/half-negative direction while also moving one residual coordinate per group |
| Calibration | Reused E11 randomized spectral construction from 128 WikiText-2 train sequences of 2048 tokens; no new exact E29 S3 eigensolve or calibration |
| Seed | One rotation seed index, 0; model RNG seed 0; sign RNG `20260902 + 1000*layer + (100000 if down else 0) + 128`, shared between paired rows |
| Inference | Batch 1, SDPA, no gradient, original BF16 model; Torch `2.11.0+cu128`, Transformers `5.16.0.dev0` |
| PPL | Frozen no-BOS WikiText-2 test tensor, 141 contiguous sequences of 2048 tokens; explicit shifted cross-entropy after upcasting logits to FP32; 2047 scored tokens per sequence and 288,627 per row |
| PPL aggregation | Exponentiate the token-weighted mean sequence NLL, without averaging sequence PPLs |
| Acceptance policy | No PPL, anchor, round-trip or source-hash numerical quality gate; row failures are recorded and remaining rows attempted |

| Quantizer | Codes and arithmetic | Metadata model | Effective format bits/value |
| --- | --- | --- | ---: |
| Asymmetric | `s = fp16((max(x)-min(x))/15)`, `z = fp16(min(x))`, `q = clamp(round((x-z)/s),0,15)`, reconstructed as `q*s+z`; zero/underflowed stored scales are replaced by 1 | One FP16 scale and one FP16 real offset per 128 features | 4.25 |
| Symmetric | `s = fp16(absmax(x)/7)`, `q = clamp(round(x/s),-8,7)`, reconstructed as `q*s`; zero stored scales are replaced by 1 | One FP16 scale per 128 features, zero fixed at 0 | 4.125 |

These bit counts describe the quantization formats. E41 uses floating reconstruction and BF16 linear operations; it does not demonstrate packed A4 storage or native integer kernels. The symmetric/asymmetric comparison changes scale definition, code range and metadata as well as the availability of an offset.

| Rotation | Asymmetric completed row | Symmetric completed row |
| --- | --- | --- |
| Hadamard | `hadamard_asym`, PPL 6.40245553 | `hadamard_sym`, PPL 6.48862309 |
| PQ constant | `pq_constant_asym`, PPL 6.33337127 | `pq_constant_sym`, PPL 6.39883259 |
| PQ signed | `pq_signed_asym`, PPL 6.36119859 | `pq_signed_sym`, PPL 6.40176998 |

The [supplied submission PDF](../../manifest.json), SHA256 `e40d8ecfbf63fcf562ae01de585b3316cfd7521483d0377f5958688799db7f87`, was read directly and Table 1 on physical page 8 was visually checked. The [saved text extraction](../../manifest.json) preserves physical pages 7, 8 and 14.

| Submitted Table 1 row | Model | PPL | Eight-task Avg. (%) | Meaning |
| --- | --- | ---: | ---: | --- |
| bf16 | Meta-Llama-3.1-8B | 6.23 | 67.81 | Published reference, not this E41 run |
| Hadamard | Meta-Llama-3.1-8B | 7.21 | 65.37 | Main-table W4A4KV4 |
| PrismQuant k=8 | Meta-Llama-3.1-8B | 6.99 | 65.36 | Main-table W4A4KV4 |
| PrismQuant k=max | Meta-Llama-3.1-8B | 6.92 | 65.43 | Main-table W4A4KV4 |
| PrismQuant k=8 | **Llama-3.2-3B** | **8.60** | **60.80** | The cited 8.60/60.80 pair belongs to 3B, not 8B |

Table 1's caption explicitly names W4A4KV4. Main text lines 408–420 describe asymmetric group-128 activations, GPTQ INT4 weights, KIVI-style KV, R1/R2/R4 rotations and three PrismQuant seeds. Appendix A.1 (physical page 14, lines 708–740) additionally states FP32 containers holding BF16 weights, asymmetric per-channel GPTQ for 8B, activation quantization at each linear input, K groups of 32 tokens, V groups of 128 head features and a 32-token full-precision residual. Those manuscript statements are distinct from the actual E41 BF16, two-site, activation-only runner and are not silently substituted for it. The manuscript's 3B/8B BF16 references are published rows according to the caption; E41 provides no matched unquantized reference or eight-task measurement.

Therefore E41's approximately 6.33–6.49 PPL range is not a replication, correction or improvement of the 8B Table 1 W4A4KV4 values. The six-row table title and caption explicitly identify its measured configuration. This audit does not resolve the separate provenance correspondence between historical Table 1 seed-averaged entries and individual checkpoint logs.

Evidence: [frozen runner](../../manifest.json), [hooks and symmetric quantizer](../../manifest.json), [BF16 loader and asymmetric quantizer](../../manifest.json), [frozen protocol](../../manifest.json), [raw completed run](raw/mechanism_8b_seed0_job1733515.json), [collection](collection_001.json). Executable source remains `4ee367bcec43869e874787bd921e9db6c6e684b7`; the current CPU analysis does not modify it.

Appendix D does not identify a real accuracy checkpoint shared with Table 1. Figure 7 on physical page 31 (line 1660), D.3 (lines 1671–1673), and Table 13 on physical page 32 (lines 1674–1676) explicitly describe random weights; the two integer methods share hashed random states. D.3 calls this a kernel-swap performance benchmark and uses an FP16 performance reference (lines 1688–1689). D.4 on physical page 33 (lines 1754–1758) explicitly distinguishes the per-token symmetric activation/KV performance interface from the group-128 asymmetric accuracy interface and disclaims checkpoint-level joint accuracy–speed evidence. Matching Appendix D therefore identifies the architecture/runtime experiment, not an already validated real Table 1 checkpoint.
