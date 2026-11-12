> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

# E38 protocol and deployment audit

Status: source audit only; no new accuracy or performance result is claimed here. The companion `raw/protocol_audit.json` records source hashes, the inspected checkout, historical evidence, and unresolved conditions. All OffQ-related experiments are excluded by the user's instruction.

## What can be used as an anchor

The user-identified [submitted PDF](../../manifest.json) has 37 pages and SHA256 `e40d8ecfbf63fcf562ae01de585b3316cfd7521483d0377f5958688799db7f87`. Its Table 1 on physical page 8 lists the following 8B PPL/eight-task values: bf16 6.23/67.81, Hadamard 7.21/65.37, PrismQuant k=8 6.99/65.36, and k=max 6.92/65.43. Its caption says the 3B/8B bf16 references come from published work; they are not local matched measurements. The inspected [public arXiv v1 Table 1](https://arxiv.org/html/2609.32429v1#S4.T1) agrees with these values. The submitted PDF is the primary document reference.

Local historical evidence is materially different from the 3B anchors in the request. `report.md`'s GPTQ-protocol table and the JSON sources under `/projects/nar/nar-validation/results/llama32_3b/` give seed-0 g128_asym values of 9.04241/59.29 (Hadamard), 8.63633/60.21 (k=8), and 8.59702/60.80 (k=max). Both submitted and public papers' k=8 task vectors match the local k=max vector to displayed precision. This is an unresolved provenance/label correspondence, not proof of a mislabeled experiment. The local `e14_nar_kmax_asym_g128_author_reported.json` and report footnote identify 8.58/61.23 as an author report from another cluster; raw harness output is absent and seed aggregation is unspecified. The submitted manuscript page 8, line 418, states that all PrismQuant results are averaged over three seeds, while the identified local rows above are single-seed records. These rows must not become newly measured rebuttal results by copying them.

There is also a numeric-protocol discrepancy to resolve: submitted Appendix A.1, page 14, lines 708–715, describes fp32 model containers and asymmetric GPTQ, whereas `nar/experiment.py::load_model` uses bf16 for the inspected E14 path and `e14_w4a4kv4.py::WEIGHT_PROTOCOLS['default']` uses symmetric per-channel GPTQ. The local 8B Hadamard/k=8/k=max values match the default E14 protocol; the 3B Hadamard value instead matches g128_asym. Record the actual source, checkpoint variant, dtype and seed separately for each model. Page 14 describes R2 as folded into V and O, whereas the main text on page 8, lines 413–414, and E14 code place R2 online after V; preserve the actual source behavior and flag the wording inconsistency for revision.

## Sources read and existing benchmark settings

The audit read E14, E17 v1/v2/v3 and folded-R4 helpers; E27; E29's shared ablation implementation and protocol; E33; E34; E36; and E28-v2's benchmark, model, cache/contract documentation and protocol. E27/E29/E34/E36 are activation-only mechanisms with bf16 weights and KV, so their PPLs cannot be substituted for E14 W4A4KV4.

The existing benchmark lives in `quarot-llama3/e28_v2/`, with the model classes in `quarot-llama3/modeling_llama3.py`. Its recorded protocol is `results/e28_v2/20260912_a40_v2_full_int4/protocol.json`. Core prefill uses 2048 tokens at B=1 and B=16 with final-position logits. Decode uses B=1, a 2048-token prefix, 128 causal steps and excludes the first eight. There are ten complete warmups and fifty runs per condition in three balanced sessions. Eager synchronization surrounds the full measured sequence; graph timings use a separate current-stream backend cohort and a 2176-token page. New B=64 runs, IQRs and exclusivity records are additional rebuttal requirements. Historical random-weight timings remain historical; equality of real/random latency has not been measured.

These settings are confirmed by submitted Appendix D.3, pages 31–32 (lines 1669–1702). Appendix D.4, page 33 (lines 1753–1758), already states that the performance backend has different activation/KV interfaces, does not constitute joint checkpoint-level accuracy/speed evidence, and has recorded large-accumulator fp16-conversion failures. E38 must close those documented gaps; the source audit alone does not close them.

## Why Stage A is not just a weight copy

1. **Artifact completeness.** At audit time the original `/projects/nar/nar-validation/artifacts/e14/{llama32_3b,llama31_8b}` roots do not contain complete checkpoints. Only two 3B k=max seed-1 layer files and a partial audit CSV remain. This does not assert that checkpoints are unavailable elsewhere: the public release catalog and cached Hub variants must be inventoried separately and verified before use. Require every decoder layer, a completion manifest, exact base-model revision, R1/R2/R4 factors, seeds and token-cache hashes.
2. **Weight representation.** E14 `_layer_state` saves seven reconstructed floating linear weights per layer, with no original integer codes or scale/zero arrays. The default GPTQ scale is computed in fp32, then reconstructed weights are rounded to bf16. E28 `Linear4bit` accepts signed nibbles and only one fp16 scale per output channel. Group-128 asymmetric GPTQ cannot in general be represented by that format. A new RTN fit is a new weight quantizer and must not silently replace the supplied checkpoint. Even default per-channel weights require a measured export reconstruction check because the saved values and metadata precision differ.
3. **Rotations.** Load the base embedding, head and norms and reproduce E14 norm/R1 folding once; then replace the decoder weights with the already-folded checkpoint. E14 R2 is still applied after `v_proj`, with compensation in `o_proj`; it is not an already-folded V projection. E14's cross-head `o_proj` Hadamard is used only for Hadamard rows, while the E28 NAR class inherits it too. The exporter must respect the actual row's structure.
4. **R4 signs and permutation.** Existing `export_nar_factors.py` uses E17's `act._seed` signs (seed + 1000*layer + 100128), whereas E14 R4 uses seed + 340000 + layer. Reusing an existing E28 factor file therefore does not establish calibrated E14 equivalence. Use the loaded E14 rotation objects. If selecting the folded R4 kernel, also permute gate rows and signed-permute up rows exactly as `fold_swiglu_weights` requires; preserve the down weight's already-transformed basis. Signed code negation needs special handling at -8: blindly negating a signed nibble produces an unrepresentable +8. Preserve exact values or fail export instead of clipping them.
5. **KV and attention.** E28 quantizes K across head channels, has no 32-token residual, rotates Q/K with a feature Hadamard, and uses original floating K/V for prefill attention. E14 groups K over 32 tokens, V over head channels and simulates a causal 32-token residual in prefill as well as decode. Matching only linear activations leaves a different model. Preserve the requested KV policy, or explicitly scope a Stage-A row as a changed-KV backend evaluation; it cannot answer the exact-format claim.
6. **Precision and arithmetic.** E28's symmetric activation kernel divides fp16 operands; E14's symmetric reference divides in fp32 after storing a fp16 scale. E28 dequantization converts the full INT32 dot product to fp16 before scaling; large integer sums can overflow that conversion even if the final scaled answer is finite. Audit real activations for finite outputs and quantify any corrected implementation separately. E14 rotation and reconstruction cast boundaries must be preserved or their errors measured, not silently removed.

The existing `build()` is specifically a random initializer and must not run after loading real weights. Its byte-equality rule between the two random integer methods is also inappropriate for method-specific calibrated checkpoints. The same loaded real model can expose all logits for accuracy and last-position logits for serving-style timing, with this mode change stated explicitly.

## Required export and equivalence checks

Before GPU timing, write an export report for every linear with checkpoint hash, shape, original dtype, weight protocol, code min/max, scale/offset dtype and hash, out-of-range count, max absolute and relative reconstruction error, unequal element count, and packing/unpacking equality. Reject missing metadata or unsupported group formats. Reconstructing an affine grid from rounded floating values is a recovery problem; success must be tested against every element, and ambiguity recorded. If original GPTQ metadata must be regenerated, preserve the calibration and algorithm and keep the regenerated checkpoint distinct.

Then compare a small real-input layer forward with the exact frozen fake-quant runtime, including R2/R4 and KV. Extend to 64 held-out sequences only after those local checks pass. Report max absolute logit difference, argmax agreement and both corpus PPLs, with absolute PPL difference <= 0.01 as the requested gate. Do not introduce a new floating simulator to make the deployed result look equivalent to Table 1. Failed attempts and their logs remain in the experiment record.

## Stage B arithmetic and storage

For exact floating weights, with `x_j = sA_g*qA_j + oA_g`, the proposed decomposition is algebraically correct: `sum_g sA_g*sum_j W_ij*qA_j + oA_g*S_ig`, where `S_ig=sum_j W_ij`. This first term is not an INT4-by-INT4 operation until W has an explicit integer representation. For general grouped affine weights `W_ij=sW_ig*qW_ij+bW_ig`, the exact expansion is

```
sum_g [sA_g*sW_ig*sum_j qA_j*qW_ij
       + sA_g*bW_ig*sum_j qA_j
       + oA_g*sW_ig*sum_j qW_ij
       + 128*oA_g*bW_ig].
```

One grouped integer accumulation followed by one global scale is insufficient. Weight-zero corrections must be included for g128_asym. Dequantized bf16 weights and bf16-rounded reconstructed activations may additionally contain rounding residuals that are not exactly affine; compare to the saved simulator instead of asserting exact equivalence from the formula.

At g=128, raw signed-by-signed INT4 partial sums lie in [-7168,8192]; unsigned activation [0,15] times signed weight [-8,7] gives [-15360,13440]; unsigned-by-unsigned gives [0,28800]. All fit signed INT16 for one group. These are mathematical bounds only: the current CUTLASS interface actually accumulates INT32, and a whole-width sum or packed intermediate may exceed INT16. Keep INT32 unless the precise instruction/layout and every intermediate have been verified. Grouped centered codes can span [-15,15] and must not be reinterpreted as signed four-bit values without correction.

Storing `S` costs `out_features*(in_features/128)*bytes_per_S_element`, separate from existing weight scales/zeros. It is 1/32 of the packed 4-bit weight bytes if S is fp16, or 1/16 if S is fp32. Record the chosen dtype and measured storage, and test its rounding contribution. These fractions exclude embeddings/head, KV, factors, padding and other metadata; they are arithmetic estimates, not model-memory measurements.

The existing E17 pack epilogue uses `floor(q+0.5)` and lacks E14's post-fp16-conversion zero-scale guard. E14 uses nearest-even rounding and replaces zero/underflowed stored scales by one. Fix and test midpoint, underflow and degenerate-group cases before reusing that epilogue. E14 casts the rotated down input back to bf16 before quantization; directly quantizing a fused fp32 rotation result changes this boundary. A successful op-level test is necessary but does not replace the whole-model gate.

## Next steps in order

1. Freeze the current anchor replay against a locally available, fully identified checkpoint and its historical JSON; keep the unresolved Table-1-to-raw-run correspondence explicit.
2. Inventory complete released checkpoint variants and missing weight metadata; confirm which 3B/8B rows can be exported without requantization. Preserve all cached factors.
3. Implement one audited real-weight linear adapter and a layer-level accuracy diagnostic. Use E14 runtime rotations and attention first to isolate packing from other changes; do not present that diagnostic as the final fast deployment path.
4. Build the full Stage-A adapter only for supported weight formats, and explicitly record activation/KV/dtype differences. Add the Stage-B grouped integer primitive, offset correction and exact KV implementation under separate gates.
5. After the required gate succeeds, evaluate all eight tasks with per-item logs, then time the same loaded model on an exclusive GPU using matched conditions and all requested batch sizes. Repeat seeds and build paired intervals only from genuine paired items.
