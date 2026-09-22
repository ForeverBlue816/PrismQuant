# E36 — Offset precision: is the aligned level really stored losslessly?

Recorded UTC: 2026-09-22T07:05:03.189671+00:00

Status at creation: PRE-REGISTERED, before any E36 model forward or PPL observation.
Parent: 025c063cbfab9ed317c52173d8eceb10f403a9a3 (completed E34).

## User hypotheses (fixed before measurement)

H36a Z32 − Z16 is within seed noise for PQ on both models (fp16 offsets cost no measurable perplexity at the default configuration).

H36b ZB16 − Z16 is positive and its interval excludes zero (the sweep is sensitive enough to see precision when it matters).

H36c The fraction of groups with offset error > 0.5 step is below 1% overall and below 5% even in the worst layer; the median ratio of (ii) to (iv) is below 1/100.

H36d Hadamard shows the same Z32 − Z16 delta as PQ within seed noise (fp16 offsets are not a PQ-specific liability).

## Frozen protocol and interpretation

- Activation-only E29/E34 protocol: Qwen3-4B-Base and Llama-3.2-3B, frozen 64 BOS-prefixed chunks of 2048 input tokens, 2047 scored next-token targets per chunk, paired seeds 0/1/2, asymmetric g=128 and k=max. Both post-RMSNorm qkv inputs and down-projection inputs are quantized. Weights/KV remain bf16. Original E29 factors, permutation, sign mapping and actual transpose are reused; no recalibration or refitting.
- Eight fixed rows per model: PQ and Hadamard each at Z16, Z32, ZB16 and S32. Z16 uses the existing `dynamic_asym_int4` routine, including fp16 scale-underflow and nonpositive-scale guards, and must reproduce all 384 corresponding E29 chunk NLLs bit for bit per model.
- The main routine rounds BOTH metadata values before computing codes. Moreover, different metadata changes upstream states. Re-encoding each row dynamically would therefore violate the requested cross-row code hash gate. This experiment freezes each Z16 chunk's per-layer/site codes, original fp32 minimum and raw scale, and fp16 metadata. The other rows replay this same quantized activation payload at every site; only reconstruction metadata changes. This is a fixed-code metadata-storage diagnostic, not a claim about a separately re-encoded end-to-end fp32 quantizer. Residual connections and all nonquantized model operations still execute normally.
- Z16: q*s16+z16. Z32: the SAME q and s16, plus original fp32 minimum z32. ZB16: SAME q and s16, plus bf16(z32). S32: SAME q and z16, but the original positive fp32 scale (nonpositive raw scale takes 1). S32 is the explicit scale-changing exception. Reconstruction arithmetic is fp32, followed by the original transpose and final bf16 cast. bf16 has 7 explicit fraction bits (8-bit significand including the implicit leading bit).
- Hash the actual uint8 code values and metadata used at every row/seed/chunk/layer/site. Codes must match across all four rows; scale hashes must match across Z16/Z32/ZB16. Assert Z16 output, codes, scale and offset against the main routine on a fixed tensor, including tie/underflow/constant cases, on CPU and allocated GPU. Round-trip and PQ anchor residual must be <=1e-6. Frozen-source and input hashes are recorded before GPU execution. Old results and the old report prefix are read only.
- The default Z16 representation is 4+(16+16)/128=4.25 effective bits/value. ZB16 has the same metadata bit width. Z32 and S32 each use an extra 16 metadata bits/group; they are diagnostic rows, excluded from default bit-accounted comparisons (nominal storage would be 4.375 bits/value). No claim that fp16 storage is mathematically lossless is made, even if its PPL effect is small.

## B: exact evaluation-group distributions from Z16 passes

- Record original fp32 group minimum z, measured range, actual stored fp16 step s, and rotated group mean c for every layer/site/group/input token/seed, during the same Z16 forward used for Part A. These underlying values and codes are shared by the frozen-payload rows, so no new forward row is needed for B. Retain raw group summaries in the project artifact store, with per-chunk hashes and no token/group subsampling.
- (i) abs(z)/range: median, P99, max. (ii) abs(z-fp16(z))/s: median, P99, max and fraction >0.5. (iv) (2*abs(c)/15)/s: median, P99, max. Also compute the per-group ratio (ii)/(iv) before taking its median; do not divide marginal medians. For PQ, c is the constant aligned Walsh level, obtained as the rotated group mean. For Hadamard, the same computation is only an incidental group-DC comparison, not a claim of an aligned direction. The 2|c| term is the isolated sign-changing component's range, not a prediction that extrema of its sum with residuals add linearly.
- Group strata: all, anchor, other. All PQ groups are anchors at k=max; Hadamard has no aligned anchor groups. Empty strata are N/A with count zero. Token strata: all, BOS, massive, BOS+massive, other, using the exact frozen E34 input-position flags (132 massive and 64 BOS per model). These describe whose activations are measured, not target-loss classes. The final input token is included in B, even though it has no scored continuation.
- Export each seed's exact distributions plus pooled distributions across all three seeds. The unit is one observed token-group, with all layers/sites weighted by their actual number of groups for model-wide summaries. Quantiles use the exact linear empirical quantile, not a sketch or a mean of per-chunk quantiles. Zero denominators are explicitly counted: nonzero/0 is +infinity; 0/0 is undefined and excluded only from that ratio, with its count retained. Infinity stays in the distribution; no clipping, epsilon denominator or silent finite-only filtering. Other metrics retain those groups. Offset/step is defined using the actual positive guarded step.
- Identify the largest pooled P99 of (ii) across layer/site pairs separately for each model/method (ties: lower layer, then qkv). Report its site, zero-based layer, median/P99, >0.5 fraction, |z|/range and aligned-level comparison explicitly. Also record the layer/site with the highest >0.5 fraction; it need not be the P99 maximizer.

## Fixed statistical decisions

- Corpus PPL is the mean of the three exp(mean chunk NLL) values. Report every row's PPL, seed SD and paired delta to its own method's Z16. Retain E29's paired 3x64 delta-method Student-t 90% intervals (df191; repeated texts across seeds), with additional paired three-seed intervals (df2). The report states this dependence.
- H36a: abs(mean PQ Z32−Z16) <= one SD of the three PQ Z16 PPL values, separately on both models. This operationalizes seed noise consistently with E34; it is not a formal equivalence test.
- H36b: mean ZB16−Z16 >0 and lower paired 90% bound >0, evaluated individually for both methods on both models. The primary hypothesis concerns PQ on both models; the Hadamard controls are reported separately, without suppressing disagreement.
- H36c: separately for PQ on each model, pooled >0.5 fraction <.01, maximum layer/site pooled fraction <.05, and pooled median of the per-group (ii)/(iv) ratio <.01. Every clause is exposed. Apply the same diagnostic checks to Hadamard but label its c as incidental DC, not aligned level.
- H36d: abs(mean[(Had_Z32−Had_Z16)−(PQ_Z32−PQ_Z16)]) <= SD of the paired Had_Z16−PQ_Z16 seed PPL differences. Use a joint paired influence calculation for its 90% CI, preserving covariance. This is a predeclared baseline seed-noise reference, not a data-tuned margin.
- No row, threshold, token selection or precision routine is changed in response to PPL. Failed hypotheses are retained. The report explicitly separates numerical nonzero rounding error, detection of a PPL effect, sensitivity of the bf16 stress control, and the limits of the fixed-code intervention.

## Deliverables

`results/<model>/e36_*.csv/json`, exact per-layer and token/group strata summaries, fixed-code/scale hashes, baseline replay and numerical gates, a four-panel figure of per-layer P99 offset error in step units (both sites and models, PQ vs Hadamard), editable SVG/PDF/PNG, source data and rendered QA, and an E36 section appended to report.md. One final E36 commit and GitHub branch push after validation.
