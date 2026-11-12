> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

E41 completed all six Meta-Llama-3.1-8B seed 0 rows, each with 141 finite sequence NLLs, using original BF16 weights and K/V with group-128 A4 only at qkv/down inputs.
The asymmetric signed-minus-constant PPL difference is 0.02782732, with paired-bootstrap 95% interval [0.02146419, 0.03418386].
The symmetric difference is 0.00293740, with interval [-0.00489953, 0.01087141], which spans zero.
The difference of contrasts is 0.02488992, with interval [0.01454039, 0.03516948], from 20,000 resamples using the same sequence indices for all six rows and aggregating token-weighted NLL before exponentiation.
The requested operational ratio is 0.40280258 [0.31590360, 0.49480952], and its denominator interval [0.06078578, 0.07748095] stays positive with no near-zero draws.

Unfinished and unfavorable findings:

- The symmetric contrast interval spans zero; the point estimate alone does not establish a reproducible advantage under that quantizer.
- The operational offset ratio is not causal attribution and was not clipped to [0, 1]; a denominator near or across zero would make the ratio unstable. The target swap also displaces one residual coordinate, and the quantizers differ in scale, code range and metadata.
- These intervals describe sequence sampling conditional on seed 0 and fixed factors, not random-seed or recalibration uncertainty. Contiguous sequence dependence and the lack of multiplicity adjustment limit broader statistical claims.
- The new six-arm g = 32, 512, d extension is packaged on one GPU in the shared compute job, with seed0 only and no Table1 gate; E41 seeds1/2 and SpinQuant training are cancelled.
- The supplied PDF labels 8.60/60.80 as the 3B k=8 row; its 8B k=max row is 6.92/65.43 under W4A4KV4, so E41 cannot be substituted for those main-table values or for native INT4 deployment evidence.

Results: [table](table.md), [LaTeX table](table.tex), [A2 configuration audit](audit_a2_config.md), [bootstrap JSON](raw/paired_bootstrap_001.json), [all paired draws](raw/paired_bootstrap_001.npz), [CPU script](../../manifest.json).
