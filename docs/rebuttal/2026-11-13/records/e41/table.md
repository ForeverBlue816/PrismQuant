> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

**E41: Meta-Llama-3.1-8B, seed 0 — BF16 weights and K/V; activation-only A4 at qkv/down inputs, g = 128**

All six rows completed all 141 WikiText-2 test sequences (2048 tokens per sequence; 288,627 scored tokens per row). These are activation-only floating simulations, distinct from the submitted Table 1 W4A4KV4 protocol; see the [A2 configuration audit](audit_a2_config.md). Lower PPL is better.

| Rotation | Asymmetric PPL (4.25 bits/value) | Symmetric PPL (4.125 bits/value) |
| --- | ---: | ---: |
| Hadamard | 6.40245553 | 6.48862309 |
| PQ constant direction | 6.33337127 | 6.39883259 |
| PQ signed direction | 6.36119859 | 6.40176998 |

| Paired quantity | Point estimate | 95% percentile interval |
| --- | ---: | --- |
| Asymmetric: signed − constant (PPL) | 0.02782732 | [0.02146419, 0.03418386] |
| Symmetric: signed − constant (PPL) | 0.00293740 | [-0.00489953, 0.01087141] |
| Asymmetric contrast − symmetric contrast (PPL) | 0.02488992 | [0.01454039, 0.03516948] |
| Operational offset fraction (ratio) | 0.40280258 | [0.31590360, 0.49480952] |

The operational ratio is `(PPL_signed_asym − PPL_constant_asym) / (PPL_Had_asym − PPL_constant_asym)`.

Its value is **40.28% [31.59%, 49.48%]**. This is an operational comparison, **not a causal attribution percentage**. Ratios were not clipped to [0, 1], and no draws were filtered.

The denominator is 0.06908426 PPL with 95% interval [0.06078578, 0.07748095]; its minimum across 20,000 draws is 0.05234972. There were no nonpositive or near-zero denominators under either declared diagnostic (absolute magnitude ≤ 0.001 PPL or ≤ 5% of the point denominator). A denominator interval spanning zero or draws near zero would make this ratio unstable; that condition was not observed here.

Intervals use 20,000 paired resamples of the 141 saved sequence NLLs (NumPy PCG64 seed 20261111). The same sequence indices are used for all six rows in each resample. Each row first aggregates token-weighted NLL and exponentiates once; differences and the ratio are then computed from those corpus PPL values. Intervals are marginal 95% percentile intervals without multiplicity adjustment. They quantify sequence sampling conditional on the frozen model/factors/rotation seed, not variability across seeds or recalibration; adjacent contiguous sequences may be dependent.

The symmetric signed-minus-constant interval crosses zero. The asymmetric interval and difference-of-contrasts interval are positive under this resampling scheme. The target swap also moves one residual coordinate per group, and symmetric/asymmetric quantizers differ in scale, code range and metadata; the matrix therefore does not isolate a pure offset effect. PQ signed still improves on Hadamard in both formats.

Provenance: [raw bootstrap JSON](raw/paired_bootstrap_001.json), [all paired draws](raw/paired_bootstrap_001.npz), [CPU script](../../manifest.json), [collection](collection_001.json), [raw run](raw/mechanism_8b_seed0_job1733515.json), [protocol](../../manifest.json). Executable model source: `4ee367bcec43869e874787bd921e9db6c6e684b7`; no new GPU run or change to that runner was made.

Latest scope: extend the same six arms to g32/512/d, seed0 only, with paired sequence bootstrap and reusable factor checkpoints. No Table1 prerequisite; no E41 seeds1/2. New measurements will appear in [the extension table](b6/table.md).
