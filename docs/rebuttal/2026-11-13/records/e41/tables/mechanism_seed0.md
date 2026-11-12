> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../../README.md) for the active scope and observed scheduler state.

**E41: Meta-Llama-3.1-8B, seed 0 — BF16 weights and K/V; activation-only A4 at qkv/down inputs, g = 128**

E41 completed all six rows on Meta-Llama-3.1-8B, seed 0, WikiText-2 test (141 sequences of 2048 tokens; 288,627 scored tokens per row). Slurm job 1733515 finished at 2026-11-10 17:40:24 UTC with exit 0. Each raw row contains all 141 finite NLLs; recomputed PPL and recorded hashes match exactly.

Only qkv and down-projection inputs are activation-quantized at group size 128. Weights and KV remain BF16. Lower PPL is better.

| Rotation | Asymmetric PPL | Symmetric PPL | Symmetric minus asymmetric |
| --- | ---: | ---: | ---: |
| Hadamard | 6.40245553 | 6.48862309 | +0.08616756 |
| PQ constant direction | 6.33337127 | 6.39883259 | +0.06546132 |
| PQ signed direction | 6.36119859 | 6.40176998 | +0.04057140 |

| PPL contrast (left minus right) | Asymmetric | Symmetric |
| --- | ---: | ---: |
| PQ constant − Hadamard | -0.06908426 | -0.08979050 |
| PQ signed − Hadamard | -0.04125694 | -0.08685311 |
| PQ signed − PQ constant | +0.02782732 | +0.00293740 |

The signed-minus-constant contrast differs by **+0.02488992 PPL** between asymmetric and symmetric quantizers. PQ constant improves on Hadamard in both formats. Replacing the constant direction with an equally flat signed direction costs 0.02782732 PPL under asymmetric quantization and 0.00293740 under symmetric quantization. This pattern is consistent with an additional benefit from the constant direction when quantization includes a groupwise offset; PQ signed still improves on Hadamard in both formats.

The target swap also moves one residual coordinate per group. The two quantizers differ in scale, code range and metadata cost (asymmetric 4.25 versus symmetric 4.125 effective bits). These results therefore do not isolate a pure offset effect or yield additive contribution percentages. They are single-seed activation-only point estimates; this is not the full three-switch factorial, a joint W4A4KV4 result, or a deployment speed measurement. Frozen E11 randomized factors were reused, and the full 141-sequence no-BOS token stream differs from historical E29/E34 evaluations.

Provenance: [collection](../collection_001.json), [raw run](../raw/mechanism_8b_seed0_job1733515.json), [protocol](../../../manifest.json). Executable source: `4ee367bcec43869e874787bd921e9db6c6e684b7`; runtime HEAD: `522ea0e980b412782e703b11bdd4f035a0b8fa9f` (submission-record addition only). No numerical acceptance gate was applied.
