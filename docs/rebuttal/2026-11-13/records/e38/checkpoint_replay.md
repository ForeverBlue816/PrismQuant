> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

E38 real-checkpoint diagnostic, seed 0, original E14 simulator on RTX PRO 6000 Blackwell Server Edition. This is not a deployed INT4 result.

| Model / row | Historical PPL | Replay PPL | Difference | Replay gate | Exact plain-integer export gate |
|---|---:|---:|---:|---|---|
| Llama-3.2-3B / PrismQuant k=max | 8.71444608 | 8.71896825 | +0.00452217 | pass (absolute difference ≤ 0.01) | fail (0/196 complete matrices) |

[Raw log](raw/checkpoint_3b_kmax_job1732989.json), [independent collection](raw/checkpoint_collection_001.json). Weights use the default E14 per-channel symmetric GPTQ protocol; activations use asymmetric group128 and KV uses E14 hooks. This row is separate from the manuscript’s groupwise-weight/author-reported3B headline values. It has one seed, no seed standard deviation or task CI.

The recovered FP32-scale representation reconstructs every saved BF16 weight after BF16 rounding, but only87,215/774,144rows are an exact plain integer linear grid. This does not rule out another packing/decoding implementation or a future model-level difference within0.01. No timing or deployed logits were measured by this diagnostic.
