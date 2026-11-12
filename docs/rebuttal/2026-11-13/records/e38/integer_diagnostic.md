> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

E38 integer-prototype diagnostic: Llama-3.2-3B, PrismQuant k=max, seed0, default E14 symmetric per-channel GPTQ weights, on one NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (`gpu-pro6000-11`). Weights are stored as signed4 nibbles and expanded to INT8 products with INT32 partial sums and FP32 affine corrections; original E14 rotation and KV simulation remain. This is not a native INT4 runtime measurement.

| 64-sequence reference PPL | Integer candidate PPL | Signed PPL difference | Absolute relative difference | Original absolute ≤0.01 | User-amended relative ≤0.5% | Max absolute logit difference | Argmax agreement |
|---:|---:|---:|---:|---|---|---:|---:|
| 8.50919942 | 8.49573069 | −0.01346873 | 0.158284% | fail | pass | 13.03125 | 86.3884% |

The user explicitly changed the PPL tolerance after seeing this result; the original job's `GATE_FAILED` and scheduler exit2 remain unchanged. The revised status means the quality tolerance is met, not that logits are identical. All64 reference chunk records exactly match the earlier original-simulator replay; candidate NLL improved on37 chunks and worsened on27. Full141-sequence candidate PPL, tasks and timing were not run by this original job because the original gate failed.

At all196 linear sites, recovered packed weights reconstruct the saved BF16 values exactly, packed-to-BF16 control outputs match exactly, and integer outputs are finite. The maximum integer-versus-same-input reference relative L2 error is0.00444360, and the maximum absolute output difference is0.125. The two captured-input QDQ mismatches arise in layer0/1 down-projection rotation replay on eight sampled tokens; same-input comparisons isolate them from integer error.

Sources: [original model-gate JSON](raw/integer_gate_3b_kmax_seed0_job1733332.json), [original operator JSON](raw/integer_operator_3b_kmax_seed0_job1733332.json), [independent collection and amended criterion](raw/integer_collection_001.json), [follow-up protocol](../../manifest.json).
