**Meta-Llama-3.1-8B — seed0, actual backend formats listed per row; exclusive A40 job 1736351**

| Method / backend | Actual configuration | Full141 PPL | Prefill ms | Decode tokens/s | Peak GiB | Status |
| --- | --- | --- | --- | --- | --- | --- |
| FlatQuant official | Packed W4 RTN / per-token A4; native KV4, unquantized prefill KV | pending | pending | pending | pending | pending; timing pending |
| FlatQuant, unused allocation removed | Same packed parameters/arithmetic as official; dead random allocation removed | pending | pending | pending | pending | pending; timing pending |
| PrismQuant k=8, optimized | Packed W4 GPTQ / per-token symmetric A4 / native per-head KV4 | pending | pending | pending | pending | pending; timing pending |
| FlatQuant official fake-quant | BF16 floating W4 RTN / per-token A4 / per-head KV4 | pending | pending | pending | pending | pending; timing pending |
| DuQuant official fake-quant | FP16 floating official W4A4 / QK/V quantizers and floating KV cache | pending | pending | pending | pending | pending; timing pending |

B1, prefix2048, decode128/drop8, token100,10 warmups/20 repeats, eager synchronized wall time, last-position logits. Each timed object first evaluates all141 WikiText2 sequences. Fresh cache construction is included in prefill latency; decode prefix construction and first8 steps are outside its timer. Peak allocated memory includes the prefix.

Packed native rows and floating fake-quant rows are distinct execution modes. FlatQuant native prefill consumes transformed unquantized KV, repeats GQA heads and retains official dequantization arithmetic; PQ native prefill reconstructs packed KV and uses 8 KV heads. These are whole-backend measurements, not an isolated comparison of rotation cost or Table1 g128 configurations. The FlatQuant cleanup removes only a random allocation immediately overwritten by a saved matrix; both versions remain reported.

The complete five-row same-GPU comparison is still pending.
