Meta-Llama-3.1-8B — seed0, actual backend formats listed per row; exclusive A40 job 1736351

B1, prefix2048, decode128/drop8, token100,10 warmups/20 repeats, eager synchronized wall time, last-position logits. Each timed object first evaluates all141 WikiText2 sequences. Fresh cache construction is included in prefill latency; decode prefix construction and first8 steps are outside its timer. Peak allocated memory includes the prefix.

Packed native rows and floating fake-quant rows are distinct execution modes. FlatQuant native prefill consumes transformed unquantized KV, repeats GQA heads and retains official dequantization arithmetic; PQ native prefill reconstructs packed KV and uses 8 KV heads. These are whole-backend measurements, not an isolated comparison of rotation cost or Table1 g128 configurations. The FlatQuant cleanup removes only a random allocation immediately overwritten by a saved matrix; both versions remain reported.

The complete five-row same-GPU comparison is still pending.

Sources and exclusivity evidence: [collection.json](collection.json). The completed original-versus-optimized PQ experiment used the earlier reused-cache boundary and is retained [separately](../../records/e38/rotation_optimization/table.md); its speedup is not substituted for this fresh-cache comparison.
