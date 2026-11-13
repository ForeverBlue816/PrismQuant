**Meta-Llama-3.1-8B — seed0, actual backend formats listed per row; exclusive A40 job 1736351**

| Method / backend | Actual configuration | Full141 PPL | Prefill ms | Decode tokens/s | Peak GiB | Status |
| --- | --- | --- | --- | --- | --- | --- |
| FlatQuant official | Packed W4 RTN / per-token A4; native KV4, unquantized prefill KV | 6.95298 | 302.482 | 9.741 | 6.244 | COMPLETE; timing COMPLETE |
| FlatQuant, unused allocation removed | Same packed parameters/arithmetic as official; dead random allocation removed | 6.95298 | 294.382 | 10.404 | 6.244 | COMPLETE; timing COMPLETE |
| PrismQuant k=8, optimized | Packed W4 GPTQ / per-token symmetric A4 / native per-head KV4 | 9.99501 | 377.209 | 16.752 | 7.791 | COMPLETE; timing COMPLETE |
| FlatQuant official fake-quant | BF16 floating W4 RTN / per-token A4 / per-head KV4 | 7.02086 | 1070.645 | 5.557 | 16.610 | COMPLETE; timing COMPLETE |
| DuQuant official fake-quant | FP16 floating official W4A4 / QK/V quantizers and floating KV cache | 8.19844 | 1163.505 | 4.845 | 16.402 | COMPLETE; timing COMPLETE |

B1, prefix2048, decode128/drop8, token100,10 warmups/20 repeats, eager synchronized wall time, last-position logits. Each timed object first evaluates all141 WikiText2 sequences. Fresh cache construction is included in prefill latency; decode prefix construction and first8 steps are outside its timer. Peak allocated memory includes the prefix.

Packed native rows and floating fake-quant rows are distinct execution modes. FlatQuant native prefill consumes transformed unquantized KV, repeats GQA heads and retains official dequantization arithmetic; PQ native prefill reconstructs packed KV and uses 8 KV heads. These are whole-backend measurements, not an isolated comparison of rotation cost or Table1 g128 configurations. The FlatQuant cleanup removes only a random allocation immediately overwritten by a saved matrix; both versions remain reported.

All five rows completed and were verified on the same physical exclusive GPU.

| PQ optimized native versus | Prefill latency ratio PQ/baseline (lower is faster) | Decode throughput ratio PQ/baseline (higher is faster) | PPL difference PQ minus baseline |
| --- | --- | --- | --- |
| FlatQuant official packed INT4 | 1.24705 | 1.71971 | 3.04203 |
| FlatQuant packed INT4, unused allocation removed | 1.28136 | 1.61014 | 3.04203 |

These ratios concern the native per-token symmetric PQ variant, whose measured PPL is 9.99501. They do not establish equivalent accuracy or performance of the asymmetric g128 reference format.
