**Meta-Llama-3.1-8B — seed0; FP16 versus PQ k=8 GPTQ W4 / asymmetric g128 A4 / simulated KV4; job 1736915**

Timing comparison: COMPLETE.

PQ prototype / FP16 speedup: prefill 0.02948×; decode 0.01227×. Values below1 mean the prototype is slower.

FP16: [stage_a_8b_fp16_seed0_g128comparison_job1736915.json](../../raw/stage_a_8b_fp16_seed0_g128comparison_job1736915.json); SHA256 d2bca2d3f985bcf6470b6780597befe2a0d9713f3ec44b7f2562c37432b15605; GPU GPU-6e3338c3-8b6c-64b7-549b-90febc02999e; torch=2.4.1+cu124; transformers=4.36.2; triton=3.0.0; CUDA=12.4; full141=True; same loaded object=True.

PrismQuant k=8 g128 integer prototype: [integer_8b_nar_k8_seed0_ppl-timing_job1736915.json](../../raw/integer_8b_nar_k8_seed0_ppl-timing_job1736915.json); SHA256 3c1ea5344ac0b3d8d0aca2eba9e16108bc82c615430e0ca766a6150bc56f8865; GPU GPU-6e3338c3-8b6c-64b7-549b-90febc02999e; torch=2.11.0+cu128; transformers=5.16.0.dev0; triton=3.6.0; CUDA=12.8; full141=True; same loaded object=True.

Candidate − FP16 full141 PPL: +0.750814. This compares the quantized model to FP16, not candidate/reference equivalence.

Target workload: B1, prefix2048, decode128 with first8 discarded, token100,10 warmups/20 repeats; synchronized eager execution with last-position logits and real incremental cache.

Fresh-cache boundary: prefill includes cache construction (and native FP16 cache clear); decode cache construction, prefix and discarded steps are outside its timer. Peak memory includes untimed decode preparation. Each row evaluates all141 WT2 windows before timing the same loaded model.

The g128 path expands stored signed4 weights to INT8 GEMM, uses INT32 partials and FP32 affine correction, and stores BF16 KV with attention-time KV4 simulation. It is not native packed INT4 compute or packed KV. FP16 uses its native paged KV and different attention implementation.

PyTorch and Transformers environments differ, as recorded below. Ratios compare complete backends and do not isolate format or rotation overhead. Missing or failed stages remain visible; no PPL tolerance or other numerical quality threshold controls timing.
