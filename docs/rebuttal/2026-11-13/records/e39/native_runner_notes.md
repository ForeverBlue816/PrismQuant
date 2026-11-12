> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

# FlatQuant packed deployment — Meta-Llama-3.1-8B, W4 RTN / per-token symmetric A4 / native asymmetric KV4

The separate `nar/e39_flatquant_native.py` runner exports the completed seed-0 trained checkpoint using the pinned official exporter. It reuses saved RTN scales and learned transforms, with no calibration, optimization, or GPTQ rerun. Packed artifacts have source hashes and an identity manifest; the original accuracy checkpoint is unchanged.

The loader uses official key mapping and matrix sharing, then verifies every used parameter and packed projection buffer. It makes the official one-element-to-scalar clipping-buffer conversion explicit. Only documented inactive quantizer buffers, unused folded diagonal scales, and shared-matrix aliases are excluded from independent source coverage. All used loaded tensor representations must match the exported tensors; this is a provenance check, with no PPL or logit threshold.

Llama-3.1's scaled RoPE stays in the model configuration. The actual native model's shared rotary frequencies are checked against the configured Hugging Face `llama3` calculation. HF 4.45 supplies these shared position embeddings to the native attention implementation. The CPU integration test confirms this configuration and exact frequency match.

The native variant must have its own PPL: official prefill attention consumes transformed **unquantized** K/V, while decode uses a packed asymmetric INT4 cache. Cache packing does not apply the learned KV clipping factors; GQA KV heads are repeated to all query heads. Residual arithmetic is FP16 (the accuracy checkpoint used BF16). The pinned upstream CUDA dequantization first truncates the accumulator divided by 10 to an integer before its subsequent half conversion, scaling and restoration of the factor 10; this arithmetic is preserved. These facts are written into every native result and must remain visible when comparing methods.

`slurm_e39_native_8b.sh` requests one A40 and runs full frozen 141×2048 PPL before timing the **same loaded native object**. Timing is batch 1, prefix 2048, 128 cached decode steps with the first 8 discarded, token 100, 10 warmups and 20 repetitions. Only the final token's vocabulary logits are formed during timing. The common helper checks physical GPU exclusivity before every recorded repetition and reports memory peaks including prefix-cache construction. Nonfinite PPL is reported and does not prevent timing; runtime errors are recorded separately. No FP16 control or random-weight performance row is created.

CPU validation: five tests cover torch 2.11 whole-model serialization → torch 2.6 official packing/loading, every used packed tensor and scaled RoPE, rejection of a missing projection, artifact reuse/corruption checks, native result labeling, and exact tensor equality between mmap/inference-mode export and the unchanged official eager export. The test temporarily bypasses only GPU-availability checks during CPU construction and installs a Triton benchmark sentinel that raises if called. No GPU forward or timing occurs in these tests. The separate native runtime build agent has exercised extension imports and small kernels on A40; full real-checkpoint model PPL and timing remain pending.

Native 3B remains outside this runner: the official deployment documentation warns about non-power-of-two hidden dimensions, and Llama-3.2-3B's hidden size 3072 has not been validated for these kernels.

Validation command:

```bash
source /home/yanlongc/.prismquant/flatquant-native-tools/environment_torch26.sh
export PYTHONPATH="$PYTHONPATH:/home/yanlongc/.e13_packages:/home/yanlongc/prismquant-rebuttal-e39"
/projects/nar/nar-validation/e28-env/bin/python -m unittest discover -s tests -p test_e39_native_export.py -v
```

Host RAM audit (8B, not a measured full-model peak): the configured BF16 weight payload is 14.958 GiB; packed projections plus FP16 embedding/head/norms are 5.208 GiB. Their sum during export is 20.165 GiB, with the largest FP32 projection temporary 0.219 GiB (two such temporaries 0.438 GiB), plus small transforms/scales and library overhead. Checkpoint loading now uses CPU `mmap=True`, so clean source pages are reclaimable; the unchanged official exporter runs under `inference_mode`. The trained source is deleted and glibc freed pages are returned before native construction. Temporary constructor allocations, mapped packed tensors after verification, and CPU weights after GPU transfer are also explicitly released. Export and load records include RSS/high-water marks. The standalone request is 40 GiB; the payload analysis supports that budget, but actual full 8B RSS and Slurm-accounted file cache have not yet been measured because the completed checkpoint is not available.
