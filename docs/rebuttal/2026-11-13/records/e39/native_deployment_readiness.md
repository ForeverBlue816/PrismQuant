> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

**FlatQuant — Meta-Llama-3.1-8B, seed0; official W4 RTN / per-token symmetric A4 / per-head asymmetric KV4 accuracy recipe**

The accuracy worker follows the official 15-epoch recipe with calibration batch size4, learning rate0.005, learned weight/activation clipping, learned transformations and diagonal scaling. The worker saves the quantized model, weight quantizers, learned matrices and calibration tokens, and records full141 sequence NLLs; it is ready for the already-submitted accuracy job and does not depend on the deployment build.

The official repository provides real-weight export and deployment in `flatquant/flat_utils.py`, `deploy/transformers/modeling_llama.py`, and `benchmarks/benchmark_model.py`. The current checkout has no compiled `deploy._CUDA`, its three build submodules are empty, and the selected main Python environment lacks FlashAttention and fast-hadamard-transform; no `nvcc` was found on PATH or under CUDA_HOME, so native INT4 timing is not ready. [CPU preflight record](raw/native_deploy_preflight_20261112.json).

The pinned source stores asymmetric per-head KV4, although [REALQUANT.md](https://github.com/ruikangliu/FlatQuant/blob/main/REALQUANT.md) contains a broader symmetric-only warning. During prefill the native cache returns unquantized transformed K/V for attention, omits learned KV clipping during packing, and repeats KV heads to the attention-head count for cache storage; these are material differences from the accuracy path and must be labelled rather than silently called the same deployed model.

After the native dependencies are built, the existing trained checkpoint can be exported without another calibration. The official deployment loader uses `strict=False` and does not act on its missing-key set, so complete used-parameter loading must be checked before measurements; native full141 PPL and timing then run on the same loaded FlatQuant model, with GPU exclusivity and cache reset between repetitions.

Only FlatQuant is in this deployment follow-up: no FP16 control, PrismQuant run, random-weight benchmark, or fake-quant latency presented as native INT4 performance. No native timing job has been submitted, and no latency numbers are claimed.
