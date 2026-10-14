# Validation scope

The reference package includes CPU tests for:

- Rotation inverses, compact-WY equivalence and Hadamard energy preservation.
- Quantizer metadata precision and underflow handling.
- SwiGLU folding and full versus selective weight restoration.
- Small Llama and Qwen3 decoders in fp32/bf16, padded cached generation and hook cleanup.
- MoE router preservation, expert folding and restoring expert forwards.
- Pinned checkpoint selection, metadata validation and incomplete-download detection.

Run the checks with:

```bash
pip install -e '.[dev]'
pytest -q
```

Tests use small models and temporary fixtures; they do not download or evaluate
the full released models. A passing test suite is not a reproduction of paper
benchmark scores. The optional Triton kernels require separate validation on
compatible CUDA hardware.

The model catalog pins artifact and base-model revisions, and the loader checks
required files and manifest consistency before loading. Model artifacts retain
their upstream licenses. See [model documentation](models.md).

The reference quantizers can produce different greedy continuations when
`use_cache` changes. Cache-mode invariance is not guaranteed; preserve the
same generation and evaluation settings when comparing results. The reference
loader uses floating-point storage and does not demonstrate packed INT4 memory
usage or serving throughput. See [reproduction guidance](reproduction.md) and
the [kernel guide](kernels.md) for the implementation scope.
