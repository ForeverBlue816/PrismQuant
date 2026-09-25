# Release validation

The public package is extracted from the validated research implementation at
`728e6e5`. Algebra and quantizer routines preserve its conventions; the new Hub
interface replaces cluster-specific orchestration.

The release checks cover:

- 21 CPU tests: inverse rotations, WY equivalence, Hadamard energy preservation,
  metadata precision/underflow, signed-permutation SwiGLU folding, full versus
  selective weight restoration, both Llama and Qwen3 tiny decoders in fp32/bf16,
  padded cached generation, hook cleanup and incomplete-download detection.
- Building a wheel and importing its model catalog, loader and optional kernel
  module from an installation outside the source checkout.
- All 37 Hub variants: complete decoder-layer and required-factor inventories,
  with immutable artifact and base-model revisions in `models.json`.
- Author figures: ten original PDFs preserved byte for byte, with companion
  previews rendered directly from those PDFs.

Detailed real-model parity and public-download checks are recorded in
[`release_validation.json`](release_validation.json).

The reference quantizers can produce different greedy continuations when
`use_cache` changes: this occurred on the Qwen and Llama random-token probes in both the
original and extracted runtime. The release test compares each cache mode with
the same mode in the original implementation; it does not claim cache-mode
invariance. Preserve the generation/evaluation settings when reproducing results.

The full numerical experiments, including negative findings and deployment
limitations, remain in the [research archive](reproduction.md).
