# Release validation

The public package is extracted from the validated research implementation at
`728e6e5`. Algebra and quantizer routines preserve its conventions; the new Hub
interface replaces cluster-specific orchestration.

The release checks cover:

- 23 CPU tests: inverse rotations, WY equivalence, Hadamard energy preservation,
  metadata precision/underflow, signed-permutation SwiGLU folding, full versus
  selective weight restoration, both Llama and Qwen3 tiny decoders in fp32/bf16,
  padded cached generation, hook cleanup and incomplete-download detection.
  MoE checks additionally cover router logits, per-expert folds, unrotated-model
  parity, cached generation and restoring expert forwards after cleanup.
- Building a wheel and importing its model catalog, loader and optional kernel
  module from an installation outside the source checkout.
- All 38 Hub variants: complete decoder-layer and required-factor inventories,
  with immutable artifact and base-model revisions in `models.json`.
- Author figures: ten original PDFs preserved byte for byte, with companion
  previews rendered directly from those PDFs.

The original dense-model parity checks are recorded in
[`release_validation.json`](release_validation.json). The paper and MoE release
checks are recorded in [`hub_release_validation.json`](hub_release_validation.json):
renamed repositories preserve the weight hashes and old-link redirects, and the
MoE checkpoint is checked against the original research runtime on two GPUs.
The Hub manifest is validated during downloads and provides Hugging Face's
standard configuration-file download counter.

The reference quantizers can produce different greedy continuations when
`use_cache` changes: this occurred on the Qwen and Llama random-token probes in both the
original and extracted runtime. The release test compares each cache mode with
the same mode in the original implementation; it does not claim cache-mode
invariance. Preserve the generation/evaluation settings when reproducing results.

The full numerical experiments, including negative findings and deployment
limitations, remain in the [research archive](reproduction.md).
