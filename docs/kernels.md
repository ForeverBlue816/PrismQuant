# Fused R4 kernels

`prismquant.kernels.r4_fused_v3` contains the tensor-core projection and group
transform/quantize/pack kernels. It shares the validated quantization and packing
helpers from `r4_fused_v2`. These require a compatible NVIDIA CUDA GPU, PyTorch
and Triton; install the optional dependencies with `pip install -e '.[kernels]'`.

The sequence is: fold the signed permutation into gate/up weights, precompute
compact-WY factors, project `U = X A`, then form `X H_block - U B` and quantize
into packed INT4 groups. `FoldedR4` in `prismquant/fused.py` builds the factor
layout. `PackedInt4`, `allocate_outputs`, `launch_nar`, `launch_hadamard`, `unpack`
and `dequantize` are retained as the low-level entry points. The projection
requires at least 16 padded rank columns; tested paper operating points are
`k=8` and `k=32` with groups of 128.

These kernels are not automatically substituted into `load_model`. The Hub
checkpoints and public loader use the numerically matched reference path.
Figure 5 reports a separate packed deployment setup described in
[the paper, Appendix D](https://arxiv.org/html/2609.32429v1#A4).
The low-level kernels here are building blocks; the public package does not
include a complete packed serving engine or the end-to-end timing harness.
Use the paper's matched precision, hardware, batch, context and execution-mode
settings when interpreting those throughput and memory measurements.
