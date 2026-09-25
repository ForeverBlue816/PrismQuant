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
Figure 5 comes from the dedicated packed deployment implementation and matched
benchmark setup, preserved in the
[deployment source archive](https://github.com/ForeverBlue816/PrismQuant/tree/research-archive-2026-09-25/quarot-llama3)
and [deployment report](https://github.com/ForeverBlue816/PrismQuant/blob/research-archive-2026-09-25/report_e28_v2.md).
Consult those records before comparing speed or peak memory.
