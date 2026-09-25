# Technical guide

PrismQuant uses a calibrated orthogonal transform `R = H_g D Pi G`. The code
accepts row-vector batches; `RotationFactor.apply(x, signs)` computes the
corresponding rotated rows and `transpose` applies the true inverse.

- `rotations.py`: sequential Householder maps, energy-balanced coordinate
  assignment, normalized Walsh/Paley transforms, factor serialization and WY.
- `calibration.py`: uncentered second-moment directions for a user-supplied
  activation matrix. `rank <= channels / group_size`; the group size is a power
  of two. The exact reference fitter forms an O(channels²) fp64 matrix.
- `quantization.py`: group INT4 QDQ and KIVI residual/padding helpers.
- `gptq.py`: the calibrated GPTQ weight quantizer adapted from QuaRot.
- `folding.py`, `model.py`, `runtime.py`: offline weight folds and online hooks
  for the released dense Llama/Qwen3 checkpoints.
- `fused.py`, `kernels/`: signed-permutation folding and fused R4 INT4 kernels.

## Affine metadata

For each group, `z = fp16(min(x))` and `s = fp16((max(x)-min(x))/15)` with the
original zero/underflow guards. Codes are computed using those rounded metadata
values. The quantizer returns reconstructed values, fp16 scale, fp16 offset and
uint8 codes; those codes are unpacked in this reference API.

At group size 128, packed INT4 values plus two fp16 metadata values account for
4.25 bits/value. This is a representation calculation, not the memory footprint
of the floating-point reference loader. Aligning a direction with group DC can
let the affine offset represent its common level while localization and flatness
reduce its range cost. The offset itself has finite precision: E36 measured small
rounding errors and rare larger tails, so the implementation does not claim
mathematically lossless offset storage.

## Orthogonality and deployment

`compact_wy` preserves the original row convention: `x - (x @ W) @ Y.T`.
The inverse reverses reflector order, undoes the permutation/signs and applies
the block transform. Do not assume every full Paley construction is symmetric.
The model loader copies calibrated factors to each device that uses them.

The signed permutation can be folded into SwiGLU gate/up weights using
`fold_swiglu_weights`; the remaining factors are constructed by `FoldedR4`.
See [kernels.md](kernels.md) for the separate packed activation path.

The public release preserves the numerical core from the
[complete research archive](https://github.com/ForeverBlue816/PrismQuant/tree/research-archive-2026-09-25).
That archive contains the full calibration, GPTQ, evaluation and ablation
commands, frozen data choices and measured limitations.
