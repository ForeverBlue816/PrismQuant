"""Standalone calibration/rotation/quantization; runs without model downloads."""
import torch
from prismquant import fit_rotation, dynamic_asym_int4

torch.manual_seed(0)
calibration = torch.randn(512, 256)
calibration[:, 0] *= 10
factor = fit_rotation(calibration, rank=2)
signs = torch.ones(256)
x = torch.randn(16, 256)
y = factor.apply(x, signs)
x_roundtrip = factor.transpose(y, signs)
print('Relative round-trip error:', float((x_roundtrip - x).norm() / x.norm()))
qdq, scale, offset, codes = dynamic_asym_int4(y, 128)
print('Quantized shape:', qdq.shape, 'Metadata:', scale.dtype, offset.dtype)
