"""Execute the saved E14 rotations with CUDA block FWHT and fixed gather maps.

The source weights, reflectors, compact-WY factors, signs and permutation are
unchanged. All rotation arithmetic remains FP32 until the original FP16 output
boundary. This module never calibrates or substitutes random rotation factors.
"""
from __future__ import annotations

import math
import types

import torch
from torch import nn


def cuda_hadamard(value, *, scale):
    from fast_hadamard_transform import hadamard_transform
    return hadamard_transform(value, scale=scale)


class _FastFixedRotation(nn.Module):
    def __init__(self, factor, signs, *, reference, hadamard_fn=None, audit=None, site=None):
        super().__init__()
        self.n, self.b = factor.n, factor.b
        if self.n % self.b or self.b < 1 or self.b & (self.b-1):
            raise ValueError('Saved factor requires a power-of-two block dividing its width')
        # Original scatter: output[:, target_order] = input[:, source_order].
        gather = torch.empty_like(factor.source_order)
        gather[factor.target_order] = factor.source_order
        self.register_buffer('gather', gather)
        self.register_buffer('signs', signs)
        self.reference = reference
        self.hadamard_fn = hadamard_fn or cuda_hadamard
        self.audit, self.site = audit, site
        self.probed = False

    def apply_g(self, value):
        raise NotImplementedError

    def transform(self, value):
        shape = value.shape
        rows = self.apply_g(value).reshape(-1, self.n)
        signed = rows.index_select(-1, self.gather) * self.signs
        blocks = signed.reshape(-1, self.b).contiguous()
        return self.hadamard_fn(blocks, scale=1/math.sqrt(self.b)).reshape(shape)

    def forward(self, value):
        if self.audit is not None and not self.probed:
            self.probed = True
            # Once per site, during the first quality forward, never in timing.
            probe = value.detach().reshape(-1, self.n)[:8]
            with torch.inference_mode():
                reference = self.reference(probe, self.signs)
                candidate = self.transform(probe)
                error = candidate-reference
                rounded_error = candidate.to(value.dtype).float()-reference.to(value.dtype).float()
                self.audit.append(dict(site=self.site, rows=probe.shape[0], width=self.n,
                    input_source='first actual quality/evaluation forward, first up to8 rows',
                    input_dtype=str(value.dtype), rotation_dtype='torch.float32',
                    max_abs_fp32_difference=float(error.abs().amax()),
                    relative_l2_fp32_difference=float(error.norm()/reference.norm().clamp_min(1e-30)),
                    max_abs_difference_after_original_output_cast=float(rounded_error.abs().amax()),
                    output_cast_changed_elements=int((rounded_error!=0).sum()),
                    compared_elements=error.numel(), numerical_execution_gate=False))
        return self.transform(value)


class FastOriginalE14R4(_FastFixedRotation):
    """The original FP32 compact-WY GEMMs followed by cached gather and CUDA H_b."""
    def __init__(self, wy, signs, **kwargs):
        super().__init__(wy.factor, signs, reference=wy.apply, **kwargs)
        self.register_buffer('w', wy.w)
        self.register_buffer('y', wy.y)

    def apply_g(self, value):
        rows = value.float().reshape(-1, self.n)
        projected = rows @ self.w
        return torch.addmm(rows, projected, self.y.T, beta=1., alpha=-1.)

    def forward(self, value):
        # Match OriginalE14R4.forward's existing output cast exactly.
        return super().forward(value).to(value.dtype)


class FastOriginalE14R2(_FastFixedRotation):
    """Original ordered reflectors without per-call CUDA scalar synchronizations."""
    def __init__(self, factor, signs, **kwargs):
        super().__init__(factor, signs, reference=factor.apply, **kwargs)
        self.register_buffer('reflectors', factor.reflectors)
        self.active_indices = tuple(index for index, active in enumerate(factor.active.detach().cpu().tolist()) if active)

    def apply_g(self, value):
        output = value.float().reshape(-1, self.n)
        for index in self.active_indices:
            vector = self.reflectors[index]
            output = output - 2*(output @ vector).unsqueeze(-1)*vector
        return output


def install_fast_rotations(model, record):
    """Opt-in replacement after original construction; packed exports stay valid."""
    rotations = model._stage_a_rotations
    if rotations.method != 'nar_k8':
        raise ValueError('Fast saved-factor backend currently supports the PrismQuant k=8 row')
    if hasattr(model, '_stage_a_fast_r2'):
        raise RuntimeError('Fast rotation backend was already installed')
    audit = []
    r2_modules = nn.ModuleDict()
    layers = []
    for index, block in enumerate(model.model.layers):
        wy = rotations.r4_wy[index]
        signs = rotations.signs('r4', index, wy.factor.n)
        previous = block.mlp.down_proj[0]
        if type(previous).__name__ != 'OriginalE14R4':
            raise TypeError(f'Unexpected trained-model R4 at layer{index}: {type(previous).__name__}')
        block.mlp.down_proj[0] = FastOriginalE14R4(wy, signs, audit=audit, site=f'layer{index}.r4').eval()
        r2 = rotations.r2[index]
        r2_modules[str(index)] = FastOriginalE14R2(r2, rotations.signs('r2', index, r2.n),
            audit=audit, site=f'layer{index}.r2')
        layers.append(dict(layer=index, r4_width=wy.factor.n, r4_block=wy.factor.b,
            r4_rank=wy.w.shape[1], r2_active_reflectors=len(r2_modules[str(index)].active_indices)))
    # Existing v_proj hooks call this same RotationSet instance. Preserve R1 and
    # other sites; only the actual R2 call is redirected to the equivalent module.
    original_apply = rotations.apply
    def apply(self, label, layer, value):
        if label == 'r2':
            return r2_modules[str(layer)](value)
        return original_apply(label, layer, value)
    rotations.apply = types.MethodType(apply, rotations)
    model._stage_a_fast_r2 = r2_modules.eval()
    record['rotation_optimization'] = dict(backend='fast',
        source_factors='unchanged saved E14 R2/R4 factors, signs and permutations',
        weights='unchanged existing packed checkpoint; no export or GPTQ recalibration',
        r4='same FP32 compact-WY GEMMs; precomputed gather; CUDA normalized H128',
        r2='same ordered FP32 reflectors; active-index list fixed once on CPU; precomputed gather; CUDA normalized H128',
        numerical_note='CUDA butterfly summation/normalization can change rounding; measured, never threshold-gated',
        layers=layers, actual_input_probes=audit,
        quality_protocol='Full141 PPL on this optimized model before timing; no numerical execution gate')
    return record['rotation_optimization']
