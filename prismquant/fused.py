"""Fold signed permutations into SwiGLU weights before the fused R4 kernels."""
from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable
import torch

from . import rotations as act
from .rotations import compact_wy, _fast_walsh_hadamard

def source_for_target(factor: act.RotationFactor) -> torch.Tensor:
    """Return p with ``(P x)[target] = x[p[target]]``."""
    result = torch.empty_like(factor.source_order)
    result[factor.target_order] = factor.source_order
    return result


def signed_permute_rows(value: torch.Tensor, source: torch.Tensor,
                        signs: torch.Tensor) -> torch.Tensor:
    """Apply Q=SP to a vector or to the rows of a tall matrix."""
    selected = value.index_select(0, source.to(value.device))
    shape = (signs.numel(),) + (1,) * (selected.ndim - 1)
    return selected * signs.to(value.device, value.dtype).reshape(shape)


def block_hadamard_columns(value: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Left-multiply every column by block-diagonal normalized H_group_size."""
    if value.shape[0] % group_size:
        raise ValueError((value.shape, group_size))
    transposed = value.T.contiguous().reshape(-1, value.shape[0] // group_size, group_size)
    return _fast_walsh_hadamard(transposed).reshape(value.shape[1], value.shape[0]).T


@dataclass
class FoldedR4:
    """Online H G' factors after Q=SP has been folded into gate/up weights."""

    n: int
    group_size: int
    rank: int
    source: torch.Tensor
    signs: torch.Tensor
    y_prime_fp32: torch.Tensor
    y_prime_bf16: torch.Tensor
    y_prime_t_fp32: torch.Tensor
    y_prime_lo_bf16: torch.Tensor
    y_prime_pad_bf16: torch.Tensor
    y_prime_pad_lo_bf16: torch.Tensor
    y_prime_third_bf16: torch.Tensor
    y_prime_pad_terms_bf16: torch.Tensor
    w_h_fp32: torch.Tensor
    w_h_t_fp32: torch.Tensor

    @classmethod
    def from_factor(cls, factor: act.RotationFactor, signs: torch.Tensor) -> "FoldedR4":
        if factor.b != 128:
            raise ValueError(f"Fused R4 requires group 128, got {factor.b}")
        source = source_for_target(factor)
        wy_w, wy_y = compact_wy(factor.reflectors, factor.active)
        # Column convention: G=I-WY^T has W=wy_y and Y=wy_w.
        y_prime = signed_permute_rows(wy_w, source, signs).float()
        w_prime = signed_permute_rows(wy_y, source, signs).float()
        w_h = block_hadamard_columns(w_prime, factor.b).float().contiguous()
        # tl.dot needs at least 16 columns, so Y' is zero-padded to KP; the
        # kernel stores only the first k, leaving the partial layout unchanged.
        rank = y_prime.shape[1]
        padded = max(16, 1 << (rank - 1).bit_length())
        y_bf16 = y_prime.to(torch.bfloat16)
        y_lo = (y_prime - y_bf16.float()).to(torch.bfloat16)
        pad = torch.zeros((y_prime.shape[0], padded), dtype=torch.bfloat16, device=y_prime.device)
        pad_lo = torch.zeros_like(pad)
        pad[:, :rank] = y_bf16
        pad_lo[:, :rank] = y_lo
        # Three bf16 terms recover ~24 mantissa bits for Y'; k=32 needs the
        # third to hold the fp16 zero-point inside one ULP.
        residual2 = (y_prime - y_bf16.float() - y_lo.float()).to(torch.bfloat16)
        pad_third = torch.zeros_like(pad)
        pad_third[:, :rank] = residual2
        terms = torch.cat((pad, pad_lo, pad_third), dim=0).contiguous()
        # The Triton kernels index both factors as (rank, channel) so that the
        # 128 channels of one group are contiguous under a single rank.
        return cls(
            n=factor.n,
            group_size=factor.b,
            rank=int(factor.active.sum()),
            source=source,
            signs=signs.float(),
            y_prime_fp32=y_prime,
            y_prime_bf16=y_prime.to(torch.bfloat16),
            y_prime_t_fp32=y_prime.T.contiguous(),
            y_prime_lo_bf16=y_lo,
            y_prime_pad_bf16=pad.contiguous(),
            y_prime_pad_lo_bf16=pad_lo.contiguous(),
            y_prime_third_bf16=residual2,
            y_prime_pad_terms_bf16=terms,
            w_h_fp32=w_h,
            w_h_t_fp32=w_h.T.contiguous(),
        )

    def q_unfolded(self, value: torch.Tensor) -> torch.Tensor:
        """Reference Qx used to validate the offline gate/up fold."""
        rows = value.float().reshape(-1, self.n)
        return signed_permute_rows(rows.T, self.source, self.signs).T.reshape_as(value)

    def apply(self, x_permuted: torch.Tensor) -> torch.Tensor:
        """Reference row form of H G' x_permuted."""
        shape = x_permuted.shape
        rows = x_permuted.float().reshape(-1, self.n)
        u = rows @ self.y_prime_fp32
        h = _fast_walsh_hadamard(
            rows.reshape(-1, self.n // self.group_size, self.group_size)
        ).reshape_as(rows)
        return (h - u @ self.w_h_fp32.T).reshape(shape)

    def save(self, path: Path, extra: dict | None = None) -> None:
        payload = {
            "n": self.n,
            "group_size": self.group_size,
            "rank": self.rank,
            "source_for_target": self.source.cpu(),
            "signs": self.signs.cpu(),
            "y_prime_fp32": self.y_prime_fp32.cpu(),
            "y_prime_bf16": self.y_prime_bf16.cpu(),
            "y_prime_t_fp32": self.y_prime_t_fp32.cpu(),
            "y_prime_lo_bf16": self.y_prime_lo_bf16.cpu(),
            "y_prime_pad_bf16": self.y_prime_pad_bf16.cpu(),
            "y_prime_pad_lo_bf16": self.y_prime_pad_lo_bf16.cpu(),
            "y_prime_third_bf16": self.y_prime_third_bf16.cpu(),
            "w_h_fp32": self.w_h_fp32.cpu(),
            "w_h_t_fp32": self.w_h_t_fp32.cpu(),
            "compact_wy_convention": "repository row form x-(x@wy_w)@wy_y.T; stored Y'=Q*wy_w and W''=H*Q*wy_y",
        }
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)


@torch.no_grad()
def fold_swiglu_weights(gate_proj: torch.nn.Linear, up_proj: torch.nn.Linear,
                        folded: FoldedR4) -> None:
    """Produce Qx by row-permuting gate and signed-row-permuting up offline."""
    if gate_proj.bias is not None or up_proj.bias is not None:
        raise AssertionError("signed-permutation fold requires bias-free gate_proj/up_proj")
    if gate_proj.weight.shape[0] != folded.n or up_proj.weight.shape[0] != folded.n:
        raise ValueError((gate_proj.weight.shape, up_proj.weight.shape, folded.n))
    source = folded.source.to(gate_proj.weight.device)
    signs = folded.signs.to(up_proj.weight.device, up_proj.weight.dtype).unsqueeze(1)
    gate_proj.weight.copy_(gate_proj.weight.detach().index_select(0, source))
    up_proj.weight.copy_(up_proj.weight.detach().index_select(0, source) * signs)
