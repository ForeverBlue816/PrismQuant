"""INT4 reference quantizers with the validated fp16 metadata rules."""
from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable
import torch

QMAX = 15
GROUP = 128
K_TOKEN_GROUP = 32
KV_RESIDUAL_LENGTH = 32

def group_view(x: torch.Tensor, group_size: int) -> torch.Tensor:
    if x.shape[-1] % group_size:
        raise ValueError(f"last dimension {x.shape[-1]} is not divisible by group size {group_size}")
    return x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)


def dynamic_asym_int4(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake-quantize with one fp16 scale and fp16 real-valued zero/offset per group.

    q = clamp(round((x - z) / s), 0, 15), x_hat = q*s + z.
    Both s and z are rounded to IEEE fp16 before they are used for q and x_hat.
    Degenerate groups use s=1 and q=0, exactly reproducing their fp16 offset.
    """
    original_dtype = x.dtype
    xg = group_view(x.float(), group_size)
    lo = xg.amin(dim=-1, keepdim=True)
    hi = xg.amax(dim=-1, keepdim=True)
    raw_scale = (hi - lo) / QMAX
    scale16 = torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale)).to(torch.float16)
    # The fp32 test above cannot see a raw scale that is positive in fp32 but
    # rounds to zero in fp16, which happens once (hi - lo)/QMAX falls below half
    # the fp16 subnormal floor.  Such a group is degenerate by the same
    # definition and must take the same s=1, q=0 path; without this the division
    # below is by zero.  Where the group minimum is exactly representable in
    # fp16 that division is 0/0, which produces a NaN that clamp does not
    # remove and that then destroys every downstream layer.
    scale16 = torch.where(scale16 > 0, scale16, torch.ones_like(scale16))
    zero16 = lo.to(torch.float16)
    scale = scale16.float()
    zero = zero16.float()
    q = torch.round((xg - zero) / scale).clamp_(0, QMAX)
    deq = q * scale + zero
    return deq.reshape_as(x).to(original_dtype), scale16.squeeze(-1), zero16.squeeze(-1), q.to(torch.uint8)
def _symmetric_per_group_int4(value: torch.Tensor, group: int) -> torch.Tensor:
    """TwinQuant's activation quantizer (arXiv 2606.01556 Eq. 1 at group 128):
    one fp16 scale max|x|/7 per contiguous group, symmetric int4 [-8, 7]."""
    original_dtype = value.dtype
    groups = group_view(value.float(), group)
    scale = (groups.abs().amax(-1, keepdim=True) / 7).clamp_min(torch.finfo(torch.float16).tiny).to(torch.float16)
    dequant = torch.round(groups / scale.float()).clamp_(-8, 7) * scale.float()
    return dequant.reshape_as(value).to(original_dtype)


def _symmetric_per_token_int4(value: torch.Tensor) -> torch.Tensor:
    original_dtype = value.dtype
    rows = value.float().reshape(-1, value.shape[-1])
    scale = (rows.abs().amax(-1, keepdim=True) / 7).clamp_min(torch.finfo(torch.float16).tiny).to(torch.float16)
    dequant = torch.round(rows / scale.float()).clamp_(-8, 7) * scale.float()
    return dequant.reshape_as(value).to(original_dtype)


def _fill_padded_tokens(key: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Replace every padded token's key with its nearest valid token's.

    Batched generation pads on the left and batched log-likelihood on the
    right. A padded position is masked out of attention, but its key would
    still enter the per-channel min/max of the 32-token chunk it shares with
    real tokens and widen their grid. Copying a neighbouring real key in its
    place keeps the chunk statistics those of real tokens only; the padded
    positions' own quantized values are never attended to.
    """
    batch, length = valid.shape
    positions = torch.arange(length, device=key.device)
    last_valid = torch.cummax(torch.where(valid, positions, torch.full_like(positions, -1)), dim=-1).values
    first_valid = valid.to(torch.int8).argmax(dim=-1)
    source = torch.where(last_valid < 0, first_valid.unsqueeze(-1), last_valid)
    index = source.view(batch, 1, length, 1).expand(batch, key.shape[1], length, key.shape[-1])
    return key.gather(-2, index)


def _key_validity(attention_mask: torch.Tensor | None, batch: int, kv_length: int) -> torch.Tensor | None:
    """[batch, kv_length] bool, or None when nothing is padded."""
    if attention_mask is None:
        return None
    if attention_mask.dim() == 4:
        row = attention_mask[:, 0, -1, :kv_length]
    elif attention_mask.dim() == 2:
        row = attention_mask[:, :kv_length]
    else:
        return None
    if row.shape[0] != batch or row.shape[-1] != kv_length:
        return None
    valid = row if row.dtype == torch.bool else row > -1.0
    return None if bool(valid.all()) else valid


def _kivi_key_qdq(key: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Quantize completed KIVI residual chunks; keep the newest chunk bf16."""
    length = key.shape[-2]
    prefix = (max(0, length - 1) // KV_RESIDUAL_LENGTH) * KV_RESIDUAL_LENGTH
    if prefix == 0:
        return key
    source = key if valid is None else _fill_padded_tokens(key, valid)
    transposed = source[..., :prefix, :].transpose(-1, -2).contiguous()
    quantized, _, _, _ = dynamic_asym_int4(transposed, K_TOKEN_GROUP)
    output = key.clone()
    output[..., :prefix, :] = quantized.transpose(-1, -2)
    return output


def _residual_masks(q_length: int, kv_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return KIVI's sawtooth K and sliding V full-precision masks."""
    query_position = torch.arange(
        kv_length - q_length, kv_length, device=device, dtype=torch.long
    ).unsqueeze(1)
    key_position = torch.arange(kv_length, device=device, dtype=torch.long).unsqueeze(0)
    causal = key_position <= query_position
    key_chunk_start = torch.div(
        query_position, KV_RESIDUAL_LENGTH, rounding_mode="floor"
    ) * KV_RESIDUAL_LENGTH
    key_full = causal & (key_position >= key_chunk_start)
    value_full = causal & (key_position > query_position - KV_RESIDUAL_LENGTH)
    return key_full.unsqueeze(0).unsqueeze(0), value_full.unsqueeze(0).unsqueeze(0)
