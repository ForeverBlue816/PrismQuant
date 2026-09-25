"""Offline weight folding for Llama and Qwen3 dense decoders."""
from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable
import torch

FOLD_DEVICE = None

def _transform_weight_rows(module: torch.nn.Linear, transform: Callable[[torch.Tensor], torch.Tensor],
                           row_batch: int) -> None:
    original = module.weight.detach()
    home = original.device
    chunks = []
    for start in range(0, original.shape[0], row_batch):
        chunk = original[start:start + row_batch].float()
        if FOLD_DEVICE is not None:
            chunk = chunk.to(FOLD_DEVICE)
        chunks.append(transform(chunk).to(original.dtype).to(home))
    module.weight.data.copy_(torch.cat(chunks, 0))


def _transform_weight_left(module: torch.nn.Linear, transform: Callable[[torch.Tensor], torch.Tensor],
                           row_batch: int) -> None:
    transposed = module.weight.detach().T
    home = transposed.device
    chunks = []
    for start in range(0, transposed.shape[0], row_batch):
        chunk = transposed[start:start + row_batch].float()
        if FOLD_DEVICE is not None:
            chunk = chunk.to(FOLD_DEVICE)
        chunks.append(transform(chunk).to(transposed.dtype).to(home))
    module.weight.data.copy_(torch.cat(chunks, 0).T)


@torch.inference_mode()
def fuse_norms_and_rotate(model: torch.nn.Module, rotations: Any,
                          row_batch: int) -> dict[str, Any]:
    """Apply QuaRot R1 plus R2/R4 folds with explicit orthogonal identities."""
    tied_embeddings = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    if tied_embeddings:
        old_head = model.lm_head
        new_head = torch.nn.Linear(old_head.in_features, old_head.out_features, bias=False,
                                   device=old_head.weight.device, dtype=old_head.weight.dtype)
        new_head.weight.copy_(old_head.weight)
        model.lm_head = new_head
        model.config.tie_word_embeddings = False
    for block in model.model.layers:
        input_scale = block.input_layernorm.weight.detach().float()
        post_scale = block.post_attention_layernorm.weight.detach().float()
        for module in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj):
            module.weight.mul_(input_scale.to(module.weight.dtype).unsqueeze(0))
        for module in (block.mlp.gate_proj, block.mlp.up_proj):
            module.weight.mul_(post_scale.to(module.weight.dtype).unsqueeze(0))
        block.input_layernorm.weight.fill_(1)
        block.post_attention_layernorm.weight.fill_(1)
    final_scale = model.model.norm.weight.detach().float()
    model.lm_head.weight.mul_(final_scale.to(model.lm_head.weight.dtype).unsqueeze(0))
    model.model.norm.weight.fill_(1)

    r1 = lambda value: rotations.apply("r1", 0, value)
    _transform_weight_rows(model.model.embed_tokens, r1, row_batch)
    _transform_weight_rows(model.lm_head, r1, row_batch)
    for block in model.model.layers:
        for module in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj,
                       block.mlp.gate_proj, block.mlp.up_proj):
            _transform_weight_rows(module, r1, row_batch)
        for module in (block.self_attn.o_proj, block.mlp.down_proj):
            _transform_weight_left(module, r1, row_batch)

    attention_heads = int(model.config.num_attention_heads)
    head_dim = rotations.head_dim
    for layer, block in enumerate(model.model.layers):
        _transform_weight_rows(
            block.mlp.down_proj, lambda value, layer=layer: rotations.apply("r4", layer, value), row_batch
        )
        weight = block.self_attn.o_proj.weight.detach()
        shaped = weight.reshape(weight.shape[0], attention_heads, head_dim)
        rotated = rotations.apply("r2", layer, shaped.reshape(-1, head_dim)).reshape_as(shaped)
        rotated = rotations.apply_r3(rotated.reshape_as(weight)).to(weight.dtype)
        weight.copy_(rotated.reshape_as(weight))
    return {
        "r1": "global residual rotation; input rows WQ, residual output Q^T W",
        "r2": "per-head V rotation with identical fold into every GQA-expanded o_proj head block",
        "r3": "QuaRot cross-head Hadamard at o_proj for Hadamard rows; omitted for the specified NAR R1/R2/R4 rows",
        "r4": "per-layer down-input rotation folded into down_proj rows",
        "norm": "all RMSNorm affine weights fused into consumer weights then set to one",
        "embedding_centering": False,
        "tied_embeddings_materialized_before_final_norm_fusion": tied_embeddings,
        "official_bug_avoided": "official head calls rotate_embeddings twice; E14 applies algebraic R1 once",
    }


def _linear_groups(layer: torch.nn.Module) -> list[list[tuple[str, torch.nn.Linear]]]:
    return [
        [("self_attn.k_proj", layer.self_attn.k_proj), ("self_attn.v_proj", layer.self_attn.v_proj),
         ("self_attn.q_proj", layer.self_attn.q_proj)],
        [("self_attn.o_proj", layer.self_attn.o_proj)],
        [("mlp.up_proj", layer.mlp.up_proj), ("mlp.gate_proj", layer.mlp.gate_proj)],
        [("mlp.down_proj", layer.mlp.down_proj)],
    ]


def _layer_state(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: module.weight.detach().cpu() for group in _linear_groups(layer) for name, module in group}


def _load_layer_state(layer: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    modules = {name: module for group in _linear_groups(layer) for name, module in group}
    if set(payload) != set(modules):
        raise RuntimeError(f"checkpoint schema mismatch: {path}")
    for name, module in modules.items():
        module.weight.data.copy_(payload[name].to(module.weight.dtype))
