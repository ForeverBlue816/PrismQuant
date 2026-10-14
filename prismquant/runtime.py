"""Online rotations, group activation quantization and KIVI-style QDQ.

This reference runtime retains floating-point weights/cache storage.
"""
from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable
import torch

from . import quantization as base
from .quantization import (GROUP, _symmetric_per_group_int4, _symmetric_per_token_int4, _fill_padded_tokens, _key_validity, _kivi_key_qdq, _residual_masks)

class RuntimeHooks:
    def __init__(self, model: torch.nn.Module, rotations: Any,
                 activation_kind: str | None, quantize_kv: bool = True):
        self.model = model
        self.rotations = rotations
        self.activation_kind = activation_kind
        self.quantize_kv = quantize_kv
        self.handles: list[Any] = []
        self.previous_attention = model.config._attn_implementation
        self.attention_key = f"prismquant_{id(self)}"

    def rotate_down(self, layer: int) -> Callable[..., tuple[torch.Tensor]]:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[torch.Tensor]:
            return (self.rotations.apply("r4", layer, inputs[0]).to(inputs[0].dtype),)
        return hook

    def rotate_v(self, layer: int) -> Callable[..., torch.Tensor]:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            shape = output.shape
            return self.rotations.apply("r2", layer, output.reshape(-1, self.rotations.head_dim)).reshape(shape).to(output.dtype)
        return hook

    def rotate_o(self) -> Callable[..., tuple[torch.Tensor, ...]]:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
            rotated = self.rotations.apply_r3(inputs[0]).to(inputs[0].dtype)
            return (rotated,) + inputs[1:]
        return hook

    def quantize_input(self, _module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
        value = inputs[0]
        if self.activation_kind == "quarot_symmetric_token":
            quantized = _symmetric_per_token_int4(value)
        elif self.activation_kind == "asymmetric_g128":
            quantized, _, _, _ = base.dynamic_asym_int4(value, GROUP)
        elif self.activation_kind == "symmetric_g128":
            quantized = _symmetric_per_group_int4(value, GROUP)
        else:
            return inputs
        return (quantized.to(value.dtype),) + inputs[1:]

    def attention(self, module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor,
                  value: torch.Tensor, attention_mask: torch.Tensor | None, **kwargs: Any) -> tuple[torch.Tensor, None]:
        from transformers.models.llama.modeling_llama import repeat_kv

        scaling = kwargs.get("scaling", getattr(module, "scaling", self.rotations.head_dim ** -0.5))
        key_full_mask, value_full_mask = _residual_masks(
            query.shape[-2], key.shape[-2], query.device
        )
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            # A boolean mask (True = attend) arrives for some attention
            # interfaces; the additive form below is what this function adds.
            attention_mask = torch.zeros(attention_mask.shape, dtype=query.dtype, device=query.device
                                         ).masked_fill_(~attention_mask, torch.finfo(query.dtype).min)
        quantized_key = _kivi_key_qdq(key, _key_validity(attention_mask, key.shape[0], key.shape[-2]))
        quantized_value, _, _, _ = base.dynamic_asym_int4(value, self.rotations.head_dim)
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
        quantized_key = repeat_kv(quantized_key, module.num_key_value_groups)
        quantized_value = repeat_kv(quantized_value, module.num_key_value_groups)

        # KIVI quantizes a completed R-token K residual chunk at once, so the
        # number of recent bf16 K tokens cycles from 1..R.
        weights = torch.matmul(query, quantized_key.transpose(-1, -2))
        correction = torch.matmul(query, (key - quantized_key).transpose(-1, -2))
        weights.add_(correction.masked_fill_(~key_full_mask, 0)).mul_(scaling)
        causal = torch.arange(key.shape[-2], device=query.device).unsqueeze(0) <= torch.arange(
            key.shape[-2] - query.shape[-2], key.shape[-2], device=query.device
        ).unsqueeze(1)
        weights.masked_fill_(~causal.unsqueeze(0).unsqueeze(0), torch.finfo(weights.dtype).min)
        if attention_mask is not None:
            weights.add_(attention_mask[..., : key.shape[-2]])
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)

        # V keeps the most recent R tokens bf16 and quantizes older V per token.
        output = torch.matmul(weights, quantized_value)
        recent_weights = weights.masked_fill(~value_full_mask, 0)
        output.add_(torch.matmul(recent_weights, value - quantized_value))
        return output.transpose(1, 2).contiguous(), None

    def install(self) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        if self.quantize_kv:
            ALL_ATTENTION_FUNCTIONS.register(self.attention_key, self.attention)
            # transformers builds the attention mask per implementation name.
            # An unregistered name receives no mask at all, so a left-padded
            # generation batch reached this hook with its padding unmasked
            # (batch-8 GSM8K 51.0 against batch-1 83.0 on Qwen3-8B). The
            # eager mask is the additive 4-D form this hook adds; for an
            # unpadded batch it is the causal mask this hook already applies,
            # and min + min = -inf rounds to the same zero weight.
            try:
                from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
                ALL_MASK_ATTENTION_FUNCTIONS.register(
                    self.attention_key, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
            except (ImportError, KeyError):
                pass
            self.model.config._attn_implementation = self.attention_key
        for layer, block in enumerate(self.model.model.layers):
            self.handles.append(block.self_attn.v_proj.register_forward_hook(self.rotate_v(layer)))
            if self.rotations.method == "hadamard":
                self.handles.append(block.self_attn.o_proj.register_forward_pre_hook(self.rotate_o()))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self.rotate_down(layer)))
            if self.activation_kind is not None:
                for module in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj,
                               block.self_attn.o_proj, block.mlp.gate_proj, block.mlp.up_proj,
                               block.mlp.down_proj):
                    self.handles.append(module.register_forward_pre_hook(self.quantize_input))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.model.config._attn_implementation = self.previous_attention
        # register() stores a bound method globally; release its model reference.
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        getattr(ALL_ATTENTION_FUNCTIONS, "_global_mapping", {}).pop(self.attention_key, None)
        try:
            from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
            getattr(ALL_MASK_ATTENTION_FUNCTIONS, "_global_mapping", {}).pop(self.attention_key, None)
        except ImportError:
            pass
