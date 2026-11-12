"""Narrow DuQuant TF4.35 -> TF4.45 adapter; retain Llama-3.1 RoPE math.

Likelihood and the official legacy-tuple cached decoding are supported through
a narrow HF DynamicCache bridge. Rotation, quantization and clipping stay official.
"""
from __future__ import annotations
import torch
from torch import nn

_ORIGINAL_DECODER_FORWARD = None


def prepare_official_catcher(model, device):
    """Move HF4.45's shared RoPE before the official first-layer catcher.

    DuQuant moves only embeddings and the first decoder to its GPU. Modern HF
    computes shared position embeddings before invoking that decoder, so its
    original, unchanged RoPE buffer must also be on the capture device.
    """
    rotary = getattr(model.model, 'rotary_emb', None)
    if rotary is None:
        return dict(shared_rotary='not present', device=str(device))
    before = str(rotary.inv_freq.device)
    rotary.to(device=device)
    return dict(shared_rotary='original HF module moved before official catcher',
        from_device=before, to_device=str(rotary.inv_freq.device),
        frequency_dtype=str(rotary.inv_freq.dtype), mathematical_change=False)


class LegacyRotary(nn.Module):
    """Expose an old full position table using the configured modern RoPE."""
    def __init__(self, original):
        super().__init__()
        self.original = original
        # Official qlayer.half() must not round Llama3.1 frequency buffers.
        # Reconstruct them from the unchanged HF RoPE configuration if a prior
        # dtype conversion already rounded the original buffer.
        if self.original.inv_freq.dtype != torch.float32:
            freq, scaling = self.original.rope_init_fn(self.original.config,
                self.original.inv_freq.device, **self.original.rope_kwargs)
            self.original.inv_freq = freq.float()
            self.original.original_inv_freq = self.original.inv_freq
            self.original.attention_scaling = scaling

    def _apply(self, function, recurse=True):
        frequency = self.original.inv_freq.detach().clone()
        super()._apply(function, recurse=recurse)
        self.original.inv_freq = frequency.to(device=self.original.inv_freq.device, dtype=torch.float32)
        self.original.original_inv_freq = self.original.inv_freq
        return self

    def forward(self, value, position_ids=None, seq_len=None):
        if seq_len is None:
            return self.original(value, position_ids)
        positions = torch.arange(seq_len, device=value.device, dtype=torch.long)[None, :]
        cos, sin = self.original(value, positions)
        return cos[0], sin[0]


def legacy_apply_rotary(query, key, cos, sin, position_ids, unsqueeze_dim=1):
    from transformers.models.llama.modeling_llama import rotate_half
    if position_ids is None:
        position_ids = torch.arange(query.shape[-2], device=query.device)[None, :]
    selected_cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    selected_sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    return (query*selected_cos + rotate_half(query)*selected_sin,
            key*selected_cos + rotate_half(key)*selected_sin)


def decoder_forward(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    cache_position=None, position_embeddings=None, **kwargs):
    from transformers.cache_utils import Cache, DynamicCache
    shared_cache = past_key_value if isinstance(past_key_value, Cache) else None
    if shared_cache is not None and not isinstance(shared_cache, DynamicCache):
        raise NotImplementedError('DuQuant bridge supports DynamicCache and legacy tuples, not static caches')
    legacy_past = past_key_value
    if shared_cache is not None:
        index = getattr(self, '_e42_layer_index', None)
        if index is None:
            raise RuntimeError('Call install(model) after loading/replacing the official decoder layers')
        legacy_past = shared_cache[index] if shared_cache.get_seq_length(index) else None
    past_length = int(legacy_past[0].shape[-2]) if legacy_past is not None else 0
    key_length = past_length + hidden_states.shape[1]
    if attention_mask is not None and attention_mask.ndim == 4:
        # TF4.45 may append one fully masked future column when no attention
        # mask is supplied. Keep every existing cached key, trim only surplus.
        attention_mask = attention_mask[..., :key_length]
    outputs = _ORIGINAL_DECODER_FORWARD(self, hidden_states,
        attention_mask=attention_mask, position_ids=position_ids,
        past_key_value=legacy_past, output_attentions=output_attentions, use_cache=use_cache)
    if shared_cache is not None and use_cache:
        slot = 2 if output_attentions else 1
        key, value = outputs[slot]
        if key.shape[-2] != key_length or value.shape[-2] != key_length:
            raise RuntimeError('Official DuQuant cache did not retain the prefix and append current tokens')
        # Official attention already concatenated its post-QDQ K and pre-PV-QDQ V.
        # Replace HF's per-layer references; DynamicCache.update would append the
        # entire prefix a second time and change both length and arithmetic.
        while len(shared_cache.key_cache) <= index:
            shared_cache.key_cache.append([])
            shared_cache.value_cache.append([])
        shared_cache.key_cache[index] = key
        shared_cache.value_cache[index] = value
        if index == 0:
            shared_cache._seen_tokens = key_length
        outputs = outputs[:slot] + (shared_cache,) + outputs[slot+1:]
    return outputs


def install(model=None):
    """Patch official decoder API; optionally wrap original per-layer RoPE.

    position_embeddings is redundant with the identical configured per-layer
    RoPE, which is recomputed by the official attention using position_ids.
    """
    global _ORIGINAL_DECODER_FORWARD
    from models import int_llama_layer as official
    if _ORIGINAL_DECODER_FORWARD is None:
        _ORIGINAL_DECODER_FORWARD = official.QuantLlamaDecoderLayer.forward
    official.QuantLlamaDecoderLayer.forward = decoder_forward
    official.apply_rotary_pos_emb = legacy_apply_rotary
    if model is not None:
        for index, layer in enumerate(model.model.layers):
            layer._e42_layer_index = index
            if not isinstance(layer.self_attn.rotary_emb, LegacyRotary):
                layer.self_attn.rotary_emb = LegacyRotary(layer.self_attn.rotary_emb)
        model.config.use_cache = False
    return dict(old_api='TF4.35 seq_len rotary + indexed cosine table',
        new_api='TF4.45 configured Llama3.1 RoPE; no replacement frequencies or scaling',
        decoder_extra_keywords='cache_position and precomputed position_embeddings are redundant with original position_ids and configured per-layer RoPE',
        frequency_dtype='FP32 configured frequencies retained across official qlayer.half()',
        mask_adapter='trim only excess future columns to past-cache length plus current query length',
        cache_support='HF4.45 DynamicCache translated to official per-layer legacy tuple; official concatenated K/V retained without requantizing or duplicating prefix',
        cache_storage='official floating post-QDQ K and pre-PV-QDQ V, not packed INT4 KV')
