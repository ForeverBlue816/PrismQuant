"""Load released PrismQuant weights without experiment-specific directory layouts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
from .hub import checkpoint_spec, download_checkpoint, validate_snapshot
from .rotations import (RotationFactor, WYFactor, compact_wy, full_hadamard_rows,
                        _fast_walsh_hadamard, _paley_hadamard_12)
from .folding import _load_layer_state
from .runtime import RuntimeHooks


class ModelRotations:
    """Device-local factor copies, preserving the released sign/permutation convention."""
    def __init__(self, snapshot, model_key, variant, config):
        self.method = variant['rotation']
        self.seed = variant['seed']
        self.layers = config.num_hidden_layers
        self.heads = config.num_attention_heads
        self.hidden = config.hidden_size
        self.intermediate = config.intermediate_size
        self.head_dim = getattr(config, 'head_dim', None) or self.hidden // self.heads
        self._factors, self._copies, self._wy, self._signs = {}, {}, {}, {}
        if self.method == 'hadamard':
            return
        root = Path(snapshot) / 'rotations' / model_key / ('e14_rotations' + (f'_seed{self.seed}' if self.seed else ''))
        rank = self.method.removeprefix('nar_')
        r1_rank = 'kmax' if rank == 'k32' else rank
        self._factors['r1', 0] = RotationFactor.load(root / f'r1_{r1_rank}.pt', torch.device('cpu'))
        for i in range(self.layers):
            self._factors['r2', i] = RotationFactor.load(root / f'r2_v_layer_{i:02d}.pt', torch.device('cpu'))
            self._factors['r4', i] = RotationFactor.load(Path(snapshot) / variant['r4_root'] / f'down_layer_{i:02d}.pt', torch.device('cpu'))
        for (label, layer), f in self._factors.items():
            expected = {'r1': self.hidden, 'r2': self.head_dim, 'r4': self.intermediate}[label]
            if f.n != expected or f.b != 128:
                raise ValueError(f'Incompatible {label} factor at layer {layer}: {f.n}, {f.b}')

    def signs(self, label, layer, n, device):
        key = label, layer, torch.device(device)
        if key not in self._signs:
            seed = self.seed + {'r1': 140_000, 'r2': 240_000, 'r4': 340_000}[label] + layer
            g = torch.Generator(device='cpu').manual_seed(seed)
            self._signs[key] = torch.randint(0, 2, (n,), generator=g, dtype=torch.int64).float().mul_(2).sub_(1).to(device)
        return self._signs[key]

    def factor(self, label, layer, device):
        key = label, layer, torch.device(device)
        if key not in self._copies:
            f = self._factors[label, layer]
            self._copies[key] = RotationFactor(f.n, f.b, f.reflectors.to(device), f.active.to(device),
                                               f.source_order.to(device), f.target_order.to(device), f.anchor_error)
        return self._copies[key]

    def apply(self, label, layer, value):
        signs = self.signs(label, layer, value.shape[-1], value.device)
        if self.method == 'hadamard':
            return full_hadamard_rows(value.float(), signs if label == 'r1' else torch.ones_like(signs))
        factor = self.factor(label, layer, value.device)
        if label != 'r4':
            return factor.apply(value, signs)
        key = label, layer, value.device
        if key not in self._wy:
            w, y = compact_wy(factor.reflectors, factor.active)
            self._wy[key] = WYFactor(factor, w, y)
        return self._wy[key].apply(value, signs)

    def apply_r3(self, value):
        if self.method != 'hadamard':
            return value.float()
        shape = value.shape
        across = value.float().reshape(-1, self.heads, self.head_dim).transpose(1, 2)
        if self.heads == 24:
            blocks = _fast_walsh_hadamard(across.reshape(-1, 12, 2))
            rotated = (blocks.transpose(1, 2) @ _paley_hadamard_12(value.device, torch.float32).T).transpose(1, 2).reshape_as(across)
        else:
            rotated = full_hadamard_rows(across, torch.ones(self.heads, device=value.device))
        return rotated.transpose(1, 2).reshape(shape)


@torch.no_grad()
def restore_unquantized_state(model, rotations, row_batch=256):
    """Fold embedding/head and restore norm state; decoder linears come from Hub.

    This exactly matches those tensors after the full research fold. Skipping
    transformations of the decoder linears avoids computing weights that the
    saved GPTQ checkpoint immediately replaces.
    """
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings = False
    scale = model.model.norm.weight.detach().float()
    model.lm_head.weight.mul_(scale.to(device=model.lm_head.weight.device, dtype=model.lm_head.weight.dtype).unsqueeze(0))
    for block in model.model.layers:
        block.input_layernorm.weight.fill_(1)
        block.post_attention_layernorm.weight.fill_(1)
    model.model.norm.weight.fill_(1)
    for module in [model.model.embed_tokens, model.lm_head]:
        for start in range(0, module.weight.shape[0], row_batch):
            rows = module.weight[start:start + row_batch]
            rows.copy_(rotations.apply('r1', 0, rows.float()).to(rows.dtype))


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    hooks: RuntimeHooks
    metadata: dict

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 64, **kwargs) -> str:
        """Greedy base-model text completion; use model/tokenizer for batched APIs."""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.get_input_embeddings().weight.device)
        options = dict(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                       pad_token_id=self.tokenizer.pad_token_id)
        options.update(kwargs)
        outputs = self.model.generate(**inputs, **options)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def close(self):
        self.hooks.close()


def load_model(model_key: str, checkpoint: str | None = None, *, snapshot_path=None,
               base_model_path=None, cache_dir=None, device='cuda', device_map=None,
               max_memory=None, token=None, local_files_only=False,
               quantize_activations=True, quantize_kv=True) -> LoadedModel:
    """Load the reference W4A4KV4 runtime from a pinned released checkpoint.

    Containers remain bf16 (Llama 3B/8B) or fp32 (Qwen3/70B), matching the
    original evaluation. This loader does not claim packed-INT4 storage or speed.
    CPU is useful for small-model debugging; 70B requires enough aggregate memory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec, name, variant = checkpoint_spec(model_key, checkpoint)
    if snapshot_path is None:
        snapshot_path = download_checkpoint(model_key, name, cache_dir=cache_dir, token=token,
                                            local_files_only=local_files_only)
    snapshot = validate_snapshot(snapshot_path, model_key, name)
    if device_map is None and torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; pass device="cpu" for a small-model CPU run.')
    base = str(base_model_path or spec['base_model'])
    kwargs = dict(cache_dir=cache_dir, token=token, local_files_only=local_files_only,
                  dtype=getattr(torch, spec['compute_dtype']), low_cpu_mem_usage=True,
                  attn_implementation='sdpa')
    if device_map is not None:
        kwargs.update(device_map=device_map)
        if max_memory is not None:
            kwargs['max_memory'] = max_memory
    if base_model_path is None:
        kwargs['revision'] = spec['base_revision']
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if model.config.model_type != spec['architecture'] or model.config.num_hidden_layers != spec['layers']:
        raise ValueError('Base model architecture/layer count does not match the released checkpoint')
    if device_map is None:
        model = model.to(device)
    if any(p.device.type == 'meta' for p in model.parameters()):
        raise ValueError('Disk/meta offloading is not supported; give the model enough CPU/GPU memory.')
    rotations = ModelRotations(snapshot, model_key, variant, model.config)
    restore_unquantized_state(model, rotations)
    for i, layer in enumerate(model.model.layers):
        _load_layer_state(layer, snapshot / 'checkpoints' / model_key / name / f'layer_{i:02d}.pt')
    model.eval()
    model.generation_config.max_length = None
    model.generation_config.max_new_tokens = None
    tokenizer = AutoTokenizer.from_pretrained(base, cache_dir=cache_dir, token=token,
                                              local_files_only=local_files_only, use_fast=True,
                                              revision=spec['base_revision'] if base_model_path is None else None)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    hooks = RuntimeHooks(model, rotations, 'asymmetric_g128' if quantize_activations else None, quantize_kv)
    hooks.install()
    return LoadedModel(model, tokenizer, hooks, dict(model=model_key, checkpoint=name,
                       repo_id=spec['repo_id'], revision=spec['revision'], base_revision=spec['base_revision'], compute_dtype=spec['compute_dtype'],
                       activation_quantization=quantize_activations, kv_quantization=quantize_kv,
                       storage='dequantized floating-point reference weights/cache'))
