"""Trained E14 8B checkpoints exported to the existing CUTLASS INT4 backend.

Only packing and FP16 scale projection are performed; no GPTQ calibration runs.
Original E14 rotation factors and compensations are preserved. R4 currently
uses the original FP32 compact-WY implementation, not the optimized E28 kernel.
"""
from __future__ import annotations

import json
from pathlib import Path

from e38 import fp16 as common

WORK = Path('/projects/nar/nar-validation')
BASE = WORK/'cache/huggingface/models--unsloth--Meta-Llama-3.1-8B/snapshots/e9a141a2091ea561b96483212645a2a05e6f99fc'
QUANT = WORK/'cache/huggingface/models--ForeverBlue--PrismQuant-Llama-3.1-8B/snapshots/fc812e6687165115feee00293c953a1b521d3bd0/checkpoints/llama31_8b'


def pack_signed4(codes):
    import torch
    if codes.shape[-1] % 2:
        raise ValueError('packed width must be even')
    unsigned = codes.to(torch.int8).to(torch.uint8) & 15
    return unsigned[..., 0::2] | (unsigned[..., 1::2] << 4)


def unpack_cache_prefix(pages, params, layer, length, attention_mask=None):
    """Read actual one-page-per-batch native KV bytes and FP16 metadata."""
    import torch
    output = []
    for kind in (0, 1):
        packed = pages[:, layer, kind, :, :length].transpose(1, 2)
        metadata = params[:, layer, kind, :, :length].transpose(1, 2)
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        decoded = codes * metadata[..., 0:1] - metadata[..., 1:2]
        if attention_mask is not None:
            restored = torch.zeros_like(decoded)
            for batch in range(decoded.shape[0]):
                valid = attention_mask[batch].bool()
                count = int(valid.sum())
                restored[batch, valid] = decoded[batch, :count]
            decoded = restored
        output.append(decoded.contiguous())
    return tuple(output)


def cache_class():
    from e28_v2.cache_adapter import SharedMetadataCache
    from quarot.transformers.kv_cache import matmul_had_cuda

    class TrainedStageACache(SharedMetadataCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            result = super().update(key_states, value_states, layer_idx, cache_kwargs)
            if self.disable_quant or not isinstance(result, tuple):
                return result
            # This first prefill has just been stored by the native init_kv_i4.
            # FlashAttention gets reconstructed KV4, not the original FP16 K/V.
            key, value = unpack_cache_prefix(self.pages, self.scales, layer_idx,
                key_states.shape[1], (cache_kwargs or {}).get('attention_mask'))
            if self.hadamard_dtype is not None:
                key = matmul_had_cuda(key, self.hadamard_dtype)
            return key, value
    return TrainedStageACache


def make_cache(model, batch, capacity):
    import torch
    return cache_class()(batch_size=batch, page_size=capacity, max_seq_len=capacity,
        device=torch.device('cuda'), n_layers=len(model.model.layers),
        num_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size//model.config.num_attention_heads,
        disable_quant=model.cache_dtype == 'float16',
        hadamard_dtype=None if model.cache_dtype == 'float16' else torch.float16)


def export_matrix(weight):
    """Recover saved GPTQ codes; expose projection error at native FP16 scale."""
    import torch
    from nar.e38_checkpoint_probe import audit_symmetric_grid
    audit, recovered = audit_symmetric_grid(weight.cuda(), row_batch=128)
    if audit['unrecovered_rows']:
        raise RuntimeError(f'Unsupported saved GPTQ grid: {audit["unrecovered_rows"]} rows could not be recovered')
    scale32 = recovered['fp32_scale'].cuda()
    scale16 = torch.where(recovered['bf16_roundtrip_fp16'], recovered['fp16_scale'],
                          recovered['fp32_scale'].half()).cuda()
    if not bool(torch.isfinite(scale16).all() and (scale16 > 0).all()):
        raise RuntimeError('Native FP16 weight-scale representation is not finite and positive')
    codes = torch.round(weight.cuda().float()/scale32[:, None]).to(torch.int8)
    if int(codes.min()) < -8 or int(codes.max()) > 7:
        raise RuntimeError('Recovered codes are outside signed INT4')
    reconstructed = codes.float()*scale16.float()[:, None]
    original = weight.cuda().float()
    audit.update(native_fp16_scale_projection_max_abs=float((reconstructed-original).abs().max()),
        native_fp16_scale_projection_relative_l2=float((reconstructed-original).norm()/original.norm().clamp_min(1e-30)),
        original_weight_modified=False, gptq_recalibrated=False,
        export_policy='recover original signed4 code grid; prefer BF16-roundtrip FP16 scale, otherwise round recovered FP32 scale to FP16')
    return dict(weight=pack_signed4(codes).cpu(), weight_scales=scale16[:, None].cpu()), audit


def layer_payload(checkpoint, cache, layer):
    import torch
    path = checkpoint/f'layer_{layer:02d}.pt'
    output = cache/f'layer_{layer:02d}.pt'
    sidecar = cache/f'layer_{layer:02d}.json'
    identity = dict(source_path=str(path), source_resolved=str(path.resolve()),
                    source_sha256=common.sha(path), exporter_sha256=common.sha(__file__))
    if output.exists() and sidecar.exists():
        info = json.loads(sidecar.read_text())
        if info['identity'] != identity:
            raise RuntimeError(f'Packed cache identity differs; choose a new --artifact-dir: {cache}')
        return torch.load(output, map_location='cpu', weights_only=True), info, True
    if output.exists() or sidecar.exists():
        raise RuntimeError(f'Incomplete packed export pair; retain it and choose a new artifact directory: {output}')
    original = torch.load(path, map_location='cpu', weights_only=True)
    packed, audits = {}, {}
    for name, weight in original.items():
        packed[name], audits[name] = export_matrix(weight)
    temporary = output.with_suffix('.pt.tmp')
    torch.save(packed, temporary)
    temporary.replace(output)
    info = dict(identity=identity, audits=audits, bytes=output.stat().st_size,
                sha256=common.sha(output), kind='actual trained GPTQ weights exported to packed signed4 plus FP16 scales')
    common.write(sidecar, info)
    return packed, info, False


def load_embedding_head(model, rotations):
    import torch
    from safetensors import safe_open
    index = json.loads((BASE/'model.safetensors.index.json').read_text())['weight_map']
    def read(name):
        with safe_open(str(BASE/index[name]), framework='pt', device='cpu') as stream:
            return stream.get_tensor(name)
    final_norm = read('model.norm.weight').cuda().bfloat16()
    details = []
    for name, module in (('model.embed_tokens.weight', model.model.embed_tokens), ('lm_head.weight', model.lm_head)):
        source = read(name)
        for start in range(0, source.shape[0], 256):
            chunk = source[start:start+256].cuda().bfloat16()
            if name == 'lm_head.weight':
                chunk.mul_(final_norm)
            rotated = rotations.apply('r1', 0, chunk.float()).bfloat16()
            module.weight.data[start:start+256].copy_(rotated.half().cpu())
        details.append(dict(name=name, file=str(BASE/index[name]), source_dtype=str(source.dtype),
                            fold_dtype='BF16 at original E14 rounding boundaries, then FP16 backend container'))
    model.model.norm.weight.data.fill_(1)
    return details


def build(row, artifact_dir, record, output):
    import torch
    import transformers
    import modeling_llama3 as ml
    from nar import e14_w4a4kv4 as e14
    if row == 'fp16':
        model, loaded = common.build_real('8b')
        record['checkpoint_loading'] = loaded
        return model
    checkpoint = QUANT/f'gptq_{row}_seed0'
    record['checkpoint_loading'] = dict(directory=str(checkpoint),
        manifest=json.loads((checkpoint/'DONE.json').read_text()),
        manifest_sha256=common.sha(checkpoint/'DONE.json'),
        native_representation_note='Same source trained checkpoint; FP16scale packing generally changes represented real-valued weights, recorded per matrix')
    cache = Path(artifact_dir)/row
    cache.mkdir(parents=True, exist_ok=True)
    config = ml.load_llama3_config(str(BASE/'config.json'))
    rotations = e14.RotationSet(WORK, 'llama31_8b', row, 0, config, torch.device('cuda'))
    if row == 'nar_k8':
        factor_root = WORK/'activations/llama31_8b'
        paths = [factor_root/'e14_rotations/DONE.json',factor_root/'e14_rotations/r1_k8.pt']
        paths += [factor_root/f'e14_rotations/r2_v_layer_{i:02d}.pt' for i in range(32)]
        paths += [factor_root/f'e11_calibration/factors/nar_b128_k8/down_layer_{i:02d}.pt' for i in range(32)]
        record['rotation_assets'] = [dict(path=str(path),sha256=common.sha(path)) for path in paths]
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with transformers.modeling_utils.no_init_weights():
            model = ml.QuarotLlamaForCausalLM(config)
    finally:
        torch.set_default_dtype(dtype)
    record['embedding_head_fold'] = load_embedding_head(model, rotations)
    record['packed_layers'] = []
    model._stage_a_rotations = rotations
    model._stage_a_handles = []

    class OriginalE14R4(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index
        def forward(self, value):
            return rotations.apply('r4', self.index, value).to(value.dtype)

    for index, block in enumerate(model.model.layers):
        packed, info, reused = layer_payload(checkpoint, cache, index)
        mapping = {'self_attn.q_proj': block.self_attn.q_proj,
            'self_attn.k_proj': block.self_attn.k_proj, 'self_attn.v_proj': block.self_attn.v_proj,
            'self_attn.o_proj': block.self_attn.o_proj[1], 'mlp.up_proj': block.mlp.up_proj,
            'mlp.gate_proj': block.mlp.gate_proj, 'mlp.down_proj': block.mlp.down_proj[2]}
        if set(packed) != set(mapping):
            raise RuntimeError(f'Unsupported checkpoint matrix mapping at layer {index}')
        for name, module in mapping.items():
            module.weight.copy_(packed[name]['weight'])
            module.weight_scales.copy_(packed[name]['weight_scales'])
        def r2_hook(_module, _inputs, value, layer=index):
            return rotations.apply('r2', layer, value.reshape(-1, 128)).reshape_as(value).to(value.dtype)
        model._stage_a_handles.append(block.self_attn.v_proj.register_forward_hook(r2_hook))
        if row == 'nar_k8':
            block.self_attn.o_proj_hadamard = torch.nn.Identity()
        block.mlp.down_proj[0] = OriginalE14R4(index)
        record['packed_layers'].append(dict(layer=index, reused=reused, export=info))
        common.write(output, record)
        print(json.dumps(dict(stage='pack', row=row, layer=index, reused=reused)), flush=True)
    model = model.eval().cuda()
    ml.install_llama3_rope(model)
    for block in model.model.layers:
        rotary = block.self_attn.rotary_emb
        rotary.inv_freq = ml.llama3_inv_freq(config, rotary.inv_freq.device).float()
        rotary._set_cos_sin_cache(rotary.max_seq_len_cached, rotary.inv_freq.device, torch.float16)
    model.logits_last_only = False
    # HFLM and any caller which omits a cache still get the same corrected KV4 prefill.
    import types
    def build_cache(self, batch_size, page_size, max_length):
        return make_cache(self, batch_size, max_length)
    model.build_cache = types.MethodType(build_cache, model)
    record['rotation_implementation'] = dict(r1='original E14 BF16 fold for embedding/head; trained GPTQ layer compensation retained',
        r2='original E14 factor after v_proj; saved o_proj compensation retained',
        r3='original head Hadamard and saved compensation' if row == 'hadamard' else 'identity, as original E14 PrismQuant',
        r4='original E14 compact-WY FP32 / full Hadamard; not optimized E28 random-factor kernel',
        kv_extra_rotation='native H128 on cached K and decode Q; prefill unpacks actual K4 then inverse H128')
    common.write(cache/'DONE.json', dict(row=row, seed=0, checkpoint=str(checkpoint),
        source_sha256=common.sha(__file__), layer_exports=[x['export']['sha256'] for x in record['packed_layers']],
        gptq_recalibrated=False))
    return model
