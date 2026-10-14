"""Qwen3 MoE reference runtime with an independently calibrated R4 per expert."""
from pathlib import Path
from types import MethodType
import torch
from .model import ModelRotations
from .rotations import RotationFactor, WYFactor, compact_wy, full_hadamard_rows
from .runtime import RuntimeHooks
from .quantization import dynamic_asym_int4, _symmetric_per_group_int4


class MoERotations(ModelRotations):
    def __init__(self, snapshot, model_key, variant, config):
        super().__init__(snapshot, model_key, variant, config)
        self.experts = config.num_experts
        self._expert_paths = {}
        if self.method != 'hadamard':
            for layer in range(self.layers):
                for expert in range(self.experts):
                    self._expert_paths[layer, expert] = (Path(snapshot) / variant['r4_root'] /
                                                       f'layer_{layer:02d}_expert_{expert:03d}.pt')

    def apply_expert(self, layer, expert, value):
        # Expert R4 uses all-positive signs (unlike the dense R4).
        signs = torch.ones(value.shape[-1], device=value.device, dtype=torch.float32)
        if self.method == 'hadamard':
            return full_hadamard_rows(value.float(), signs)
        key = 'expert', layer, expert, value.device
        if key not in self._wy:
            factor = RotationFactor.load(self._expert_paths[layer, expert], value.device)
            if factor.n != self.intermediate or factor.b != 128:
                raise ValueError(f'Incompatible expert factor at layer {layer}, expert {expert}')
            if bool(factor.active.any()):
                w, y = compact_wy(factor.reflectors, factor.active)
                self._wy[key] = WYFactor(factor, w, y)
            else:
                self._wy[key] = factor
        return self._wy[key].apply(value, signs)


@torch.no_grad()
def load_moe_layer(layer, path):
    """Restore all attention/expert GPTQ tensors; the router is restored separately."""
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    parameters = {f'self_attn.{name}': getattr(layer.self_attn, name).weight
                  for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')}
    parameters.update({'mlp.experts.gate_up_proj': layer.mlp.experts.gate_up_proj,
                       'mlp.experts.down_proj': layer.mlp.experts.down_proj})
    if set(state) != set(parameters):
        raise ValueError(f'Unexpected MoE checkpoint tensors in {path}')
    for name, parameter in parameters.items():
        if parameter.shape != state[name].shape:
            raise ValueError(f'Incompatible shape for {name}: {state[name].shape}')
        parameter.copy_(state[name])


class MoERuntimeHooks(RuntimeHooks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._expert_forwards = []

    def _quantize(self, value):
        if self.activation_kind == 'asymmetric_g128':
            return dynamic_asym_int4(value, 128)[0].to(value.dtype)
        if self.activation_kind == 'symmetric_g128':
            return _symmetric_per_group_int4(value, 128).to(value.dtype)
        return value

    def expert_forward(self, layer):
        def forward(module, hidden_states, top_k_index, top_k_weights):
            # Routing is already complete; only the expert's input is quantized.
            x = self._quantize(hidden_states)
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask = torch.nn.functional.one_hot(top_k_index, num_classes=module.num_experts).permute(2, 1, 0)
                hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()
            for index in hit:
                expert = int(index[0])
                slot, token = torch.where(mask[expert])
                gate, up = torch.nn.functional.linear(x[token], module.gate_up_proj[expert]).chunk(2, dim=-1)
                h = module.act_fn(gate) * up
                h = self.rotations.apply_expert(layer, expert, h).to(h.dtype)
                out = torch.nn.functional.linear(self._quantize(h), module.down_proj[expert])
                out = out * top_k_weights[token, slot, None]
                final.index_add_(0, token, out.to(final.dtype))
            return final
        return forward

    def install(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        if self.quantize_kv:
            ALL_ATTENTION_FUNCTIONS.register(self.attention_key, self.attention)
            try:
                from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
                ALL_MASK_ATTENTION_FUNCTIONS.register(self.attention_key, ALL_MASK_ATTENTION_FUNCTIONS['eager'])
            except (ImportError, KeyError):
                pass
            self.model.config._attn_implementation = self.attention_key
        for layer, block in enumerate(self.model.model.layers):
            self.handles.append(block.self_attn.v_proj.register_forward_hook(self.rotate_v(layer)))
            if self.rotations.method == 'hadamard':
                self.handles.append(block.self_attn.o_proj.register_forward_pre_hook(self.rotate_o()))
            if self.activation_kind is not None:
                for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                    self.handles.append(getattr(block.self_attn, name).register_forward_pre_hook(self.quantize_input))
            experts = block.mlp.experts
            # Preserve Accelerate's device-transfer wrapper when it is installed.
            attribute = '_old_forward' if hasattr(experts, '_hf_hook') else 'forward'
            self._expert_forwards.append((experts, attribute, getattr(experts, attribute)))
            setattr(experts, attribute, MethodType(self.expert_forward(layer), experts))

    def close(self):
        super().close()
        for module, attribute, original in self._expert_forwards:
            setattr(module, attribute, original)
        self._expert_forwards.clear()
