import copy
import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
from prismquant import fit_rotation
from prismquant.model import ModelRotations, restore_unquantized_state
from prismquant.folding import fuse_norms_and_rotate, _layer_state, _load_layer_state
from prismquant.runtime import RuntimeHooks
from prismquant.hub import list_models, checkpoint_spec


def make_rotations(tmp_path, config, method):
    root = tmp_path / 'rotations/test/e14_rotations_seed2';root.mkdir(parents=True)
    r4 = tmp_path / 'r4';r4.mkdir()
    torch.manual_seed(3)
    for label, n in [('r1', config.hidden_size), ('r2', 128), ('r4', config.intermediate_size)]:
        f = fit_rotation(torch.randn(160, n), rank=n//128)
        if label == 'r1':f.save(root / 'r1_kmax.pt')
        else:
            for i in range(config.num_hidden_layers):
                f.save((root / f'r2_v_layer_{i:02d}.pt') if label=='r2' else (r4 / f'down_layer_{i:02d}.pt'))
    return ModelRotations(tmp_path, 'test', dict(rotation=method,seed=2,r4_root='r4'), config)


@pytest.mark.parametrize('architecture', ['llama', 'qwen3'])
@pytest.mark.parametrize('method', ['hadamard', 'nar_kmax'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_restore_matches_full_fold_and_cached_padded_generation(tmp_path, architecture, method, dtype):
    cls, ctor = (LlamaConfig, LlamaForCausalLM) if architecture=='llama' else (Qwen3Config, Qwen3ForCausalLM)
    cfg = cls(hidden_size=256, intermediate_size=512, num_hidden_layers=2,
              num_attention_heads=2, num_key_value_heads=1, head_dim=128,
              vocab_size=64, tie_word_embeddings=True, pad_token_id=0)
    torch.manual_seed(2);original=ctor(cfg).to(dtype).eval();original.config._attn_implementation='sdpa'
    full, restored = copy.deepcopy(original), copy.deepcopy(original)
    rotations=make_rotations(tmp_path,cfg,method)
    with torch.no_grad():
        fuse_norms_and_rotate(full, rotations, 64)
        restore_unquantized_state(restored, rotations, 64)
        for i, layer in enumerate(full.model.layers):
            path=tmp_path/f'layer_{i}.pt';torch.save(_layer_state(layer),path)
            _load_layer_state(restored.model.layers[i],path)
    for name,value in full.state_dict().items():assert torch.equal(value,restored.state_dict()[name]),name
    h1=RuntimeHooks(full,rotations,'asymmetric_g128',True);h1.install()
    h2=RuntimeHooks(restored,rotations,'asymmetric_g128',True);h2.install()
    tokens=torch.tensor([[0,0,1,2,3],[1,2,3,4,5]]);mask=(tokens!=0).long()
    with torch.inference_mode():
        a=full(input_ids=tokens,attention_mask=mask,use_cache=False).logits
        b=restored(input_ids=tokens,attention_mask=mask,use_cache=False).logits
        assert torch.isfinite(b).all();assert torch.equal(a,b)
        generated=restored.generate(tokens,attention_mask=mask,max_new_tokens=3,do_sample=False,pad_token_id=0,eos_token_id=None)
        assert generated.shape==(2,8)
    h1.close();h2.close()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    assert h1.attention_key not in ALL_ATTENTION_FUNCTIONS
    assert h2.attention_key not in ALL_ATTENTION_FUNCTIONS
    assert not restored.model.layers[0].mlp.down_proj._forward_pre_hooks


def test_catalog_only_exposes_complete_pinned_variants():
    catalog=list_models();assert len(catalog)==8
    for key,spec in catalog.items():
        _,name,row=checkpoint_spec(key)
        assert len(spec['revision'])==40
        assert sum(p.startswith('checkpoints/') and '/layer_' in p for p in row['required_files'])==spec['layers']
        assert all(not p.startswith('/') and '..' not in p.split('/') for p in row['required_files'])
        assert row['download_bytes']>0
    with pytest.raises(ValueError):checkpoint_spec('unknown')
    with pytest.raises(ValueError):checkpoint_spec('qwen3_0.6b_base','../../bad')
