"""Check expert folding against an unrotated MoE, including router decisions."""
import copy
import torch
import pytest
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from prismquant import fit_rotation
from prismquant.model import restore_unquantized_state
from prismquant.moe import MoERotations, MoERuntimeHooks, load_moe_layer


def rotations(tmp_path, config):
    root = tmp_path/'rotations/test/e14_rotations'
    r4 = tmp_path/'experts'
    root.mkdir(parents=True); r4.mkdir()
    torch.manual_seed(7)
    for label, width in [('r1',256), ('r2',128)]:
        factor = fit_rotation(torch.randn(160, width), rank=width//128)
        factor.save(root/('r1_kmax.pt' if label=='r1' else 'r2_v_layer_00.pt'))
    for e in range(config.num_experts):
        factor = fit_rotation(torch.randn(160, 256), rank=2)
        factor.save(r4/f'layer_00_expert_{e:03d}.pt')
    return MoERotations(tmp_path,'test',dict(rotation='nar_kmax',seed=0,r4_root='experts'),config)


def test_moe_rotation_preserves_routing_logits_and_forward(tmp_path):
    cfg=Qwen3MoeConfig(hidden_size=256,moe_intermediate_size=256,intermediate_size=512,
                      num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
                      head_dim=128,num_experts=4,num_experts_per_tok=2,vocab_size=64,
                      tie_word_embeddings=False,pad_token_id=0)
    torch.manual_seed(5)
    original=Qwen3MoeForCausalLM(cfg).eval()
    original.config._attn_implementation='sdpa'
    # Exercise norm fusion: initialized norms would all be one.
    with torch.no_grad():
        for name,p in original.named_parameters():
            if name.endswith('layernorm.weight') or name=='model.norm.weight':p.uniform_(0.8,1.2)
    model=copy.deepcopy(original);rot=rotations(tmp_path,cfg)
    block=original.model.layers[0]
    # Independently express the full fold as explicit matrices; these are what
    # the saved decoder tensors already contain before the loader restores them.
    r1=rot.apply('r1',0,torch.eye(256)); r2=rot.apply('r2',0,torch.eye(128))
    inputs=block.input_layernorm.weight.detach();post=block.post_attention_layernorm.weight.detach()
    state={}
    for name in ['q_proj','k_proj','v_proj']:
        state['self_attn.'+name]=(getattr(block.self_attn,name).weight.detach()*inputs)@r1
    o= r1.T@block.self_attn.o_proj.weight.detach()
    state['self_attn.o_proj']=(o.reshape(256,2,128)@r2).reshape(256,256)
    experts=block.mlp.experts
    state['mlp.experts.gate_up_proj']=(experts.gate_up_proj.detach()*post)@r1
    state['mlp.experts.down_proj']=torch.stack([
        r1.T@experts.down_proj[e].detach()@rot.apply_expert(0,e,torch.eye(256)) for e in range(4)])
    path=tmp_path/'layer.pt';torch.save(state,path)
    restore_unquantized_state(model,rot)
    load_moe_layer(model.model.layers[0],path)
    x=torch.randn(8,256)
    expected=torch.nn.functional.linear(x*post,block.mlp.gate.weight)
    actual=torch.nn.functional.linear(rot.apply('r1',0,x),model.model.layers[0].mlp.gate.weight)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    prior_forward=model.model.layers[0].mlp.experts.forward
    untouched_forward=original.model.layers[0].mlp.experts.forward
    hooks=MoERuntimeHooks(model,rot,None,False);hooks.install()
    tokens=torch.tensor([[1,2,3,4,5]])
    with torch.inference_mode():
        expected=original(input_ids=tokens,use_cache=False).logits
        actual=model(input_ids=tokens,use_cache=False).logits
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)
    hooks.close()
    assert model.model.layers[0].mlp.experts.forward==prior_forward
    assert original.model.layers[0].mlp.experts.forward==untouched_forward
    # Enable W4A4KV4 reference hooks and exercise padded cached generation.
    hooks=MoERuntimeHooks(model,rot,'asymmetric_g128',True);hooks.install()
    tokens=torch.tensor([[0,0,1,2,3],[1,2,3,4,5]])
    with torch.inference_mode():
        generated=model.generate(tokens,attention_mask=(tokens!=0).long(),max_new_tokens=3,
                                 do_sample=False,pad_token_id=0,eos_token_id=None)
    assert generated.shape==(2,8)
    hooks.close()
    bad=dict(state);bad.pop('mlp.experts.down_proj');torch.save(bad,path)
    with pytest.raises(ValueError,match='checkpoint tensors'):load_moe_layer(model.model.layers[0],path)
