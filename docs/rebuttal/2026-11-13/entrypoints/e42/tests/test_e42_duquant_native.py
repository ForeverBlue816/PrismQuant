import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,apply_rotary_pos_emb,LlamaForCausalLM
from nar.e42_duquant_compat import LegacyRotary,legacy_apply_rotary,install,prepare_official_catcher
from nar.e42_duquant_native import native_args,rotation_table,sample_windows,MODELS,reuse_preparation,identity


class DuQuantNative(unittest.TestCase):
    def config(self):
        return LlamaConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=1,
            num_attention_heads=4,num_key_value_heads=2,vocab_size=128,max_position_embeddings=131072,
            rope_theta=500000.,rope_scaling={'factor':8.,'low_freq_factor':1.,'high_freq_factor':4.,
                'original_max_position_embeddings':8192,'rope_type':'llama3'},
            attention_dropout=0.,_attn_implementation='eager',use_cache=False)

    def test_shared_rotary_runs_before_official_style_catcher(self):
        model=LlamaForCausalLM(self.config()).eval()
        rotary=model.model.rotary_emb
        before=rotary.inv_freq.clone();events=[]
        handle=rotary.register_forward_hook(lambda *unused:events.append('shared_rotary'))
        class Catcher(torch.nn.Module):
            def forward(self,hidden_states,**kwargs):
                events.append('catcher')
                self.hidden=hidden_states
                self.positions=kwargs['position_ids']
                raise ValueError('captured')
        catcher=Catcher();model.model.layers[0]=catcher
        with patch.object(rotary,'to',wraps=rotary.to) as move:
            report=prepare_official_catcher(model,torch.device('cpu'))
            move.assert_called_once_with(device=torch.device('cpu'))
        with torch.inference_mode(),self.assertRaisesRegex(ValueError,'captured'):
            model(input_ids=torch.tensor([[2,3,4,5]]),use_cache=False)
        handle.remove()
        self.assertEqual(events,['shared_rotary','catcher'])
        self.assertEqual(tuple(catcher.hidden.shape),(1,4,128))
        self.assertEqual(catcher.positions.tolist(),[[0,1,2,3]])
        self.assertIs(model.model.rotary_emb,rotary)
        self.assertTrue(torch.equal(before,rotary.inv_freq))
        self.assertFalse(report['mathematical_change'])

    @unittest.skipUnless(torch.cuda.is_available(),'requires an allocated CUDA GPU')
    def test_catcher_shared_rotary_cpu_cuda_failure_is_fixed(self):
        model=LlamaForCausalLM(self.config()).eval()
        model.model.embed_tokens.cuda()
        class Catcher(torch.nn.Module):
            def forward(self,hidden_states,**kwargs):
                raise ValueError('captured')
        model.model.layers[0]=Catcher()
        ids=torch.tensor([[2,3,4,5]],device='cuda')
        with torch.inference_mode(),self.assertRaises(RuntimeError):
            model(input_ids=ids,use_cache=False)
        before=model.model.rotary_emb.inv_freq.clone()
        prepare_official_catcher(model,torch.device('cuda'))
        with torch.inference_mode(),self.assertRaisesRegex(ValueError,'captured'):
            model(input_ids=ids,use_cache=False)
        self.assertTrue(torch.equal(before,model.model.rotary_emb.inv_freq.cpu()))

    def test_preparation_reuse_checks_identity_and_replays_sampling_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact=Path(directory);args=native_args(artifact,'llama32_3b')
            batches,positions=sample_windows(torch.arange(5000)[None,:])
            torch.save(dict(batches=batches,positions=positions),artifact/'calibration.pt')
            torch.save({'model.layers.0.self_attn.q_proj':torch.ones(128)},artifact/'act_scales.pt')
            (artifact/'Rot.pkl').write_bytes(b'identified-rotation-fixture')
            manifest=dict(model_key='llama32_3b',base_revision=MODELS['llama32_3b']['base'].name,
                official_arguments=vars(args),calibration=dict(artifact=identity(artifact/'calibration.pt'),
                    positions=positions,token_count=5000),
                smoothing=dict(scales=identity(artifact/'act_scales.pt')),
                rotation_table=identity(artifact/'Rot.pkl'),smooth_scale_collection_seconds=12.)
            path=artifact/'failed.json';path.write_text(json.dumps(manifest));record={}
            loaded,scales=reuse_preparation(path,artifact,'llama32_3b',args,record)
            self.assertEqual(len(loaded),128)
            self.assertTrue(torch.equal(loaded[-1][0],batches[-1][0]))
            self.assertTrue(torch.equal(scales['model.layers.0.self_attn.q_proj'],torch.ones(128)))
            expected=random.Random(0)
            for _ in range(128):expected.randint(0,5000-2048-1)
            self.assertEqual(random.random(),expected.random())
            self.assertFalse(record['preparation_reuse']['calibration_and_scale_collection_repeated'])
            with self.assertRaisesRegex(ValueError,'different base model'):
                reuse_preparation(path,artifact,'llama31_8b',native_args(artifact),{})
            args.alpha=.5
            with self.assertRaisesRegex(ValueError,'recipe differs'):
                reuse_preparation(path,artifact,'llama32_3b',args,{})
            args.alpha=.6;(artifact/'Rot.pkl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                reuse_preparation(path,artifact,'llama32_3b',args,{})

    def test_llama31_rope_noncontiguous_positions_matches_modern(self):
        config=self.config();modern=LlamaRotaryEmbedding(config=config);legacy=LegacyRotary(copy.deepcopy(modern))
        q=torch.randn(2,4,3,32);k=torch.randn(2,2,3,32)
        positions=torch.tensor([[0,17,999],[4,32,1234]])
        cos,sin=modern(k,positions)
        expected=apply_rotary_pos_emb(q,k,cos,sin)
        table_cos,table_sin=legacy(k,seq_len=1235)
        actual=legacy_apply_rotary(q,k,table_cos,table_sin,positions)
        self.assertTrue(torch.equal(actual[0],expected[0]));self.assertTrue(torch.equal(actual[1],expected[1]))
        legacy.half()
        self.assertEqual(legacy.original.inv_freq.dtype,torch.float32)
        half_expected=modern(k.half(),positions)
        half_tables=legacy(k.half(),seq_len=1235)
        self.assertTrue(torch.equal(half_tables[0][positions],half_expected[0]))

    def test_actual_official_decoder_accepts_new_model_keywords(self):
        config=self.config();model=LlamaForCausalLM(config).half().eval();install(model)
        from models.int_llama_layer import QuantLlamaDecoderLayer
        args=native_args(Path('/tmp/duquant-test-unused'))
        for key in ('q','k','v','gate','up','down','o'):
            setattr(args,key+'_weight_quant_params',copy.copy(args.weight_quant_params))
            setattr(args,key+'_act_quant_params',copy.copy(args.act_quant_params))
        original=model.model.layers[0]
        quant=QuantLlamaDecoderLayer(config,original,args).eval()
        quant.set_quant_state(False,False)
        from quantize.utils import set_init_duquant_params_state
        set_init_duquant_params_state(quant,True)
        model.model.layers[0]=quant
        with torch.inference_mode():
            out=model(input_ids=torch.tensor([[2,3,4,5]]),use_cache=False)
        self.assertEqual(tuple(out.logits.shape),(1,4,128));self.assertTrue(torch.isfinite(out.logits).all())
        data=io.BytesIO();torch.save(model,data);data.seek(0)
        restored=torch.load(data,weights_only=False)
        with torch.inference_mode():
            replay=restored(input_ids=torch.tensor([[2,3,4,5]]),use_cache=False)
        self.assertTrue(torch.equal(out.logits,replay.logits))

    def test_official_quantized_cached_decode_matches_full_sequence(self):
        torch.manual_seed(7)
        config=self.config();config.num_hidden_layers=2
        config.num_attention_heads=1;config.num_key_value_heads=1;config.head_dim=128
        model=LlamaForCausalLM(config).half().eval();install(model)
        from models.int_llama_layer import QuantLlamaDecoderLayer
        from quantize.utils import set_init_duquant_params_state
        args=native_args(Path('/tmp/duquant-test-unused'))
        for key in ('q','k','v','gate','up','down','o'):
            setattr(args,key+'_weight_quant_params',copy.copy(args.weight_quant_params))
            setattr(args,key+'_act_quant_params',copy.copy(args.act_quant_params))
        for index in range(config.num_hidden_layers):
            quant=QuantLlamaDecoderLayer(config,model.model.layers[index],args).eval()
            # Identity rotation factors isolate the cache bridge while retaining
            # the actual official W4/A4, Q/K and V quantizer operations.
            set_init_duquant_params_state(quant,True)
            quant.set_quant_state(True,True)
            model.model.layers[index]=quant
        install(model)
        ids=torch.tensor([[2,3,4,5,6,7]])
        from transformers.cache_utils import DynamicCache
        with torch.inference_mode():
            full=model(input_ids=ids,use_cache=False).logits
            for cache in (None,DynamicCache()):
                out=model(input_ids=ids[:,:3],past_key_values=cache,use_cache=True)
                pieces=[out.logits];cache=out.past_key_values
                for position in range(3,6):
                    old_keys=(cache.key_cache[0] if isinstance(cache,DynamicCache) else cache[0][0]).clone()
                    out=model(input_ids=ids[:,position:position+1],past_key_values=cache,use_cache=True)
                    cache=out.past_key_values;pieces.append(out.logits)
                    pairs=list(cache) if isinstance(cache,DynamicCache) else cache
                    self.assertEqual([p[0].shape[-2] for p in pairs],[position+1]*2)
                    self.assertTrue(torch.equal(pairs[0][0][...,:position,:],old_keys))
                actual=torch.cat(pieces,dim=1)
                self.assertTrue(torch.allclose(actual,full,atol=2e-3,rtol=2e-3),
                    msg=f'max cached/full logit discrepancy: {(actual-full).abs().max().item()}')

    def test_official_rotation_recurrence_and_seeded_sampling(self):
        matrix=rotation_table(8,'cpu')
        self.assertTrue(torch.allclose(matrix@matrix.T,torch.eye(8),atol=3e-7))
        self.assertTrue(torch.allclose(matrix[0].abs(),torch.full((8,),1/(8**.5)),atol=1e-7))
        ids=torch.arange(30)[None,:]
        batches,positions=sample_windows(ids,nsamples=3,seq_len=4,seed=0)
        self.assertEqual(positions,[12,24,13])
        self.assertEqual(batches[0][0].tolist(),[[12,13,14,15]])
        self.assertEqual(batches[0][1].tolist(),[[-100,-100,-100,15]])

    def test_official_run_sh_defaults(self):
        a=native_args(Path('/tmp/duquant-test-unused'))
        self.assertEqual((a.wbits,a.abits,a.block_size,a.max_rotation_step,a.epochs),(4,4,128,256,0))
        self.assertEqual((a.alpha,a.lac,a.swc),(.6,.9,.8))
        self.assertTrue(a.lwc and a.smooth)
        self.assertFalse(a.symmetric or a.let)
        self.assertIsNone(a.group_size);self.assertIsNone(a.act_group_size)

    def test_three_b_and_eight_b_assets_are_distinct(self):
        a=native_args(Path('/tmp/duquant-test-unused'),'llama32_3b')
        self.assertEqual(a.net,'Llama-3.2-3B')
        self.assertIn('d4446454',a.model)
        self.assertNotEqual(MODELS['llama31_8b']['tokens'],MODELS['llama32_3b']['tokens'])


if __name__=='__main__':unittest.main()
