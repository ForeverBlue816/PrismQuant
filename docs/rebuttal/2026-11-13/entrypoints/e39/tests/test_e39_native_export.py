"""CPU checks only: torch2.11 checkpoint -> official torch2.6 packed export/load.

The tiny synthetic fixture is for serialization and parameter-coverage checks,
not a model accuracy result or a performance benchmark. No GPU kernel executes.
"""
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedModel
from nar import e39_flatquant as accuracy
from nar import e39_flatquant_native as native

REPO=Path(__file__).resolve().parents[1]


def tiny_config():
    config=LlamaConfig.from_pretrained(accuracy.BASE,local_files_only=True)
    config.hidden_size=256;config.intermediate_size=256
    config.num_hidden_layers=1;config.num_attention_heads=2;config.num_key_value_heads=1
    config.vocab_size=128;config.pad_token_id=0;config.bos_token_id=1;config.eos_token_id=2
    config._attn_implementation_internal='eager'
    return config


def make_fixture(path):
    from flatquant.model_tools.llama31_utils import apply_flatquant_to_llama_31
    from flatquant.flat_utils import reparameterize_model
    import gptq_utils
    torch.set_num_threads(2);torch.manual_seed(7)
    logging.disable(logging.CRITICAL)
    path.mkdir(parents=True)
    model=LlamaForCausalLM(tiny_config()).to(torch.bfloat16).eval()
    args,_=accuracy.official_arguments(path)
    with patch.object(torch.Tensor,'cuda',lambda tensor,*a,**kw:tensor):
        model=apply_flatquant_to_llama_31(args,model)
    reparameterize_model(model)
    quantizers=gptq_utils.rtn_fwrd(model,torch.device('cpu'),args)
    torch.save(model,path/'quantized_model.pt')
    torch.save(quantizers,path/'weight_quantizers.pt')
    accuracy.write(path/'DONE.json',dict(status='COMPLETE',base_checkpoint=str(accuracy.BASE),
        official_commit=native.OFFICIAL_COMMIT,official_args=vars(args),fixture='tiny CPU structural test only',
        producer_torch=torch.__version__))


def no_gpu_benchmark(*args,**kwargs):
    raise AssertionError('CPU mapping tests must never execute GPU kernels or benchmarking')


class NativeExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temporary.name)
        cls.source=cls.root/'source'
        env=dict(os.environ)
        env['PYTHONPATH']=(':'.join(['/home/yanlongc/.prismquant/llama31-baseline-packages',
            '/home/yanlongc/FlatQuant-upstream','/home/yanlongc/.e13_packages',str(REPO)]))
        completed=subprocess.run(['/projects/nar/nar-validation/venv/bin/python',str(Path(__file__).resolve()),
            '--make-fixture',str(cls.source)],env=env,capture_output=True,text=True,timeout=120)
        if completed.returncode:
            raise RuntimeError(completed.stdout+'\n'+completed.stderr)
        # Triton decorators request a benchmark function at import. Supply a
        # sentinel solely in this CPU test; production imports are untouched.
        import triton.runtime.autotuner as autotuner
        cls.driver_patch=patch.object(autotuner.driver,'active',
            SimpleNamespace(get_benchmarker=lambda:no_gpu_benchmark))
        cls.driver_patch.start()
        # Only bypass HF's GPU-availability check while constructing CPU
        # modules. No forward, FlashAttention, or native kernel is invoked.
        cls.flash_check=patch.object(PreTrainedModel,'_check_and_enable_flash_attn_2',
            classmethod(lambda cls, config, *args, **kwargs: config))
        cls.flash_check.start()
        cls.packed=cls.root/'packed'
        cls.exported,_=native.export_checkpoint(cls.source,cls.packed)

    @classmethod
    def tearDownClass(cls):
        cls.flash_check.stop()
        cls.driver_patch.stop()
        cls.temporary.cleanup()

    def test_cross_version_packed_load_rope_and_every_used_tensor(self):
        self.assertTrue(json.loads((self.source/'DONE.json').read_text())['producer_torch'].startswith('2.11.'))
        self.assertTrue(torch.__version__.startswith('2.6.'))
        model,coverage,rope=native.build_native(tiny_config(),self.packed)
        self.assertEqual(coverage['packed_linear_count'],7)
        self.assertTrue(coverage['every_used_tensor_matches_export'])
        self.assertEqual(rope['max_abs_difference'],0)
        self.assertEqual(rope['shared_inv_freq_dtype'],'torch.float32')
        # Check native source preserves the real Llama-3.1 shared RoPE path.
        self.assertEqual(model.model.rotary_emb.rope_type,'llama3')
        self.assertIs(model.model.layers[0].self_attn.inp_trans_q.left_matrix,
                      model.model.layers[0].self_attn.left_matrix)
        self.assertIs(model.model.layers[0].mlp.inp_trans_u.right_matrix,
                      model.model.layers[0].mlp.right_matrix)

    def test_missing_real_projection_never_leaves_random_buffer(self):
        loader=native.load_module('flatquant_test_official_loader',native.OFFICIAL/'benchmarks/benchmark_model.py')
        mapped=loader.rename_keys(loader.load_from_safetensors(str(self.packed)))
        model=native.construct_native(tiny_config())
        del mapped['model_state_dict']['model.layers.0.self_attn.q_proj.weight']
        with self.assertRaisesRegex(ValueError,'Missing used native parameters.*q_proj.weight'):
            native.load_state_verified(model,mapped)

    def test_mapped_inference_export_matches_unmodified_eager_export(self):
        # Compare every exported tensor against the exact official function
        # called with eager source loading and its original autograd context.
        from flatquant.flat_utils import save_quantized_weights_with_safetensors
        from safetensors.torch import load_file
        before=native.sha(self.source/'quantized_model.pt')
        model=torch.load(self.source/'quantized_model.pt',map_location='cpu',weights_only=False)
        quantizers=torch.load(self.source/'weight_quantizers.pt',map_location='cpu',weights_only=False)
        args=SimpleNamespace(**json.loads((self.source/'DONE.json').read_text())['official_args'])
        args.exp_dir=str(self.root/'eager_reference')
        save_quantized_weights_with_safetensors(args,model,quantizers,sym=True)
        expected=load_file(str(Path(args.exp_dir)/'model.safetensors'))
        actual=load_file(str(self.packed/'model.safetensors'))
        self.assertEqual(set(actual),set(expected))
        for name in expected:
            self.assertEqual(actual[name].dtype,expected[name].dtype,name)
            self.assertTrue(torch.equal(actual[name],expected[name]),name)
        self.assertEqual(before,native.sha(self.source/'quantized_model.pt'))
        self.assertEqual(self.exported['host_memory'][-1]['stage'],'after_source_release')
        self.assertIn('mmap=True',self.exported['input_loading'])

    def test_export_reuse_and_corruption_detection(self):
        _,reused=native.export_checkpoint(self.source,self.packed)
        self.assertTrue(reused)
        manifest=json.loads((self.packed/'DONE.json').read_text())
        manifest['files']['model.safetensors']='invalid'
        native.write(self.packed/'DONE.json',manifest)
        try:
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                native.export_checkpoint(self.source,self.packed)
        finally:
            native.write(self.packed/'DONE.json',self.exported)

    def test_native_metadata_replaces_generic_fake_quant_note(self):
        record=dict(completed_sequences=141,timing=dict(execution_note='fake-quant'))
        path=self.root/'record.json'
        native.save_record(path,record)
        self.assertIn('packed INT4 CUDA',json.loads(path.read_text())['timing']['execution_note'])
        self.assertTrue(record['timing']['same_loaded_instance_after_ppl'])


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--make-fixture':
        make_fixture(Path(sys.argv[2]))
    else:
        unittest.main()
