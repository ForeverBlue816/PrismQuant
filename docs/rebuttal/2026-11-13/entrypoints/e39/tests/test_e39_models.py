"""CPU integration checks for model routing and real cached decode semantics."""
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nar import e39_flatquant as runner


class Models(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.logging_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.init_functions = {name: getattr(torch.nn.init, name)
            for name in ('kaiming_uniform_', 'uniform_', 'normal_')}

    def tearDown(self):
        logging.disable(self.logging_level)
        for name, function in self.init_functions.items():
            setattr(torch.nn.init, name, function)

    def test_default_8b_arguments_and_distinct_3b_identity(self):
        from flatquant import args_utils
        initial_choices = list(args_utils.supported_models)
        with tempfile.TemporaryDirectory() as temporary:
            a, _ = runner.official_arguments(Path(temporary)/'8b')
            b, _ = runner.official_arguments(Path(temporary)/'3b', 'llama32_3b')
        self.assertEqual(a.model, str(runner.BASE))
        self.assertEqual(a.model_name, 'Llama-3.1-8B')
        self.assertEqual(b.model_name, 'Llama-3.2-3B')
        self.assertNotEqual(a.model, b.model)
        self.assertNotEqual(a.exp_dir, b.exp_dir)
        self.assertEqual(args_utils.supported_models, initial_choices)
        for args in (a, b):
            self.assertEqual((args.w_bits, args.a_bits, args.k_bits, args.v_bits), (4,4,4,4))
            self.assertFalse(args.gptq)
            self.assertEqual((args.epochs,args.cali_bsz,args.flat_lr), (15,4,0.005))

    def test_3b_uses_modern_official_loader_without_changing_8b_dispatch(self):
        from flatquant import model_utils
        with patch.object(model_utils, 'get_llama_31', return_value=('3b','modern')) as modern, \
             patch.object(model_utils, 'get_model', return_value=('8b','original')) as original:
            self.assertEqual(runner.load_official_model('llama32_3b'), ('3b','modern'))
            modern.assert_called_once_with(str(runner.MODELS['llama32_3b']['base']), None)
            self.assertEqual(runner.load_official_model('llama31_8b'), ('8b','original'))
            original.assert_called_once_with(str(runner.BASE), None)

    def test_official_quantized_3b_architecture_cached_decode_matches_full_prefix(self):
        import numpy as np
        from flatquant.model_tools.llama31_utils import apply_flatquant_to_llama_31
        from flatquant.flat_utils import reparameterize_model
        import gptq_utils
        torch.manual_seed(0); np.random.seed(0)
        config = LlamaConfig.from_pretrained(runner.MODELS['llama32_3b']['base'], local_files_only=True)
        original_rope = dict(config.rope_scaling)
        config.hidden_size=256; config.intermediate_size=256
        config.num_hidden_layers=1; config.num_attention_heads=2; config.num_key_value_heads=1
        config.vocab_size=64; config.pad_token_id=0; config.bos_token_id=1; config.eos_token_id=2
        config._attn_implementation_internal='eager'
        model=LlamaForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = runner.official_arguments(Path(temporary), 'llama32_3b')
            # The official constructor puts activation-statistic buffers on
            # CUDA unconditionally. Keep just those allocations on CPU for
            # this arithmetic/cache integration test; production is unchanged.
            with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor):
                model = apply_flatquant_to_llama_31(args, model)
            reparameterize_model(model)
            gptq_utils.rtn_fwrd(model, torch.device('cpu'), args)
        model.eval()
        ids=torch.tensor([[3,7,11,19,23,31]], dtype=torch.long)
        with torch.inference_mode():
            full=model(input_ids=ids,use_cache=False).logits[:,-1]
            prefix=model(input_ids=ids[:,:-1],use_cache=True)
            output=model(input_ids=ids[:,-1:],use_cache=True,past_key_values=prefix.past_key_values)
        self.assertEqual(config.rope_scaling, original_rope)
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertEqual(output.logits.shape, (1,1,64))
        cache=output.past_key_values
        length=cache.get_seq_length() if hasattr(cache,'get_seq_length') else cache[0][0].shape[-2]
        self.assertEqual(length, ids.shape[1])
        torch.testing.assert_close(output.logits[:,-1],full,rtol=1e-4,atol=1e-4)


if __name__ == '__main__':
    unittest.main()
