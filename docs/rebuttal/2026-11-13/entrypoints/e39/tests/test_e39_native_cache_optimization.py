"""CPU provenance and real update-method parity checks; no GPU kernels execute."""
import ast
import copy
import functools
import hashlib
from pathlib import Path
import types
import typing
import unittest
from unittest.mock import patch

import torch
from nar import e39_native_cache_optimization as adapter


def official_method(native_calls):
    source=adapter.OFFICIAL_SOURCE.read_text()
    tree=ast.parse(source)
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='MultiLayerPagedKVCache4Bit')
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='update')
    def native_call(**kwargs):
        native_calls.append({k:v.clone() if torch.is_tensor(v) else v for k,v in kwargs.items()})
    def quantize(tensor,**kwargs):
        # Stub only the external packed kernel. All official update-method
        # transform/metadata/branch operations execute unchanged on CPU.
        return tensor[...,::2].to(torch.uint8),torch.ones_like(tensor[...,:1]),torch.zeros_like(tensor[...,:1])
    class AttentionStub:
        def __init__(self,*args,**kwargs):pass
        def forward(self,*args,**kwargs):pass
    namespace=dict(torch=torch,Optional=typing.Optional,Dict=typing.Dict,Any=typing.Any,functools=functools,
        _AttentionStub=AttentionStub,asym_quantize_and_pack_i4=quantize,
        init_kv_i4=native_call,append_kv_i4=native_call)
    exec(compile(ast.Module(body=[copy.deepcopy(method)],type_ignores=[]),str(adapter.OFFICIAL_SOURCE),'exec'),namespace)
    return namespace['update']


def cache():
    obj=types.SimpleNamespace(trans='matmul',org_head_dim=4,trans_dtype=torch.float32,device='cpu',
        page_size=16,n_layers=1,disable_quant=False,group_size=1,batch_size=1,
        pages=torch.zeros(1,2,dtype=torch.uint8),length=0,_needs_init=[True])
    obj.page_cnt_from_length=lambda length:1
    obj._ensure_page_cnt_per_batch=lambda count:None
    obj.get_cache_specs_for_flash_infer=lambda mask:dict(cache_fixture='CPU test')
    return obj


def inputs(length):
    key=torch.arange(length*8,dtype=torch.float32).reshape(1,length,2,4)/10
    value=key+2
    transform=torch.eye(4)*.5
    kwargs=dict(trans_matrix_k=transform,trans_matrix_k_inv_t=torch.eye(4)*2,
        trans_matrix_v=torch.eye(4),kclip_factor_a_max=torch.tensor(1.),
        kclip_factor_a_min=torch.tensor(1.),vclip_factor_a_max=torch.tensor(1.),vclip_factor_a_min=torch.tensor(1.))
    return key,value,kwargs


class NativeCacheOptimization(unittest.TestCase):
    def test_pinned_source_and_only_one_ast_statement_removed(self):
        before=adapter.OFFICIAL_SOURCE.read_bytes()
        self.assertEqual(hashlib.sha256(before).hexdigest(),adapter.OFFICIAL_SOURCE_SHA256)
        original=official_method([])
        optimized,audit=adapter._compile_optimized_update(original)
        self.assertEqual(audit['removed_statement_count'],1)
        self.assertTrue(audit['all_remaining_ast_nodes_unchanged'])
        self.assertNotEqual(audit['original_function_ast_sha256'],audit['transformed_function_ast_sha256'])
        self.assertEqual(adapter.OFFICIAL_SOURCE.read_bytes(),before)

    def test_actual_official_prefill_and_decode_keep_outputs_and_kernel_arguments(self):
        calls=[];original=official_method(calls);optimized,_=adapter._compile_optimized_update(original)
        def execute(method):
            calls.clear();obj=cache();key,value,kwargs=inputs(3)
            out=method(obj,key,value,0,kwargs)
            first=tuple(item.clone() for item in out)
            key,value,kwargs=inputs(1)
            out=method(obj,key,value,0,kwargs)
            return first,out.keywords,copy.deepcopy(calls),obj.length
        with patch.object(torch,'randn',wraps=torch.randn) as random:
            a=execute(original)
            self.assertEqual(random.call_count,2)
        with patch.object(torch,'randn',side_effect=AssertionError('Unused draw still executed')):
            b=execute(optimized)
        self.assertEqual(a[3],4);self.assertEqual(a[3],b[3])
        for x,y in zip(a[0],b[0]):self.assertTrue(torch.equal(x,y))
        self.assertEqual(a[1],b[1])
        self.assertEqual(len(a[2]),2)
        for old,new in zip(a[2],b[2]):
            self.assertEqual(old.keys(),new.keys())
            for key,value in old.items():
                if torch.is_tensor(value):self.assertTrue(torch.equal(value,new[key]),key)
                else:self.assertEqual(value,new[key])

    def test_reject_changed_adjacent_overwrite_and_missing_assignment(self):
        import inspect
        source=inspect.getsource(official_method([]))
        with self.assertRaisesRegex(ValueError,'immediately overwritten'):
            adapter._remove_one_dead_assignment(source.replace(adapter._OVERWRITE,'self.head_dim = self.trans_matrix_v'))
        with self.assertRaisesRegex(ValueError,'exactly one'):
            adapter._remove_one_dead_assignment(source.replace(adapter._DEAD,'pass'))

    def test_install_is_opt_in_reversible_and_preserves_source(self):
        original=official_method([])
        cls=type('FixtureCache',(),dict(update=original))
        module=types.SimpleNamespace(__file__=str(adapter.OFFICIAL_SOURCE),MultiLayerPagedKVCache4Bit=cls)
        before=adapter.OFFICIAL_SOURCE.read_bytes()
        try:
            with patch.object(adapter.importlib,'import_module',return_value=module):
                official=adapter.install_native_cache_optimization('official')
                self.assertIs(cls.update,original);self.assertFalse(official['applied'])
                changed=adapter.install_native_cache_optimization('remove-dead-random')
                self.assertIsNot(cls.update,original);self.assertTrue(changed['applied'])
                same=cls.update
                adapter.install_native_cache_optimization('remove-dead-random')
                self.assertIs(cls.update,same)
                adapter.install_native_cache_optimization('official')
                self.assertIs(cls.update,original)
        finally:
            adapter._INSTALLED.pop(cls,None)
        self.assertEqual(adapter.OFFICIAL_SOURCE.read_bytes(),before)

if __name__=='__main__':unittest.main()
