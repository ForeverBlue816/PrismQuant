import contextlib
import types
import unittest
from unittest import mock

import torch
from scripts import baseline_timing as timing


class Cache:
    def __init__(self, length): self.length=length
    def get_seq_length(self): return self.length


class CachedModel:
    def __init__(self, stale=False, nonfinite=False):
        self.calls=[];self.stale=stale;self.nonfinite=nonfinite
    def eval(self): return self
    def __call__(self, input_ids, past_key_values=None, use_cache=True, return_dict=True, num_logits_to_keep=0):
        prior=0 if past_key_values is None else past_key_values.get_seq_length()
        self.calls.append((input_ids.shape[1], prior))
        length=prior if self.stale and prior else prior+input_ids.shape[1]
        logits=torch.full((1,num_logits_to_keep or input_ids.shape[1],4),float('nan') if self.nonfinite else 0.)
        return types.SimpleNamespace(logits=logits,past_key_values=Cache(length))


class TimingProtocol(unittest.TestCase):
    def run_timing(self, model, exclusive=True):
        tensor_to=torch.Tensor.to;full=torch.full
        def cpu_to(tensor,*args,**kwargs):
            if kwargs.get('device')=='cuda':kwargs['device']='cpu'
            return tensor_to(tensor,*args,**kwargs)
        def cpu_full(*args,**kwargs):
            if kwargs.get('device')=='cuda':kwargs['device']='cpu'
            return full(*args,**kwargs)
        record={}
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(torch.Tensor,'to',cpu_to))
            stack.enter_context(mock.patch.object(torch,'full',cpu_full))
            for name,value in [('get_device_name','CPU protocol test'),('synchronize',None),
                ('empty_cache',None),('reset_peak_memory_stats',None),
                ('max_memory_allocated',123),('max_memory_reserved',456)]:
                stack.enter_context(mock.patch.object(torch.cuda,name,return_value=value))
            stack.enter_context(mock.patch.object(timing,'exclusivity',return_value={
                'status':'EXCLUSIVE' if exclusive else 'CONTENDED_OR_UNVERIFIED'}))
            timing.measure_timing(model,torch.arange(4)[None],record,None,lambda *_:None,
                warmup=1,repetitions=2,prefix=4,decode_steps=5,discard=2)
        return record['timing']

    def test_fresh_prefix_and_real_incremental_decode_every_repetition(self):
        model=CachedModel();result=self.run_timing(model)
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(model.calls.count((4,0)),6)
        self.assertEqual([prior for width,prior in model.calls if width==1],list(range(4,9))*3)
        self.assertEqual(result['workloads']['decode']['cache_lengths'],[9,9])
        self.assertEqual(result['workloads']['decode']['seconds']['n'],2)
        self.assertEqual(result['model_object_id'],result['same_model_object_id_after_timing'])

    def test_stale_cache_never_yields_a_decode_throughput(self):
        result=self.run_timing(CachedModel(stale=True))
        self.assertEqual(result['status'],'PARTIAL')
        self.assertNotIn('tokens_per_second',result['workloads']['decode'])

    def test_numerical_values_do_not_veto_timing(self):
        result=self.run_timing(CachedModel(nonfinite=True))
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(result['workloads']['prefill']['finite_last_position_logits'],[False,False])

    def test_shared_gpu_is_not_timed(self):
        model=CachedModel();result=self.run_timing(model,exclusive=False)
        self.assertEqual(result['status'],'NOT_MEASURED')
        self.assertEqual(model.calls,[])


if __name__=='__main__':unittest.main()
