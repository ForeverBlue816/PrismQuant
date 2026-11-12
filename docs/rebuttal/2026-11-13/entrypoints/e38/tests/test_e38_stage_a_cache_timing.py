"""CPU control-flow tests for the real Stage A timing entry point."""
import contextlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'quarot-llama3'))
from e38 import stage_a_8b as timing


class StageACacheTiming(unittest.TestCase):
    def run_protocol(self, policy=None):
        events=[];caches=[];inside=False;clock_value=0
        def clock():
            nonlocal inside,clock_value
            inside=not inside;clock_value+=1
            events.append(('timer_start' if inside else 'timer_end',))
            return float(clock_value)
        def make_cache(model,batch,capacity):
            cache=types.SimpleNamespace(index=len(caches),capacity=capacity,length=0)
            caches.append(cache);events.append(('make',cache.index,inside))
            return cache
        def clear(cache):
            cache.length=0;events.append(('clear',cache.index,inside))
        class Model:
            def __call__(self,ids,past_key_values,use_cache):
                cache=past_key_values
                events.append(('forward',cache.index,int(ids.shape[1]),cache.length,inside))
                cache.length+=int(ids.shape[1])
                return types.SimpleNamespace(logits=torch.zeros(1,1,4))
        model=Model();record={};real_full=torch.full
        def cpu_full(*args,**kwargs):
            if kwargs.get('device')=='cuda':kwargs['device']='cpu'
            return real_full(*args,**kwargs)
        adapter=types.ModuleType('e28_v2.cache_adapter');adapter.clear=clear
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules,{'e28_v2.cache_adapter':adapter}))
            stack.enter_context(mock.patch.object(torch.Tensor,'cuda',lambda value:value))
            stack.enter_context(mock.patch.object(torch,'full',cpu_full))
            for name,value in [('synchronize',None),('empty_cache',None),('reset_peak_memory_stats',None),
                ('memory_allocated',100),('max_memory_allocated',200),('max_memory_reserved',300)]:
                stack.enter_context(mock.patch.object(torch.cuda,name,return_value=value))
            stack.enter_context(mock.patch.object(timing.backend,'make_cache',side_effect=make_cache))
            stack.enter_context(mock.patch.object(timing,'exclusivity',return_value={'status':'EXCLUSIVE'}))
            stack.enter_context(mock.patch.object(timing.common,'write'))
            stack.enter_context(mock.patch.object(timing.time,'perf_counter',side_effect=clock))
            kwargs={} if policy is None else dict(cache_timing=policy)
            timing.measure_timing(model,torch.arange(2048)[None],record,None,
                warmup=1,repetitions=2,**kwargs)
        self.assertFalse(inside)
        self.assertEqual(record['timing']['status'],'COMPLETE')
        return record['timing'],events,caches

    def test_fresh_prefill_includes_new_cache_and_clear_each_time(self):
        result,events,caches=self.run_protocol('fresh')
        self.assertEqual(len(caches),6)
        self.assertEqual([cache.capacity for cache in caches],[2048]*3+[2176]*3)
        for cache in caches[:3]:
            related=[event for event in events if len(event)>1 and event[1]==cache.index]
            self.assertEqual(related,[('make',cache.index,True),('clear',cache.index,True),
                ('forward',cache.index,2048,0,True)])
        self.assertTrue(result['workloads']['prefill']['cache_construction_inside_timer'])
        self.assertFalse(result['workloads']['prefill']['cache_reset_and_decode_prefix_outside_timer'])

    def test_fresh_decode_builds_new_prefix_before_timer_and_appends_cache(self):
        result,events,caches=self.run_protocol('fresh')
        for cache in caches[3:]:
            related=[event for event in events if len(event)>1 and event[1]==cache.index]
            self.assertEqual(related[:3],[('make',cache.index,False),('clear',cache.index,False),
                ('forward',cache.index,2048,0,False)])
            decode=related[3:]
            self.assertEqual(len(decode),128)
            self.assertEqual([event[3] for event in decode],list(range(2048,2176)))
            self.assertEqual([event[4] for event in decode],[False]*8+[True]*120)
            self.assertEqual(cache.length,2176)
        self.assertTrue(result['workloads']['decode']['cache_reset_and_decode_prefix_outside_timer'])
        self.assertFalse(result['workloads']['decode']['cache_construction_inside_timer'])

    def test_default_reuses_one_cache_per_workload_and_clears_before_timer(self):
        result,events,caches=self.run_protocol()
        self.assertEqual(result['cache_timing'],'reused')
        self.assertEqual(len(caches),2)
        self.assertEqual([event for event in events if event[0]=='make'],[('make',0,False),('make',1,False)])
        for cache in caches:
            clears=[event for event in events if event[0]=='clear' and event[1]==cache.index]
            self.assertEqual(clears,[('clear',cache.index,False)]*3)
            prefill=[event for event in events if event[0]=='forward' and event[1]==cache.index and event[2]==2048]
            self.assertEqual(len(prefill),3)
            self.assertTrue(all(event[3]==0 for event in prefill))
        self.assertEqual([event[4] for event in events if event[0]=='forward' and event[1]==0],[True]*3)


if __name__=='__main__':unittest.main()
