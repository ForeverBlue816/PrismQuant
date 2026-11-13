"""CPU checks for the 8B same-instance audit and no-quality-veto lifecycle."""
import copy
import contextlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import torch
from nar import e38_8b_integer as runner
from nar.e38_8b_integer import (MODEL, accuracy_then_timing, compare_logits_cpu,
    candidate_ppl, make_switchable, parse_args, release_reference, stats, timing_cohort,
    validate_cache_identity)
from nar.e38_integer_operator import PackedSymmetric4


class EightBInteger(unittest.TestCase):
    def test_streaming_comparison_all_vocabulary_and_tie_rule(self):
        r=torch.tensor([[[1.,1.,0.],[2.,3.,0.],[1.,0.,-1.]]],dtype=torch.bfloat16)
        a=r.clone();a[0,0,1]=2.;a[0,2,2]=-3.
        result=compare_logits_cpu(r,a,token_block=1)
        self.assertEqual(result['max_abs_logit_difference'],2.)
        self.assertEqual(result['argmax_matches'],2)
        self.assertEqual(result['positions'],3)
        self.assertEqual(result['reference_argmax'].tolist(),[[0,1,0]])
        expected=float((a.float()-r.float()).norm()/r.float().norm())
        self.assertAlmostEqual(result['relative_l2'],expected,places=6)
        a[0,1,1]=float('nan')
        bad=compare_logits_cpu(r,a,token_block=2)
        self.assertFalse(bad['finite']);self.assertEqual(bad['nonfinite_candidate'],1)
        self.assertIsNone(bad['max_abs_logit_difference'])

    def test_failed_accuracy_still_releases_then_times(self):
        events=[]
        def quality():
            events.append('accuracy');raise RuntimeError('measurement failed')
        result=accuracy_then_timing(quality,lambda:events.append('timing') or 'done',lambda:events.append('release'))
        self.assertEqual(result,'done');self.assertEqual(events,['accuracy','release','timing'])
        events.clear()
        accuracy_then_timing(lambda:events.append('large numerical difference') or {'status':'NONFINITE'},
            lambda:events.append('timing'),lambda:events.append('release'))
        self.assertEqual(events[-1],'timing')

    def test_cache_identity_does_not_inspect_accuracy(self):
        identity={'directory':'actual8b','files':[{'sha256':'abc'}]}
        manifest=dict(model=MODEL,rotation='nar_k8',seed=0,checkpoint=identity,status='COMPLETE',
            layers=[{}]*32,ppl_difference=1000.)
        validate_cache_identity(manifest,'nar_k8',0,identity)
        with self.assertRaises(ValueError):validate_cache_identity(manifest,'hadamard',0,identity)
        changed=copy.deepcopy(identity);changed['files'][0]['sha256']='different'
        with self.assertRaises(ValueError):validate_cache_identity(manifest,'nar_k8',0,changed)

    def test_switch_release_retains_integer_weights_and_result(self):
        q=torch.arange(-8,8,dtype=torch.int8).repeat(8,8)
        scales=torch.full((8,),.125,dtype=torch.float32)
        w=(q.float()*scales[:,None]).bfloat16()
        linear=torch.nn.Linear(128,8,bias=False,dtype=torch.bfloat16)
        linear.weight.data.copy_(w)
        packed,_=PackedSymmetric4.from_weight(w,scales)
        wrapper=make_switchable(linear,packed)
        x=torch.arange(128,dtype=torch.float32).bfloat16()[None,:]
        reference=wrapper(x)
        wrapper.use_integer=True;before=wrapper(x)
        pointer=wrapper.integer.packed.data_ptr()
        release_reference([wrapper])
        self.assertFalse(hasattr(wrapper,'reference'));self.assertEqual(wrapper.integer.packed.data_ptr(),pointer)
        self.assertTrue(torch.equal(before,wrapper(x)))
        self.assertEqual(reference.shape,before.shape)

    def test_parser_limits_scope_and_controlled_workload(self):
        common=['--output','run.json','--artifact-dir','artifact','--packed-cache','cache']
        args=parse_args(common)
        self.assertEqual(args.rotation,'nar_k8');self.assertEqual(args.repetitions,20)
        self.assertEqual(args.decode_steps-args.discard_decode_steps,120)
        self.assertIn('llama31_8b',str(args.checkpoint_directory))
        self.assertEqual(parse_args(common+['--phase','ppl-timing']).phase,'ppl-timing')
        with self.assertRaises(SystemExit):parse_args(common+['--rotation','nar_kmax'])
        with self.assertRaises(SystemExit):parse_args(common+['--seed','1','--phase','eight-tasks'])

    def test_checkpoint_parent_cannot_record_one_model_and_load_another(self):
        with self.assertRaises(SystemExit):
            parse_args(['--output','run.json','--artifact-dir','artifact','--packed-cache','cache',
                '--checkpoint-directory','/tmp/wrong_model/gptq_nar_k8_seed0'])

    def test_timing_statistics_retain_population(self):
        result=stats([1.,2.,3.,4.])
        self.assertEqual(result['n'],4);self.assertEqual(result['median'],2.5)
        self.assertEqual(result['p25'],1.75);self.assertEqual(result['p75'],3.25)


class IntegerTimingProtocol(unittest.TestCase):
    def run_cohort(self,phase,nonfinite=False,full_logits=False,stale=False):
        events=[];calls=[]
        class Cache:
            def __init__(self):self.length=0
            def get_seq_length(self):return self.length
        def new_cache(_model):
            events.append('cache');return Cache()
        def model(input_ids,past_key_values,use_cache,logits_to_keep):
            events.append('forward')
            calls.append((input_ids.shape[1],past_key_values.length,use_cache,logits_to_keep))
            if not stale:past_key_values.length+=input_ids.shape[1]
            width=input_ids.shape[1] if full_logits else logits_to_keep
            return SimpleNamespace(logits=torch.full((1,width,4),float('nan') if nonfinite else 0.),
                past_key_values=past_key_values)
        ticks=iter(range(100))
        def clock():events.append('clock');return next(ticks)
        args=SimpleNamespace(repetitions=2,warmups=1,decode_steps=5,discard_decode_steps=2,output=None)
        record={'timing':{'cohorts':{}}}
        with contextlib.ExitStack() as stack:
            for name,value in [('synchronize',None),('empty_cache',None),('reset_peak_memory_stats',None),
                ('memory_allocated',123),('memory_reserved',456),('max_memory_allocated',789),
                ('max_memory_reserved',999)]:
                stack.enter_context(mock.patch.object(torch.cuda,name,return_value=value))
            stack.enter_context(mock.patch.object(runner,'new_cache',new_cache))
            stack.enter_context(mock.patch.object(runner,'exclusive_guard',return_value={'status':'PASS'}))
            stack.enter_context(mock.patch.object(runner,'flush'))
            stack.enter_context(mock.patch.object(runner.time,'perf_counter',clock))
            result=timing_cohort(model,torch.arange(4)[None,:],args,record,phase,'cpu-protocol-test')
        return result,events,calls

    def test_prefill_includes_fresh_cache_construction(self):
        result,events,calls=self.run_cohort('prefill')
        self.assertEqual(events,['clock','cache','forward','clock']*3)
        self.assertEqual(calls,[(4,0,True,1)]*3)
        self.assertTrue(result['cache_creation_in_timed_region'])
        self.assertEqual(result['cache_lengths'],[4,4])

    def test_decode_excludes_preparation_and_advances_real_cache(self):
        result,events,calls=self.run_cohort('decode')
        self.assertEqual(events,['cache']+['forward']*3+['clock']+['forward']*3+['clock']
            +['cache']+['forward']*3+['clock']+['forward']*3+['clock']
            +['cache']+['forward']*3+['clock']+['forward']*3+['clock'])
        self.assertEqual(calls,([(4,0,True,1)]+[(1,x,True,1) for x in range(4,9)])*3)
        self.assertFalse(result['cache_creation_in_timed_region'])
        self.assertEqual(result['cache_lengths'],[9,9])
        self.assertEqual(result['tokens_per_second']['median'],3.)

    def test_full_position_logits_and_stale_cache_are_invalid_workloads(self):
        with self.assertRaisesRegex(RuntimeError,'last-position logits'):
            self.run_cohort('prefill',full_logits=True)
        with self.assertRaisesRegex(RuntimeError,'cache length'):
            self.run_cohort('decode',stale=True)

    def test_nonfinite_logits_are_recorded_without_timing_veto(self):
        result,_,_=self.run_cohort('decode',nonfinite=True)
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(result['output_finite_per_repetition'],[False,False])

    def test_candidate_ppl_saves_all_nll_without_reference_or_logit_comparison(self):
        wrapper=SimpleNamespace(use_integer=False)
        calls=[]
        def model(input_ids,labels,use_cache):
            self.assertTrue(wrapper.use_integer)
            self.assertIs(input_ids,labels);self.assertFalse(use_cache)
            calls.append(input_ids.clone())
            return SimpleNamespace(loss=torch.tensor(float(len(calls))/100))
        tokens=torch.arange(141*4).reshape(141,4)
        record={}
        with mock.patch.object(torch.Tensor,'cuda',lambda x:x), \
             mock.patch.object(torch.cuda,'empty_cache'),mock.patch.object(runner,'flush'), \
             mock.patch.object(runner,'compare_logits_cpu',side_effect=AssertionError('unwanted comparison')):
            result=candidate_ppl(model,[wrapper],tokens,record,None)
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(result['completed_sequences'],141)
        self.assertEqual(result['tokens_scored'],141*3)
        self.assertEqual(result['model_instance_id'],id(model))
        self.assertTrue(torch.equal(torch.cat(calls),tokens))
        self.assertEqual(len({x['input_ids_sha256'] for x in result['chunks']}),141)
        self.assertAlmostEqual(result['mean_nll'],.71,places=6)
        self.assertNotIn('equivalence',record)


if __name__=='__main__':unittest.main()
