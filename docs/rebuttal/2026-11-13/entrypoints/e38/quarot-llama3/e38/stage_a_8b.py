"""8B trained FP16 / native per-token INT4 accuracy then same-instance timing.

The packed rows use actual E14 default GPTQ checkpoints and original rotations.
Native per-head KV4 is consumed by both prefill and decode. This is a different
activation/KV protocol from Table1. Numerical differences never veto timing.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

from e38 import fp16 as common
from e38 import stage_a_8b_model as backend

REPO = Path(__file__).resolve().parents[2]
TOKENS = backend.WORK/'cache/tokenized/ffcc87608d60-wikitext2-test-full-l2048.pt'
SOURCES = ('quarot-llama3/e38/stage_a_8b.py', 'quarot-llama3/e38/stage_a_8b_model.py',
    'quarot-llama3/e38/fp16.py', 'quarot-llama3/modeling_llama3.py',
    'quarot-llama3/e28_v2/cache_adapter.py', 'nar/e14_w4a4kv4.py',
    'nar/activation_experiments.py', 'nar/experiment.py', 'nar/e12_wy.py',
    'nar/e38_checkpoint_probe.py', 'quarot-llama3/e38/optimized_rotation.py')


def exclusivity():
    rows = []
    for line in common.command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']).splitlines():
        if line.strip():
            uuid, pid = [x.strip() for x in line.split(',')]
            rows.append(dict(gpu_uuid=uuid, pid=int(pid)))
    own = [row for row in rows if row['pid'] == os.getpid()]
    uuids = {row['gpu_uuid'] for row in own}
    other = [row for row in rows if row['gpu_uuid'] in uuids and row['pid'] != os.getpid()]
    return dict(status='EXCLUSIVE' if own and not other else 'CONTENDED_OR_UNVERIFIED',
                own=own, other=other, processes=rows, load_average=list(os.getloadavg()),
                host_cpu_count=os.cpu_count(), checked_utc=common.now())


def measure_ppl(model, windows, record, output):
    import torch
    from e28_v2.cache_adapter import clear
    result = dict(status='RUNNING', chunks=[], requested_sequences=141, sequence_length=2048,
                  logits='all positions; FP32 cross entropy', same_loaded_instance=True)
    record['ppl'] = result
    model.logits_last_only = False
    with torch.inference_mode():
        for index, sequence in enumerate(windows):
            cache = None
            out = None
            row = dict(chunk=index, tokens_scored=int(sequence.numel())-1)
            try:
                ids = sequence[None].cuda().long()
                cache = backend.make_cache(model, 1, int(ids.shape[1]))
                clear(cache)
                out = model(ids, past_key_values=cache, use_cache=True)
                row.update(status='COMPLETE', nll=common.logit_nll(out.logits, ids))
            except Exception as exc:
                row.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
            result['chunks'].append(row)
            result['completed_sequences'] = sum(x['status'] == 'COMPLETE' for x in result['chunks'])
            common.write(output, record)
            del cache, out
            if row['status'] != 'COMPLETE':
                gc.collect(); torch.cuda.empty_cache()
            if index % 16 == 0:
                print(json.dumps(dict(stage='ppl', **row)), flush=True)
    valid = [row for row in result['chunks'] if row['status'] == 'COMPLETE']
    if len(valid) == 141 and len(windows) == 141:
        try:
            value = math.exp(sum(x['nll']*x['tokens_scored'] for x in valid)/sum(x['tokens_scored'] for x in valid))
            result.update(status='COMPLETE', value=value)
        except OverflowError:
            result.update(status='FAILED', value=None, nonfinite_value='inf',
                          reason='Corpus PPL overflow; timing still attempted')
    else:
        result.update(status='PARTIAL', value=None, reason='Failed or missing sequences; timing still attempted')
    common.write(output, record)


def measure_timing(model, windows, record, output, repetitions=20, warmup=10, batch=1,
                   cache_timing='reused'):
    import torch
    from e28_v2.cache_adapter import clear
    if cache_timing not in ('reused', 'fresh'):
        raise ValueError('cache_timing must be reused or fresh')
    record['timing'] = dict(status='RUNNING', same_loaded_instance_after_ppl=True,
        mode='eager synchronized wall time; no CUDA Graph claim',
        logits='last position only, matched across all Stage A rows',
        cache='all pages/scales/length/init flags cleared; decode prefix rebuilt before every repetition',
        cache_timing=cache_timing,
        cache_boundaries=dict(
            prefill=('new cache construction and clear inside each timer, before forward'
                     if cache_timing == 'fresh' else 'one cache per workload; clear before each timer'),
            decode=('new cache construction and clear before each prefix, outside timer'
                    if cache_timing == 'fresh' else 'one cache per workload; clear before each prefix, outside timer'),
            decode_prefix_and_discard='outside timer for both policies'),
        memory_scope='peak allocated/reserved includes decode prefix construction outside its latency timer',
        prefix=2048, decode_steps=128, discarded_decode_steps=8, measured_decode_steps=120,
        fixed_decode_token_id=100,
        warmup=warmup, repetitions=repetitions, batch_size=batch, workloads={})
    model.logits_last_only = True
    ids = windows[:batch].cuda().long()
    next_ids = torch.full((batch,128),100,dtype=torch.long,device='cuda')
    guards = record['timing']['guards'] = []
    guard = exclusivity(); guards.append(guard)
    if guard['status'] != 'EXCLUSIVE':
        record['timing'].update(status='NOT_MEASURED', reason='GPU is not verifiably exclusive')
        common.write(output, record)
        return
    with torch.inference_mode():
        for phase in ('prefill', 'decode'):
            result = dict(status='RUNNING', samples_seconds=[], finite_output_samples=[],
                cache_reset_and_decode_prefix_outside_timer=cache_timing == 'reused' or phase == 'decode',
                fresh_cache_per_repetition=cache_timing == 'fresh',
                cache_construction_inside_timer=cache_timing == 'fresh' and phase == 'prefill')
            record['timing']['workloads'][phase] = result
            capacity = 2176 if phase == 'decode' else 2048
            cache = backend.make_cache(model, batch, capacity) if cache_timing == 'reused' else None
            def one_run():
                run_cache = cache
                if cache_timing == 'fresh' and phase == 'decode':
                    run_cache = backend.make_cache(model, batch, capacity)
                if cache_timing == 'reused' or phase == 'decode':
                    clear(run_cache)
                if phase == 'decode':
                    model(ids, past_key_values=run_cache, use_cache=True)
                    for position in range(8):
                        model(next_ids[:, position:position+1], past_key_values=run_cache, use_cache=True)
                torch.cuda.synchronize()
                begin = time.perf_counter()
                if cache_timing == 'fresh' and phase == 'prefill':
                    run_cache = backend.make_cache(model, batch, capacity)
                    clear(run_cache)
                if phase == 'decode':
                    for position in range(8, 128):
                        out = model(next_ids[:, position:position+1], past_key_values=run_cache, use_cache=True)
                else:
                    out = model(ids, past_key_values=run_cache, use_cache=True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-begin
                finite = bool(torch.isfinite(out.logits).all())
                return elapsed, finite
            try:
                for _ in range(warmup): one_run()
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                result['allocated_before_bytes'] = torch.cuda.memory_allocated()
                for repeat in range(repetitions):
                    guard = exclusivity(); guards.append(guard)
                    if guard['status'] != 'EXCLUSIVE':
                        result.update(status='CONTENDED', reason='Samples retained, no timing summary')
                        break
                    duration, finite = one_run()
                    result['samples_seconds'].append(duration)
                    result['finite_output_samples'].append(finite)
                    result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    common.write(output, record)
                final_guard = exclusivity(); guards.append(final_guard)
                if len(result['samples_seconds']) == repetitions and final_guard['status'] == 'EXCLUSIVE':
                    result.update(status='COMPLETE', seconds=common.sample_summary(result['samples_seconds']))
                    if phase == 'decode':
                        result['milliseconds_per_token'] = common.sample_summary([x*1000/120 for x in result['samples_seconds']])
                    else:
                        result['tokens_per_second'] = common.sample_summary([batch*2048/x for x in result['samples_seconds']])
                elif result['status'] == 'RUNNING':
                    result.update(status='CONTENDED', reason='Final exclusivity check failed; no summary')
            except Exception as exc:
                result.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
            del cache
            gc.collect(); torch.cuda.empty_cache()
            common.write(output, record)
    record['timing']['status'] = ('COMPLETE' if all(x['status'] == 'COMPLETE'
        for x in record['timing']['workloads'].values()) else 'PARTIAL')
    common.write(output, record)


def run_eight_tasks(model, harness, record, output):
    import torch
    from transformers import AutoTokenizer
    from e28_v2.cache_adapter import clear
    from nar import e38_direct_accuracy as tasks
    evaluation, hflm, manager_class = harness
    class NativeHFLM(hflm):
        def _model_call(self, inps, attn_mask=None, labels=None):
            if attn_mask is not None or labels is not None:
                raise ValueError('Only causal likelihood is supported')
            self.model.logits_last_only = False
            cache = backend.make_cache(self.model, int(inps.shape[0]), int(inps.shape[1]))
            clear(cache)
            with torch.inference_mode():
                return self.model(inps, past_key_values=cache, use_cache=True).logits
    tokenizer = AutoTokenizer.from_pretrained(backend.BASE, use_fast=True, local_files_only=True)
    lm = NativeHFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, max_batch_size=1, max_length=2048)
    tasks.run_tasks(lm, manager_class(), record, output, evaluator=evaluation.simple_evaluate, batch_size=1)


def profile_components(model, windows, record, output):
    """Separate diagnostic pass; inclusive CUDA-event groups are not additive."""
    import types
    import torch
    from e28_v2.cache_adapter import clear
    result = dict(status='RUNNING', scope='independent diagnostic pass after main timing',
        units='CUDA-event milliseconds', interpretation='inclusive, non-additive; does not partition end-to-end wall latency',
        categories={'act_quant_pack':'native per-token quantizer including INT4 packing',
                    'linear':'Linear4bit CUTLASS+dequant or FP16 Linear; R2 post-hook excluded',
                    'r4':'actual full original E14 online R4 including signs/permutation/Hadamard/WY',
                    'r2':'optimized original E14 online R2 after v_proj, when exposed as a module',
                    'kv_cache_update':'native packed write, metadata and prefill unpack/inverse-H128'}, workloads={})
    record['component_profile'] = result
    guard = exclusivity()
    result['exclusivity'] = guard
    if guard['status'] != 'EXCLUSIVE':
        result.update(status='NOT_MEASURED',reason='GPU not verifiably exclusive')
        common.write(output,record)
        return
    ids = windows[:1].cuda().long()
    token = torch.full((1,1),100,dtype=torch.long,device='cuda')
    model.logits_last_only = True
    with torch.inference_mode():
        for phase in ('prefill','decode'):
            cache = backend.make_cache(model,1,2176 if phase == 'decode' else 2048)
            clear(cache)
            if phase == 'decode':
                model(ids,past_key_values=cache,use_cache=True)
                for _ in range(8): model(token,past_key_values=cache,use_cache=True)
            pairs = {key:[] for key in result['categories']}
            handles = []
            starts = {}
            def pre(module,_inputs):
                event=torch.cuda.Event(enable_timing=True);event.record()
                starts[id(module)]=event
            def post(category):
                def finish(module,_inputs,_output):
                    event=torch.cuda.Event(enable_timing=True);event.record()
                    pairs[category].append((starts.pop(id(module)),event))
                return finish
            for module in model.modules():
                name=type(module).__name__
                category=('act_quant_pack' if name=='Quantizer' else 'r2' if name=='FastOriginalE14R2'
                          else 'r4' if name in ('OriginalE14R4','FastOriginalE14R4')
                          else 'linear' if name=='Linear4bit' or isinstance(module,torch.nn.Linear) else None)
                if category:
                    handles.append(module.register_forward_pre_hook(pre))
                    handles.append(module.register_forward_hook(post(category),prepend=True))
            original_update=cache.update
            def traced_update(self,*args,**kwargs):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                value=original_update(*args,**kwargs)
                end.record();pairs['kv_cache_update'].append((start,end))
                return value
            cache.update=types.MethodType(traced_update,cache)
            steps=8 if phase=='decode' else 1
            try:
                torch.cuda.synchronize();begin=time.perf_counter()
                for _ in range(steps):
                    model(token if phase=='decode' else ids,past_key_values=cache,use_cache=True)
                torch.cuda.synchronize();wall=time.perf_counter()-begin
                categories={}
                for category,events in pairs.items():
                    durations=[a.elapsed_time(b) for a,b in events]
                    categories[category]=dict(calls=len(events),inclusive_total_ms=sum(durations),
                        inclusive_ms_per_model_call=sum(durations)/steps,event_ms=durations)
                result['workloads'][phase]=dict(status='COMPLETE',model_calls=steps,
                    wall_seconds_with_profiling=wall,categories=categories,
                    profile_decode_positions='2056..2063' if phase=='decode' else None)
            except Exception as exc:
                result['workloads'][phase]=dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
            finally:
                for handle in handles:handle.remove()
                cache.update=original_update
                del cache
            common.write(output,record)
    result['status']='COMPLETE' if all(x['status']=='COMPLETE' for x in result['workloads'].values()) else 'PARTIAL'
    result['final_exclusivity']=exclusivity()
    common.write(output,record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--row', choices=('fp16','hadamard','nar_k8'), required=True)
    parser.add_argument('--phase', choices=('ppl-timing','eight-tasks','all'), default='ppl-timing')
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit')
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--batch-size', type=int, choices=(1,16,64), default=1)
    parser.add_argument('--rotation-backend', choices=('original','fast'), default='original')
    parser.add_argument('--cache-timing', choices=('reused','fresh'), default='reused',
        help='fresh includes per-repetition cache creation/clear in prefill; decode prefix remains outside timer')
    args = parser.parse_args()
    if args.rotation_backend=='fast' and args.row!='nar_k8':
        parser.error('--rotation-backend fast currently requires --row nar_k8')
    if args.repetitions < 20: parser.error('Timing requires at least20 repetitions')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(dict(status='STARTING'), stream)
    record = dict(kind='e38_stage_a_trained8b_native', status='STARTING', model='llama31_8b',
        rotation=args.row, row=args.row, rotation_backend=args.rotation_backend,
        cache_timing=args.cache_timing, seed=0, backend='FP16' if args.row == 'fp16' else 'native CUTLASS signed INT4 GEMM',
        scope='Actual trained8B weights; independent per-token symmetric activation/native per-head KV4 protocol; no Table1 equivalence claim',
        source_commit=args.source_commit, command_line=list(sys.orig_argv), started_utc=common.now(),
        hostname=platform.node(), python=sys.version, pid=os.getpid(),
        source_sha256={name:common.sha(REPO/name) for name in SOURCES},
        base_revision=backend.BASE.name, quant_snapshot='fc812e6687165115feee00293c953a1b521d3bd0',
        task_seed=0, quality_execution_gates='none',
        ppl=dict(status='not run'), eight_tasks=dict(status='not run'), timing=dict(status='not run'),
        quantization=dict(weights='original default E14 GPTQ symmetric per-channel saved BF16 values recovered into signed4 codes; native FP16scale projection drift recorded' if args.row != 'fp16' else 'released BF16 base cast to FP16',
            activation='symmetric per-token maxabs/7; native packed4' if args.row != 'fp16' else 'FP16',
            kv='native asymmetric packed4 per-token/head128, FP16 scale+offset, no residual window; prefill reads actual packed pages and reconstructs K/V' if args.row != 'fp16' else 'native FP16 paged cache',
            table1_difference='Table1 uses asymmetric group128 activations and KIVI token-axis grouped K plus residual32',
            native_int4_tensorcore=args.row != 'fp16', whole_graph_integer=False),
        slurm={k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','CUDA_VISIBLE_DEVICES')})
    code = 1
    try:
        os.environ.update(HF_HOME=str(backend.WORK/'cache/huggingface'), HF_DATASETS_CACHE=str(backend.WORK/'cache/datasets'),
            HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        harness = None
        if args.phase in ('eight-tasks','all'):
            from e38 import eight_tasks
            harness = eight_tasks.load_harness()
            record['dataset_metadata_compat'] = eight_tasks.install_dataset_metadata_compat()
            record['tokenizer_runtime'] = eight_tasks.tokenizer_runtime_manifest()
        import torch
        import inspect
        import quarot
        import quarot.transformers.kv_cache
        torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        record.update(gpu_model=torch.cuda.get_device_name(), cuda_runtime=torch.version.cuda,
            actual_head_commit=common.command(['git','-C',str(REPO),'rev-parse','HEAD']), libraries={})
        if torch.cuda.get_device_capability() != (8,6):
            raise RuntimeError('This existing compiled CUTLASS native backend requires A40/sm86')
        for name in ('torch','transformers','triton','datasets','numpy','tokenizers','flash-attn','fast-hadamard-transform'):
            try: record['libraries'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: record['libraries'][name] = None
        record['native_operator_sources'] = []
        for value in (quarot.nn.Linear4bit,quarot.nn.Quantizer,
                      quarot.transformers.kv_cache.MultiLayerPagedKVCache4Bit):
            path = Path(inspect.getfile(value))
            record['native_operator_sources'].append(dict(path=str(path),sha256=common.sha(path)))
        record['native_extension'] = dict(path=quarot._CUDA.__file__,
            bytes=Path(quarot._CUDA.__file__).stat().st_size,content_hash_checked=False)
        record['evaluation_tokens'] = dict(path=str(TOKENS), sha256=common.sha(TOKENS))
        record['source_observation'] = record['source_sha256']
        common.write(args.output, record)
        with torch.inference_mode():
            model = backend.build(args.row, args.artifact_dir, record, args.output)
            if args.rotation_backend=='fast':
                from e38.optimized_rotation import install_fast_rotations
                install_fast_rotations(model,record)
        record.update(status='RUNNING', model_object_id=id(model), parameter_dtype=str(next(model.parameters()).dtype))
        windows = torch.load(TOKENS,map_location='cpu',weights_only=True)
        record['evaluation_tokens']['shape'] = list(windows.shape)
        if args.phase in ('ppl-timing','all'):
            measure_ppl(model, windows, record, args.output)
        if args.phase in ('eight-tasks','all'):
            try:
                run_eight_tasks(model, harness, record, args.output)
            except Exception as exc:
                record['eight_tasks'] = dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
        if args.phase in ('ppl-timing','all'):
            measure_timing(model, windows, record, args.output, args.repetitions, args.warmup,
                           args.batch_size, cache_timing=args.cache_timing)
            try:
                profile_components(model, windows, record, args.output)
            except Exception as exc:
                record['component_profile']=dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
        record['same_model_object_id_after_measurements'] = id(model)
        expected = [record['ppl']['status'],record['timing']['status']] if args.phase == 'ppl-timing' else [record['eight_tasks']['status']]
        if args.phase == 'all': expected += [record['ppl']['status'],record['timing']['status']]
        record['status'] = 'COMPLETE' if all(x in ('COMPLETE','PASS') for x in expected) else 'PARTIAL'
        code = 0
    except Exception as exc:
        record.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        record['finished_utc'] = common.now()
        common.write(args.output, record)
        print(json.dumps(dict(status=record['status'], output=str(args.output))), flush=True)
    return code


if __name__ == '__main__': raise SystemExit(main())
