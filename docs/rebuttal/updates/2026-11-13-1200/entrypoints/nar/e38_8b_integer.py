"""8B trained-checkpoint, same-instance g128 asymmetric integer prototype.

Signed4 weight storage expands to INT8 torch._int_mm with INT32 partial sums;
FP32 affine correction and original E14 floating KV simulation remain. This is
NOT native INT4 tensor-core or packed-KV deployment. Quality has no execution
threshold. The ppl-timing and optional tasks phases reuse exported packed
layers, not GPTQ. Ppl-timing records candidate-only full WT2 NLL before timing
the same loaded model; it does not repeat the completed logit equivalence audit.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

from nar import e38_anchor as provenance

REPO = Path(__file__).resolve().parents[1]
WORK = Path('/projects/nar/nar-validation')
BASE = WORK/'cache/huggingface/models--unsloth--Meta-Llama-3.1-8B/snapshots/e9a141a2091ea561b96483212645a2a05e6f99fc'
RELEASE = WORK/'cache/huggingface/models--ForeverBlue--PrismQuant-Llama-3.1-8B/snapshots/fc812e6687165115feee00293c953a1b521d3bd0/checkpoints'
TOKENS = WORK/'cache/tokenized/ffcc87608d60-wikitext2-test-full-l2048.pt'
MODEL = 'llama31_8b'
SCOPE = '4-bit weight storage expanded to INT8 GEMM, INT32 group partials and FP32 affine correction; original E14 online rotations and floating KV simulation; not native INT4 or packed KV'
SOURCE_FILES = ('nar/e38_8b_integer.py', 'slurm_e38_8b_integer.sh',
    'nar/e38_8b_ops.py',
    'nar/e38_integer_operator.py', 'nar/e38_checkpoint_probe.py',
    'nar/e38_direct_accuracy.py', 'nar/e38_reference_tasks.py', 'nar/e38_anchor.py',
    'nar/e14_w4a4kv4.py', 'nar/experiment.py', 'nar/activation_experiments.py',
    'nar/e12_wy.py', 'nar/extended_experiment.py', 'nar/quarot_gptq.py')


def flush(output, record):
    from nar.e38_direct_accuracy import safe_json
    provenance.write(output, safe_json(record))


def file_identity(path, content=False):
    path = Path(path)
    item = dict(path=str(path), resolved=str(path.resolve()), bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns)
    if content:
        item['sha256'] = provenance.sha(path)
    return item


def tensor_sha(tensor):
    import torch
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def stats(values):
    values = list(values)
    if not values or not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError('finite nonnegative timing samples required')
    import numpy as np
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values), p25=float(np.percentile(values,25)),
                p75=float(np.percentile(values,75)), stdev=statistics.stdev(values) if len(values)>1 else 0.)


def compare_logits_cpu(reference, candidate, token_block=128):
    """Stream candidate blocks to CPU; compare every vocabulary element.

    Reference is already CPU. Only block-sized FP32/FP64 temporaries are made;
    the full vocabulary logits are not serialized to the output JSON.
    """
    import torch
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError('expected matching [batch, tokens, vocabulary] logits')
    maximum = delta_sq = reference_sq = 0.
    matches = positions = nonfinite_ref = nonfinite_candidate = 0
    ref_argmax, candidate_argmax = [], []
    for start in range(0, reference.shape[1], token_block):
        r = reference[:, start:start+token_block].float()
        a = candidate[:, start:start+token_block].detach().to('cpu', dtype=torch.float32)
        finite_r, finite_a = torch.isfinite(r), torch.isfinite(a)
        nonfinite_ref += int((~finite_r).sum()); nonfinite_candidate += int((~finite_a).sum())
        if bool(finite_r.all() and finite_a.all()):
            delta = a-r
            maximum = max(maximum, float(delta.abs().max()))
            delta_sq += float(delta.square().sum(dtype=torch.float64))
            reference_sq += float(r.square().sum(dtype=torch.float64))
        ra, aa = r.argmax(-1), a.argmax(-1)
        ref_argmax.append(ra); candidate_argmax.append(aa)
        matches += int((ra==aa).sum()); positions += ra.numel()
    finite = nonfinite_ref == nonfinite_candidate == 0
    return dict(finite=finite, nonfinite_reference=nonfinite_ref,
        nonfinite_candidate=nonfinite_candidate, max_abs_logit_difference=maximum if finite else None,
        squared_error=delta_sq if finite else None, reference_squared_norm=reference_sq if finite else None,
        relative_l2=math.sqrt(delta_sq/max(reference_sq,1e-300)) if finite else None,
        argmax_matches=matches, positions=positions, argmax_agreement=matches/positions,
        reference_argmax=torch.cat(ref_argmax,dim=1), candidate_argmax=torch.cat(candidate_argmax,dim=1))


def model_packed_identity(wrappers):
    return [dict(packed_ptr=w.integer.packed.data_ptr(), scale_ptr=w.integer.scale.data_ptr(),
                 group_sums_ptr=w.integer.group_sums.data_ptr(), shape=[w.out_features,w.in_features])
            for w in wrappers]


def make_switchable(reference, packed):
    import torch
    from nar import experiment as base
    class Switchable(torch.nn.Module):
        def __init__(self, original, integer):
            super().__init__()
            if original.bias is not None:
                raise ValueError('8B Llama linears must be bias-free')
            self.reference, self.integer = original, integer
            self.in_features, self.out_features = original.in_features, original.out_features
            self.use_integer = False
            self.bias = None
        def forward(self, value):
            if self.use_integer:
                return self.integer(value)
            qdq, _, _, _ = base.dynamic_asym_int4(value,128)
            return self.reference(qdq)
    return Switchable(reference, packed)


def set_integer(wrappers, enabled=True):
    for wrapper in wrappers:
        wrapper.use_integer = enabled


def release_reference(wrappers):
    set_integer(wrappers)
    for wrapper in wrappers:
        if hasattr(wrapper,'reference'):
            del wrapper.reference
    gc.collect()
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def checkpoint_identity(checkpoint):
    return dict(directory=str(checkpoint.resolve()),
        files=[file_identity(checkpoint/'DONE.json',True)]+
              [file_identity(checkpoint/f'layer_{i:02d}.pt') for i in range(32)])


def validate_cache_identity(manifest, rotation, seed, checkpoint):
    """Protect cache identity, not model quality; no metric enters this check."""
    if manifest.get('model') != MODEL or manifest.get('rotation') != rotation or manifest.get('seed') != seed:
        raise ValueError('packed cache model/rotation/seed mismatch')
    if manifest.get('checkpoint') != checkpoint:
        raise ValueError('packed cache checkpoint files differ; choose a new cache directory')
    if manifest.get('status') != 'COMPLETE' or len(manifest.get('layers',[])) != 32:
        raise ValueError('packed cache is incomplete')


def prepare_model(args, record):
    import torch
    from nar import experiment as base
    from nar import e14_w4a4kv4 as e14
    from nar.e38_checkpoint_probe import audit_symmetric_grid
    from nar.e38_integer_operator import PackedSymmetric4
    previous = e14.LOAD_MODEL
    e14.LOAD_MODEL = lambda _id, wd: base.load_model(str(BASE),wd)
    try:
        model, rotations = e14.load_quantized_model(WORK,args.checkpoint_directory.parent.parent,
            MODEL,args.rotation,args.seed,256,protocol='default')
    finally:
        e14.LOAD_MODEL = previous
    model.eval()
    record.update(model_instance_id=id(model), actual_model_dtype=str(next(model.parameters()).dtype))
    manifest_path=args.packed_cache/'manifest.json'
    identity=checkpoint_identity(args.checkpoint_directory)
    reuse=manifest_path.exists()
    if reuse:
        manifest=json.loads(manifest_path.read_text())
        validate_cache_identity(manifest,args.rotation,args.seed,identity)
    else:
        if args.phase in ('eight-tasks','ppl-timing'):
            raise FileNotFoundError(f'{args.phase} phase requires completed --packed-cache; no scale recovery in that phase')
        args.packed_cache.mkdir(parents=True,exist_ok=False)
        manifest=dict(kind='e38_8b_real_packed_cache',status='BUILDING',model=MODEL,
            rotation=args.rotation,seed=args.seed,checkpoint=identity,base_revision=BASE.name,
            source_sha256=record['source_sha256'],scope=SCOPE,layers=[])
        provenance.write(manifest_path,manifest)
    wrappers=[]; export=[]
    record['packing']=dict(status='RUNNING',reused_cache=reuse,cache=str(args.packed_cache),rows=export,
        recovery='fit FP32 scales to each actual saved BF16 weight; no training or GPTQ recalibration')
    flush(args.output,record)
    with torch.inference_mode():
        for index,block in enumerate(model.model.layers):
            path=args.packed_cache/f'layer_{index:02d}.pt'
            if reuse:
                entry=manifest['layers'][index]
                if entry['index']!=index or provenance.sha(path)!=entry['sha256']:
                    raise ValueError(f'packed cache file corruption at layer{index}')
                cached=torch.load(path,map_location='cpu',weights_only=True)
            else:
                cached={}
            for name,original in [p for group in e14._linear_groups(block) for p in group]:
                key=f'layer_{index:02d}.{name}'
                if reuse:
                    item=cached[name]; state=item['state']
                    packed=PackedSymmetric4(state['packed'].to(original.weight.device),
                        state['scale'].to(original.weight.device),state['group_sums'].to(original.weight.device))
                    report=item['export']; audit=item['audit']
                    # Cache corruption/identity is a real input error, distinct
                    # from numerical approximation in the model comparison.
                    if not torch.equal(packed.decode(),original.weight):
                        raise ValueError(f'cached weight differs from loaded checkpoint: {key}')
                else:
                    audit,candidates=audit_symmetric_grid(original.weight.detach(),row_batch=128)
                    packed,report=PackedSymmetric4.from_weight(original.weight.detach(),candidates['fp32_scale'])
                    cached[name]=dict(state={k:v.detach().cpu() for k,v in packed.state_dict().items()},
                        candidates=candidates,audit=audit,export=report)
                wrapper=make_switchable(original,packed)
                parent,leaf=name.rsplit('.',1)
                setattr(block.get_submodule(parent),leaf,wrapper)
                wrappers.append(wrapper); export.append(dict(site=key,export=report,audit=audit))
            if not reuse:
                with path.open('xb') as stream: torch.save(cached,stream)
                manifest['layers'].append(dict(index=index,**file_identity(path,True)))
                provenance.write(manifest_path,manifest)
            del cached
            flush(args.output,record)
            print(f'packed 8B {args.rotation} layer {index+1}/32',flush=True)
    if not reuse:
        manifest.update(status='COMPLETE',completed_utc=provenance.now(),linear_count=len(wrappers))
        provenance.write(manifest_path,manifest)
    record['packing'].update(status='COMPLETE',linear_count=len(wrappers),manifest=file_identity(manifest_path,True))
    runtime=e14.RuntimeHooks(model,rotations,activation_kind=None,quantize_kv=True)
    runtime.install()
    return model,wrappers,runtime


def paired_ppl(model,wrappers,tokens,record,output,artifact_dir):
    import torch
    result=dict(status='RUNNING',chunks=[],sequence_length=int(tokens.shape[1]),
        sequences_expected=len(tokens),batch_size=1,full_vocabulary=True,
        logit_storage='per-chunk full-vocabulary comparison statistics; token argmax arrays saved; full logits streamed and discarded',
        numerical_execution_gate=False,comparison='same loaded model and same GPU, reference vs integer on identical input_ids')
    record['equivalence']=result; flush(output,record)
    argmax_dir=artifact_dir/'token_argmax'; argmax_dir.mkdir()
    try:
        with torch.inference_mode():
            for index in range(len(tokens)):
                ids=tokens[index:index+1].cuda()
                set_integer(wrappers,False)
                ref=model(input_ids=ids,labels=ids,use_cache=False)
                ref_nll=float(ref.loss.float())
                ref_logits=ref.logits.detach().cpu()
                del ref
                set_integer(wrappers,True)
                candidate=model(input_ids=ids,labels=ids,use_cache=False)
                candidate_nll=float(candidate.loss.float())
                comp=compare_logits_cpu(ref_logits,candidate.logits)
                path=argmax_dir/f'chunk_{index:03d}.pt'
                with path.open('xb') as stream:
                    torch.save(dict(reference=comp.pop('reference_argmax'),integer=comp.pop('candidate_argmax'),
                        input_ids=tokens[index:index+1]),stream)
                row=dict(chunk=index,reference_nll=ref_nll,candidate_nll=candidate_nll,
                    tokens_scored=int(ids.shape[1])-1,input_ids_sha256=tensor_sha(ids),
                    token_argmax=file_identity(path,True),**comp)
                result['chunks'].append(row)
                result['completed_sequences']=len(result['chunks'])
                flush(output,record)
                del candidate,ref_logits,ids
                print(f'8B paired PPL {index+1}/{len(tokens)}',flush=True)
        rows=result['chunks']; total=sum(x['tokens_scored'] for x in rows)
        ref_nll=sum(x['reference_nll']*x['tokens_scored'] for x in rows)/total
        cand_nll=sum(x['candidate_nll']*x['tokens_scored'] for x in rows)/total
        finite=all(x['finite'] for x in rows) and math.isfinite(ref_nll) and math.isfinite(cand_nll)
        result.update(status='COMPLETE' if finite else 'NONFINITE',tokens_scored=total,
            reference_ppl=math.exp(ref_nll),candidate_ppl=math.exp(cand_nll),
            max_abs_logit_difference=max(x['max_abs_logit_difference'] for x in rows) if finite else None,
            argmax_matches=sum(x['argmax_matches'] for x in rows),positions=sum(x['positions'] for x in rows))
        result.update(ppl_difference=result['candidate_ppl']-result['reference_ppl'],
            relative_ppl_difference_percent=100*(result['candidate_ppl']/result['reference_ppl']-1),
            argmax_agreement=result['argmax_matches']/result['positions'],
            relative_l2=math.sqrt(sum(x['squared_error'] for x in rows)/max(sum(x['reference_squared_norm'] for x in rows),1e-300)) if finite else None)
    except Exception as exc:
        result.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),
            continuation='quality failure is recorded; timing still attempted on same loaded instance')
        traceback.print_exc()
    finally:
        set_integer(wrappers)
        gc.collect(); torch.cuda.empty_cache(); flush(output,record)
    return result


def candidate_ppl(model,wrappers,tokens,record,output):
    """Record every candidate NLL without redoing the reference/logit audit."""
    import torch
    result=dict(status='RUNNING',chunks=[],sequence_length=int(tokens.shape[1]),
        sequences_expected=len(tokens),batch_size=1,use_cache=False,
        model_instance_id=id(model),numerical_execution_gate=False,
        scope='candidate integer path only; full frozen WT2 windows; no reference/logit comparison')
    record['ppl']=result; flush(output,record)
    set_integer(wrappers)
    try:
        with torch.inference_mode():
            for index in range(len(tokens)):
                ids=tokens[index:index+1].cuda()
                candidate=model(input_ids=ids,labels=ids,use_cache=False)
                nll=float(candidate.loss.float())
                result['chunks'].append(dict(chunk=index,nll=nll,candidate_nll=nll,
                    tokens_scored=int(ids.shape[1])-1,input_ids_sha256=tensor_sha(ids)))
                result['completed_sequences']=len(result['chunks'])
                flush(output,record)
                del candidate,ids
                print(f'8B candidate PPL {index+1}/{len(tokens)}',flush=True)
        rows=result['chunks']; total=sum(x['tokens_scored'] for x in rows)
        mean_nll=sum(x['nll']*x['tokens_scored'] for x in rows)/total
        ppl=math.exp(mean_nll) if mean_nll<math.log(sys.float_info.max) else float('inf')
        finite=math.isfinite(mean_nll) and math.isfinite(ppl)
        result.update(status='COMPLETE' if finite else 'NONFINITE',tokens_scored=total,
            mean_nll=mean_nll,ppl=ppl,candidate_ppl=ppl)
    except Exception as exc:
        result.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),
            continuation='quality failure is recorded; timing still attempted on same loaded instance')
        traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache(); flush(output,record)
    return result


def compute_processes():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True,timeout=20)
    rows=[]
    for line in raw.splitlines():
        if not line.strip(): continue
        gpu,pid=line.split(',',1)
        rows.append(dict(gpu_uuid=gpu.strip(),pid=int(pid.strip())))
    return rows


def exclusive_guard(gpu_uuid=None):
    rows=compute_processes()
    own={x['gpu_uuid'] for x in rows if x['pid']==os.getpid()}
    if gpu_uuid is None:
        if len(own)!=1: raise RuntimeError(f'cannot identify one process GPU: {own}')
        gpu_uuid=next(iter(own))
    foreign=[x for x in rows if x['gpu_uuid']==gpu_uuid and x['pid']!=os.getpid()]
    return dict(status='PASS' if gpu_uuid in own and not foreign else 'BLOCKED',gpu_uuid=gpu_uuid,
        processes=[x for x in rows if x['gpu_uuid']==gpu_uuid],foreign_processes=foreign,
        checked_utc=provenance.now(),scope='only GPU exclusivity; no host-load or numerical threshold')


def new_cache(model):
    from transformers import DynamicCache
    return DynamicCache(config=model.config)


def timing_cohort(model,ids,args,record,phase,gpu_uuid):
    import torch
    row=dict(status='RUNNING',phase=phase,batch_size=1,prefix=int(ids.shape[1]),
        repetitions=args.repetitions,warmups=args.warmups,samples_seconds=[],guards=[],
        decode_steps=args.decode_steps if phase=='decode' else 0,
        discarded_decode_steps=args.discard_decode_steps if phase=='decode' else 0,
        output='last-position logits only; eager Python launches; no CUDA graph',
        fixed_decode_token_id=100,cache_reset='fresh DynamicCache for every repetition',
        cache_creation_in_timed_region=phase=='prefill',
        timing_boundary=('fresh DynamicCache construction and full prefix forward'
            if phase=='prefill' else 'incremental single-token forwards after untimed cache construction, prefix and discarded steps'),
        cache_lengths=[],
        cache_format='BF16 DynamicCache with E14 attention-time KV4 QDQ simulation; not packed KV')
    record['timing']['cohorts'][phase]=row
    token=torch.full((1,1),100,device=ids.device,dtype=torch.long)
    def run():
        if phase=='decode':
            cache=new_cache(model)
            prepared=model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=prepared.past_key_values; del prepared
            for _ in range(args.discard_decode_steps):
                prepared=model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=prepared.past_key_values; del prepared
        torch.cuda.synchronize()
        start=time.perf_counter()
        if phase=='prefill':
            cache=new_cache(model)
            actual=model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
        else:
            for _ in range(args.decode_steps-args.discard_decode_steps):
                actual=model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=actual.past_key_values
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        expected=ids.shape[1]+(args.decode_steps if phase=='decode' else 0)
        cache_length=int(actual.past_key_values.get_seq_length())
        if actual.logits.ndim!=3 or actual.logits.shape[1]!=1:
            raise RuntimeError(f'timing requires last-position logits, got {tuple(actual.logits.shape)}')
        finite=bool(torch.isfinite(actual.logits).all())
        del actual,cache
        if cache_length!=expected:
            raise RuntimeError(f'cache length {cache_length}, expected {expected}')
        # Nonfinite output is recorded, never used as a timing veto.
        return elapsed,finite,cache_length
    for _ in range(args.warmups): run()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    row['allocated_before_bytes']=torch.cuda.memory_allocated()
    row['reserved_before_bytes']=torch.cuda.memory_reserved()
    row['output_finite_per_repetition']=[]
    for index in range(args.repetitions):
        guard=exclusive_guard(gpu_uuid); row['guards'].append(guard)
        if guard['status']!='PASS':
            row.update(status='BLOCKED',reason='another compute process uses this GPU; partial raw samples retained without summary')
            flush(args.output,record); return row
        elapsed,finite,cache_length=run()
        row['samples_seconds'].append(elapsed); row['output_finite_per_repetition'].append(finite)
        row['cache_lengths'].append(cache_length)
        flush(args.output,record)
        print(f'8B integer {phase} {index+1}/{args.repetitions}: {elapsed:.6f}s',flush=True)
    final=exclusive_guard(gpu_uuid); row['guards'].append(final)
    if final['status']!='PASS':
        row.update(status='BLOCKED',reason='GPU exclusivity changed; raw samples retained without summary')
    else:
        count=ids.shape[1] if phase=='prefill' else args.decode_steps-args.discard_decode_steps
        row.update(status='COMPLETE',latency_seconds=stats(row['samples_seconds']),
            tokens_per_second=stats([count/x for x in row['samples_seconds']]),
            ms_per_token=stats([1000*x/count for x in row['samples_seconds']]),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            memory_scope='PyTorch peak includes resident integer model, rotations, cache and temporary buffers; decode also includes untimed prefix preparation',
            metric_scope='prototype elapsed latency only; no native INT4 or packed-KV speedup claim')
    flush(args.output,record)
    return row


def measure_timing(model,wrappers,tokens,args,record):
    import torch
    quality_stage='ppl' if args.phase=='ppl-timing' else 'equivalence'
    result=dict(status='RUNNING',cohorts={},scope=SCOPE,same_loaded_model_as_quality=True,
        quality_stage=quality_stage,same_loaded_model_as_equivalence=quality_stage=='equivalence',
        model_instance_id=id(model),numerical_execution_gate=False,graph_mode=False,
        reference_linear_weights_removed=all(not hasattr(x,'reference') for x in wrappers),
        input_ids_sha256=tensor_sha(tokens[:1]),peak_memory_includes_simulated_KV=True)
    record['timing']=result; flush(args.output,record)
    try:
        guard=exclusive_guard(); result['initial_guard']=guard
        if guard['status']!='PASS':
            result.update(status='BLOCKED',reason='GPU is not exclusive; no timing summary')
            return result
        result['gpu_uuid']=guard['gpu_uuid']
        with torch.inference_mode():
            ids=tokens[:1].cuda()
            for phase in ('prefill','decode'):
                try: timing_cohort(model,ids,args,record,phase,guard['gpu_uuid'])
                except Exception as exc:
                    result['cohorts'].setdefault(phase,{}).update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
                    gc.collect();torch.cuda.empty_cache();traceback.print_exc()
                    flush(args.output,record)
        states=[x['status'] for x in result['cohorts'].values()]
        result['status']='COMPLETE' if states==['COMPLETE','COMPLETE'] else 'PARTIAL'
    except Exception as exc:
        result.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result['finished_utc']=provenance.now();flush(args.output,record)
    return result


def optional_tasks(model,args,record):
    from transformers import AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    import lm_eval
    from nar import e38_direct_accuracy as direct
    from nar import e38_reference_tasks as task_utils
    if args.seed!=0:
        raise ValueError('optional eight-task phase is seed0 only; other seeds are E38 timing/equivalence only')
    tokenizer=AutoTokenizer.from_pretrained(BASE,use_fast=True,local_files_only=True)
    lm=HFLM(pretrained=model,tokenizer=tokenizer,batch_size=args.batch_size,
        max_batch_size=args.batch_size,max_length=2048)
    record['harness']=dict(expected_commit_recorded=task_utils.HARNESS_COMMIT,
        installed_module=str(Path(lm_eval.__file__).resolve()),module_sha256=provenance.sha(Path(lm_eval.__file__).resolve()))
    record['tokenizer_files']=[file_identity(BASE/name,True) for name in
        ('tokenizer.json','tokenizer_config.json','special_tokens_map.json')]
    record['harness_model']=dict(max_length=lm.max_length,batch_size=args.batch_size,
        pad_token_id=tokenizer.pad_token_id,padding_side=tokenizer.padding_side)
    direct.run_tasks(lm,TaskManager(),record,args.output,batch_size=args.batch_size)


def accuracy_then_timing(accuracy,timing,release):
    """No metric or status blocks timing; even a failed accuracy stage proceeds."""
    try: accuracy()
    except Exception: traceback.print_exc()
    finally: release()
    return timing()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rotation',choices=('nar_k8','hadamard'),default='nar_k8')
    p.add_argument('--seed',type=int,choices=(0,1,2),default=0)
    p.add_argument('--checkpoint-directory',type=Path)
    p.add_argument('--packed-cache',type=Path,required=True)
    p.add_argument('--artifact-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit')
    p.add_argument('--phase',choices=('equivalence-timing','ppl-timing','eight-tasks'),default='equivalence-timing')
    p.add_argument('--eight-tasks-after-timing',action='store_true')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--repetitions',type=int,default=20)
    p.add_argument('--warmups',type=int,default=10)
    p.add_argument('--decode-steps',type=int,default=128)
    p.add_argument('--discard-decode-steps',type=int,default=8)
    args=p.parse_args(argv)
    if args.repetitions<20 or args.warmups<1: p.error('at least20 repetitions and1 warmup required')
    if not 0<=args.discard_decode_steps<args.decode_steps: p.error('decode steps must exceed discarded steps')
    if args.batch_size<1: p.error('positive batch size required')
    if (args.phase=='eight-tasks' or args.eight_tasks_after_timing) and args.seed!=0:
        p.error('eight-task evaluation is seed0 only')
    if args.checkpoint_directory is None:
        args.checkpoint_directory=RELEASE/MODEL/f'gptq_{args.rotation}_seed{args.seed}'
    if args.checkpoint_directory.name!=f'gptq_{args.rotation}_seed{args.seed}':
        p.error('checkpoint directory name must identify the selected rotation and seed')
    if args.checkpoint_directory.parent.name!=MODEL:
        p.error('checkpoint directory parent must be llama31_8b; the E14 loader resolves this model directory explicitly')
    return args


def main(argv=None):
    args=parse_args(argv)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(dict(status='STARTING'),stream)
    args.artifact_dir.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HOME=str(WORK/'cache/huggingface'),HF_DATASETS_CACHE=str(WORK/'cache/datasets'),
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    record=dict(kind='e38_8b_asymmetric_same_instance_integer',status='RUNNING',backend='integer',scope=SCOPE,
        model=MODEL,rotation=args.rotation,seed=args.seed,phase=args.phase,source_commit=args.source_commit,
        command_line=list(sys.orig_argv),started_utc=provenance.now(),hostname=platform.node(),python=sys.version,
        numerical_execution_gates='none',native_int4_tensorcore=False,packed_KV=False,
        base_revision=BASE.name,task_batch_size=args.batch_size,ppl_batch_size=1,
        equivalence=dict(status='not run'),ppl=dict(status='not run'),timing=dict(status='not run'),eight_tasks=dict(status='not run'),
        slurm={k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','CUDA_VISIBLE_DEVICES')})
    runtime=None; code=1
    try:
        import torch
        from nar import experiment as base
        record.update(actual_head_commit=provenance.git('rev-parse','HEAD'),
            source_sha256={x:provenance.sha(REPO/x) for x in SOURCE_FILES},libraries=provenance.packages(),
            source_policy='record actual identity; no source commit equality gate')
        if not torch.cuda.is_available(): raise RuntimeError('allocated GPU required')
        torch.set_num_threads(4);base.seed_everything(args.seed)
        record.update(hardware=base.hardware_info(),gpu_model=torch.cuda.get_device_name(),cuda_runtime=torch.version.cuda)
        record['checkpoint']=checkpoint_identity(args.checkpoint_directory)
        record['evaluation_tokens']=file_identity(TOKENS,True)
        tokens=torch.load(TOKENS,map_location='cpu',weights_only=True)
        if tuple(tokens.shape)!=(141,2048): raise ValueError(f'expected full141 WT2 windows, found{tuple(tokens.shape)}')
        record['evaluation_tokens']['shape']=list(tokens.shape)
        model,wrappers,runtime=prepare_model(args,record)
        before=model_packed_identity(wrappers)
        if args.phase in ('equivalence-timing','ppl-timing'):
            capture_handles=[]
            if args.phase=='equivalence-timing':
                from nar.e38_8b_ops import capture_first_layer, benchmark
                captured, capture_handles = capture_first_layer(wrappers,args.artifact_dir)
                quality=lambda: paired_ppl(model,wrappers,tokens,record,args.output,args.artifact_dir)
            else:
                release_reference(wrappers)
                quality=lambda: candidate_ppl(model,wrappers,tokens,record,args.output)
            def release_for_timing():
                for handle in capture_handles: handle.remove()
                release_reference(wrappers)
            accuracy_then_timing(
                quality,
                lambda: measure_timing(model,wrappers,tokens,args,record),
                release_for_timing)
            if args.phase=='equivalence-timing': benchmark(wrappers,captured,args,record)
            record['same_instance_evidence']=dict(model_id=id(model),reference_weights_removed=True,
                quality_stage='ppl' if args.phase=='ppl-timing' else 'equivalence',
                packed_buffer_pointer_and_shapes_unchanged=before==model_packed_identity(wrappers),
                no_reload_between_quality_and_timing=True,packed_cache_manifest=record['packing']['manifest'])
        else: release_reference(wrappers)
        if args.phase=='eight-tasks' or args.eight_tasks_after_timing:
            try: optional_tasks(model,args,record)
            except Exception as exc:
                record['eight_tasks'].update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
                traceback.print_exc()
        wanted={'eight-tasks':['eight_tasks'],'equivalence-timing':['equivalence','timing'],
            'ppl-timing':['ppl','timing']}[args.phase]
        if args.eight_tasks_after_timing and 'eight_tasks' not in wanted: wanted.append('eight_tasks')
        complete=[record[k].get('status') in ('COMPLETE','PASS') for k in wanted]
        record['status']='COMPLETE' if all(complete) else 'PARTIAL' if any(complete) else 'FAILED'
        code=0 if any(complete) else 1
    except Exception as exc:
        record.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        if runtime is not None: runtime.close()
        record['ended_utc']=provenance.now();flush(args.output,record)
        print(json.dumps(dict(status=record['status'],output=str(args.output))),flush=True)
    return code


if __name__=='__main__': raise SystemExit(main())
