"""Same-loaded-model timing for official FlatQuant and DuQuant implementations.

The runners identify their actual execution mode; this helper never labels
floating fake-quant arithmetic as native INT4. No FP16 control is constructed.
"""
from __future__ import annotations

import gc
import os
import platform
import statistics
import subprocess
import time
import traceback


def cache_length(cache):
    """Read the actual cached sequence length, including legacy tuple caches."""
    if cache is None:
        return None
    if hasattr(cache, 'get_seq_length'):
        return int(cache.get_seq_length())
    if isinstance(cache, (tuple, list)) and cache and cache[0]:
        return int(cache[0][0].shape[-2])
    return None


def summary(values):
    ordered = sorted(values)
    def quantile(p):
        position = (len(ordered)-1)*p
        lower = int(position)
        upper = min(lower+1, len(ordered)-1)
        return ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)
    return dict(n=len(ordered), median=statistics.median(ordered),
                q25=quantile(.25), q75=quantile(.75), iqr=quantile(.75)-quantile(.25))


def exclusivity():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode:
        return dict(status='UNVERIFIED', error=result.stderr)
    processes = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid = (value.strip() for value in line.split(','))
        processes.append(dict(gpu_uuid=uuid, pid=int(pid)))
    own = [item for item in processes if item['pid']==os.getpid()]
    devices = {item['gpu_uuid'] for item in own}
    other = [item for item in processes if item['gpu_uuid'] in devices and item['pid']!=os.getpid()]
    return dict(status='EXCLUSIVE' if own and not other else 'CONTENDED_OR_UNVERIFIED',
                own=own, other=other, hostname=platform.node())


def measure_timing(model, windows_tensor, record, output_path, write_record, *,
                   repetitions=20, warmup=10, prefix=2048, decode_steps=128, discard=8):
    import torch

    if repetitions<1 or warmup<0 or not 0<=discard<decode_steps:
        raise ValueError('Invalid measurement lengths')
    if windows_tensor.ndim!=2 or windows_tensor.shape[1]<prefix:
        raise ValueError('Frozen evaluation tokens do not contain the requested prefix')
    timing = record['timing'] = dict(status='RUNNING', model_object_id=id(model),
        same_loaded_instance_after_ppl=True, hostname=platform.node(), gpu_model=torch.cuda.get_device_name(),
        implementation='official model implementation; execution mode is recorded by the method runner',
        execution_note='Floating fake-quant results are software-runtime measurements, not native INT4 deployment speed',
        mode='eager synchronized wall time; real autoregressive KV cache',
        batch_size=1, prefix=prefix, decode_steps=decode_steps, discarded_decode_steps=discard,
        measured_decode_steps=decode_steps-discard, fixed_decode_token_id=100,
        warmup=warmup, repetitions=repetitions, logits='full vocabulary at last token position only; num_logits_to_keep=1',
        cache_reset='fresh past_key_values=None for every repetition; prefix and discarded decode outside decode timer',
        memory_scope='peak allocated/reserved includes decode prefix construction outside its latency timer',
        numerical_execution_gate=False, guards=[], workloads={})
    model.eval()
    ids = windows_tensor[:1, :prefix].to(device='cuda', dtype=torch.long)
    token = torch.full((1, 1), 100, device='cuda', dtype=torch.long)
    def save():
        write_record(output_path, record)
    def guard():
        value = exclusivity();timing['guards'].append(value)
        return value['status']=='EXCLUSIVE'
    def call(tokens, cache=None):
        result=model(input_ids=tokens, past_key_values=cache, use_cache=True, return_dict=True,
                     num_logits_to_keep=1)
        if result.logits.shape[1]!=1:
            raise RuntimeError('Backend did not honor the matched last-position-logit timing workload')
        return result
    save()
    if not guard():
        timing.update(status='NOT_MEASURED', reason='Timing GPU is not verifiably exclusive')
        save();return timing
    with torch.inference_mode():
        for workload in ('prefill', 'decode'):
            measured = timing['workloads'][workload] = dict(status='RUNNING', samples_seconds=[],
                finite_last_position_logits=[], cache_lengths=[], peak_allocated_bytes=0, peak_reserved_bytes=0)
            def one_run():
                cached = None
                if workload=='decode':
                    initial = call(ids)
                    cached = initial.past_key_values
                    if cache_length(cached)!=prefix:
                        raise RuntimeError('Model did not return a verifiable prefix KV cache; cached decode unsupported')
                    del initial
                    for _ in range(discard):
                        out = call(token, cached);cached = out.past_key_values;del out
                torch.cuda.synchronize()
                begin = time.perf_counter()
                if workload=='prefill':
                    out = call(ids)
                else:
                    for _ in range(discard, decode_steps):
                        out = call(token, cached);cached = out.past_key_values
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-begin
                allocated = torch.cuda.max_memory_allocated()
                reserved = torch.cuda.max_memory_reserved()
                length = cache_length(out.past_key_values)
                if workload=='decode' and length!=prefix+decode_steps:
                    raise RuntimeError(f'Cached decode length is {length}, expected {prefix+decode_steps}')
                finite = bool(torch.isfinite(out.logits[:, -1]).all())
                return elapsed, finite, length, allocated, reserved
            try:
                for _ in range(warmup):
                    one_run()
                gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                for _ in range(repetitions):
                    if not guard():
                        measured.update(status='CONTENDED', reason='GPU is not exclusive; no timing summary')
                        break
                    elapsed, finite, length, allocated, reserved = one_run()
                    measured['samples_seconds'].append(elapsed)
                    measured['finite_last_position_logits'].append(finite)
                    measured['cache_lengths'].append(length)
                    measured['peak_allocated_bytes']=max(measured['peak_allocated_bytes'],allocated)
                    measured['peak_reserved_bytes']=max(measured['peak_reserved_bytes'],reserved)
                    save()
                if len(measured['samples_seconds'])==repetitions and guard():
                    measured.update(status='COMPLETE', seconds=summary(measured['samples_seconds']))
                    if workload=='prefill':
                        measured['milliseconds']=summary([value*1000 for value in measured['samples_seconds']])
                    else:
                        measured['tokens_per_second']=summary([(decode_steps-discard)/value for value in measured['samples_seconds']])
                elif measured['status']=='RUNNING':
                    measured.update(status='CONTENDED',reason='Final exclusivity check failed')
            except Exception as exc:
                measured.update(status='UNSUPPORTED_OR_FAILED', error=repr(exc), traceback=traceback.format_exc())
            gc.collect();torch.cuda.empty_cache();save()
    timing['same_model_object_id_after_timing']=id(model)
    timing['status']='COMPLETE' if all(item['status']=='COMPLETE' for item in timing['workloads'].values()) else 'PARTIAL'
    save()
    return timing
