"""Real-checkpoint FlatQuant packed INT4 deployment, Meta-Llama-3.1-8B only.

Official arithmetic/export are unchanged. This adapter checks checkpoint coverage,
retains Llama-3.1 RoPE, and records the native backend's actual KV semantics.
"""
from __future__ import annotations
import argparse
import ctypes
import gc
import importlib.util
import json
import math
import os
import resource
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from nar.e39_flatquant import BASE, TOKENS, TIMING_HELPER, sha, write

OFFICIAL = Path('/home/yanlongc/.prismquant/FlatQuant-native-torch26-cu124')
OFFICIAL_COMMIT = '9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1'
RUNTIME_MANIFEST = Path('/home/yanlongc/.prismquant/flatquant-native-tools/native_runtime_manifest.json')
NATIVE_NOTE = ('Official packed INT4 CUDA GEMM and paged INT4 decode kernels; real trained weights. '
    'Prefill attention consumes transformed unquantized K/V; native KV packing omits learned clipping '
    'and repeats GQA KV heads. This is a distinct deployment variant from fake-quant W4A4KV4 accuracy.')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_record(path, record):
    if 'timing' in record:
        record['timing']['execution_note'] = NATIVE_NOTE
        record['timing']['implementation'] = 'FlatQuant official deploy packed INT4 backend'
        record['timing']['same_loaded_instance_after_ppl'] = record.get('completed_sequences') == 141
    write(path, record)


def cpu_memory_snapshot(stage):
    """Process RSS and high-water mark; Linux values are reported in bytes."""
    result = dict(stage=stage, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS','VmHWM'):
                result[key+'_bytes'] = int(value.split()[0])*1024
    except (OSError, ValueError):
        pass
    return result


def release_cpu_memory():
    """Release unreachable tensors and glibc's freed pages between large stages."""
    gc.collect()
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return bool(trim(0))
    except (AttributeError, OSError):
        return None


def source_identity(checkpoint):
    done = checkpoint/'DONE.json'
    manifest = json.loads(done.read_text())
    if manifest['status'] != 'COMPLETE' or manifest['base_checkpoint'] != str(BASE):
        raise ValueError('Requires completed pinned Meta-Llama-3.1-8B checkpoint')
    if manifest['official_commit'] != OFFICIAL_COMMIT:
        raise ValueError('Checkpoint official source commit mismatch')
    args = manifest['official_args']
    if (args['w_bits'], args['a_bits'], args['k_bits'], args['v_bits']) != (4,4,4,4) or args['gptq']:
        raise ValueError('Requires official W4A4KV4 RTN checkpoint, without GPTQ recalibration')
    return manifest, dict(base_checkpoint=str(BASE), official_commit=OFFICIAL_COMMIT,
        source_checkpoint=str(checkpoint.resolve()), files={name:sha(checkpoint/name)
            for name in ('DONE.json','quantized_model.pt','weight_quantizers.pt')},
        exporter_sha256=sha(OFFICIAL/'flatquant/flat_utils.py'), representation='official symmetric packed INT4 export')


def export_checkpoint(checkpoint, destination):
    """Export saved RTN codes/scales only; never recalibrate or quantize new weights."""
    import torch
    from flatquant.flat_utils import save_quantized_weights_with_safetensors
    manifest, identity = source_identity(checkpoint)
    done = destination/'DONE.json'
    if done.exists():
        exported = json.loads(done.read_text())
        if exported['identity'] != identity:
            raise ValueError('Packed artifact identity mismatch; choose a distinct destination')
        for name, digest in exported['files'].items():
            if sha(destination/name) != digest:
                raise ValueError(f'Packed artifact hash mismatch: {name}')
        return exported, True
    if destination.exists():
        raise FileExistsError(f'Incomplete destination exists; preserve for diagnosis: {destination}')
    staging = destination.with_name(destination.name+f'.staging.{os.getpid()}')
    staging.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    host_memory = [cpu_memory_snapshot('before_source_load')]
    # Read-only operations on trusted local ZIP checkpoints. mmap lets clean
    # source pages be reclaimed instead of keeping ~15 GiB anonymous weights.
    model = torch.load(checkpoint/'quantized_model.pt', map_location='cpu', weights_only=False, mmap=True)
    quantizers = torch.load(checkpoint/'weight_quantizers.pt', map_location='cpu', weights_only=False, mmap=True)
    host_memory.append(cpu_memory_snapshot('after_mapped_source_load'))
    args = SimpleNamespace(**manifest['official_args'])
    args.exp_dir = str(staging)
    with torch.inference_mode():
        save_quantized_weights_with_safetensors(args, model, quantizers, sym=True)
    host_memory.append(cpu_memory_snapshot('after_official_export'))
    del model, quantizers
    trimmed = release_cpu_memory()
    host_memory.append(cpu_memory_snapshot('after_source_release'))
    exported = dict(status='COMPLETE', identity=identity, export_seconds=time.monotonic()-started,
        original_calibration_cost=manifest.get('calibration_cost'),
        files={path.name:sha(path) for path in sorted(staging.iterdir()) if path.is_file()},
        exporter='flatquant.flat_utils.save_quantized_weights_with_safetensors(sym=True)',
        recalibration=False, gptq_rerun=False,
        host_memory=host_memory, malloc_trim_after_export=trimmed,
        input_loading='torch.load mmap=True, CPU; official export under inference_mode')
    write(staging/'DONE.json', exported)
    staging.rename(destination)
    return exported, False


def required_keys(model):
    """Every used native parameter/buffer must originate in the saved checkpoint.

    Explicit exclusions are unused diagonal scales, pass-through quantizers, and
    matrices aliased from the same layer's loaded shared transform below.
    """
    from deploy.nn import OnlineTrans, Quantizer
    state = model.state_dict()
    excluded = {}
    for name, module in model.named_modules():
        if isinstance(module, OnlineTrans) and hasattr(module, 'diag_scale'):
            excluded[name+'.diag_scale'] = 'unused in official OnlineTrans.forward; folded before export'
        if isinstance(module, Quantizer):
            for attr in ('clip_factor_a_max','clip_factor_a_min'):
                excluded[name+'.'+attr] = 'unused fallback; transformed input already packed or branch inactive'
    for i, layer in enumerate(model.model.layers):
        for branch in ('q','k','v'):
            for matrix in ('left_matrix','right_matrix'):
                excluded[f'model.layers.{i}.self_attn.inp_trans_{branch}.{matrix}'] = 'alias of loaded attention shared matrix'
        for branch in ('g','u'):
            for matrix in ('left_matrix','right_matrix'):
                excluded[f'model.layers.{i}.mlp.inp_trans_{branch}.{matrix}'] = 'alias of loaded MLP shared matrix'
    unknown = set(excluded)-set(state)
    if unknown:
        raise ValueError(f'Native module structure differs from audited official version: {sorted(unknown)}')
    return set(state)-set(excluded), excluded


def load_state_verified(model, checkpoint):
    """Official key mapping and aliases with mandatory used-parameter coverage."""
    import torch
    from deploy.nn import Linear4bit
    required, excluded = required_keys(model)
    state = dict(checkpoint['model_state_dict'])
    duplicate = set(state) & set(checkpoint['quantizers'])
    if duplicate:
        raise ValueError(f'Duplicate exported keys: {sorted(duplicate)}')
    state.update(checkpoint['quantizers'])
    missing = sorted(required-set(state))
    if missing:
        raise ValueError(f'Missing used native parameters: {missing}')
    target = model.state_dict()
    linears = [(name, module) for name,module in model.named_modules() if isinstance(module,Linear4bit)]
    if len(linears) != len(model.model.layers)*7:
        raise ValueError('Expected seven packed projection matrices per layer')
    for name, module in linears:
        if state[name+'.weight'].dtype != torch.uint8:
            raise ValueError(f'Expected packed UINT8 weight: {name}')
    scalar_shape_adapters = []
    for name in required:
        if state[name].shape != target[name].shape:
            # Official training stores one-element clip Parameters; native
            # modules use scalar buffers. PyTorch's official loader already
            # accepts this scalar compatibility case; make it explicit.
            if 'clip_factor_a_' in name and state[name].numel()==1 and target[name].ndim==0:
                state[name] = state[name].reshape(())
                scalar_shape_adapters.append(name)
            else:
                raise ValueError(f'Native shape mismatch: {name}: {state[name].shape} vs {target[name].shape}')
    incompatible = model.load_state_dict(state, strict=False)
    for layer in model.model.layers:
        for branch in ('q','k','v'):
            transform = getattr(layer.self_attn,'inp_trans_'+branch)
            for matrix in ('left_matrix','right_matrix'):
                transform.register_buffer(matrix,getattr(layer.self_attn,matrix))
        for branch in ('u','g'):
            transform = getattr(layer.mlp,'inp_trans_'+branch)
            for matrix in ('left_matrix','right_matrix'):
                transform.register_buffer(matrix,getattr(layer.mlp,matrix))
    # Verify actual copied tensors, including dtype conversion from the exporter.
    loaded = model.state_dict()
    for name in required:
        expected = state[name].to(dtype=loaded[name].dtype)
        # Compare representation bits, including any NaN payloads; this is a
        # checkpoint-copy check, not a numerical-quality execution gate.
        if not torch.equal(loaded[name].reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)):
            raise ValueError(f'Native parameter copy differs from exported tensor: {name}')
    for module in model.modules():
        for attr in ('clip_factor_a_max','clip_factor_a_min'):
            value = getattr(module,attr,None)
            if isinstance(value,torch.Tensor):
                delattr(module,attr)
                setattr(module,attr,value.item())
    return dict(status='COMPLETE', packed_linear_count=len(linears),
        required_loaded_keys=sorted(required), excluded_unused_or_aliased_keys=excluded,
        official_missing_keys=list(incompatible.missing_keys),
        official_unexpected_keys=list(incompatible.unexpected_keys),
        every_used_tensor_matches_export=True, scalar_clip_shape_adapters=sorted(scalar_shape_adapters))


def verify_rope(model):
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    config = model.config
    rope_type = config.rope_scaling.get('rope_type',config.rope_scaling.get('type'))
    if rope_type != 'llama3':
        raise ValueError('Llama-3.1 scaled RoPE configuration was lost')
    expected, attention_factor = ROPE_INIT_FUNCTIONS['llama3'](config, device=torch.device('cpu'))
    actual = model.model.rotary_emb.inv_freq.detach().cpu()
    if not torch.equal(actual,expected):
        raise ValueError('Native shared rotary frequencies differ from configured Llama-3.1 scaling')
    return dict(status='COMPLETE', config=config.rope_scaling, theta=config.rope_theta,
        shared_inv_freq_dtype=str(actual.dtype), max_abs_difference=float((actual-expected).abs().max()),
        attention_factor=float(attention_factor),
        path='HF 4.45 LlamaModel shared position_embeddings -> native attention apply_rotary_pos_emb')


def construct_native(config):
    import torch
    from transformers.modeling_utils import no_init_weights
    from deploy.transformers.modeling_llama import FlatQuantLlamaForCausalLM
    args = SimpleNamespace(fuseLN=False, trans='matmul',
        online_trans={'qk','o_proj','down_proj','qkv_proj','up_gate_proj'})
    config._attn_implementation = 'flash_attention_2'
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with no_init_weights():
            model = FlatQuantLlamaForCausalLM(args=args, config=config)
    finally:
        torch.set_default_dtype(previous_dtype)
    return model


def build_native(config, packed):
    release_cpu_memory()
    host_memory = [cpu_memory_snapshot('before_native_construction')]
    model = construct_native(config)
    # The official constructor replaces temporary floating projection modules.
    # Return their freed pages before mapping the packed checkpoint.
    release_cpu_memory()
    host_memory.append(cpu_memory_snapshot('after_native_construction'))
    loader = load_module('flatquant_official_benchmark_loader',OFFICIAL/'benchmarks/benchmark_model.py')
    mapped = loader.rename_keys(loader.load_from_safetensors(str(packed)))
    coverage = load_state_verified(model,mapped)
    host_memory.append(cpu_memory_snapshot('after_native_copy_verification'))
    del mapped
    release_cpu_memory()
    host_memory.append(cpu_memory_snapshot('after_export_mapping_release'))
    coverage['host_memory'] = host_memory
    rope = verify_rope(model)
    return model.eval(), coverage, rope


def ppl(model, windows, output, record):
    import torch
    record['evaluation'] = dict(tokens=str(TOKENS),sha256=sha(TOKENS),batch_size=1,
        sequence_length=2048,added_bos=False,use_cache=True,cache_reset='fresh for every window',
        loss='FP32 shifted CE; 2047 scored tokens per window',
        attention_KV='official native prefill uses transformed unquantized K/V; packed cache also constructed')
    record['chunks'] = []
    with torch.inference_mode():
        for i,row in enumerate(windows):
            batch=row[None].cuda()
            result=model(input_ids=batch,past_key_values=None,use_cache=True,num_logits_to_keep=0)
            logits=result.logits
            nll=float(torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),batch[:,1:].reshape(-1)))
            record['chunks'].append(dict(chunk=i,nll=nll if math.isfinite(nll) else None,tokens_scored=2047))
            record['completed_sequences']=i+1
            save_record(output,record)
            if i%16==0:
                print(f'Native FlatQuant PPL {i+1}/141 NLL={nll}',flush=True)
            del result,logits,batch
    nlls=[chunk['nll'] for chunk in record['chunks']]
    try:
        record['ppl']=math.exp(sum(nlls)/len(nlls)) if all(value is not None for value in nlls) else None
    except OverflowError:
        record['ppl']=None
    record['ppl_status']='COMPLETE' if record['ppl'] is not None else 'NONFINITE'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--packed-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--cache-optimization',choices=('official','remove-dead-random'),default='official')
    cli=parser.parse_args()
    record=dict(kind='flatquant_packed_native',status='RUNNING',phase='native-ppl-timing',
        model='Meta-Llama-3.1-8B',model_key='llama31_8b',method='FlatQuant official packed INT4',seed=0,
        source_commit=cli.source_commit,source_sha256=sha(__file__),job_id=os.getenv('SLURM_JOB_ID'),
        checkpoint=str(cli.checkpoint),packed_checkpoint=str(cli.packed_checkpoint),base_checkpoint=str(BASE),
        official_commit=OFFICIAL_COMMIT,command=sys.orig_argv,execution_backend=NATIVE_NOTE,
        config='W4 RTN per-channel symmetric packed; A4 per-token symmetric; native asymmetric KV4 per head128; FP16 residual',
        native_int4_kernel=True,numerical_execution_gates='none',
        cache_optimization=dict(mode=cli.cache_optimization,status='not installed'),
        variant=dict(prefill_attention='transformed unquantized K/V',decode_attention='packed asymmetric INT4 KV',
            kv_learned_clipping=False,gqa_cache_heads='repeated to full query-head count',
            source_fake_quant_residual='BF16',native_residual='FP16',
            same_format_as_fake_quant_accuracy=False,
            official_dequantization='Pinned CUDA helper truncates accumulator / 10 to integer before restoring factor 10; unchanged upstream arithmetic'))
    save_record(cli.output,record)
    started=time.monotonic()
    try:
        import torch
        import transformers
        import triton
        if not torch.cuda.is_available():
            raise RuntimeError('Requires allocated exclusive A40 GPU; no login-node timing')
        if torch.cuda.get_device_capability()!=(8,6):
            raise RuntimeError('This native extension was built for SM86/A40 only')
        if subprocess.check_output(['git','-C',str(OFFICIAL),'rev-parse','HEAD'],text=True).strip()!=OFFICIAL_COMMIT:
            raise ValueError('Official native source checkout changed')
        import deploy
        if not Path(deploy.__file__).resolve().is_relative_to(OFFICIAL):
            raise ValueError('Wrong native source imported')
        from nar.e39_native_cache_optimization import install_native_cache_optimization
        record['cache_optimization']=install_native_cache_optimization(cli.cache_optimization)
        torch.set_num_threads(4)
        record['libraries']=dict(torch=torch.__version__,transformers=transformers.__version__,triton=triton.__version__)
        record['runtime_manifest']=dict(path=str(RUNTIME_MANIFEST),sha256=sha(RUNTIME_MANIFEST),
            content=json.loads(RUNTIME_MANIFEST.read_text()))
        record['gpu']=torch.cuda.get_device_name()
        record['export'],record['packed_checkpoint_reused']=export_checkpoint(cli.checkpoint,cli.packed_checkpoint)
        save_record(cli.output,record)
        config=transformers.AutoConfig.from_pretrained(BASE,local_files_only=True)
        model,record['parameter_coverage'],record['rope_verification']=build_native(config,cli.packed_checkpoint)
        model.cuda().eval()
        release_cpu_memory()
        record['host_memory_after_gpu_transfer']=cpu_memory_snapshot('after_gpu_transfer')
        record['model_object_id']=id(model)
        windows=torch.load(TOKENS,map_location='cpu',weights_only=True)
        if tuple(windows.shape)!=(141,2048):
            raise ValueError('Expected frozen 141 x 2048 evaluation tokens')
        try:
            ppl(model,windows,cli.output,record)
        except Exception as exc:
            record.update(ppl_status='FAILED',ppl_error=repr(exc),ppl_traceback=traceback.format_exc())
        save_record(cli.output,record)
        helper=load_module('rebuttal_baseline_timing',TIMING_HELPER)
        record['timing_helper']=dict(path=str(TIMING_HELPER),sha256=sha(TIMING_HELPER))
        helper.measure_timing(model,windows,record,cli.output,save_record,
            repetitions=20,warmup=10,prefix=2048,decode_steps=128,discard=8)
        record['status']='COMPLETE' if record.get('ppl_status')=='COMPLETE' and record['timing']['status']=='COMPLETE' else 'PARTIAL'
    except Exception as exc:
        record.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds']=time.monotonic()-started
        save_record(cli.output,record)


if __name__=='__main__':
    main()
