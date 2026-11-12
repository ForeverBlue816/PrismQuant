"""Official DuQuant native-configuration W4A4, Llama-3.1-8B/3.2-3B, seed0.

Native configuration means official algorithm/default quantization granularity;
it does NOT mean native INT4 GPU arithmetic. Official FP16 fake-quant modules
are retained. A local trusted full-model pickle enables tasks without calibration.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
from importlib import metadata
import json
import logging
import math
import os
from pathlib import Path
import pickle
import platform
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1]
OFFICIAL=Path('/home/yanlongc/DuQuant-upstream')
WORK=Path('/projects/nar/nar-validation')
BASE=WORK/'cache/huggingface/models--unsloth--Meta-Llama-3.1-8B/snapshots/e9a141a2091ea561b96483212645a2a05e6f99fc'
TOKENS=WORK/'cache/tokenized/ffcc87608d60-wikitext2-test-full-l2048.pt'
MODELS={
    'llama31_8b':dict(name='Meta-Llama-3.1-8B',base=BASE,tokens=TOKENS),
    'llama32_3b':dict(name='Llama-3.2-3B',base=WORK/'cache/huggingface/models--unsloth--Llama-3.2-3B/snapshots/d4446454d87d51aa42e1fb174f25acc5f8762331',
        tokens=WORK/'cache/tokenized/llama32_3b-wikitext2-test-full-l2048.pt'),
}
TIMING_HELPER=Path('/home/yanlongc/prismquant-rebuttal-e38/scripts/baseline_timing.py')
TRAIN=WORK/'cache/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train.arrow'
TASKS=('arc_easy','arc_challenge','boolq','hellaswag','openbookqa','piqa','social_iqa','winogrande')
METRICS={x:('acc,none' if x in ('boolq','social_iqa','winogrande') else 'acc_norm,none') for x in TASKS}
SCOPE='Official DuQuant W4A4 native configuration in FP16 floating fake-quant modules; not E14 g128/KIVI and not native INT4 deployment'


def now(): return datetime.now(timezone.utc).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def identity(path, content=True):
    path=Path(path); out=dict(path=str(path),resolved=str(path.resolve()),bytes=path.stat().st_size)
    if content: out['sha256']=sha(path)
    return out


def serial(value):
    def default(item):
        if hasattr(item,'item'): return item.item()
        return str(item)
    value=json.loads(json.dumps(value,default=default))
    def clean(v):
        if isinstance(v,float) and not math.isfinite(v): return str(v)
        if isinstance(v,list): return [clean(x) for x in v]
        if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
        return v
    return clean(value)


def write(path,record):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(serial(record),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def native_args(artifact,model_key='llama31_8b'):
    specification=MODELS[model_key]
    a=SimpleNamespace(model=str(specification['base']),net=specification['name'],seed=0,nsamples=128,batch_size=1,
        wbits=4,abits=4,epochs=0,smooth_epochs=0,lwc=True,smooth=True,alpha=.6,let_alpha=.8,
        let=False,lac=.9,swc=.8,block_size=128,max_rotation_step=256,permutation_times=1,
        symmetric=False,a_dynamic_method='per_token',w_dynamic_method='per_channel',
        group_size=None,act_group_size=None,quant_method='duquant',deactive_amp=False,
        aug_loss=False,resume=None,save_dir=str(artifact/'official_parameters'),multigpu=False,
        let_lr=.005,smooth_lr=.0001,lwc_lr=.01,wd=0)
    a.weight_quant_params=dict(n_bits=4,per_channel_axes=[0],symmetric=False,dynamic_method='per_channel',
        group_size=None,lwc=True,swc=.8,quant_method='duquant',block_size=128,max_rotation_step=256,permutation_times=1)
    a.act_quant_params=dict(n_bits=4,per_channel_axes=[],symmetric=False,lac=.9,act_group_size=None,
        dynamic_method='per_token',quant_method='duquant',block_size=128,max_rotation_step=256,permutation_times=1)
    # Official duquant overwrites Q/K parameters with act_quant_params per layer.
    a.q_quant_params=dict(n_bits=4,per_channel_axes=[],symmetric=False,dynamic_method='per_token',
        quant_method='duquant',block_size=128,max_rotation_step=256)
    a.k_quant_params=dict(n_bits=4,per_channel_axes=[],symmetric=False,dynamic_method='per_token',
        quant_method='duquant',block_size=128)
    a.v_quant_params=dict(n_bits=4,per_channel_axes=[],symmetric=False,dynamic_method='per_token')
    a.p_quant_params=dict(n_bits=16,metric='fix0to1')
    return a


def rotation_table(n=128,device='cuda'):
    """Exactly the get_rot.py Givens recursion, limited to requested block128.

    Official utils.get_rot subsequently applies its seeded random QR factor;
    that code and its RNG calls are not replaced here.
    """
    import numpy as np
    import torch
    R=torch.eye(n,device=device)
    for i in range(n-1):
        rotation=torch.eye(n,device=device)
        cosine=1/np.sqrt(i+2);sine=np.sqrt(i+1)/np.sqrt(i+2)
        rotation[i,i]=rotation[i+1,i+1]=cosine
        rotation[i,i+1]=-sine;rotation[i+1,i]=sine
        R=torch.matmul(rotation,R)
    rows=torch.arange(n);rows[0],rows[n-1]=rows[n-1].item(),rows[0].item()
    R=R[rows]
    columns=torch.arange(n);columns[0],columns[n-1]=columns[n-1].item(),columns[0].item()
    return R[:,columns].cpu()


def sample_windows(ids,nsamples=128,seq_len=2048,seed=0):
    rng=random.Random(seed);batches=[];positions=[]
    for _ in range(nsamples):
        start=rng.randint(0,ids.shape[1]-seq_len-1)
        value=ids[:,start:start+seq_len];target=value.clone();target[:,:-1]=-100
        batches.append((value,target));positions.append(start)
    return batches,positions


def calibration(tokenizer,artifact,record):
    import torch
    from datasets import Dataset
    dataset=Dataset.from_file(str(TRAIN))
    text='\n\n'.join(dataset['text'])
    encoded=tokenizer(text,return_tensors='pt').input_ids
    batches,positions=sample_windows(encoded)
    # Match the official global Python-RNG state after calibration sampling.
    random.seed(0)
    for _ in range(128): random.randint(0,encoded.shape[1]-2048-1)
    path=artifact/'calibration.pt'
    with path.open('xb') as f:torch.save(dict(batches=batches,positions=positions),f)
    record['calibration']=dict(source=identity(TRAIN),dataset_fingerprint=dataset._fingerprint,
        nsamples=128,sequence_length=2048,seed=0,positions=positions,token_count=encoded.numel(),
        tokenizer_default_special_tokens=True,first_token_id=int(encoded[0,0]),
        algorithm='Official datautils.get_wikitext2: join all train text with two newlines; random.Random(0).randint(0,N-2048-1), with replacement',
        artifact=identity(path))
    return batches


def source_record():
    names=['main.py','run.sh','get_rot.py','datautils.py','generate_act_scale_shift.py','utils.py']
    names += [str(p.relative_to(OFFICIAL)) for folder in ('quantize','models') for p in sorted((OFFICIAL/folder).glob('*.py'))]
    return dict(official_commit=subprocess.check_output(['git','-C',str(OFFICIAL),'rev-parse','HEAD'],text=True).strip(),
        official_files={name:sha(OFFICIAL/name) for name in names},
        runner_files={name:sha(REPO/name) for name in ('nar/e42_duquant_native.py','nar/e42_duquant_compat.py','slurm_e42_duquant_native.sh')},
        policy='record only; no source or numerical execution veto')


def reuse_preparation(manifest_path,artifact,model_key,official_args,record):
    """Reuse only identified completed preprocessing, never an unverified folder."""
    import torch
    previous=json.loads(Path(manifest_path).read_text())
    specification=MODELS[model_key]
    if previous.get('model_key')!=model_key or previous.get('base_revision')!=specification['base'].name:
        raise ValueError('Preparation manifest belongs to a different base model')
    recipe_keys=('seed','nsamples','wbits','abits','epochs','smooth_epochs','lwc','smooth','alpha','let_alpha',
        'let','lac','swc','block_size','max_rotation_step','permutation_times','symmetric',
        'a_dynamic_method','w_dynamic_method','group_size','act_group_size')
    saved=previous.get('official_arguments',{})
    if Path(saved.get('model','')).resolve()!=specification['base'].resolve():
        raise ValueError('Preparation manifest has a different base checkpoint path')
    differences={key:(saved.get(key),getattr(official_args,key)) for key in recipe_keys
        if saved.get(key)!=getattr(official_args,key)}
    if differences:raise ValueError(f'Preparation recipe differs: {differences}')
    entries=(('calibration.pt',previous['calibration']['artifact']),
        ('act_scales.pt',previous['smoothing']['scales']),('Rot.pkl',previous['rotation_table']))
    for name,source in entries:
        path=artifact/name
        if path.resolve()!=Path(source['path']).resolve() or sha(path)!=source['sha256']:
            raise ValueError(f'Preparation identity mismatch: {path}')
    saved_calibration=torch.load(artifact/'calibration.pt',map_location='cpu',weights_only=True)
    batches=saved_calibration['batches'];positions=saved_calibration['positions']
    if len(batches)!=official_args.nsamples or positions!=previous['calibration']['positions']:
        raise ValueError('Saved calibration windows/positions are incomplete or inconsistent')
    if any(tuple(batch[0].shape)!=(1,2048) or tuple(batch[1].shape)!=(1,2048) for batch in batches):
        raise ValueError('Saved calibration windows have unexpected shapes')
    random.seed(official_args.seed)
    replay=[random.randint(0,previous['calibration']['token_count']-2048-1) for _ in range(official_args.nsamples)]
    if replay!=positions:raise ValueError('Saved positions do not match the original seeded sampling protocol')
    scales=torch.load(artifact/'act_scales.pt',map_location='cpu',weights_only=True)
    for key in ('calibration','smoothing','rotation_table'):
        record[key]=previous[key]
    record['preparation_reuse']=dict(manifest=identity(manifest_path),
        files=[identity(artifact/name) for name,_ in entries],
        calibration_and_scale_collection_repeated=False,
        original_smooth_scale_collection_seconds=previous.get('smooth_scale_collection_seconds'),
        python_rng='restored by replaying saved sampling positions; torch/NumPy preprocessing consumes no random draws')
    return batches,scales


def loaded_model_identity(model,base):
    """Reject an accidentally mixed 3B/8B artifact, independent of its score."""
    expected=json.loads((base/'config.json').read_text())
    keys=('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads',
          'num_key_value_heads','vocab_size','rope_theta','rope_scaling')
    actual={key:getattr(model.config,key,None) for key in keys}
    differences={key:dict(expected=expected.get(key),actual=actual[key]) for key in keys
        if actual[key]!=expected.get(key)}
    if differences:raise ValueError(f'Checkpoint model configuration does not match selected base: {differences}')
    return dict(base_config=identity(base/'config.json'),matched_fields=actual,
        actual_decoder_layers=len(model.model.layers))


def measure_ppl(model,record,output,tokens=TOKENS):
    import torch
    start=time.perf_counter();chunks=[]
    record['ppl']=dict(status='RUNNING',chunks=chunks,batch_size=1,sequence_length=2048,
        no_bos=True,scoring='FP32 next-token cross entropy, 2047 labels per complete2048-token window',
        tokens=identity(tokens),numerical_execution_gate=False)
    write(output,record)
    try:
        windows=torch.load(tokens,map_location='cpu',weights_only=True)
        record['ppl']['sequences_expected']=len(windows)
        with torch.inference_mode():
            for index in range(len(windows)):
                ids=windows[index:index+1].cuda()
                result=model(input_ids=ids,labels=ids,use_cache=False)
                nll=float(result.loss.float())
                chunks.append(dict(chunk=index,nll=nll,tokens_scored=ids.shape[1]-1,finite=math.isfinite(nll)))
                record['ppl']['completed_sequences']=len(chunks)
                write(output,record);del result,ids
                print(f'DuQuant PPL {index+1}/{len(windows)} nll={nll:.8f}',flush=True)
        finite=all(x['finite'] for x in chunks)
        total=sum(x['tokens_scored'] for x in chunks)
        record['ppl'].update(status='COMPLETE' if finite else 'NONFINITE',tokens_scored=total,
            value=math.exp(sum(x['nll']*x['tokens_scored'] for x in chunks)/total) if finite else None,
            completed_141_sequences=len(chunks)==141)
    except Exception as exc:
        record['ppl'].update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
        traceback.print_exc()
    record['ppl']['elapsed_seconds']=time.perf_counter()-start
    write(output,record)


def evaluate_tasks(model,tokenizer,args,record):
    # Official DuQuant bundles an older lm_eval. Explicitly select the recorded
    # frozen modern harness for this separate comparability phase.
    harness='/home/yanlongc/.e13_packages'
    if harness in sys.path:sys.path.remove(harness)
    sys.path.insert(0,harness)
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    record['harness']=dict(module=identity(lm_eval.__file__),expected_revision_recorded='b954108c9baaaa934b4ad842033b31a97ee30816')
    lm=HFLM(pretrained=model,tokenizer=tokenizer,batch_size=args.batch_size,max_batch_size=args.batch_size,max_length=2048)
    manager=TaskManager();rows=[]
    record['eight_tasks']=dict(status='RUNNING',tasks=rows,seed=0,num_fewshot=0,batch_size=args.batch_size,
        max_length=2048,log_samples=True,limit=None)
    for task in TASKS:
        target=args.output.with_name(args.output.stem+'_'+task+'.json')
        payload=dict(task=task,scope=SCOPE,seed=0,status='RUNNING',result=None)
        begin=time.perf_counter();write(args.output,record)
        try:
            result=lm_eval.simple_evaluate(model=lm,tasks=[task],num_fewshot=0,batch_size=args.batch_size,
                max_batch_size=args.batch_size,task_manager=manager,log_samples=True,limit=None,bootstrap_iters=0,
                random_seed=0,numpy_random_seed=0,torch_random_seed=0,fewshot_random_seed=0,
                cache_requests=False,apply_chat_template=False,fewshot_as_multiturn=False)
            payload['result']=result
            expected=result['n-samples'][task]['original'];samples=result['samples'][task]
            if len(samples)!=expected or len({(x['doc_id'],x['filter']) for x in samples})!=expected:
                raise ValueError('incomplete or duplicate per-item task outputs')
            metric=METRICS[task];value=float(result['results'][task][metric]);field=metric.split(',')[0]
            reconstructed=sum(float(x[field]) for x in samples)/len(samples)
            if not math.isfinite(value) or abs(reconstructed-value)>1e-12:
                raise ValueError('per-item accuracy differs from aggregate')
            payload.update(status='COMPLETE',accuracy=value,metric=metric,samples=len(samples))
        except Exception as exc:
            payload.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());traceback.print_exc()
        payload['elapsed_seconds']=time.perf_counter()-begin
        with target.open('x') as f:json.dump(serial(payload),f,indent=2,allow_nan=False);f.write('\n')
        rows.append(dict(task=task,status=payload['status'],output=identity(target),accuracy=payload.get('accuracy')))
        write(args.output,record)
    passed=[x for x in rows if x['status']=='COMPLETE']
    record['eight_tasks'].update(status='COMPLETE' if len(passed)==8 else 'PARTIAL',completed_tasks=len(passed),
        mean_accuracy=sum(x['accuracy'] for x in passed)/8 if len(passed)==8 else None)
    write(args.output,record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('quantize-ppl','eight-tasks','ppl-timing'),default='quantize-ppl')
    parser.add_argument('--model',choices=tuple(MODELS),default='llama31_8b')
    parser.add_argument('--artifact-dir',required=True,type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--resume-preparation-from',type=Path,
        help='Prior local run JSON identifying complete calibration/scales/rotation preprocessing')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--source-commit')
    parser.add_argument('--batch-size',type=int,choices=(1,),default=1,help='Official DuQuant rotation indexing expects batch1')
    args=parser.parse_args();args.output=args.output.resolve();args.artifact_dir=args.artifact_dir.resolve()
    specification=MODELS[args.model];base=specification['base'];tokens=specification['tokens']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(dict(status='STARTING'),f)
    record=dict(kind='E42_DuQuant_official_native_'+('8b' if args.model=='llama31_8b' else '3b'),status='RUNNING',scope=SCOPE,phase=args.phase,seed=0,
        model=specification['name'],model_key=args.model,base_revision=base.name,command_line=list(sys.orig_argv),source_commit=args.source_commit,
        hostname=platform.node(),started_utc=now(),numerical_execution_gates='none',
        ppl=dict(status='not run'),eight_tasks=dict(status='not run'),
        slurm={k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','CUDA_VISIBLE_DEVICES')})
    started=time.perf_counter();code=1
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HOME=str(WORK/'cache/huggingface'),HF_DATASETS_CACHE=str(WORK/'cache/datasets'))
    try:
        import torch
        import numpy as np
        import transformers,tokenizers
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from nar import e42_duquant_compat as compat
        if not torch.cuda.is_available():raise RuntimeError('allocated GPU required')
        if torch.cuda.device_count()!=1:
            raise RuntimeError('Expose exactly one GPU: official DuQuant otherwise automatically places the MLP on cuda:1')
        torch.set_num_threads(4);random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark=True
        record.update(source_record(),libraries=dict(torch=torch.__version__,cuda=torch.version.cuda,
            transformers=transformers.__version__,transformers_file=transformers.__file__,tokenizers=tokenizers.__version__,
            packages={name:metadata.version(name) for name in ('datasets','accelerate','peft','numpy')}),
            gpu_model=torch.cuda.get_device_name(),gpu_capability=list(torch.cuda.get_device_capability()),
            tokenizer_files=[identity(base/name) for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json')])
        write(args.output,record)
        tokenizer=AutoTokenizer.from_pretrained(base,use_fast=True,local_files_only=True)
        record['tokenizer_adapter']='Fast loader for the same pinned tokenizer.json: TF4.45 cannot instantiate this snapshot via use_fast=False; no token vocabulary or special-token policy changed'
        if args.phase=='quantize-ppl':
            existing=args.artifact_dir/'quantized_model.pt'
            if existing.is_file():
                record['compatibility']=compat.install()
                model=torch.load(existing,map_location='cpu',weights_only=False).eval().cuda()
                record['compatibility']=compat.install(model)
                model.config.use_cache=False
                record['checkpoint']=dict(**identity(existing),reused_without_calibration=True,
                    format='local trusted full torch module pickle')
                previous=args.artifact_dir/'quantization_manifest.json'
                if previous.is_file():
                    record['prior_quantization']=dict(manifest=identity(previous),record=json.loads(previous.read_text()))
                record['quantization_seconds']=0.
                record['reuse_policy']='Existing complete serialized model reused directly; no calibration, smoothing or rotation search repeated'
                write(args.output,record)
            else:
                preparation_manifest=args.resume_preparation_from
                if preparation_manifest is None and (args.artifact_dir/'preparation_manifest.json').is_file():
                    preparation_manifest=args.artifact_dir/'preparation_manifest.json'
                args.artifact_dir.mkdir(parents=True,exist_ok=preparation_manifest is not None)
                official_args=native_args(args.artifact_dir,args.model)
                Path(official_args.save_dir).mkdir(exist_ok=preparation_manifest is not None)
                record['official_arguments']=vars(official_args).copy()
                record['calibration_dtype']='official FP16 model/activations; no E14 weights or rotations reused'
                model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,attn_implementation='eager',local_files_only=True).eval()
                model.config.use_cache=False
                for parameter in model.parameters():parameter.requires_grad=False
                if preparation_manifest is not None:
                    batches,act_scales=reuse_preparation(preparation_manifest,args.artifact_dir,args.model,official_args,record)
                else:
                    batches=calibration(tokenizer,args.artifact_dir,record);write(args.output,record)
                    # Use the official scale collector; Llama's branch never reads shifts.
                    from generate_act_scale_shift import get_act_scales
                    begin=time.perf_counter();model.cuda()
                    with torch.no_grad():act_scales=get_act_scales(model,batches,num_samples=128)
                    torch.cuda.synchronize()
                    record['smooth_scale_collection_seconds']=time.perf_counter()-begin
                    scale_file=args.artifact_dir/'act_scales.pt'
                    with scale_file.open('xb') as f:torch.save(act_scales,f)
                    record['smoothing']=dict(scales=identity(scale_file),shift='unused by official Llama branch; passed empty dict',
                        source='official generate_act_scale_shift.get_act_scales, per-input-channel absolute maximum over128 sequences')
                model.cpu();torch.cuda.empty_cache()
                record['compatibility']=compat.install(model)
                if preparation_manifest is None:
                    rotation=rotation_table()
                    with (args.artifact_dir/'Rot.pkl').open('xb') as f:pickle.dump({128:rotation},f)
                    record['rotation_table']=dict(**identity(args.artifact_dir/'Rot.pkl'),block_size=128,
                        construction='official get_rot.py sequential Givens recursion and exchange(0,127); subsequent official random-QR draws unchanged')
                write(args.artifact_dir/'preparation_manifest.json',record)
                os.chdir(args.artifact_dir)
                import utils as official_utils
                official_utils.Rot.clear()
                logger=official_utils.create_logger(str(args.artifact_dir),name='duquant-e42')
                from quantize.duquant import duquant
                lm=SimpleNamespace(model=model,device=torch.device('cuda'),seqlen=2048)
                record['catcher_device_compatibility']=compat.prepare_official_catcher(model,lm.device)
                begin=time.perf_counter();write(args.output,record)
                with torch.no_grad():model=duquant(lm,official_args,batches,act_scales,{},logger=logger)
                torch.cuda.synchronize()
                record['quantization_seconds']=time.perf_counter()-begin
                model.eval();model.config.use_cache=False
                record['compatibility']=compat.install(model)
                checkpoint=args.artifact_dir/'quantized_model.pt'
                begin=time.perf_counter()
                # This checkpoint is a locally created trusted pickle. Only this
                # explicit artifact is loaded with weights_only=False in task phase.
                model.cpu()
                temporary=checkpoint.with_suffix('.pt.tmp')
                with temporary.open('xb') as f:torch.save(model,f)
                os.replace(temporary,checkpoint)
                record['checkpoint']=dict(**identity(checkpoint),format='local trusted full torch module pickle',
                    reuse='contains official weights, rotations, permutation and quantizer state; no recalibration required')
                record['checkpoint_save_seconds']=time.perf_counter()-begin
                write(args.artifact_dir/'quantization_manifest.json',record)
                write(args.output,record);gc.collect();model.cuda()
            record['loaded_model_identity']=loaded_model_identity(model,base)
            measure_ppl(model,record,args.output,tokens)
            record['status']='COMPLETE' if record['ppl']['status']=='COMPLETE' else 'PARTIAL'
            code=0
        else:
            checkpoint=(args.checkpoint or args.artifact_dir/'quantized_model.pt').resolve()
            record['checkpoint']=identity(checkpoint)
            record['compatibility']=compat.install()
            # Only load the user-authorized local artifact created above.
            model=torch.load(checkpoint,map_location='cpu',weights_only=False).eval().cuda()
            record['compatibility']=compat.install(model)
            model.config.use_cache=False
            record['loaded_model_identity']=loaded_model_identity(model,base)
            if args.phase=='eight-tasks':
                evaluate_tasks(model,tokenizer,args,record)
                record['status']=record['eight_tasks']['status'];code=0 if record['eight_tasks']['completed_tasks'] else 1
            else:
                record['model_object_id_before_ppl']=id(model)
                measure_ppl(model,record,args.output,tokens)
                helper_spec=importlib.util.spec_from_file_location('baseline_timing',TIMING_HELPER)
                helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
                record['timing_helper']=identity(TIMING_HELPER)
                record['timing_backend']='Official DuQuant FP16 floating fake-quant with legacy tuple KV cache; not native INT4 throughput'
                windows=torch.load(tokens,map_location='cpu',weights_only=True)
                helper.measure_timing(model,windows,record,args.output,write)
                record['model_object_id_after_timing']=id(model)
                record['status']='COMPLETE' if record['ppl']['status']=='COMPLETE' and record['timing']['status']=='COMPLETE' else 'PARTIAL'
                code=0
        record['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        record['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    except Exception as exc:
        record.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        record.update(ended_utc=now(),total_elapsed_seconds=time.perf_counter()-started,
            cost_scope='wall clock of this process, phase breakdown and PyTorch peak memory; excludes queue wait; not a native kernel speed claim')
        write(args.output,record)
        print(json.dumps(dict(status=record['status'],output=str(args.output))),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
