"""Official FlatQuant W4A4KV4 on pinned Llama-3.1-8B / Llama-3.2-3B.

Adapters only resolve local data/model paths and persist reusable checkpoints.
Native published recipe uses RTN weights and per-token symmetric activations;
it must not be described as the E14 GPTQ/g128 reference configuration.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

WORK = Path('/projects/nar/nar-validation')
BASE = WORK/'cache/huggingface/models--unsloth--Meta-Llama-3.1-8B/snapshots/e9a141a2091ea561b96483212645a2a05e6f99fc'
TRAIN = WORK/'cache/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train.arrow'
TOKENS = WORK/'cache/tokenized/ffcc87608d60-wikitext2-test-full-l2048.pt'
MODELS = {
    'llama31_8b': dict(name='Meta-Llama-3.1-8B', official_name='Llama-3.1-8B',
                      base=BASE, tokens=TOKENS),
    'llama32_3b': dict(name='Llama-3.2-3B', official_name='Llama-3.2-3B',
        base=WORK/'cache/huggingface/models--unsloth--Llama-3.2-3B/snapshots/d4446454d87d51aa42e1fb174f25acc5f8762331',
        tokens=WORK/'cache/tokenized/llama32_3b-wikitext2-test-full-l2048.pt'),
}
TIMING_HELPER = Path('/home/yanlongc/prismquant-rebuttal-e38/scripts/baseline_timing.py')
OFFICIAL = Path('/home/yanlongc/FlatQuant-upstream')
TASKS = ('arc_easy','arc_challenge','boolq','hellaswag','openbookqa','piqa','social_iqa','winogrande')


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str)+'\n'); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8<<20), b''): h.update(block)
    return h.hexdigest()


def official_arguments(checkpoint, model_key='llama31_8b'):
    from flatquant import args_utils
    selected = MODELS[model_key]
    alias = './modelzoo/meta-llama/'+selected['official_name']
    original = sys.argv
    original_models = args_utils.supported_models
    # The official CLI omits 3.2 although its modern Llama implementation can
    # consume that architecture/config. Extend only the local parser choice.
    if alias not in original_models:
        args_utils.supported_models = [*original_models, alias]
    sys.argv = ['official-flatquant','--model',alias,
        '--seed','0','--w_bits','4','--a_bits','4','--k_bits','4','--k_asym',
        '--k_groupsize','128','--v_bits','4','--v_asym','--v_groupsize','128',
        '--cali_bsz','4','--epochs','15','--flat_lr','5e-3','--lwc','--lac',
        '--cali_trans','--add_diag','--save_matrix','--output_dir',str(checkpoint),
        '--exp_name','official_native_seed0']
    try: args, logger = args_utils.parser_gen()
    finally:
        sys.argv = original
        args_utils.supported_models = original_models
    args.model = str(selected['base'])
    return args, logger


def load_official_model(model_key):
    from flatquant import model_utils
    path = str(MODELS[model_key]['base'])
    if model_key == 'llama32_3b':
        # Preserve the official modern attention implementation and the 3.2
        # configuration/RoPE; its string dispatcher otherwise chooses legacy.
        return model_utils.get_llama_31(path, None)
    return model_utils.get_model(path, None)


def calibration(args, tokenizer, checkpoint):
    import torch
    from flatquant import data_utils
    cached = checkpoint/'calibration_tokens.pt'
    if cached.exists():
        saved = torch.load(cached, map_location='cpu', weights_only=True)
        if saved.shape != (128,2048): raise ValueError('calibration cache shape')
        loader = []
        for row in saved:
            inp = row[None]; target = inp.clone(); target[:,:-1] = -100
            loader.append((inp,target))
        return loader
    original = data_utils.datasets.load_dataset
    def local_dataset(path, name=None, **kwargs):
        if path == './datasets/wikitext' and name == 'wikitext-2-raw-v1' and kwargs.get('split') == 'train':
            return data_utils.datasets.Dataset.from_file(str(TRAIN))
        return original(path,name,**kwargs)
    data_utils.datasets.load_dataset = local_dataset
    try:
        loader = data_utils.get_loaders(args,'wikitext2',tokenizer,nsamples=128,seqlen=2048,eval_mode=False)
    finally: data_utils.datasets.load_dataset = original
    torch.save(torch.cat([x[0] for x in loader]), cached)
    return loader


def ppl(model, output, record, tokens_path=TOKENS):
    import torch
    tokens = torch.load(tokens_path, map_location='cpu', weights_only=True)
    if tuple(tokens.shape) != (141,2048): raise ValueError('Expected frozen 141x2048 tokens')
    record['evaluation'] = dict(tokens=str(tokens_path),sha256=sha(tokens_path),batch_size=1,
        sequence_length=2048,added_bos=False,use_cache=False,loss='FP32 shifted CE; token weighted')
    record['chunks'] = []
    with torch.inference_mode():
        for i, token_row in enumerate(tokens):
            batch = token_row[None].cuda()
            logits = model(input_ids=batch,use_cache=False).logits
            nll = float(torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),batch[:,1:].reshape(-1)))
            record['chunks'].append(dict(chunk=i,nll=nll if math.isfinite(nll) else None,tokens_scored=2047))
            record['completed_sequences'] = i+1
            write(output,record)
            if i%16 == 0: print(f'FlatQuant PPL {i+1}/141 NLL={nll}',flush=True)
            del logits
    nlls = [x['nll'] for x in record['chunks']]
    try:
        record['ppl'] = math.exp(sum(nlls)/len(nlls)) if all(x is not None for x in nlls) else None
    except OverflowError:
        record['ppl'] = None
    record['ppl_status'] = 'COMPLETE' if record['ppl'] is not None else 'NONFINITE'
    return tokens


def timing(model, windows, record, output):
    spec = importlib.util.spec_from_file_location('rebuttal_baseline_timing', TIMING_HELPER)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load shared timing helper: {TIMING_HELPER}')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    record['timing_helper'] = dict(path=str(TIMING_HELPER), sha256=sha(TIMING_HELPER))
    helper.measure_timing(model, windows, record, output, write,
        repetitions=20, warmup=10, prefix=2048, decode_steps=128, discard=8)


def tasks(model, tokenizer, output, record):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    tokenizer.pad_token_id = 128004; tokenizer.padding_side = 'left'
    hflm = HFLM(pretrained=model,tokenizer=tokenizer,batch_size=1,max_length=2048,device='cuda')
    record['eight_tasks'] = {}
    for task in TASKS:
        result = lm_eval.simple_evaluate(model=hflm,tasks=[task],num_fewshot=0,batch_size=1,
            log_samples=True,random_seed=0,numpy_random_seed=0,torch_random_seed=0,fewshot_random_seed=0)
        destination = output.with_name(output.stem+'_'+task+'.json')
        write(destination,result)
        record['eight_tasks'][task] = dict(metrics=result['results'][task],samples=str(destination))
        write(output,record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--model',choices=tuple(MODELS),default='llama31_8b')
    parser.add_argument('--phase',choices=('train-ppl','tasks','ppl-timing'),default='train-ppl')
    cli = parser.parse_args()
    selected = MODELS[cli.model]
    started = time.monotonic()
    record = dict(model=selected['name'],model_key=cli.model,method='FlatQuant official native',seed=0,
        status='RUNNING',phase=cli.phase,source_commit=cli.source_commit,
        official_commit=subprocess.check_output(['git','-C',str(OFFICIAL),'rev-parse','HEAD'],text=True).strip(),
        base_checkpoint=str(selected['base']),checkpoint=str(cli.checkpoint),job_id=os.getenv('SLURM_JOB_ID'),
        config='W4 RTN per-channel symmetric; A4 per-token symmetric; native K4/V4 asymmetric per head (128); BF16 residual',
        source_sha256=sha(__file__),command=sys.orig_argv,numerical_execution_gates='none',
        execution_backend='official floating fake-quant PyTorch implementation; not packed INT4 deployment',
        native_int4_kernel=False,
        model_dispatch_adapter=('official get_llama_31 for Llama3.2; retain its original config and scaled RoPE'
            if cli.model=='llama32_3b' else 'unchanged official model dispatcher'))
    write(cli.output,record)
    try:
        import torch
        import transformers
        from flatquant import model_utils,train_utils,flat_utils,utils
        import gptq_utils
        if not torch.cuda.is_available(): raise RuntimeError('Requires allocated GPU')
        torch.set_num_threads(4)
        cli.checkpoint.mkdir(parents=True,exist_ok=True)
        args, logger = official_arguments(cli.checkpoint, cli.model)
        record['official_args'] = vars(args)
        record['libraries'] = {x:importlib.metadata.version(x) for x in ('torch','transformers','datasets','numpy')}
        record['gpu'] = torch.cuda.get_device_name()
        utils.seed_everything(0)
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(selected['base']),use_fast=True,local_files_only=True)
        record['tokenizer_adapter'] = 'Fast tokenizer from pinned tokenizer.json; no token or BOS setting changed'
        done = cli.checkpoint/'DONE.json'; saved = cli.checkpoint/'quantized_model.pt'
        if done.exists():
            manifest = json.loads(done.read_text())
            if manifest['base_checkpoint'] != str(selected['base']) or manifest['official_commit'] != record['official_commit'] or manifest['official_args'] != vars(args):
                raise ValueError('Checkpoint identity does not match; do not recalibrate or silently reuse another configuration')
            model = torch.load(saved,map_location='cpu',weights_only=False)
            record['checkpoint_reused'] = True
            record['original_calibration_cost'] = manifest['calibration_cost']
        else:
            if cli.phase != 'train-ppl': raise FileNotFoundError(f'{cli.phase} requires completed cached checkpoint; no recalibration')
            model, apply = load_official_model(cli.model)
            model.eval(); model.config.use_cache=False
            loader = calibration(args,tokenizer,cli.checkpoint)
            record['calibration_tokens_sha256'] = sha(cli.checkpoint/'calibration_tokens.pt')
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t0 = time.monotonic()
            model = apply(args,model)
            train_utils.cali_flat_quant(args,model,loader,utils.DEV,logger=logger)
            torch.cuda.synchronize()
            record['calibration_cost'] = dict(wall_seconds=time.monotonic()-t0,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                physical_gpu_count=1)
            record['calibration_cost']['gpu_hours'] = record['calibration_cost']['wall_seconds']/3600
            write(cli.output,record)
            flat_utils.save_flat_matrices(args,model)
            flat_utils.reparameterize_model(model)
            t0=time.monotonic()
            quantizers=gptq_utils.rtn_fwrd(model,utils.DEV,args)
            torch.cuda.synchronize(); record['rtn_seconds']=time.monotonic()-t0
            torch.save(quantizers,cli.checkpoint/'weight_quantizers.pt')
            model.cpu().eval(); model.config.use_cache=False
            t0=time.monotonic(); torch.save(model,saved.with_suffix('.tmp')); saved.with_suffix('.tmp').replace(saved)
            record['checkpoint_write_seconds']=time.monotonic()-t0
            write(done,dict(status='COMPLETE',base_checkpoint=str(selected['base']),model_key=cli.model,
                model=selected['name'],official_commit=record['official_commit'],
                official_args=vars(args),config=record['config'],calibration_cost=record['calibration_cost'],
                quantized_model=str(saved),checkpoint_bytes=saved.stat().st_size,
                calibration_tokens_sha256=record['calibration_tokens_sha256'],native_quantization='RTN, not GPTQ'))
            del loader, quantizers
        model.cuda().eval()
        record['model_object_id'] = id(model)
        if cli.phase in ('train-ppl','ppl-timing'):
            windows = ppl(model,cli.output,record,selected['tokens'])
            if cli.phase == 'ppl-timing':
                timing(model,windows,record,cli.output)
        else:
            tasks(model,tokenizer,cli.output,record)
        record['same_model_object_id_after_measurements'] = id(model)
        record['status']='COMPLETE'
    except Exception as exc:
        record.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        record['total_wall_seconds']=time.monotonic()-started
        record['allocated_gpu_hours']=record['total_wall_seconds']/3600
        write(cli.output,record)
    return 0 if record['status']=='COMPLETE' else 1


if __name__=='__main__': raise SystemExit(main())
