"""E36: fixed-code metadata precision, with exact E29 baseline replays."""
from __future__ import annotations
import argparse, gc, hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a

MODELS=('qwen3_4b_base','llama32_3b')
METHODS=('pq','hadamard')
ROWS=('Z16','Z32','ZB16','S32')
PARENT='025c063cbfab9ed317c52173d8eceb10f403a9a3'
PREREG=a.REPO/'experiments/e36_preregistration.md'

def root(m): return a.REPO/'results'/m
def assets(m): return a.ASSETS/m/'e36'
def sha_tensor(t):
    t=t.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
def plan():return [dict(method=method,precision=row,row=method+'_'+row) for method in METHODS for row in ROWS]
def checkpoint(m,method,seed,chunk):return assets(m)/method/f'seed{seed}'/f'chunk{chunk:02d}.pt'

def freeze():
    path=a.REPO/'experiments/e36_execution_manifest.json'
    if path.exists():return a.js(path)
    source=[PREREG,Path(__file__),a.REPO/'nar/reviewer_ablations.py',a.REPO/'nar/experiment.py',a.REPO/'nar/activation_experiments.py',a.REPO/'nar/run_reviewer_experiments.py',a.REPO/'slurm_e36.sh']
    inputs={};factors=[]
    for m in MODELS:
        for p in [a.tokenpath(m,'wt2_eval'),a.ASSETS/m/'e34/flags.pt',root(m)/'e34_flags.json',root(m)/'e29_per_sequence.csv']:
            inputs[str(p)]=a.sha(p)
        for site,layer,n in a.keys(m):
            p=a.factor_path(m,'wt2_n128_s0','S3','max','P1',0,site,layer)
            factors.append(dict(model=m,site=site,layer=layer,path=str(p),sha256=a.sha(p)))
    old=subprocess.check_output(['git','ls-tree','-r','--name-only',PARENT,'results/'],cwd=a.REPO,text=True).splitlines()
    manifest=dict(utc=a.utc(),status='FROZEN_BEFORE_FIRST_FORWARD',parent=PARENT,source_sha256={str(p.relative_to(a.REPO)):a.sha(p) for p in source},input_sha256=inputs,factors=factors,rows=plan(),old_results_sha256={p:a.sha(a.REPO/p) for p in old},old_report_sha256=hashlib.sha256(subprocess.check_output(['git','show',PARENT+':report.md'],cwd=a.REPO)).hexdigest())
    a.savej(path,manifest);return manifest

def verify_frozen(m):
    f=a.js(a.REPO/'experiments/e36_execution_manifest.json')
    for p,sha in f['source_sha256'].items():assert a.sha(a.REPO/p)==sha,('changed source',p)
    for p,sha in f['input_sha256'].items():
        if m in p:assert a.sha(p)==sha,('changed input',p)
    for row in f['factors']:
        if row['model']==m:assert a.sha(row['path'])==row['sha256']
    return f

def payload(x):
    deq,s,z,q=a.base.dynamic_asym_int4(x,128)
    g=a.base.group_view(x.float(),128);lo=g.amin(-1);ran=g.amax(-1)-lo;raw_scale=ran/15
    p=dict(codes=q,scale16=s,zero16=z,lo=lo,raw_scale=raw_scale)
    metrics=torch.stack((lo,ran,s.float(),g.mean(-1)),dim=-1).squeeze(0)
    return deq,p,metrics

def metadata(p,row):
    z=p['lo'] if row=='Z32' else (p['lo'].bfloat16() if row=='ZB16' else p['zero16'])
    s=torch.where(p['raw_scale']>0,p['raw_scale'],torch.ones_like(p['raw_scale'])) if row=='S32' else p['scale16']
    return s,z

def decode(p,row):
    s,z=metadata(p,row)
    return (p['codes'].float()*s.float().unsqueeze(-1)+z.float().unsqueeze(-1)).flatten(-2)

def hashes(p,row):
    s,z=metadata(p,row)
    return dict(code_sha256=sha_tensor(p['codes']),scale_sha256=sha_tensor(s),offset_sha256=sha_tensor(z),code_values=p['codes'].numel(),scale_dtype=str(s.dtype),offset_dtype=str(z.dtype))

def fixed_tensor(device='cpu'):
    u=torch.arange(128,dtype=torch.float32,device=device)
    x=torch.stack((123.4567+u*.00137,-2.013+u*.017,torch.full_like(u,.5)+u*2e-9,torch.zeros_like(u),torch.full_like(u,17.1234))).unsqueeze(0)
    native,s,z,q=a.base.dynamic_asym_int4(x,128);got,p,_=payload(x)
    assert torch.equal(native,got) and torch.equal(native,decode(p,'Z16'))
    assert torch.equal(s,p['scale16']) and torch.equal(z,p['zero16']) and torch.equal(q,p['codes'])
    assert s.dtype==z.dtype==torch.float16 and torch.isfinite(native).all()
    assert torch.equal(z,p['lo'].half())
    checks={row:hashes(p,row) for row in ROWS}
    assert len({v['code_sha256'] for v in checks.values()})==1
    assert len({checks[row]['scale_sha256'] for row in ('Z16','Z32','ZB16')})==1
    assert checks['S32']['offset_sha256']==checks['Z16']['offset_sha256']
    assert any(checks[row]['offset_sha256']!=checks['Z16']['offset_sha256'] for row in ('Z32','ZB16'))
    return dict(status='PASS',device=device,main_routine='nar.experiment.dynamic_asym_int4',input_sha256=sha_tensor(x),rounding_matches_main=True,hashes=checks)

class Capture(a.Hooks):
    def __init__(self,model,rot):
        super().__init__(model,rot,128,False);self.payloads={};self.metrics={};self.audit=[]
    def transform(self,site,layer,v):
        x=self.rot.apply(site,layer,v);deq,p,metric=payload(x)
        assert torch.isfinite(p['zero16']).all() and torch.isfinite(p['scale16']).all()
        self.payloads[site,layer]={k:t.detach().cpu() for k,t in p.items()}
        self.metrics[site,layer]=metric.cpu()
        self.audit.append(dict(precision='Z16',site=site,layer=layer,**hashes(p,'Z16')))
        return self.rot.apply(site,layer,deq,True).to(v.dtype)

class Replay(a.Hooks):
    def __init__(self,model,rot,stored,row):
        super().__init__(model,rot,128,False);self.stored=stored;self.row=row;self.audit=[]
    def transform(self,site,layer,v):
        p={k:t.to(v.device) for k,t in self.stored[site,layer].items()}
        self.audit.append(dict(precision=self.row,site=site,layer=layer,**hashes(p,self.row)))
        deq=decode(p,self.row);assert deq.shape==v.shape and torch.isfinite(deq).all()
        return self.rot.apply(site,layer,deq,True).to(v.dtype)

@torch.inference_mode()
def forward_nll(model,batch,hooks):
    hooks.install()
    try:
        logits=model(input_ids=batch,use_cache=False).logits
        loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),batch[:,1:].reshape(-1))
        assert loss.dtype==torch.float32 and torch.isfinite(loss)
        return float(loss)
    finally:hooks.close()

def verify_payload_hashes(rows):
    by={(r['precision'],r['site'],r['layer']):r for r in rows}
    sites={(r['site'],r['layer']) for r in rows};assert len(rows)==len(sites)*4
    for site,layer in sites:
        rr=[by[row,site,layer] for row in ROWS]
        assert len({r['code_sha256'] for r in rr})==1
        assert len({r['scale_sha256'] for r in rr[:3]})==1
        assert rr[0]['offset_sha256']==rr[3]['offset_sha256']
    return True

@torch.inference_mode()
def run(m):
    verify_frozen(m);a.LOG.info('E36 prereg SHA256 %s',a.sha(PREREG));a.LOG.info('E36 manifest SHA256 %s',a.sha(a.REPO/'experiments/e36_execution_manifest.json'))
    a.savej(root(m)/'e36_fixed_tensor_gate.json',fixed_tensor('cuda'))
    a.savej(root(m)/'e36_metadata.json',dict(model=m,created_utc=a.utc(),rows=plan(),seeds=list(a.SEEDS),sites=['qkv','down'],protocol='fixed Z16 payload; change reconstruction metadata only',hardware=a.base.hardware_info(),preregistration_sha256=a.sha(PREREG),execution_manifest_sha256=a.sha(a.REPO/'experiments/e36_execution_manifest.json'),default_bits=4.25,diagnostic_fp32_metadata_bits=4.375,raw_metric_fields=['minimum_fp32','range_fp32','stored_step_fp16_as_fp32','group_mean_fp32']))
    model=a.base.load_model(a.act.MODEL_IDS[m],a.WORK);tok=a.loadt(a.tokenpath(m,'wt2_eval'))
    old={(r['row'],int(r['seed']),int(r['chunk'])):r['nll'] for r in a.rows(root(m)/'e29_per_sequence.csv') if r['eval_set']=='wt2'}
    token_sha=a.sha(a.tokenpath(m,'wt2_eval'));cfg,_=a.config(m)
    for method in METHODS:
        for seed in a.SEEDS:
            rot=a.Rotation(m,seed,method);gates=rot.gate()
            assert all(r['round_trip']<=1e-6 and r['anchor_residual']<=1e-6 for r in gates)
            a.append(root(m)/'e36_gates.csv',[dict(model=m,method=method,seed=seed,precision=row,**r) for row in ROWS for r in gates],['method','seed','precision','site','layer'])
            for chunk in range(64):
                cp=checkpoint(m,method,seed,chunk);meta_path=cp.with_suffix('.json')
                if meta_path.exists():
                    saved=a.js(meta_path);assert cp.exists() and saved['raw_sha256']==a.sha(cp)
                else:
                    capture=Capture(model,rot);batch=tok[chunk:chunk+1].cuda()
                    losses={'Z16':forward_nll(model,batch,capture)};reference=old['Q1_'+method,seed,chunk]
                    if np.float32(losses['Z16']).tobytes()!=np.float32(float(reference)).tobytes():
                        a.savej(root(m)/'e36_BASELINE_FAILURE.json',dict(method=method,seed=seed,chunk=chunk,reference=reference,observed=losses['Z16']))
                        raise RuntimeError('Z16 did not reproduce E29 bit for bit')
                    audit=capture.audit
                    for row in ROWS[1:]:
                        replay=Replay(model,rot,capture.payloads,row)
                        losses[row]=forward_nll(model,batch,replay);audit+=replay.audit
                    verify_payload_hashes(audit)
                    raw={site:torch.stack([capture.metrics[site,l] for l in range(cfg.num_hidden_layers)]) for site in ['qkv','down']}
                    a.savet(cp,raw)
                    saved=dict(model=m,method=method,seed=seed,chunk=chunk,nll=losses,baseline_source_nll=reference,baseline_bit_identical=True,payload_hashes=audit,raw_sha256=a.sha(cp),raw_bytes=cp.stat().st_size)
                    a.savej(meta_path,saved)
                    del capture,replay,raw,batch;gc.collect()
                a.append(root(m)/'e36_per_sequence.csv',[dict(model=m,method=method,precision=row,row=method+'_'+row,seed=seed,chunk=chunk,nll=saved['baseline_source_nll'] if row=='Z16' else saved['nll'][row],tokens=2047,token_sha256=token_sha) for row in ROWS],['row','seed','chunk'])
                if chunk%8==0:a.LOG.info('E36 %s %s seed%d chunk%d/64 all four precisions complete',m,method,seed,chunk+1)
            del rot;gc.collect();torch.cuda.empty_cache()
    del model;gc.collect();torch.cuda.empty_cache()
    audit=[];replay=[];artifacts=[]
    for method in METHODS:
        for seed in a.SEEDS:
            for chunk in range(64):
                cp=checkpoint(m,method,seed,chunk);r=a.js(cp.with_suffix('.json'))
                audit += [dict(model=m,method=method,seed=seed,chunk=chunk,**v) for v in r['payload_hashes']]
                replay.append(dict(model=m,method=method,seed=seed,chunk=chunk,source_row='Q1_'+method,source_nll=r['baseline_source_nll'],replayed_nll=r['nll']['Z16'],float32_bit_identical=r['baseline_bit_identical']))
                artifacts.append(dict(method=method,seed=seed,chunk=chunk,path=str(cp),sha256=r['raw_sha256'],bytes=r['raw_bytes']))
    a.write(root(m)/'e36_payload_hashes.csv',audit);a.write(root(m)/'e36_baseline_replay.csv',replay);a.write(root(m)/'e36_raw_artifacts.csv',artifacts)
    verify_frozen(m);rr=a.rows(root(m)/'e36_per_sequence.csv');assert len(rr)==1536
    a.savej(root(m)/'e36_DONE.json',dict(model=m,status='COMPLETE',utc=a.utc(),rows=8,chunk_records=len(rr),baseline_chunks=len(replay),payload_hash_records=len(audit),raw_group_summary_bytes=sum(r['bytes'] for r in artifacts),hardware=a.base.hardware_info()))
    a.LOG.info('E36 %s all measurements COMPLETE',m)

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--model',choices=MODELS);p.add_argument('--fixed-tensor',action='store_true');args=p.parse_args()
    if args.freeze:print(json.dumps({'manifest_sha256':a.sha(a.REPO/'experiments/e36_execution_manifest.json')} if (a.REPO/'experiments/e36_execution_manifest.json').exists() else {'frozen_utc':freeze()['utc']}));return
    if args.fixed_tensor:print(json.dumps(fixed_tensor(),indent=2));return
    assert args.model;a.setup();run(args.model)
if __name__=='__main__':main()
