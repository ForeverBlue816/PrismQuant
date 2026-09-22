"""Exact E36 group quantiles and pre-registered precision comparisons."""
from __future__ import annotations
import argparse,gc,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a
from nar import e36_offset_precision as e
from nar.run_reviewer_experiments import stats
from nar.e34_report import paired_ppl_combo

METRICS=('abs_offset_over_range','offset_error_steps','unaligned_step_steps','error_to_unaligned')
T=a.LENGTH*64

def divide(num,den):
    with np.errstate(divide='ignore',invalid='ignore'):return np.asarray(num,dtype=np.float64)/np.asarray(den,dtype=np.float64)

def exact_quantiles(values,probabilities=(.5,.99),inplace=False):
    x=np.asarray(values).reshape(-1)
    undefined=int(np.isnan(x).sum())
    if undefined:x=x[~np.isnan(x)];inplace=True
    if not len(x):return [None for _ in probabilities],dict(defined_count=0,undefined_count=undefined,infinite_count=0,max=None)
    infinite=int(np.isinf(x).sum());maximum=float(x.max());positions=[p*(len(x)-1) for p in probabilities]
    indices=sorted({int(math.floor(q)) for q in positions}|{int(math.ceil(q)) for q in positions})
    if inplace:x.partition(indices)
    else:x=np.partition(x,indices)
    out=[]
    for q in positions:
        lo,hi=int(math.floor(q)),int(math.ceil(q));weight=q-lo;left,right=float(x[lo]),float(x[hi])
        value=left if lo==hi or left==right or weight==0 else (float('inf') if np.isposinf(right) else (1-weight)*left+weight*right)
        out.append(value)
    return out,dict(defined_count=len(x),undefined_count=undefined,infinite_count=infinite,max=maximum)

def exact_stream_median(values,block_size=2**20):
    """Exact float64 order selection in four radix passes; bounded RAM.

    Nonnegative IEEE-754 bit patterns preserve numeric order, including +inf.
    NaNs have already been removed with explicit counts by the caller.
    """
    n=len(values)
    if not n:return None
    ranks=[(n-1)//2,n//2]
    active={0:[(label,rank) for label,rank in enumerate(ranks)]}
    for shift in [48,32,16,0]:
        hist={prefix:np.zeros(65536,dtype=np.int64) for prefix in active}
        for start in range(0,n,block_size):
            chunk=np.asarray(values[start:start+block_size]);assert chunk.dtype==np.float64
            assert np.all(chunk>=0) and not np.isnan(chunk).any()
            bits=chunk.view(np.uint64)
            for prefix in active:
                selected=bits if shift==48 else bits[(bits>>(shift+16))==prefix]
                digits=((selected>>shift)&65535).astype(np.int64)
                hist[prefix]+=np.bincount(digits,minlength=65536)
        following={}
        for prefix,targets in active.items():
            cumulative=np.cumsum(hist[prefix])
            for label,rank in targets:
                digit=int(np.searchsorted(cumulative,rank,side='right'))
                below=int(cumulative[digit-1]) if digit else 0
                following.setdefault((prefix<<16)|digit,[]).append((label,rank-below))
        active=following
    result={label:np.array([prefix],dtype=np.uint64).view(np.float64)[0] for prefix,targets in active.items() for label,rank in targets}
    left,right=float(result[0]),float(result[1])
    return left if left==right else (float('inf') if np.isposinf(right) else .5*left+.5*right)

def measure(raw):
    z=raw[...,0].astype(np.float64);ran=raw[...,1].astype(np.float64);step=raw[...,2].astype(np.float64);c=raw[...,3].astype(np.float64)
    assert np.all(step>0) and np.isfinite(raw).all() and np.all(ran>=0)
    rounded=raw[...,0].astype(np.float16).astype(np.float64)
    err=np.abs(z-rounded);level=2*np.abs(c)/15
    return dict(zip(METRICS,[divide(np.abs(z),ran),divide(err,step),divide(level,step),divide(err,level)]))

def summarize_values(metrics):
    size=next(iter(metrics.values())).size
    result=dict(group_token_observations=size,over_half_step_count=int((metrics['offset_error_steps']>.5).sum()),fraction_over_half_step=float((metrics['offset_error_steps']>.5).mean()) if size else None)
    for name,value in metrics.items():
        (median,p99),info=exact_quantiles(value)
        result.update({name+'_median':median,name+'_p99':p99,**{name+'_'+k:v for k,v in info.items()}})
    return result

def assemble(m,method):
    folder=e.assets(m)/'analysis'/method;folder.mkdir(parents=True,exist_ok=True)
    cfg,ds=a.config(m);shape={s:(3,cfg.num_hidden_layers,T,ds[s]//128,4) for s in ['qkv','down']}
    done=folder/'assembled.json'
    if done.exists():
        for s in shape:assert tuple(a.js(done)['shapes'][s])==shape[s]
        return {s:np.load(folder/(s+'.npy'),mmap_mode='r') for s in shape}
    maps={s:np.lib.format.open_memmap(folder/(s+'.npy'),mode='w+',dtype=np.float32,shape=shape[s]) for s in shape}
    for seed in a.SEEDS:
        for chunk in range(64):
            cp=e.checkpoint(m,method,seed,chunk);meta=a.js(cp.with_suffix('.json'));assert a.sha(cp)==meta['raw_sha256']
            raw=a.loadt(cp)
            for site in shape:
                x=raw[site].numpy();assert x.shape==(cfg.num_hidden_layers,2048,ds[site]//128,4)
                references={(r['precision'],r['site'],r['layer']):r for r in meta['payload_hashes']}
                for layer in range(cfg.num_hidden_layers):
                    reference=references['Z16',site,layer]
                    assert e.sha_tensor(raw[site][layer,:,:,0].half())==reference['offset_sha256'], 'CPU/GPU fp16 offset rounding mismatch'
                    assert e.sha_tensor(raw[site][layer,:,:,2].half())==reference['scale_sha256'], 'Stored-step reconstruction mismatch'
                maps[site][seed,:,chunk*2048:(chunk+1)*2048]=x
            del raw
        for x in maps.values():x.flush()
        print(f'E36 assemble {m} {method} seed{seed} complete',flush=True)
    a.savej(done,dict(status='COMPLETE',shapes=shape,raw_validation='each source checkpoint SHA256 matched before copying',utc=a.utc()))
    del maps;gc.collect()
    return {s:np.load(folder/(s+'.npy'),mmap_mode='r') for s in shape}

def group_statistics(m):
    flags=a.loadt(a.ASSETS/m/'e34/flags.pt');bos=flags['bos'].numpy().reshape(-1);massive=flags['massive'].numpy().reshape(-1)
    masks={'all':np.ones(T,dtype=bool),'BOS':bos,'massive':massive,'BOS+massive':bos|massive,'other':~(bos|massive)}
    cfg,ds=a.config(m);output=[];overall=[]
    for method in e.METHODS:
        maps=assemble(m,method);nmax=sum(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3] for x in maps.values())
        ratio_path=e.assets(m)/'analysis'/method/'global_ratio.npy'
        ratios=np.lib.format.open_memmap(ratio_path,mode='w+',dtype=np.float64,shape=(nmax,));used=0;undefined_total=0;total=0;over=0
        for site in ['qkv','down']:
            for layer in range(cfg.num_hidden_layers):
                raw=np.array(maps[site][:,layer]);metrics=measure(raw)
                for seed in [0,1,2,'pooled']:
                    chosen=metrics if seed=='pooled' else {k:v[seed:seed+1] for k,v in metrics.items()}
                    for token_class,mask in masks.items():
                        selected={k:v[:,mask,:] for k,v in chosen.items()};summary=summarize_values(selected)
                        nonempty=['all','anchor'] if method=='pq' else ['all','other']
                        for group_class in ['all','anchor','other']:
                            values=summary if group_class in nonempty else summarize_values({k:np.empty(0) for k in METRICS})
                            output.append(dict(model=m,method=method,seed=seed,site=site,layer=layer,group_class=group_class,token_class=token_class,aligned_level_interpretation='aligned constant Walsh level' if method=='pq' else 'incidental group DC only',status='MEASURED' if group_class in nonempty else 'N/A: empty group stratum',**values))
                ratio=metrics['error_to_unaligned'].reshape(-1);valid=~np.isnan(ratio);n=int(valid.sum());ratios[used:used+n]=ratio[valid];used+=n;undefined_total+=len(ratio)-n
                total+=metrics['offset_error_steps'].size;over+=int((metrics['offset_error_steps']>.5).sum())
                ratios.flush()
                del raw,metrics,chosen,selected,ratio,valid;gc.collect()
                if layer%6==0:print(f'E36 exact quantiles {m} {method} {site} layer{layer}',flush=True)
        ratios.flush();del maps;gc.collect()
        median=exact_stream_median(ratios[:used])
        infinite_count=sum(int(np.isinf(ratios[start:min(start+2**20,used)]).sum()) for start in range(0,used,2**20))
        rr=[r for r in output if r['method']==method and r['seed']=='pooled' and r['group_class']=='all' and r['token_class']=='all']
        worst=max(rr,key=lambda r:(r['fraction_over_half_step'],-r['layer'],r['site']=='qkv'))
        peak=max(rr,key=lambda r:(r['offset_error_steps_p99'],-r['layer'],r['site']=='qkv'))
        overall.append(dict(model=m,method=method,group_token_observations=total,over_half_step_count=over,fraction_over_half_step=over/total,error_to_unaligned_median=median,error_to_unaligned_undefined_count=undefined_total,error_to_unaligned_infinite_count=infinite_count,worst_fraction_site=worst['site'],worst_fraction_layer=worst['layer'],worst_layer_fraction=worst['fraction_over_half_step'],largest_p99_site=peak['site'],largest_p99_layer=peak['layer'],largest_p99=peak['offset_error_steps_p99'],aligned_level_interpretation='aligned constant Walsh level' if method=='pq' else 'incidental group DC only'))
        del ratios;gc.collect()
    a.write(e.root(m)/'e36_group_statistics.csv',output);a.write(e.root(m)/'e36_overall.csv',overall)
    peaks=[]
    for r in overall:
        peaks += [row for row in output if row['method']==r['method'] and row['seed']=='pooled' and row['site']==r['largest_p99_site'] and row['layer']==r['largest_p99_layer'] and row['group_class']=='all']
    a.write(e.root(m)/'e36_worst_layer.csv',peaks)
    return output,overall

def matrices(m):
    rows=a.rows(e.root(m)/'e36_per_sequence.csv');result={}
    for spec in e.plan():
        chosen=[r for r in rows if r['row']==spec['row']];assert len(chosen)==192
        x=np.empty((3,64));seen=set()
        for r in chosen:
            key=int(r['seed']),int(r['chunk']);assert key not in seen;seen.add(key);x[key]=float(r['nll'])
        result[spec['row']]=x
    return result

def precision_statistics(m,overall):
    data=matrices(m);summary=[]
    for spec in e.plan():
        reference=spec['method']+'_Z16'
        summary.append(dict(model=m,reference=reference,**spec,**stats(data[spec['row']],data[reference])))
    a.write(e.root(m)/'e36_summary.csv',summary);lookup={r['row']:r for r in summary}
    h_a=dict(delta=lookup['pq_Z32']['delta'],Z16_seed_sd=lookup['pq_Z16']['seed_std'])
    h_a['supported']=abs(h_a['delta'])<=h_a['Z16_seed_sd']
    h_b=[dict(method=method,delta=lookup[method+'_ZB16']['delta'],ci90_low=lookup[method+'_ZB16']['ci90_low'],ci90_high=lookup[method+'_ZB16']['ci90_high'],supported=lookup[method+'_ZB16']['delta']>0 and lookup[method+'_ZB16']['ci90_low']>0) for method in e.METHODS]
    h_c=[]
    for r in overall:
        checks=dict(overall_below_one_percent=r['fraction_over_half_step']<.01,worst_layer_below_five_percent=r['worst_layer_fraction']<.05,median_ratio_below_one_percent=r['error_to_unaligned_median'] is not None and r['error_to_unaligned_median']<.01)
        h_c.append(dict(**r,**checks,supported=all(checks.values())))
    dd=paired_ppl_combo([(1,data['hadamard_Z32'],data['hadamard_Z16']),(-1,data['pq_Z32'],data['pq_Z16'])])
    baseline_seed=np.exp(data['hadamard_Z16'].mean(1))-np.exp(data['pq_Z16'].mean(1));noise=float(baseline_seed.std(ddof=1))
    h_d=dict(**dd,baseline_paired_seed_sd=noise,supported=abs(dd['delta'])<=noise)
    hypotheses=dict(model=m,H36a=h_a,H36b=h_b,H36c=h_c,H36d=h_d,primary_scope='PQ on both models; method-specific Hadamard stress and DC-comparison controls also reported')
    a.savej(e.root(m)/'e36_hypotheses.json',hypotheses);a.write(e.root(m)/'e36_method_precision_contrast.csv',[dict(model=m,contrast='Had(Z32-Z16) minus PQ(Z32-Z16)',**h_d)])
    return summary,hypotheses

def analyze(m):
    assert a.js(e.root(m)/'e36_DONE.json')['status']=='COMPLETE';e.verify_frozen(m)
    rows,overall=group_statistics(m);summary,hyp=precision_statistics(m,overall)
    a.savej(e.root(m)/'e36_ANALYSIS_DONE.json',dict(status='COMPLETE',utc=a.utc(),statistics_rows=len(rows),summary_rows=len(summary),source_sha256=a.sha(Path(__file__))))
    print(json.dumps(hyp,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=e.MODELS,required=True);arg=p.parse_args();analyze(arg.model)
if __name__=='__main__':main()
