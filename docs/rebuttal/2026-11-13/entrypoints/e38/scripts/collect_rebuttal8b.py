"""CPU-only, evidence-preserving collection of the active 8B rebuttal runs.

No runner, raw measurement, or original E41 six-arm table is modified. A caller
may serialize concurrent invocations with flock. Missing metrics remain pending.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

HOME=Path('/home/yanlongc')
TASKS=('arc_easy','arc_challenge','boolq','hellaswag','openbookqa','piqa','social_iqa','winogrande')
METRICS={name:'acc_norm,none' for name in TASKS}
METRICS.update(boolq='acc,none',social_iqa='acc,none',winogrande='acc,none')
ARMS=('hadamard_asym','pq_constant_asym','pq_signed_asym','hadamard_sym','pq_constant_sym','pq_signed_sym')
OK=('COMPLETE','PASS')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def finite(value):return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)
def fmt(value,digits=5):return f'{value:.{digits}f}' if finite(value) else 'pending'
def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(text);temp.replace(path)
def dump(path,data):write(path,json.dumps(data,indent=2,allow_nan=False)+'\n')
def link(path,label=None):return f'[{label or Path(path).name}]({Path(path).resolve()})'
def escape(value):return str(value).replace('|','/').replace('\n',' ')
def source(run):return link(run['_path']) if run else 'pending'
def state(run):return run.get('status','UNKNOWN') if run else 'pending'
def run_state(run):
    if not run:return 'pending'
    parts=[state(run)]
    for name in ('ppl','equivalence','timing','eight_tasks'):
        value=run.get(name)
        status=value.get('status') if isinstance(value,dict) else run.get('ppl_status') if name=='ppl' else None
        if status in ('FAILED','NONFINITE','PARTIAL','BLOCKED','CONTENDED'):parts.append(name+' '+status)
    return '; '.join(parts)
def phase(run):
    if run.get('phase'):return run['phase']
    args=run.get('command_line',run.get('command',[]))
    if isinstance(args,list) and '--phase' in args:
        index=args.index('--phase')
        if len(args)>index+1:return args[index+1]
    return 'ppl-timing'
def has_task_attempt(run):
    tasks=run.get('eight_tasks',{})
    return phase(run) in ('tasks','eight-tasks','all') or (isinstance(tasks,dict) and
        (tasks.get('status') not in (None,'not run') or any(name in tasks for name in TASKS)))

def discover(root,recursive=False):
    records=[];errors=[]
    paths=root.rglob('*.json') if recursive else root.glob('*.json')
    for path in paths:
        if any(path.stem.endswith('_'+name) for name in (*TASKS,*ARMS)):continue
        if 'statistics' in path.parts or path.name.startswith(('collection_','bootstrap_')):continue
        try:
            record=read(path)
            if isinstance(record,dict):
                record['_path']=str(path.resolve());record['_sha256']=sha(path)
                record['_mtime_ns']=path.stat().st_mtime_ns
                records.append(record)
        except Exception as exc:errors.append(dict(path=str(path),error=repr(exc)))
    return records,errors

def latest(rows):
    def order(row):
        match=re.search(r'job(\d+)',Path(row['_path']).name)
        job=int(match.group(1)) if match else int((row.get('slurm') or {}).get('SLURM_JOB_ID') or row.get('job_id') or 0)
        return job,row['_mtime_ns'],row['_path']
    return max(rows,key=order) if rows else None

def identity(row):
    if not row:return None
    packed=row.get('packing',{}).get('manifest',{})
    if packed.get('sha256'):return ('packed_manifest',packed['sha256'])
    checkpoint=row.get('checkpoint')
    if isinstance(checkpoint,str):return ('checkpoint',checkpoint,row.get('base_checkpoint'),row.get('official_commit'))
    if isinstance(checkpoint,dict):
        return ('checkpoint',checkpoint.get('sha256') or checkpoint.get('directory') or checkpoint.get('snapshot'),
                row.get('base_revision') or row.get('base_model_revision'))
    return ('stage_a',row.get('row'),row.get('seed'),row.get('base_revision'),row.get('quant_snapshot'),
            row.get('source_sha256',{}).get('quarot-llama3/e38/stage_a_8b_model.py'))

def task_result(row):
    if not row:return None,0
    result=row.get('eight_tasks',{})
    if not isinstance(result,dict):return None,0
    values={}
    if isinstance(result.get('tasks'),list):
        for task in result['tasks']:
            if task.get('status') in OK and finite(task.get('accuracy')):
                values[task['task']]=task['accuracy']
    else:
        for name in TASKS:
            value=result.get(name,{}).get('metrics',{}).get(METRICS[name])
            if finite(value):values[name]=value
    return (sum(values[name] for name in TASKS)/8 if set(TASKS)<=values.keys() else None),len(values)

def chunks(row,candidate=False):
    if not row:return None
    if candidate:
        items=row.get('equivalence',{}).get('chunks',[])
        key='candidate_nll'
    else:
        ppl=row.get('ppl',{})
        items=ppl.get('chunks',[]) if isinstance(ppl,dict) else row.get('chunks',[])
        if not items:items=row.get('chunks',[])
        key='nll'
    if len(items)!=141 or [x.get('chunk') for x in items]!=list(range(141)):return None
    if not all(finite(x.get(key)) and x.get('tokens_scored',0)>0 for x in items):return None
    return [x[key] for x in items],[x['tokens_scored'] for x in items]

def ppl_value(row,candidate=False):
    if not row:return None
    if candidate:return row.get('equivalence',{}).get('candidate_ppl') if chunks(row,True) else None
    value=row.get('ppl')
    return (value.get('value') if isinstance(value,dict) else value) if chunks(row) else None

def token_hash(row):
    if not row:return None
    for key in ('evaluation_tokens_sha256',):
        if row.get(key):return row[key]
    for entry in (row.get('evaluation_tokens'),row.get('evaluation'),
                  row.get('ppl',{}).get('tokens') if isinstance(row.get('ppl'),dict) else None,
                  row.get('ppl',{}).get('token_file') if isinstance(row.get('ppl'),dict) else None):
        if isinstance(entry,dict) and entry.get('sha256'):return entry['sha256']
    return None

def paired_bootstrap(losses,weights,replicates=20000):
    import numpy as np
    losses=np.asarray(losses,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    generator=np.random.Generator(np.random.PCG64(20261111))
    indices=generator.integers(0,len(weights),size=(replicates,len(weights)),dtype=np.uint16)
    draws=np.empty((replicates,len(losses)),dtype=np.float64)
    for start in range(0,replicates,512):
        take=indices[start:start+512];w=weights[take]
        draws[start:start+512]=np.exp((losses[:,take]*w[None]).sum(-1)/w.sum(-1)[None]).T
    point=np.exp((losses*weights).sum(-1)/weights.sum())
    return point,draws,indices

def interval(values):
    import numpy as np
    return np.quantile(values,[.025,.975],method='linear').tolist() if np.isfinite(values).all() else None

def pair_ci(candidate,reference,candidate_integer=False):
    a=chunks(candidate,candidate_integer);b=chunks(reference)
    h=token_hash(candidate)
    if not a or not b or not h or h!=token_hash(reference) or a[1]!=b[1]:
        return dict(status='unavailable',reason='Need complete141 paired NLL arrays, equal token-file SHA256 and equal sequence token counts')
    point,draws,_=paired_bootstrap([b[0],a[0]],a[1])
    return dict(status='COMPLETE',point=float(point[1]-point[0]),percentile95=interval(draws[:,1]-draws[:,0]),
        replicates=20000,rng_seed=20261111,evaluation_tokens_sha256=h,reference_source=source(reference),
        candidate_source=source(candidate),scope='paired test-sequence PPL difference, candidate minus reference; fixed runs, not seed uncertainty or a causal implementation effect')

def ci_text(result):
    if result.get('status')!='COMPLETE':return 'pending / unpaired'
    ci=result['percentile95']
    return f"{result['point']:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}]" if ci else 'nonfinite interval'

def baseline(e38repo):
    rows,_=discover(e38repo/'results/e38/raw')
    active=[r for r in rows if r.get('kind')=='e38_stage_a_trained8b_native' and r.get('row')=='fp16' and chunks(r)]
    if active:return latest(active)
    path=e38repo/'results/e38/raw/direct_fp16_8b_seed0_job1733513.json'
    if path.exists():
        row=read(path);row.update(_path=str(path),_sha256=sha(path),_mtime_ns=path.stat().st_mtime_ns)
        if chunks(row):return row
    return None

def archive(out):
    folder=out/'history_before_collect8b';folder.mkdir(parents=True,exist_ok=True)
    for name in ('table.md','table.tex','summary.md'):
        if (out/name).exists() and not (folder/name).exists():shutil.copy2(out/name,folder/name)
    return folder

def submitted(experiment,e38repo):
    path=e38repo/'results/e38/raw/parallel_8b_submissions_001.json'
    aliases={'e38':('e38','stagea','stage a','timing'),'e39':('e39','flatquant'),
             'e41':('e41','mechanism','b6'),'e42':('e42','duquant')}[experiment]
    if path.exists():
        data=read(path);selected=[]
        for job in data.get('jobs',[]):
            description=json.dumps({k:job.get(k) for k in ('kind','tasks')}).lower()
            if any(word in description for word in aliases):
                selected.append(f"submitted {job['job_id']} ({job.get('state','state unqueried')} in submission manifest)")
        if selected:return '; '.join(selected),str(path)
    if experiment=='e38':
        # Preserve a real documented submission while the unified manifest is
        # being written; this is not a fresh scheduler query.
        old=e38repo/'results/e38/history_before_collect8b/table.md'
        if old.exists():
            jobs=sorted(set(re.findall(r'Submitted\s*(\d+)',old.read_text())))
            if jobs:return '; '.join('submitted '+job+' (documented; state unqueried)' for job in jobs),str(old)
    return 'pending',None

def render_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |',
                      *['| '+' | '.join(escape(x) for x in row)+' |' for row in rows]])

def tex_table(headers,rows,title):
    def tex(value):
        value=re.sub(r'\[([^]]+)\]\([^)]+\)',r'\1',str(value))
        for char,replacement in (('\\',r'\textbackslash{}'),('_',r'\_'),('%',r'\%'),('&',r'\&'),('#',r'\#')):
            value=value.replace(char,replacement)
        return value
    return '\n'.join(['% '+title,r'\begin{tabular}{'+'l'*len(headers)+'}',r'\toprule',
        ' & '.join(map(tex,headers))+r' \\',r'\midrule',
        *[' & '.join(map(tex,row))+r' \\' for row in rows],r'\bottomrule',r'\end{tabular}'])+'\n'

def timing(row):
    if not row:return ['pending']*3
    group=row.get('timing',{});cohorts=group.get('cohorts',group.get('workloads',{}))
    pre=cohorts.get('prefill',{});dec=cohorts.get('decode',{})
    def display(summary,scale=1):
        if not finite(summary.get('median')):return 'pending'
        spread=summary.get('iqr')
        if not finite(spread):
            low=summary.get('p25',summary.get('q25'));high=summary.get('p75',summary.get('q75'))
            spread=high-low if finite(low) and finite(high) else None
        return fmt(summary['median']*scale,3)+' ['+fmt(spread*scale if finite(spread) else None,3)+']'
    pre_value=display(pre.get('latency_seconds',pre.get('seconds',{})),1000) if pre.get('status') in OK else 'pending'
    if dec.get('status') in OK:
        speed=dec.get('tokens_per_second',{})
        if not speed and dec.get('samples_seconds'):
            import statistics
            values=sorted(120/x for x in dec['samples_seconds'])
            def q(f):
                pos=(len(values)-1)*f;lo=int(pos);hi=min(lo+1,len(values)-1)
                return values[lo]+(values[hi]-values[lo])*(pos-lo)
            speed=dict(median=statistics.median(values),iqr=q(.75)-q(.25))
        speed_value=display(speed)
    else:speed_value='pending'
    memory='/'.join(fmt(x.get('peak_allocated_bytes',0)/2**30,3) if x.get('status') in OK else 'pending' for x in (pre,dec))
    return [pre_value,speed_value,memory]

def collect_e38(repo,reference):
    out=repo/'results/e38';all_rows,errors=discover(out/'raw')
    runs=[r for r in all_rows if r.get('kind') in ('e38_8b_asymmetric_same_instance_integer','e38_stage_a_trained8b_native')]
    table=[];details=[]
    formats=[('native','fp16','FP16 W/A/KV'),('native','hadamard','Packed W4 / per-token symmetric A4 / native per-head KV4'),
        ('native','nar_k8','Packed W4 / per-token symmetric A4 / native per-head KV4'),
        ('integer','hadamard','W4 expanded INT8 GEMM / asymmetric g128 A4 / simulated E14 KV4'),
        ('integer','nar_k8','W4 expanded INT8 GEMM / asymmetric g128 A4 / simulated E14 KV4')]
    for family,rotation,label in formats:
        matching=[r for r in runs if r.get('seed')==0 and r.get('rotation')==rotation and
            (r.get('kind')=='e38_stage_a_trained8b_native')==(family=='native')]
        primary=latest([r for r in matching if phase(r) not in ('eight-tasks',)])
        task_run=latest([r for r in matching if has_task_attempt(r) and (primary is None or identity(r)==identity(primary))])
        avg,count=task_result(task_run)
        value=ppl_value(primary,family=='integer')
        comparison=dict(status='unavailable')
        if family=='integer' and primary:
            eq=primary.get('equivalence',{});cs=eq.get('chunks',[])
            if len(cs)==141 and all(finite(x.get('reference_nll')) for x in cs):
                ref=dict(primary,ppl=dict(chunks=[dict(x,nll=x['reference_nll']) for x in cs]))
                comparison=pair_ci(primary,ref,True)
            diff=f"{fmt(eq.get('max_abs_logit_difference'))} / {fmt(eq.get('argmax_agreement'),6)}"
        else:
            diff='pending' if family=='integer' else 'format variant' if rotation!='fp16' else 'n/a'
            if primary and rotation!='fp16' and reference:comparison=pair_ci(primary,reference)
        table.append([label,rotation,fmt(value),fmt(avg*100 if avg is not None else None,3),f'{count}/8' if task_run else 'pending',diff,
                      *timing(primary),run_state(primary)+(('; tasks '+state(task_run)) if task_run and task_run!=primary else ''),source(primary)+(('; '+source(task_run)) if task_run and task_run!=primary else '')])
        details.append(dict(family=family,rotation=rotation,source=source(primary),tasks_source=source(task_run),paired_ppl=comparison))
    headers=['Actual format/backend','Rotation','Full141 PPL','Eight-task %','Tasks','Max logits / argmax',
             'Prefill ms','Decode tok/s','Peak GiB P/D','Run status','Source']
    text='**E38 — Meta-Llama-3.1-8B, seed0; GPTQ W4 / asymmetric g128 A4 / KV4 reference, with five explicitly labelled deployment formats**\n\n'
    text+=render_table(headers,table)
    text+='\n\nTiming: median [IQR], same loaded model after PPL, exclusive A40, B1/prefix2048, 10 warmups/20 repetitions, decode128/drop8/token100; component profiles are independent inclusive, non-additive diagnostics. Native per-head KV4 and per-token symmetric A4 differ from Table1 asymmetric g128/KIVI; the g128 backend expands W4 storage to INT8 arithmetic and is not native INT4. Cross-backend latency differences include implementation and KV differences and do not isolate pure format overhead.\n\n'
    text+=render_table(['Row','Paired PPL delta [95% CI]','Comparison'],[[x['family']+'/'+x['rotation'],ci_text(x['paired_ppl']),
        'integer minus same-run reference' if x['family']=='integer' else 'variant minus matched-token FP16; different format'] for x in details])
    component_rows=[]
    native_measurements=[]
    native_devices={}
    for rotation in ('fp16','hadamard','nar_k8'):
        measured=latest([r for r in runs if r.get('kind')=='e38_stage_a_trained8b_native' and
            r.get('rotation')==rotation and r.get('seed')==0 and phase(r)!='eight-tasks'])
        if not measured:continue
        uuids=tuple(sorted({item['gpu_uuid'] for guard in measured.get('timing',{}).get('guards',[])
            for item in guard.get('own',[]) if item.get('gpu_uuid')}))
        if uuids:native_devices[rotation]=(measured.get('hostname'),uuids)
        for workload,profile in measured.get('component_profile',{}).get('workloads',{}).items():
            if profile.get('status')!='COMPLETE':continue
            categories=profile.get('categories',{})
            component_rows.append([rotation,workload,*[fmt(categories.get(name,{}).get('inclusive_ms_per_model_call'),3)
                for name in ('act_quant_pack','linear','r4','kv_cache_update')],source(measured)])
        if rotation!='fp16' and finite(ppl_value(measured)):
            native_measurements.append(rotation+' PPL '+fmt(ppl_value(measured)))
    if component_rows:
        text+='\n\nNative Stage A component diagnostics (milliseconds per model call, separate instrumented pass; inclusive and non-additive):\n\n'
        text+=render_table(['Rotation','Workload','Activation quant/pack ms','Linear ms','R4 rotation ms','KV update ms','Source'],component_rows)
        gpu_comparison=('The three native-backend control rows used the same physical GPU.'
            if len(native_devices)==3 and len(set(native_devices.values()))==1 else
            'The selected native-backend control rows were not all verified on the same physical GPU; GPU identities are retained in each raw record.')
        text+='\n\nThe selected records identify original or optimized E14 R2/R4 implementations through rotation_backend; the observed Had/PQ latency gap is not an intrinsic rotation-cost comparison. '+gpu_comparison+' KV update includes prefill unpack/inverse-Hadamard where applicable.'
    seed_runs=[r for r in all_rows if r.get('kind')=='e38_8b_seed_checkpoint_worker']
    seed_rows=[]
    for seed in (1,2):
        for rotation in ('hadamard','nar_k8'):
            candidates=[r for r in seed_runs if seed in r.get('seeds',[])]
            run=latest(candidates);item=None
            if run:
                item=next((m for s in run.get('results',[]) if s.get('seed')==seed for m in s.get('methods',[]) if m.get('rotation')==rotation),None)
            seed_rows.append([seed,rotation,fmt(item.get('ppl') if item else None),item.get('status','pending') if item else state(run),
                              link(item['ppl_output']) if item and item.get('ppl_output') else source(run)])
    text+='\n\nSeed1/2 checkpoint/NLL generation uses fresh E14 R1/R2 and GPTQ samples, fixed original E11 R4 factors with seed-specific signs; these are separate from seed0 timing.\n\n'+render_table(['Seed','Rotation','PPL','Status','Source'],seed_rows)
    historical=out/'raw/fp16_collection_002.json'
    if historical.exists():text+='\n\nHistorical FP16 timing/accuracy comes from separate jobs and is not inserted into the new cohort: '+link(historical)+'.'
    table1_deltas=[]
    output_differences=[]
    for rotation,paper_ppl,paper_accuracy in (('hadamard',7.21,65.37),('nar_k8',6.99,65.36)):
        measured=latest([r for r in runs if r.get('kind')=='e38_8b_asymmetric_same_instance_integer' and r.get('rotation')==rotation and r.get('seed')==0 and phase(r)=='equivalence-timing'])
        ref_ppl=measured.get('equivalence',{}).get('reference_ppl') if measured else None
        task_run=latest([r for r in all_rows if r.get('kind')=='e38_8b_reference_eight_tasks' and r.get('rotation')==rotation and r.get('quantization_seed')==0])
        ref_accuracy,_=task_result(task_run)
        table1_deltas.append(rotation+': ΔPPL '+fmt(ref_ppl-paper_ppl if finite(ref_ppl) else None)+
            ', Δtask mean '+fmt(100*ref_accuracy-paper_accuracy if finite(ref_accuracy) else None,4)+'pp')
        eq=measured.get('equivalence',{}) if measured else {}
        if finite(eq.get('argmax_agreement')):
            output_differences.append(rotation+': max |Δlogit| '+fmt(eq.get('max_abs_logit_difference'),4)+
                ', argmax agreement '+fmt(100*eq['argmax_agreement'],4)+'%')
    statuses='; '.join(('native' if index<3 else 'integer')+'/'+row[1]+': '+row[-2]
        for index,row in enumerate(table))
    summary=[f'Collected {len(runs)} actual 8B deployment-run artifacts; current row states: {statuses}; absent metrics remain pending.',
        'Seed0 reference minus Table1 reported values (not a three-seed equivalence claim): '+'; '.join(table1_deltas)+'.',
        ('Native per-token symmetric A4/per-head KV4 format results: '+'; '.join(native_measurements)+
         '; these are measured format variants rather than Table1 g128 results.' if native_measurements else
         'Native per-token symmetric A4/per-head KV4 format results remain pending.'),
        ('Integer-path output diagnostics: '+'; '.join(output_differences)+
         '; close corpus PPL does not establish logit or prediction equivalence.' if output_differences else 'Integer-path output diagnostics remain pending.'),
        'The g128 prototype uses expanded INT8 GEMM, component profiles are non-additive, bootstrap intervals condition on fixed runs rather than calibration seeds, and no numerical difference blocks execution.']
    return out,text,headers,table,summary,dict(rows=details,observed_runs=[dict(path=r['_path'],sha256=r['_sha256'],status=state(r),error=r.get('error')) for r in runs],read_errors=errors)

def collect_baseline(repo,experiment,reference):
    out=repo/'results'/experiment;runs,errors=discover(out/'raw')
    if experiment=='e39':
        runs=[r for r in runs if r.get('model')=='Meta-Llama-3.1-8B' and r.get('method')=='FlatQuant official native']
        method='FlatQuant official';default_config='W4 RTN per-channel symmetric / A4 per-token symmetric / native asymmetric per-head K4V4, BF16 residual'
    else:
        runs=[r for r in runs if r.get('kind')=='E42_DuQuant_official_native_8b']
        method='DuQuant official';default_config='Official native W4A4 floating fake-quant; actual K/V format from run, not E14 g128/KIVI'
    primary=latest([r for r in runs if r.get('phase') in ('train-ppl','quantize-ppl')])
    task_run=latest([r for r in runs if has_task_attempt(r) and (primary is None or identity(r)==identity(primary))])
    avg,count=task_result(task_run);cost=None;cost_scope='pending';gpu_hours=None;peak=None
    if primary:
        c=primary.get('calibration_cost',primary.get('original_calibration_cost',{}))
        if finite(c.get('wall_seconds')):cost=c['wall_seconds'];cost_scope='transform calibration'
        elif finite(primary.get('quantization_seconds')):cost=primary['quantization_seconds'];cost_scope='official DuQuant quantization; includes rotation/smoothing work'
        elif primary.get('prior_quantization'):
            cost=primary['prior_quantization'].get('record',{}).get('quantization_seconds');cost_scope='original reused-checkpoint quantization'
        elapsed=primary.get('total_wall_seconds',primary.get('total_elapsed_seconds'))
        gpu_hours=elapsed/3600 if finite(elapsed) else None
        peak_bytes=c.get('peak_allocated_bytes',primary.get('peak_allocated_bytes'))
        peak=peak_bytes/(1<<30) if finite(peak_bytes) else None
    config=primary.get('config',primary.get('scope',default_config)) if primary else default_config
    if experiment=='e42' and primary:
        actual=primary.get('official_arguments',primary.get('prior_quantization',{}).get('record',{}).get('official_arguments',{}))
        if actual:
            config='FP16 fake-quant; '+ '; '.join(f'{name}={json.dumps(actual[key],sort_keys=True)}' for name,key in
                (('W','weight_quant_params'),('A','act_quant_params'),('Q','q_quant_params'),('K','k_quant_params'),('V','v_quant_params')) if key in actual)
    table=[['Meta-Llama-3.1-8B',method,config,fmt(ppl_value(primary)),fmt(avg*100 if avg is not None else None,3),
            f'{count}/8' if task_run else 'pending',fmt(cost,2),fmt(gpu_hours,4),fmt(peak,3),run_state(primary)+(('; tasks '+state(task_run)) if task_run and task_run!=primary else ''),source(primary)+(('; '+source(task_run)) if task_run else '')]]
    headers=['Model','Method','Actual configuration','Full141 PPL','Eight-task %','Tasks','Calib/quant seconds','Quantize+PPL GPU hours','Peak allocated GiB','Run status','Source']
    ci=pair_ci(primary,reference) if primary and reference else dict(status='unavailable',reason='No complete same-token baseline pair yet')
    text=f'**{experiment.upper()} — Meta-Llama-3.1-8B, {method}, seed0; {default_config}**\n\n'+render_table(headers,table)
    text+=f'\n\nCost scope: {cost_scope}; checkpoint serialization and total allocated wall time are separate fields in raw results. No E14/GPTQ/group128 compatibility is implied.\n\nPaired PPL difference from the explicitly linked same-token FP16 control: {ci_text(ci)}; '+(source(reference) if reference else 'baseline pending')+'.'
    summary=[f'{method} delivery row is Meta-Llama-3.1-8B seed0, status {state(primary)}.',
        f'Complete task metrics currently cover {count}/8 tasks; an eight-task mean is shown only when all eight are available.',
        f'Reported calibration cost scope is {cost_scope}.',
        'Any paired interval requires equal token-file hashes and all141 per-sequence NLLs, and describes fixed-run sequence uncertainty only.',
        'Historical implementation-validation material remains linked separately.']
    return out,text,headers,table,summary,dict(paired_ppl=ci,observed_runs=[dict(path=r['_path'],sha256=r['_sha256'],status=state(r),error=r.get('error')) for r in runs],read_errors=errors)

def e41_statistics(rows,folder,group):
    import numpy as np
    if not all(state(row) in OK and finite(ppl_value(row)) for row in rows):
        return dict(status='unavailable',reason='Six complete finite PPL measurements required')
    parsed=[chunks(row) for row in rows]
    hashes=[token_hash(row) for row in rows]
    if not all(parsed) or not hashes[0] or len(set(hashes))!=1 or any(p[1]!=parsed[0][1] for p in parsed):
        return dict(status='unavailable',reason='Six complete same-token141-sequence rows required')
    sources=[dict(path=r['_path'],sha256=sha(r['_path'])) for r in rows]
    digest=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:16]
    path=folder/f'bootstrap_g{group}_seed0_{digest}.json'
    if path.exists():return read(path)
    point,draws,indices=paired_bootstrap([x[0] for x in parsed],parsed[0][1])
    asym=draws[:,2]-draws[:,1];sym=draws[:,5]-draws[:,4];denom=draws[:,0]-draws[:,1]
    with np.errstate(divide='ignore',invalid='ignore'):ratio=asym/denom
    quantities=(asym,sym,asym-sym,ratio)
    p_asym=float(point[2]-point[1]);p_sym=float(point[5]-point[4]);p_denom=float(point[0]-point[1])
    names=('asym_signed_minus_constant','sym_signed_minus_constant','difference_of_contrasts','operational_offset_fraction')
    pvals=(p_asym,p_sym,p_asym-p_sym,p_asym/p_denom if p_denom else None)
    record=dict(status='COMPLETE',kind='e41_b6_paired_sequence_bootstrap',group=group,model='Meta-Llama-3.1-8B',
        configuration=f'W16(BF16)/A4 qkv+down only/KV16(BF16), g{group}',rotation_seed=0,
        row_order=list(ARMS),input_records=sources,evaluation_tokens_sha256=hashes[0],sequences=141,
        replicates=20000,bootstrap_rng_seed=20261111,random_generator='NumPy PCG64',
        recomputed_point_ppl=point.tolist(),resampling_unit='Same141 sequence indices for every arm in each replicate',
        aggregation='Token-weighted NLL -> exp -> PPL contrasts',
        interval_method='Marginal two-sided95% percentile, linear interpolation; no multiplicity adjustment',
        interpretation='Conditional sequence uncertainty, not seed/calibration uncertainty; neighboring sequences can be dependent',
        quantities={name:dict(point=value,percentile95=interval(draw),finite_draws=int(np.isfinite(draw).sum()))
                    for name,value,draw in zip(names,pvals,quantities)},
        versus_matched_hadamard={name:dict(point=float(point[i]-point[0 if i<3 else 3]),
            percentile95=interval(draws[:,i]-draws[:,0 if i<3 else 3])) for i,name in enumerate(ARMS)},
        denominator_diagnostics=dict(point=p_denom,percentile95=interval(denom),
            nonpositive_draws=int((denom<=0).sum()),exact_zero_draws=int((denom==0).sum()),
            absolute_near_zero_draws=int((np.abs(denom)<=.001).sum()),nonfinite_ratio_draws=int((~np.isfinite(ratio)).sum()),
            policy='No clipping, trimming or deletion of denominator/ratio draws'))
    folder.mkdir(parents=True,exist_ok=True);archive=path.with_suffix('.npz')
    np.savez_compressed(archive,paired_sequence_indices=indices,bootstrap_ppl=draws,row_order=np.array(ARMS),
        bootstrap_quantities=np.stack(quantities,axis=-1),bootstrap_denominator=denom)
    record['raw_archive']=dict(path=str(archive),sha256=sha(archive));record['_statistics_source']=str(path)
    dump(path,record)
    return record

def collect_e41(repo):
    out=repo/'results/e41/b6';masters,errors=discover(out,True)
    masters=[r for r in masters if r.get('kind')=='e41_b6_mechanism' and r.get('seed')==0]
    table=[];statistics={}
    for group in ('32','512','d'):
        master=latest([r for r in masters if str(r.get('group'))==group]);rows=[]
        for arm in ARMS:
            entry=next((x for x in master.get('rows',[]) if x['name']==arm),None) if master else None
            path=Path(entry['output']) if entry and entry.get('output') else None
            row=read(path) if path and path.exists() else None
            if row:row['_path']=str(path)
            rows.append(row)
        stats=e41_statistics(rows,out/'statistics',group) if all(rows) else dict(status='unavailable')
        statistics[group]=stats
        for arm,row in zip(ARMS,rows):
            diff=stats.get('versus_matched_hadamard',{}).get(arm)
            ci=(f"{diff['point']:+.5f} [{diff['percentile95'][0]:+.5f}, {diff['percentile95'][1]:+.5f}]"
                if diff and diff['percentile95'] else 'pending')
            table.append([group,arm,fmt(ppl_value(row)),ci,state(row) if row else state(master),source(row) if row else source(master)])
    headers=['Group','Arm','Full141 PPL','Delta vs matched Had [95% paired CI]','Status','Source']
    text='**E41 B6 — Meta-Llama-3.1-8B, seed0, W16(BF16)/A4 only at qkv and down inputs/KV16(BF16); six arms × g32,g512,full-width d**\n\n'+render_table(headers,table)
    text+='\n\nFull-width d means4096 at qkv and14336 at down; BF16 weights/KV are retained, without GPTQ/R2. Each completed group uses20000 paired sequence resamples for all six arms, matching the original g128 reporting convention.\n\n'
    contrasts=[]
    for group,stats in statistics.items():
        for name in ('asym_signed_minus_constant','sym_signed_minus_constant','difference_of_contrasts','operational_offset_fraction'):
            value=stats.get('quantities',{}).get(name,{})
            ci=value.get('percentile95')
            contrasts.append([group,name,fmt(value.get('point')),f'[{ci[0]:+.5f}, {ci[1]:+.5f}]' if ci else 'pending',
                              link(stats['_statistics_source']) if stats.get('_statistics_source') else 'pending'])
    text+=render_table(['Group','Contrast','Point','95% paired CI','Statistics source'],contrasts)
    text+='\n\nRatios are operational quantities, not causal attribution; near-zero/negative denominators and nonfinite draws are retained in the statistics files. Original g128 six-arm table and bootstrap are immutable and remain separate: '+link(repo/'results/e41/table.md')+'; '+link(repo/'results/e41/raw/paired_bootstrap_001.json')+'.'
    complete=sum(r[4] in OK for r in table)
    summary=[f'Collected {complete}/18 complete B6 arm measurements for Meta-Llama-3.1-8B seed0.',
        'The actual setting is BF16 weights/KV and activation-only INT4 at qkv/down, not W4A4KV4.',
        'Each complete six-arm group receives20000 paired sequence-bootstrap resamples with the g128 contrast definitions.',
        'Intervals are conditional sequence uncertainty; ratios are neither clipped nor interpreted causally.',
        'The original g128 table, raw results, and bootstrap remain unchanged.']
    return out,text,headers,table,summary,dict(statistics=statistics,read_errors=errors)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=('e38','e39','e41','e42'),required=True)
    parser.add_argument('--repo',type=Path)
    parser.add_argument('--e38-repo',type=Path,default=HOME/'prismquant-rebuttal-e38')
    args=parser.parse_args();repo=args.repo or HOME/f'prismquant-rebuttal-{args.experiment}'
    reference=baseline(args.e38_repo) if args.experiment!='e41' else None
    if args.experiment=='e38':result=collect_e38(repo,reference)
    elif args.experiment=='e41':result=collect_e41(repo)
    else:result=collect_baseline(repo,args.experiment,reference)
    out,text,headers,rows,summary,record=result
    submission,submission_path=submitted(args.experiment,args.e38_repo)
    old_table=render_table(headers,rows)
    status_column=headers.index('Run status') if 'Run status' in headers else headers.index('Status')
    for row in rows:
        if row[status_column]=='pending':row[status_column]=submission
    text=text.replace(old_table,render_table(headers,rows),1)
    summary=[sentence.replace('status pending.',f'status {submission}.') for sentence in summary]
    if submission_path:
        text+='\n\nSubmission record (scheduler states are observations at submission, not a live query): '+link(submission_path)+'.'
        record['submission_provenance']=dict(path=submission_path,sha256=sha(submission_path),description=submission)
    historical=archive(out)
    if (historical/'table.md').exists():text+='\n\nPreserved pre-collector table: '+link(historical/'table.md')+'.'
    observations=record.get('observed_runs',[])
    if observations:
        text+='\n\nAll observed attempts (no selection by score):\n\n'+render_table(['Source','Status','Error'],
            [[link(r['path']),r['status'],r.get('error') or ''] for r in observations])
    text+='\n\nCollector: '+link(Path(__file__))+'; '+link(out/'collection_8b.json','collection provenance')+'.\n'
    record.update(experiment=args.experiment,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        collector=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        table_rows=rows,selection_policy='latest attempt by job ID then mtime per phase; never choose a better score',
        preserved_table=str(historical/'table.md'))
    dump(out/'collection_8b.json',record)
    write(out/'table.md',text);write(out/'table.tex',tex_table(headers,rows,text.splitlines()[0]))
    write(out/'summary.md','\n\n'.join(summary[:5])+f'\n\n{link(out/"table.md","Table and raw source links")}\n')
    print(json.dumps(dict(experiment=args.experiment,output=str(out),rows=len(rows))))

if __name__=='__main__':main()
