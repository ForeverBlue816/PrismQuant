"""Collect one allocation's measured 8B backend comparison, preserving formats."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import collect_rebuttal8b as common
from collect_requested_baselines import model_key
ROOT=Path(__file__).resolve().parents[1]
SPECS=[
 ('flat_official','FlatQuant official','Packed W4 RTN / per-token A4; native KV4, unquantized prefill KV'),
 ('flat_clean','FlatQuant, unused allocation removed','Same packed parameters/arithmetic as official; dead random allocation removed'),
 ('pq_fast','PrismQuant k=8, optimized','Packed W4 GPTQ / per-token symmetric A4 / native per-head KV4'),
 ('flat_fake','FlatQuant official fake-quant','BF16 floating W4 RTN / per-token A4 / per-head KV4'),
 ('du_fake','DuQuant official fake-quant','FP16 floating official W4A4 / QK/V quantizers and floating KV cache'),
]

def classify(row):
    if model_key(row)!='llama31_8b':return None
    if row.get('kind')=='flatquant_packed_native':
        command=row.get('command',[])
        mode=command[command.index('--cache-optimization')+1] if '--cache-optimization' in command else 'official'
        return 'flat_clean' if mode=='remove-dead-random' else 'flat_official'
    if row.get('kind')=='e38_stage_a_trained8b_native' and row.get('rotation')=='nar_k8' and row.get('rotation_backend')=='fast':
        command=row.get('command_line',[])
        if '--cache-timing' in command and command[command.index('--cache-timing')+1]=='fresh':return 'pq_fast'
        return None
    if row.get('method')=='FlatQuant official native' and common.phase(row)=='ppl-timing':return 'flat_fake'
    if str(row.get('kind','')).startswith('E42_DuQuant_official_native_') and common.phase(row)=='ppl-timing':return 'du_fake'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',required=True,type=int)
    args=parser.parse_args()
    paths=[ROOT/'results/e38/raw',Path('/home/yanlongc/prismquant-rebuttal-e39/results/e39/raw'),
        Path('/home/yanlongc/prismquant-rebuttal-e42/results/e42/raw')]
    rows={};sources={}
    for directory in paths:
        for path in sorted(directory.glob(f'*job{args.job_id}.json'),key=lambda p:p.stat().st_mtime):
            row=json.loads(path.read_text());kind=classify(row)
            if kind:
                rows[kind]=row;sources[kind]=dict(path=str(path),sha256=common.sha(path))
    table=[];details={};devices=[]
    for key,label,configuration in SPECS:
        row=rows.get(key,{})
        timing=row.get('timing',{});work=timing.get('workloads',{})
        pre=work.get('prefill',{});dec=work.get('decode',{})
        pre_ms=pre.get('seconds',{}).get('median') if pre.get('status')=='COMPLETE' else None
        if pre_ms is not None:pre_ms*=1000
        samples=dec.get('samples_seconds',[])
        dec_tps=statistics.median(timing.get('measured_decode_steps',120)/x for x in samples) if samples and dec.get('status')=='COMPLETE' else None
        peaks=[x['peak_allocated_bytes'] for x in work.values() if x.get('status')=='COMPLETE' and x.get('peak_allocated_bytes')]
        guards=timing.get('guards',[])
        uuids=sorted({item['gpu_uuid'] for guard in guards for item in guard.get('own',[])})
        verified=bool(guards) and all(g.get('status')=='EXCLUSIVE' for g in guards) and len(uuids)==1
        if verified:devices.append(uuids[0])
        value=common.ppl_value(row) if row else None
        status=row.get('status','pending')+'; timing '+timing.get('status','pending')
        table.append([label,configuration,common.fmt(value,5),common.fmt(pre_ms,3),common.fmt(dec_tps,3),
            common.fmt(max(peaks)/(1<<30) if peaks else None,3),status])
        details[key]=dict(source=sources.get(key),status=status,ppl=value,prefill_ms=pre_ms,decode_tokens_per_second=dec_tps,
            run_status=row.get('status','pending'),timing_status=timing.get('status','pending'),
            peak_allocated_gib=max(peaks)/(1<<30) if peaks else None,gpu_uuids=uuids,exclusive=verified,
            cache_timing=timing.get('cache_timing',timing.get('cache_reset',timing.get('cache'))))
    same_gpu=len(devices)==len(SPECS) and len(set(devices))==1
    all_complete=all(row['run_status']=='COMPLETE' and row['timing_status']=='COMPLETE' for row in details.values())
    comparisons={};comparison_rows=[]
    pq=details['pq_fast']
    for key,label in [('flat_official','FlatQuant official packed INT4'),
                      ('flat_clean','FlatQuant packed INT4, unused allocation removed')]:
        baseline=details[key]
        if (pq['timing_status']==baseline['timing_status']=='COMPLETE'
            and pq['exclusive'] and baseline['exclusive']
            and pq['gpu_uuids']==baseline['gpu_uuids']
            and all(common.finite(row[metric]) and row[metric]>0
                for row in (pq,baseline) for metric in ('prefill_ms','decode_tokens_per_second'))):
            result=dict(prefill_latency_ratio=pq['prefill_ms']/baseline['prefill_ms'],
                decode_throughput_ratio=pq['decode_tokens_per_second']/baseline['decode_tokens_per_second'],
                ppl_difference=pq['ppl']-baseline['ppl'] if all(common.finite(row['ppl']) for row in (pq,baseline)) else None,
                interpretation='Whole-backend comparison with distinct quantization/KV formats and measured PPL; not equal-quality superiority')
            comparisons[key]=result
            comparison_rows.append([label,common.fmt(result['prefill_latency_ratio'],5),
                common.fmt(result['decode_throughput_ratio'],5),common.fmt(result['ppl_difference'],5)])
    comparison_text=('\n\n'+common.render_table(
        ['PQ optimized native versus','Prefill latency ratio PQ/baseline (lower is faster)',
         'Decode throughput ratio PQ/baseline (higher is faster)','PPL difference PQ minus baseline'],comparison_rows)+
        '\n\nThese ratios concern the native per-token symmetric PQ variant, whose measured PPL is '+common.fmt(pq['ppl'],5)+
        '. They do not establish equivalent accuracy or performance of the asymmetric g128 reference format.') if comparison_rows else ''
    title='Meta-Llama-3.1-8B — seed0, actual backend formats listed per row; exclusive A40 job '+str(args.job_id)
    notes=(
        'B1, prefix2048, decode128/drop8, token100,10 warmups/20 repeats, eager synchronized wall time, last-position logits. '
        'Each timed object first evaluates all141 WikiText2 sequences. Fresh cache construction is included in prefill latency; '
        'decode prefix construction and first8 steps are outside its timer. Peak allocated memory includes the prefix.\n\n'
        'Packed native rows and floating fake-quant rows are distinct execution modes. FlatQuant native prefill consumes transformed '
        'unquantized KV, repeats GQA heads and retains official dequantization arithmetic; PQ native prefill reconstructs packed KV and uses '
        '8 KV heads. These are whole-backend measurements, not an isolated comparison of rotation cost or Table1 g128 configurations. '
        'The FlatQuant cleanup removes only a random allocation immediately overwritten by a saved matrix; both versions remain reported.\n\n'
        +('All five rows completed and were verified on the same physical exclusive GPU.' if same_gpu and all_complete
          else 'The recorded GPU observations agree, but the five-row comparison is not yet complete.' if same_gpu
          else 'The complete five-row same-GPU comparison is still pending.')
    )
    folder=ROOT/'results/baselines/native_8b'
    common.write(folder/'table.md','**'+title+'**\n\n'+common.render_table(
        ['Method / backend','Actual configuration','Full141 PPL','Prefill ms','Decode tokens/s','Peak GiB','Status'],table)+'\n\n'+notes+comparison_text+'\n')
    common.write(folder/'summary.md',title+'\n\n'+notes+comparison_text+'\n\n'+
        'Sources and exclusivity evidence: [collection.json](collection.json). '
        'The completed original-versus-optimized PQ experiment used the earlier reused-cache boundary and is retained '
        '[separately](../../e38/rotation_optimization/table.md); its speedup is not substituted for this fresh-cache comparison.\n')
    common.dump(folder/'collection.json',dict(job_id=args.job_id,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        same_physical_gpu=same_gpu,all_complete=all_complete,pq_native_comparisons=comparisons,
        selection='all latest attempts from specified allocation, never selected by score',rows=details))
    print(f'Collected same-allocation 8B backend results for job {args.job_id}; {len(rows)}/5 rows present.')

if __name__=='__main__':main()
