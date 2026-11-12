"""Collect the requested 3B/8B FlatQuant and DuQuant rows without FP16 runs."""
from __future__ import annotations

import datetime
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import collect_rebuttal8b as common

ROOT=Path(__file__).resolve().parents[1]
METHODS={
    'FlatQuant':(Path('/home/yanlongc/prismquant-rebuttal-e39/results/e39'),
        'W4 RTN / per-token symmetric A4 / asymmetric per-head KV4; BF16 floating fake-quant'),
    'DuQuant':(Path('/home/yanlongc/prismquant-rebuttal-e42/results/e42'),
        'Official run.sh W4A4 / native QK/V quantizers; FP16 floating fake-quant'),
}
MODELS={'llama32_3b':'Llama-3.2-3B','llama31_8b':'Meta-Llama-3.1-8B'}


def model_key(row):
    if row.get('model_key') in MODELS:return row['model_key']
    value=str(row.get('model','')).lower()
    if value in ('llama32_3b','llama-3.2-3b','meta-llama-3.2-3b'):return 'llama32_3b'
    if value in ('llama31_8b','llama-3.1-8b','meta-llama-3.1-8b'):return 'llama31_8b'
    return None


def method_record(row,method):
    if method=='FlatQuant':return row.get('method')=='FlatQuant official native'
    return str(row.get('kind','')).startswith('E42_DuQuant_official_native_')


def timing_values(row):
    if not row:return ['pending']*3
    workloads=row.get('timing',{}).get('workloads',{})
    prefill=workloads.get('prefill',{});decode=workloads.get('decode',{})
    prefill_ms=(prefill.get('milliseconds',{}).get('median') if prefill.get('status')=='COMPLETE' else None)
    decode_tps=(decode.get('tokens_per_second',{}).get('median') if decode.get('status')=='COMPLETE' else None)
    peaks=[x.get('peak_allocated_bytes') for x in workloads.values() if x.get('status')=='COMPLETE']
    peaks=[x for x in peaks if common.finite(x)]
    return [common.fmt(prefill_ms,3),common.fmt(decode_tps,3),common.fmt(max(peaks)/(1<<30) if peaks else None,3)]


def collect_model(key,inputs):
    output=ROOT/'results/baselines'/key
    rows=[];details={};ppl_rows=[];devices=[]
    for method,(_,config) in METHODS.items():
        candidates=[r for r in inputs[method] if model_key(r)==key and method_record(r,method)]
        accuracy=common.latest([r for r in candidates if common.phase(r) in ('train-ppl','quantize-ppl')])
        benchmark=common.latest([r for r in candidates if common.phase(r)=='ppl-timing'])
        primary=benchmark if benchmark and common.ppl_value(benchmark) is not None else accuracy
        tasks=common.latest([r for r in candidates if common.phase(r) in ('tasks','eight-tasks') and
            (primary is None or common.identity(r)==common.identity(primary))])
        if benchmark and primary and common.identity(benchmark)!=common.identity(primary):
            benchmark=None
        mean,count=common.task_result(tasks)
        value=common.ppl_value(primary)
        status='PPL '+common.run_state(primary)+'; tasks '+common.state(tasks)+'; timing '+(
            benchmark.get('timing',{}).get('status','pending') if benchmark else 'pending')
        rows.append([method,config,common.fmt(value),common.fmt(100*mean if mean is not None else None,3),
            str(count)+'/8',*timing_values(benchmark),status])
        details[method]=dict(ppl_source=common.source(primary),task_source=common.source(tasks),
            timing_source=common.source(benchmark),calibration_source=common.source(accuracy),
            prior_attempts=[dict(path=r['_path'],sha256=r['_sha256'],phase=common.phase(r),status=common.state(r)) for r in candidates])
        ppl_rows.append(primary)
        if benchmark:
            uuids={item['gpu_uuid'] for guard in benchmark.get('timing',{}).get('guards',[])
                for item in guard.get('own',[]) if item.get('gpu_uuid')}
            devices.append(uuids)
    paired=common.pair_ci(ppl_rows[0],ppl_rows[1]) if all(ppl_rows) else dict(status='unavailable',reason='Both complete NLL rows are required')
    same_gpu=len(devices)==2 and all(len(x)==1 for x in devices) and devices[0]==devices[1]
    title=f'**{MODELS[key]} — seed0; official FlatQuant and DuQuant INT4 quantization recipes, actual floating fake-quant execution**'
    headers=['Method','Actual configuration','Full141 PPL','Eight-task %','Tasks','Prefill ms','Decode tokens/s','Peak GiB','Status']
    table=common.render_table(headers,rows)
    text=title+'\n\n'+table+'\n\n'
    text+='Timing uses the same loaded checkpoint after full141 PPL, B1/prefix2048/decode128/drop8,10 warmups/20 repeats, real KV-cache decode and an exclusive A40; full-vocabulary logits at the last token position only are computed for both methods. These numbers measure the official floating fake-quant implementations and are not native INT4 kernel throughput.\n\n'
    text+=('Both timing rows are verified on the same physical GPU.' if same_gpu else 'A same-physical-GPU timing pair is not yet complete.')+'\n\n'
    text+='Paired PPL difference FlatQuant − DuQuant [95% sequence-bootstrap CI]: '+common.ci_text(paired)+'.\n\n'
    text+=common.render_table(['Method','PPL source','Eight-task source','Timing source','Calibration source'],
        [[method,*(details[method][name] for name in ('ppl_source','task_source','timing_source','calibration_source'))] for method in METHODS])
    text+='\n\nNo new FP16 control is run; saved checkpoints and NLLs are reused, and missing values remain pending. Native packed FlatQuant performance and optimized PrismQuant are separate backend results and must not be mixed into these floating-implementation timings.\n'
    summary=[f'{MODELS[key]} has two requested official-method rows; currently '+', '.join(row[0]+' PPL '+row[2]+' / tasks '+row[4] for row in rows)+'.',
        'Actual weight, activation and KV formats are stated per method, including floating fake-quant execution.',
        'Timing follows complete PPL on the same loaded checkpoint and uses an exclusive GPU, real cached decode and common workload settings.',
        'The paired PPL interval uses the same saved sequence resamples and conditions on seed0 rather than measuring calibration-seed variation.',
        'No FP16 run or missing benchmark value is introduced; unsupported native backends remain separate from measured implementation speed.']
    common.write(output/'table.md',text)
    common.write(output/'table.tex',common.tex_table(headers,rows,title))
    common.write(output/'summary.md','\n\n'.join(summary)+'\n')
    common.dump(output/'collection.json',dict(model=MODELS[key],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rows=details,paired_ppl=paired,same_physical_gpu=same_gpu,selection='latest attempt per method/model/phase, never score based'))
    return text


def main():
    inputs={method:common.discover(folder/'raw')[0] for method,(folder,_) in METHODS.items()}
    outputs=[collect_model(key,inputs) for key in MODELS]
    common.write(ROOT/'results/baselines/table.md','\n\n'.join(outputs))
    common.write(ROOT/'results/baselines/summary.md','Only the requested FlatQuant/DuQuant rows on Llama-3.2-3B and Meta-Llama-3.1-8B are collected here.\n\nNative INT4 and optimized PrismQuant timing are separate results; there is no new FP16 run.\n\n[3B results](llama32_3b/table.md) · [8B results](llama31_8b/table.md) · [8B native backend speed comparison](native_8b/table.md) · [PQ rotation optimization](../e38/rotation_optimization/table.md)\n')
    print('Collected requested 3B/8B baselines; missing measurements remain pending.')


if __name__=='__main__':main()
