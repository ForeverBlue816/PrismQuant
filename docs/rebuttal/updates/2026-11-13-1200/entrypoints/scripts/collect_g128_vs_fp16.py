"""Collect one exact-job 8B g128 prototype/FP16 cohort without GPU work.

Only measurement identity and workload completeness determine whether a speed
ratio can be reported. PPL values and numerical differences are never gates.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Meta-Llama-3.1-8B'
FORMATS = {
    'fp16': 'FP16 W/A; native FP16 paged KV; released BF16 base cast to FP16',
    'g128': 'PQ k=8; GPTQ signed W4 storage expanded to INT8 GEMM; asymmetric g128 A4; INT32 partials/FP32 affine correction; simulated KV4 in BF16 DynamicCache',
}
EXPECTED = dict(batch_size=1, prefix=2048, decode_steps=128, discarded_decode_steps=8,
                fixed_decode_token_id=100, warmups=10, repetitions=20,
                fresh_prefill_cache_inside_timer=True, fresh_decode_cache_outside_timer=True,
                last_position_logits=True, eager=True)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clean(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    return value


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(text)
    temp.replace(path)


def read_record(path):
    if not path.exists(): return {}, dict(path=str(path), status='PENDING', reason='Expected raw file does not exist yet')
    try:
        data = path.read_bytes()
        record = json.loads(data)
        if not isinstance(record, dict): raise ValueError('Expected a JSON object')
        return record, dict(path=str(path), status='READ', sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc:
        return {}, dict(path=str(path), status='READ_ERROR', reason=repr(exc))


def protocol(record, key):
    timing = record.get('timing', {})
    if key == 'fp16':
        workloads = timing.get('workloads', {})
        prefill, decode = (workloads.get(x, {}) for x in ('prefill', 'decode'))
        values = {x: timing.get(x) for x in ('batch_size', 'prefix', 'decode_steps',
            'discarded_decode_steps', 'fixed_decode_token_id', 'repetitions')}
        values.update(warmups=timing.get('warmup'),
            fresh_prefill_cache_inside_timer=timing.get('cache_timing') == 'fresh'
                and prefill.get('cache_construction_inside_timer') is True,
            fresh_decode_cache_outside_timer=decode.get('fresh_cache_per_repetition') is True
                and decode.get('cache_construction_inside_timer') is False
                and decode.get('cache_reset_and_decode_prefix_outside_timer') is True,
            last_position_logits='last position only' in timing.get('logits', ''),
            eager='eager' in timing.get('mode', '') and 'no CUDA Graph' in timing.get('mode', ''))
    else:
        workloads = timing.get('cohorts', {})
        prefill, decode = (workloads.get(x, {}) for x in ('prefill', 'decode'))
        values = {x: decode.get(x) for x in ('batch_size', 'prefix', 'decode_steps',
            'discarded_decode_steps', 'fixed_decode_token_id', 'warmups', 'repetitions')}
        values.update(fresh_prefill_cache_inside_timer=prefill.get('cache_creation_in_timed_region') is True,
            fresh_decode_cache_outside_timer=decode.get('cache_creation_in_timed_region') is False
                and 'fresh DynamicCache' in decode.get('cache_reset', ''),
            last_position_logits=all('last-position logits only' in x.get('output', '') for x in (prefill, decode)),
            eager=timing.get('graph_mode') is False)
        values['prefill_workload_matches_decode'] = all(prefill.get(x) == decode.get(x)
            for x in ('batch_size', 'prefix', 'fixed_decode_token_id', 'warmups', 'repetitions'))
        values['cache_lengths_verified'] = (prefill.get('cache_lengths') == [2048] * 20
            and decode.get('cache_lengths') == [2176] * 20)
    return values, workloads


def collect_row(record, source, key, job_id):
    timing = record.get('timing', {})
    ppl = record.get('ppl', {})
    chunks = ppl.get('chunks', [])
    full141 = (len(chunks) == 141 and [x.get('chunk') for x in chunks] == list(range(141))
        and all(x.get('tokens_scored') == 2047 and x.get('status', 'COMPLETE') == 'COMPLETE' for x in chunks)
        and ppl.get('sequence_length') == 2048)
    ppl_value = ppl.get('value') if key == 'fp16' else ppl.get('candidate_ppl', ppl.get('ppl'))
    settings, workloads = protocol(record, key)
    guards = timing.get('guards', []) if key == 'fp16' else (
        [timing['initial_guard']] if timing.get('initial_guard') else []) + [
            guard for phase in workloads.values() for guard in phase.get('guards', [])]
    uuids = sorted({item['gpu_uuid'] for guard in guards for item in guard.get('own', [])
                    if item.get('gpu_uuid')} | {guard['gpu_uuid'] for guard in guards if guard.get('gpu_uuid')})
    exclusive = bool(guards) and all(g.get('status') == ('EXCLUSIVE' if key == 'fp16' else 'PASS') for g in guards)
    if key == 'fp16':
        identity = record.get('model_object_id')
        same_instance = identity is not None and identity == record.get('same_model_object_id_after_measurements') \
            and timing.get('same_loaded_instance_after_ppl') is True
    else:
        identity = record.get('model_instance_id')
        same_instance = (identity is not None and identity == ppl.get('model_instance_id')
            == timing.get('model_instance_id') == record.get('same_instance_evidence', {}).get('model_id')
            and record.get('same_instance_evidence', {}).get('no_reload_between_quality_and_timing') is True
            and record.get('same_instance_evidence', {}).get('packed_buffer_pointer_and_shapes_unchanged') is True)
    measurements = {}
    for phase in ('prefill', 'decode'):
        raw = workloads.get(phase, {})
        samples = raw.get('samples_seconds', [])
        complete = raw.get('status') == 'COMPLETE' and len(samples) == 20 \
            and all(finite(x) and x > 0 for x in samples)
        measurements[phase] = dict(status=raw.get('status', 'PENDING'), samples=len(samples),
            latency_ms=statistics.median(samples) * 1000 if complete else None,
            tokens_per_second=statistics.median([120/x for x in samples]) if complete and phase == 'decode' else None,
            peak_allocated_gib=raw.get('peak_allocated_bytes') / (1 << 30)
                if complete and finite(raw.get('peak_allocated_bytes')) else None,
            peak_reserved_gib=raw.get('peak_reserved_bytes') / (1 << 30)
                if complete and finite(raw.get('peak_reserved_bytes')) else None,
            complete=complete, error=raw.get('error'), reason=raw.get('reason'))
    issues = []
    if source['status'] != 'READ': issues.append(source.get('reason', source['status']))
    for label, valid in (
        ('record must identify the requested job', str(record.get('slurm', {}).get('SLURM_JOB_ID')) == job_id),
        ('record must identify llama31_8b seed0', record.get('model') == 'llama31_8b' and record.get('seed') == 0),
        ('record must identify the requested row', record.get('row') == 'fp16' if key == 'fp16'
            else record.get('rotation') == 'nar_k8' and record.get('phase') == 'ppl-timing'),
        ('both timing workloads must be complete', timing.get('status') == 'COMPLETE'
            and all(x['complete'] for x in measurements.values())),
        ('single physical GPU with all exclusivity observations passing', exclusive and len(uuids) == 1),
        ('quality and timing must use the same model object', same_instance),
        ('expected common timing workload and fresh-cache boundary', all(settings.get(k) == v for k, v in EXPECTED.items())),
        ('prefill settings must match decode', settings.get('prefill_workload_matches_decode', True)),
        ('incremental cache length must be verified', settings.get('cache_lengths_verified', True)),
    ):
        if not valid: issues.append(label)
    libraries = {name: value.get('version') if isinstance(value, dict) else value
                 for name, value in record.get('libraries', {}).items()}
    return dict(key=key, label='FP16' if key == 'fp16' else 'PrismQuant k=8 g128 integer prototype',
        format=FORMATS[key], source=source, status=record.get('status', source['status']),
        ppl=dict(status=ppl.get('status', 'PENDING'), full141=full141, completed_sequences=ppl.get('completed_sequences', 0),
                 value=ppl_value if full141 else None, error=ppl.get('error'), reason=ppl.get('reason')),
        timing_status=timing.get('status', 'PENDING'), timing_error=timing.get('error'),
        timing_reason=timing.get('reason'), measurements=measurements, protocol=settings,
        gpu_uuids=uuids, gpu_model=record.get('gpu_model'), exclusive=exclusive, same_loaded_instance=same_instance,
        base_revision=record.get('base_revision'), evaluation_tokens=record.get('evaluation_tokens'),
        libraries=libraries, cuda_runtime=record.get('cuda_runtime'), source_commit=record.get('source_commit'),
        actual_head_commit=record.get('actual_head_commit'), checkpoint=record.get('checkpoint', record.get('checkpoint_loading')),
        packed_manifest=record.get('packing', {}).get('manifest'), error=record.get('error'),
        timing_issues=issues)


def comparison(rows, job_id):
    baseline, candidate = rows
    reasons = [f"{row['key']}: {issue}" for row in rows for issue in row['timing_issues']]
    if len(baseline['gpu_uuids']) != 1 or baseline['gpu_uuids'] != candidate['gpu_uuids']:
        reasons.append('Timing rows do not yet identify the same physical GPU')
    for name in ('base_revision',):
        if not baseline[name] or baseline[name] != candidate[name]: reasons.append(f'Missing or different {name}')
    hashes = [(row.get('evaluation_tokens') or {}).get('sha256') for row in rows]
    shapes = [(row.get('evaluation_tokens') or {}).get('shape') for row in rows]
    if not hashes[0] or hashes[0] != hashes[1] or shapes != [[141, 2048], [141, 2048]]:
        reasons.append('Missing or different frozen full141 evaluation-token identity')
    ratios = None
    if not reasons:
        b, c = baseline['measurements'], candidate['measurements']
        ratios = dict(prefill=b['prefill']['latency_ms']/c['prefill']['latency_ms'],
                      decode=c['decode']['tokens_per_second']/b['decode']['tokens_per_second'])
    return dict(schema_version=1, model=MODEL, seed=0, job_id=job_id,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows,
        timing_comparison_status='COMPLETE' if ratios else 'UNAVAILABLE', timing_comparison_reasons=reasons,
        speedup_vs_fp16=ratios, numerical_execution_gate=False,
        interpretation='Whole backend elapsed-time comparison; differing Torch/Transformers, attention and KV implementations; not pure-format overhead or native INT4 acceleration',
        speedup_rule='Only completed matched timing evidence determines ratios; PPL values do not veto timing or ratios')


def fmt(value, digits=3):
    return f'{value:.{digits}f}' if finite(value) else '—'


def escape(value):
    return str(value).replace('|', '/').replace('\n', ' ')


def render(report):
    title = f'**{MODEL} — seed0; FP16 versus PQ k=8 GPTQ W4 / asymmetric g128 A4 / simulated KV4; job {report["job_id"]}**'
    headers = ['Backend', 'Actual format', 'Full141 PPL', 'Prefill ms', 'Decode tokens/s',
               'Peak allocated GiB (prefill/decode)', 'Speedup × (prefill/decode)', 'Status']
    table = [title, '', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |']
    for row in report['rows']:
        prefill, decode = (row['measurements'][x] for x in ('prefill', 'decode'))
        ratio = report['speedup_vs_fp16']
        speed = '—' if ratio is None else ('1.000 / 1.000' if row['key'] == 'fp16'
            else f"{fmt(ratio['prefill'], 5)} / {fmt(ratio['decode'], 5)}")
        values = [row['label'], row['format'], fmt(row['ppl']['value'], 6), fmt(prefill['latency_ms']),
            fmt(decode['tokens_per_second'], 5), f"{fmt(prefill['peak_allocated_gib'])} / {fmt(decode['peak_allocated_gib'])}",
            speed, f"run {row['status']}; PPL {row['ppl']['status']} ({row['ppl']['completed_sequences']}/141); timing {row['timing_status']}"]
        table.append('| ' + ' | '.join(escape(x) for x in values) + ' |')
    notes = [
        'Target workload: B1, prefix2048, decode128 with first8 discarded, token100,10 warmups/20 repeats; synchronized eager execution with last-position logits and real incremental cache.',
        'Fresh-cache boundary: prefill includes cache construction (and native FP16 cache clear); decode cache construction, prefix and discarded steps are outside its timer. Peak memory includes untimed decode preparation. Each row evaluates all141 WT2 windows before timing the same loaded model.',
        'The g128 path expands stored signed4 weights to INT8 GEMM, uses INT32 partials and FP32 affine correction, and stores BF16 KV with attention-time KV4 simulation. It is not native packed INT4 compute or packed KV. FP16 uses its native paged KV and different attention implementation.',
        'PyTorch and Transformers environments differ, as recorded below. Ratios compare complete backends and do not isolate format or rotation overhead. Missing or failed stages remain visible; no PPL tolerance or other numerical quality threshold controls timing.',
    ]
    table.extend(['', *['\n'+x for x in notes]])
    summary = [title, f"Timing comparison: {report['timing_comparison_status']}."]
    if report['speedup_vs_fp16']:
        ratio = report['speedup_vs_fp16']
        summary.append(f"PQ prototype / FP16 speedup: prefill {fmt(ratio['prefill'], 5)}×; decode {fmt(ratio['decode'], 5)}×. Values below1 mean the prototype is slower.")
    else:
        summary.append('No matched speed ratio is reported: ' + '; '.join(report['timing_comparison_reasons']) + '.')
    for row in report['rows']:
        source = Path(row['source']['path'])
        link = f'[{source.name}](../../raw/{source.name})'
        env = '; '.join(f'{name}={row["libraries"].get(name, "pending")}' for name in ('torch', 'transformers', 'triton'))
        detail = (f"{row['label']}: {link}; SHA256 {row['source'].get('sha256', 'pending')}; "
            f"GPU {', '.join(row['gpu_uuids']) or 'pending'}; {env}; CUDA={row['cuda_runtime'] or 'pending'}; "
            f"full141={row['ppl']['full141']}; same loaded object={row['same_loaded_instance']}.")
        table.extend(['', detail]); summary.append(detail)
        errors = [x for x in [row['source'].get('reason'), row.get('error'), row['timing_error'],
            row['timing_reason'], row['ppl'].get('error'),
            row['ppl'].get('reason'), *[m.get('error') or m.get('reason') for m in row['measurements'].values()]] if x]
        if errors: summary.append(f"{row['label']} recorded issues: " + '; '.join(errors))
    ppls = [row['ppl']['value'] for row in report['rows']]
    if all(finite(x) for x in ppls):
        summary.append(f'Candidate − FP16 full141 PPL: {ppls[1]-ppls[0]:+.6f}. This compares the quantized model to FP16, not candidate/reference equivalence.')
    summary.extend(notes)
    return '\n'.join(table)+'\n', '\n\n'.join(summary)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--repo', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if not args.job_id.isdecimal(): parser.error('--job-id must contain only decimal digits')
    names = {'fp16': f'stage_a_8b_fp16_seed0_g128comparison_job{args.job_id}.json',
             'g128': f'integer_8b_nar_k8_seed0_ppl-timing_job{args.job_id}.json'}
    rows = []
    for key, name in names.items():
        record, source = read_record(args.repo/'results/e38/raw'/name)
        rows.append(collect_row(record, source, key, args.job_id))
    report = comparison(rows, args.job_id)
    output = args.repo/'results/e38/g128_vs_fp16'/f'job{args.job_id}'
    table, summary = render(report)
    write(output/'table.md', table)
    write(output/'summary.md', summary)
    write(output/'comparison.json', json.dumps(clean(report), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), status=report['timing_comparison_status'])))
    return 0


if __name__ == '__main__': raise SystemExit(main())
