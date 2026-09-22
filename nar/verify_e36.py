"""Final E36 audit: preservation, pairing, payload invariance and source coverage."""
from pathlib import Path
import json,sys,subprocess
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a
from nar import e36_offset_precision as e


def verify():
    manifest=a.js(a.REPO/'experiments/e36_execution_manifest.json')
    for path,sha in manifest['old_results_sha256'].items():assert a.sha(a.REPO/path)==sha,('changed old result',path)
    old=subprocess.check_output(['git','show',e.PARENT+':report.md'],cwd=a.REPO)
    assert (a.REPO/'report.md').read_bytes().startswith(old)
    cpu=a.js(a.REPO/'experiments/e36_protocol_checks.json')['fixed_tensor_cpu'];records=[]
    figure_data=a.rows(a.REPO/'figures/fig_offset_precision_source.csv')
    for m in e.MODELS:
        e.verify_frozen(m);done=a.js(e.root(m)/'e36_DONE.json');analysis=a.js(e.root(m)/'e36_ANALYSIS_DONE.json')
        assert done['status']==analysis['status']=='COMPLETE' and analysis['source_sha256']==a.sha(a.REPO/'nar/e36_analysis.py')
        gpu=a.js(e.root(m)/'e36_fixed_tensor_gate.json');assert gpu['status']=='PASS' and gpu['input_sha256']==cpu['input_sha256']
        assert all(gpu['hashes'][row]==cpu['hashes'][row] for row in ['Z16','Z32','ZB16'])
        assert gpu['hashes']['S32']['code_sha256']==cpu['hashes']['S32']['code_sha256'] and gpu['hashes']['S32']['offset_sha256']==cpu['hashes']['S32']['offset_sha256']
        cfg,ds=a.config(m);nl=cfg.num_hidden_layers
        rows=a.rows(e.root(m)/'e36_per_sequence.csv');wanted={(method+'_'+row,str(seed),str(c)) for method in e.METHODS for row in e.ROWS for seed in a.SEEDS for c in range(64)}
        actual={(r['row'],r['seed'],r['chunk']) for r in rows};assert actual==wanted and len(actual)==len(rows)==1536
        token_sha=a.sha(a.tokenpath(m,'wt2_eval'))
        assert all(r['token_sha256']==token_sha and r['tokens']=='2047' and np.isfinite(float(r['nll'])) for r in rows)
        frozen={(r['row'],r['seed'],r['chunk']):r for r in a.rows(e.root(m)/'e29_per_sequence.csv') if r['eval_set']=='wt2'}
        for r in rows:
            if r['precision']=='Z16':assert r['nll']==frozen['Q1_'+r['method'],r['seed'],r['chunk']]['nll']
        replay=a.rows(e.root(m)/'e36_baseline_replay.csv');assert len(replay)==384
        assert all(r['float32_bit_identical']=='True' and np.float32(float(r['source_nll'])).tobytes()==np.float32(float(r['replayed_nll'])).tobytes() for r in replay)
        gates=a.rows(e.root(m)/'e36_gates.csv')
        expected_g={(method,row,str(seed),site,str(l)) for method in e.METHODS for row in e.ROWS for seed in a.SEEDS for site in ['qkv','down'] for l in range(nl)}
        actual_g={(r['method'],r['precision'],r['seed'],r['site'],r['layer']) for r in gates};assert actual_g==expected_g and len(actual_g)==len(gates)
        assert all(float(r['round_trip'])<=1e-6 and float(r['anchor_residual'])<=1e-6 for r in gates)
        hashes=a.rows(e.root(m)/'e36_payload_hashes.csv');assert len(hashes)==4*2*3*64*2*nl
        grouped=defaultdict(dict)
        for r in hashes:
            key=r['method'],r['seed'],r['chunk'],r['site'],r['layer'];assert r['precision'] not in grouped[key]
            grouped[key][r['precision']]=r
            assert int(r['code_values'])==2048*ds[r['site']]
        for key,rr in grouped.items():
            assert set(rr)==set(e.ROWS)
            assert len({r['code_sha256'] for r in rr.values()})==1
            assert len({rr[row]['scale_sha256'] for row in ['Z16','Z32','ZB16']})==1
            assert rr['Z16']['offset_sha256']==rr['S32']['offset_sha256']
            for row,r in rr.items():
                assert r['scale_dtype']==('torch.float32' if row=='S32' else 'torch.float16')
                assert r['offset_dtype']=={'Z16':'torch.float16','Z32':'torch.float32','ZB16':'torch.bfloat16','S32':'torch.float16'}[row]
        artifacts=a.rows(e.root(m)/'e36_raw_artifacts.csv');assert len(artifacts)==384
        for r in artifacts:
            cp=e.checkpoint(m,r['method'],int(r['seed']),int(r['chunk']));meta=a.js(cp.with_suffix('.json'))
            assert r['sha256']==meta['raw_sha256'] and int(r['bytes'])==cp.stat().st_size
        flags=a.js(e.root(m)/'e34_flags.json');counts={'all':64*2048,'BOS':64,'massive':132,'BOS+massive':196,'other':64*2048-196}
        stats_rows=a.rows(e.root(m)/'e36_group_statistics.csv');assert len(stats_rows)==nl*2*2*4*3*5
        lookup={}
        for r in stats_rows:
            key=r['method'],r['seed'],r['site'],r['layer'],r['group_class'],r['token_class'];assert key not in lookup;lookup[key]=r
            populated=r['group_class']=='all' or r['group_class']==('anchor' if r['method']=='pq' else 'other')
            multiplier=3 if r['seed']=='pooled' else 1
            expected=counts[r['token_class']]*(ds[r['site']]//128)*multiplier if populated else 0
            assert int(r['group_token_observations'])==expected
            assert int(r['offset_error_steps_defined_count'])==expected and int(r['offset_error_steps_undefined_count'])==0
            if expected:
                assert abs(float(r['fraction_over_half_step'])-int(r['over_half_step_count'])/expected)<1e-15
                assert float(r['offset_error_steps_median'])<=float(r['offset_error_steps_p99'])<=float(r['offset_error_steps_max'])
            else:assert r['offset_error_steps_median']==r['offset_error_steps_p99']==r['fraction_over_half_step']==''
            for metric in ['abs_offset_over_range','offset_error_steps','unaligned_step_steps','error_to_unaligned']:
                assert int(r[metric+'_defined_count'])+int(r[metric+'_undefined_count'])==expected
        for r in [x for x in stats_rows if x['group_class']=='all']:
            counterpart=lookup[r['method'],r['seed'],r['site'],r['layer'],'anchor' if r['method']=='pq' else 'other',r['token_class']]
            assert all(r[field]==counterpart[field] for field in r if field!='group_class')
        overall=a.rows(e.root(m)/'e36_overall.csv');assert len(overall)==2
        for r in overall:
            rr=[x for x in stats_rows if x['method']==r['method'] and x['seed']=='pooled' and x['group_class']==x['token_class']=='all']
            peak=max(rr,key=lambda x:(float(x['offset_error_steps_p99']),-int(x['layer']),x['site']=='qkv'))
            assert r['largest_p99_layer']==peak['layer'] and r['largest_p99_site']==peak['site']
            assert int(r['group_token_observations'])==sum(int(x['group_token_observations']) for x in rr)
            assert int(r['over_half_step_count'])==sum(int(x['over_half_step_count']) for x in rr)
        event_classes=a.rows(e.root(m)/'e36_exceedance_classes.csv');assert len(event_classes)==6
        for r in event_classes:
            selected=[x for x in stats_rows if x['method']==r['method'] and x['seed']=='pooled' and x['group_class']=='all' and x['token_class']==r['token_class']]
            assert int(r['group_token_observations'])==sum(int(x['group_token_observations']) for x in selected)
            assert int(r['over_half_step_count'])==sum(int(x['over_half_step_count']) for x in selected)
        for r in a.rows(e.root(m)/'e36_threshold_layer.csv'):
            assert r==lookup[r['method'],r['seed'],r['site'],r['layer'],r['group_class'],r['token_class']]
        frows=[r for r in figure_data if r['model']==m];assert len(frows)==4*nl
        for r in frows:
            for field,seed in [('pooled_p99','pooled'),('seed0_p99','0'),('seed1_p99','1'),('seed2_p99','2')]:
                expected=lookup[r['method'],seed,r['site'],r['layer'],'all','all']['offset_error_steps_p99']
                assert float(r[field])==float(expected)
        assert len(a.rows(e.root(m)/'e36_summary.csv'))==8
        files={p.name:a.sha(p) for p in sorted(e.root(m).glob('e36_*')) if p.is_file()}
        largest=max((e.root(m)/name).stat().st_size for name in files);assert largest<90*1024*1024
        records.append(dict(model=m,status='PASS',ppl_chunk_records=len(rows),baseline_replayed_chunks=len(replay),payload_hash_records=len(hashes),code_and_Z_scale_hashes_all_identical=True,fp16_and_Z_rows_cpu_gpu_hashes_identical=True,numerical_gate_rows=len(gates),max_round_trip=max(float(r['round_trip']) for r in gates),max_anchor_residual=max(float(r['anchor_residual']) for r in gates),group_statistics_rows=len(stats_rows),raw_group_summary_bytes=sum(int(r['bytes']) for r in artifacts),largest_result_file_bytes=largest,sha256=files))
    qa=a.js(a.REPO/'figures/qa/e36/QA_summary.json');assert qa['status']=='PASS' and qa['visual_review_completed']
    alignment=a.js(a.REPO/'figures/qa/e36/fig_offset_precision.alignment.json');assert alignment['verdict']=='PASS'
    for name,sha in qa['figure_sha256'].items():assert a.sha(a.REPO/name)==sha
    source_files=['nar/e36_analysis.py','nar/e36_report.py','nar/verify_e36.py','figures/plot_e36_offset_precision.py','slurm_e36_analysis.sh']
    result=dict(status='PASS',utc=a.utc(),models=records,old_result_files_preserved=len(manifest['old_results_sha256']),old_report_prefix_preserved=True,figure_rows=len(figure_data),figure_QA='PASS',analysis_source_sha256={p:a.sha(a.REPO/p) for p in source_files})
    a.savej(a.REPO/'experiments/e36_final_verification.json',result)
    print(json.dumps({**result,'models':[{k:v for k,v in r.items() if k!='sha256'} for r in records]},indent=2))
if __name__=='__main__':verify()
