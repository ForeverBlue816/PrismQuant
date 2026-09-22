"""Render the completed E36 tables and decision-conditioned interpretation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a
from nar import e36_offset_precision as e
from nar.reviewer_report import mdtable

NAMES={'qwen3_4b_base':'Qwen3-4B-Base','llama32_3b':'Llama-3.2-3B'}
METHOD={'pq':'PrismQuant','hadamard':'Hadamard'}
def number(x):return 'N/A' if x is None or x=='' else f'{float(x):.6g}'
def percent(x):return 'N/A' if x is None or x=='' else f'{100*float(x):.6g}%'
def ci(r):return f"{number(r['delta'])} [{number(r['ci90_low'])}, {number(r['ci90_high'])}]"
def outcome(x):return 'supported' if x else 'not supported'

def report():
    lines=['The complete fixed grid comprises eight precision/method rows per model, three paired seeds and 64 frozen 2048-token chunks (3,072 chunk evaluations across both models). Z16 is the original activation-only fp16-metadata pipeline. Each Z16 chunk must reproduce its frozen E29 fp32 NLL bit for bit before its precision controls run.',
    '**Scope of the intervention.** All four rows use the same saved Z16 codes at every layer/site. Z32 and ZB16 also use exactly the same stored scale. Only the reconstruction metadata changes; S32 is the explicit scale-changing control. This satisfies the requested across-row hash gate. A dynamically re-encoded precision sweep would change codes and downstream quantizer inputs, so these results are a fixed-code storage diagnostic, not an estimate for that different experiment. Residual connections and unquantized model operations still execute normally.',
    'Z16 stores a 16-bit scale and real offset, giving 4.25 effective bits/value at g=128. ZB16 keeps that bit width but changes offset precision to the 8-bit bf16 significand (7 explicit fraction bits). Z32 and S32 are diagnostics excluded from default bit-accounted comparisons; their nominal metadata storage would give 4.375 bits/value. All model weights and KV remain bf16.',
    'PPL is the mean of three seed-level corpus PPLs. The displayed paired 90% intervals retain E29’s 3×64 delta-method convention (t191; the same texts recur across seeds); the CSVs additionally report paired three-seed intervals (t2). “Within seed noise” uses the fixed pre-registered baseline SD, not a formal equivalence test. Group statistics use every Z16 input position, including the final unscored predictor, and exact pooled empirical quantiles across seeds, without subsampling.']
    for m in e.MODELS:
        assert a.js(e.root(m)/'e36_ANALYSIS_DONE.json')['status']=='COMPLETE'
        summary=a.rows(e.root(m)/'e36_summary.csv');lookup={r['row']:r for r in summary};h=a.js(e.root(m)/'e36_hypotheses.json');overall=a.rows(e.root(m)/'e36_overall.csv');worst=a.rows(e.root(m)/'e36_worst_layer.csv');flags=a.js(e.root(m)/'e34_flags.json')
        lines += [f'### {NAMES[m]}','**A. Metadata precision at fixed codes**',mdtable(['Method','Precision','PPL','Seed SD','Δ vs method Z16 [90% paired CI]'],[[METHOD[r['method']],r['precision'],number(r['mean_ppl']),number(r['seed_std']),ci(r)] for r in summary])]
        z32=lookup['pq_Z32'];low,high=float(z32['ci90_low']),float(z32['ci90_high'])
        if low>0 or high<0:
            direction='positive' if low>0 else 'negative'
            lines.append(f"The PQ Z32−Z16 paired interval excludes zero on the {direction} side ({ci(z32)}), even though the separate baseline-SD rule may classify the change as within seed noise. Thus that operational H36a rule must not be paraphrased as no detectable change. Z32 retains codes selected using rounded Z16 metadata; a finer reconstruction offset is not guaranteed to improve those fixed codes or their PPL.")
        lines += ['**B. Where fp16 offset rounding lives**',mdtable(['Method','Token-group observations','Error > 0.5 step, overall','Worst layer/site fraction','Median error / unaligned-step comparison'],[[METHOD[r['method']],r['group_token_observations'],percent(r['fraction_over_half_step']),f"{percent(r['worst_layer_fraction'])} (layer {r['worst_fraction_layer']}, {r['worst_fraction_site']})",number(r['error_to_unaligned_median'])] for r in overall])]
        peak=[r for r in worst if r['token_class']=='all']
        lines.append('The following is the layer/site with the **largest pooled P99 offset error in step units** for each method; layer indices are zero based. This need not be the layer/site with the largest threshold-exceedance fraction above.')
        lines.append(mdtable(['Method','Layer/site','|z| / range: median / P99 / max','Offset error / step: median / P99','Fraction > 0.5 step','(2|c|/15) / step: median / P99'],[[METHOD[r['method']],f"{r['layer']} / {r['site']}", ' / '.join(number(r['abs_offset_over_range_'+f]) for f in ['median','p99','max']), ' / '.join(number(r['offset_error_steps_'+f]) for f in ['median','p99']),percent(r['fraction_over_half_step']),' / '.join(number(r['unaligned_step_steps_'+f]) for f in ['median','p99'])] for r in peak]))
        full=a.rows(e.root(m)/'e36_group_statistics.csv');threshold_rows=[]
        for item in overall:
            if int(item['over_half_step_count'])>0:
                threshold_rows += [r for r in full if r['method']==item['method'] and r['seed']=='pooled' and r['site']==item['worst_fraction_site'] and r['layer']==item['worst_fraction_layer'] and r['group_class']=='all' and r['token_class'] in ['all','BOS','massive','other']]
        class_totals=[]
        for method in e.METHODS:
            for token_class in ['BOS','massive','other']:
                selected=[r for r in full if r['method']==method and r['seed']=='pooled' and r['group_class']=='all' and r['token_class']==token_class]
                count=sum(int(r['group_token_observations']) for r in selected);events=sum(int(r['over_half_step_count']) for r in selected)
                class_totals.append(dict(model=m,method=method,token_class=token_class,group_token_observations=count,over_half_step_count=events,fraction_over_half_step=events/count))
        a.write(e.root(m)/'e36_exceedance_classes.csv',class_totals)
        for method in e.METHODS:
            selected=[r for r in class_totals if r['method']==method]
            if sum(r['over_half_step_count'] for r in selected):
                lines.append(METHOD[method]+' half-step exceedances over all layers/sites split as '+', '.join(f"{r['token_class']}={r['over_half_step_count']}" for r in selected)+f"; [class counts and denominators](results/{m}/e36_exceedance_classes.csv) are retained.")
        if threshold_rows:
            a.write(e.root(m)/'e36_threshold_layer.csv',threshold_rows)
            lines.append('Rare half-step exceedances can be missed by the all-token P99. The following disjoint input-class breakdown uses each method’s predeclared maximum-exceedance layer/site when any exceedances exist; the all row is a total, not an additional disjoint class.')
            lines.append(mdtable(['Method / layer / site','Input class','Token-group count','Count > 0.5 step','Fraction > 0.5 step','Maximum error / step'],[[f"{METHOD[r['method']]} / {r['layer']} / {r['site']}",r['token_class'],r['group_token_observations'],r['over_half_step_count'],percent(r['fraction_over_half_step']),number(r['offset_error_steps_max'])] for r in threshold_rows]))
        rare=[r for r in worst if r['token_class'] in ['BOS','massive','BOS+massive']]
        lines.append(mdtable(['Method at its P99-max layer','Input class','Token-group count','|z| / range: median / P99','Offset error / step: median / P99','Fraction > 0.5 step'],[[METHOD[r['method']],r['token_class'],r['group_token_observations'],' / '.join(number(r['abs_offset_over_range_'+f]) for f in ['median','p99']),' / '.join(number(r['offset_error_steps_'+f]) for f in ['median','p99']),percent(r['fraction_over_half_step'])] for r in rare]))
        lines.append(f"BOS and massive positions are the frozen E34 **input** classes: {flags['bos_count']} BOS and {flags['massive_count']} massive positions, combined fraction {percent(flags['flagged_fraction'])}, selected from residual norms at E34 layer {flags['selected_layer']}. Token-group counts include three repeated rotation seeds and the number of groups, not independent text samples. The full CSV retains every layer/site/seed, all/anchor/other groups and all/BOS/massive/BOS+massive/other input classes. At k=max, every PQ group is anchored and Hadamard has no aligned anchor groups; empty strata are N/A with zero count.")
        lines.append('Here c is the measured rotated group mean: the constant aligned Walsh level for PQ, and only an incidental group-DC comparison for Hadamard. The 2|c|/15 quantity describes the isolated sign-changing level’s step cost; it is not a claim that the extrema of level plus residual add linearly. The reported median ratio is the median of per-group ratios, not a ratio of medians. Zero-denominator and infinite-value counts remain in the full tables; no epsilon clipping or silent finite-only filtering is applied.')
        ca=next(r for r in h['H36c'] if r['method']=='pq');cb=next(r for r in h['H36b'] if r['method']=='pq');hb_had=next(r for r in h['H36b'] if r['method']=='hadamard')
        lines += ['**Pre-registered decisions**',mdtable(['Hypothesis','Outcome'],[
            ['H36a',f"{outcome(h['H36a']['supported'])}; PQ Z32−Z16={number(h['H36a']['delta'])}, Z16 seed SD={number(h['H36a']['Z16_seed_sd'])}"],
            ['H36b',f"PQ: {outcome(cb['supported'])}, {ci(cb)}; Hadamard control: {outcome(hb_had['supported'])}, {ci(hb_had)}"],
            ['H36c',f"PQ: {outcome(ca['supported'])}; overall <1%: {ca['overall_below_one_percent']}; worst layer <5%: {ca['worst_layer_below_five_percent']}; median ratio <0.01: {ca['median_ratio_below_one_percent']}"],
            ['H36d',f"{outcome(h['H36d']['supported'])}; Had−PQ precision-effect contrast {ci(h['H36d'])}; baseline paired seed SD={number(h['H36d']['baseline_paired_seed_sd'])}"]])]
        interaction=h['H36d']
        if interaction['ci90_low']>0 or interaction['ci90_high']<0:
            lines.append(f"The method-interaction paired-chunk interval also excludes zero ({ci(interaction)}), despite satisfying the separate baseline-SD margin. Its three-seed interval is [{number(interaction['seed_ci90_low'])}, {number(interaction['seed_ci90_high'])}]. These uncertainty views do not establish identical precision responses between methods.")
        pqpeak=next(r for r in peak if r['method']=='pq')
        lines.append(f"**Reading.** PQ’s Z32−Z16 change is {number(h['H36a']['delta'])} PPL, so the fixed baseline-seed-noise criterion is {outcome(h['H36a']['supported'])}. The bf16-offset stress test is {outcome(cb['supported'])} for PQ; its paired interval is {ci(cb)}, which limits what can be claimed about sensitivity. The largest PQ P99 offset error is {number(pqpeak['offset_error_steps_p99'])} steps at layer {pqpeak['layer']} ({pqpeak['site']}), while the overall >half-step fraction is {percent(ca['fraction_over_half_step'])}. The median per-group rounding-error/unaligned-step ratio is {number(ca['error_to_unaligned_median'])}; all three H36c clauses are reported rather than inferred from this one number. The comparison of PQ and Hadamard precision effects is {outcome(h['H36d']['supported'])} under H36d’s fixed noise reference. Small or undetected PPL changes in this frozen-payload experiment do not establish mathematically lossless storage and do not substitute for a dynamically re-encoded precision sweep.")
        prefix=f'results/{m}/'
        lines.append('Source tables: '+', '.join(f'[{label}]({prefix}{name})' for label,name in [('PPL sweep','e36_summary.csv'),('all group statistics','e36_group_statistics.csv'),('model-wide checks','e36_overall.csv'),('P99-max layers and token classes','e36_worst_layer.csv'),('method precision contrast','e36_method_precision_contrast.csv'),('hypotheses','e36_hypotheses.json'),('payload hashes','e36_payload_hashes.csv'),('E29 replays','e36_baseline_replay.csv')])+'.')
    lines += ['### Per-layer precision figure','[PDF](figures/fig_offset_precision.pdf), [editable SVG](figures/fig_offset_precision.svg), [PNG](figures/fig_offset_precision.png), [source data](figures/fig_offset_precision_source.csv), and [caption](figures/fig_offset_precision_caption.md). The curve is the exact pooled P99; its envelope spans the three individual seed P99s and is not a confidence interval. All layers and both activation sites are shown for both models.',
    'The [final audit](experiments/e36_final_verification.json) checks baseline bit identity, actual code/scale hashes, numerical gates, frozen inputs, full row coverage and preservation of old results. The [run manifest](experiments/e36_run_manifest.json) and [preregistration](experiments/e36_preregistration.md) preserve execution and decision provenance.']
    p=a.REPO/'report.md';s=p.read_text();start='<!-- E36 RESULTS START -->';end='<!-- E36 RESULTS END -->';block=start+'\n\n'+'\n\n'.join(lines)+'\n\n'+end
    if start in s:s=s[:s.index(start)]+block+s[s.index(end)+len(end):]
    else:s=s.rstrip()+'\n\n'+block+'\n'
    s=s.replace('Status: PRE-REGISTERED; no E36 model forward has run.','Status: all measurements and exact distribution analyses complete; fixed hypotheses retained.')
    p.write_text(s.rstrip()+'\n')
if __name__=='__main__':report()
