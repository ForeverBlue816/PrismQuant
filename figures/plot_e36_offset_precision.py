"""Per-layer tails of measured fp16 offset error, all sites/models/seeds."""
from pathlib import Path
import csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator,NullFormatter,FuncFormatter
from figure_style import configure_style
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'figures/fig_offset_precision'
QA=ROOT/'figures/qa/e36'
MODELS=[('qwen3_4b_base','Qwen3-4B-Base'),('llama32_3b','Llama-3.2-3B')]
STYLE={'pq':('#1D3557','-', 'PrismQuant'),'hadamard':('#56858B',(0,(3.1,1.8)), 'Hadamard')}

def main():
    QA.mkdir(parents=True,exist_ok=True);source=[];expected_layers={}
    for model,name in MODELS:
        done=json.loads((ROOT/'results'/model/'e36_ANALYSIS_DONE.json').read_text());assert done['status']=='COMPLETE'
        rows=list(csv.DictReader((ROOT/'results'/model/'e36_group_statistics.csv').open()))
        selected=[r for r in rows if r['token_class']=='all' and r['group_class']=='all']
        lookup={(r['method'],r['site'],int(r['layer']),r['seed']):r for r in selected}
        layers=sorted({int(r['layer']) for r in selected});assert layers==list(range(len(layers)));expected_layers[model]=len(layers)
        for site in ['qkv','down']:
            for method in STYLE:
                for layer in layers:
                    values=[float(lookup[method,site,layer,str(seed)]['offset_error_steps_p99']) for seed in range(3)]
                    pooled=float(lookup[method,site,layer,'pooled']['offset_error_steps_p99'])
                    source.append(dict(model=model,site=site,method=method,layer=layer,pooled_p99=pooled,seed0_p99=values[0],seed1_p99=values[1],seed2_p99=values[2],seed_min=min(values),seed_max=max(values),source=f'results/{model}/e36_group_statistics.csv'))
    assert len(source)==sum(expected_layers.values())*4
    values=np.array([r[field] for r in source for field in ['pooled_p99','seed_min','seed_max']]);assert np.isfinite(values).all() and np.all(values>=0)
    configure_style();plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':8,'axes.titlesize':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.05,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'E36-offset-precision'})
    fig,axes=plt.subplots(1,4,figsize=(183/25.4,70/25.4),sharey=True)
    fig.subplots_adjust(left=.10,right=.985,bottom=.28,top=.79,wspace=.24)
    positive=bool(np.all(values>0));ymin=float(values.min()*.60) if positive else 0.;ymax=float(max(.5,values.max())*1.45)
    panels=[]
    for index,(ax,(model,site)) in enumerate(zip(axes,[(m,s) for m,_ in MODELS for s in ['qkv','down']])):
        if positive:ax.set_yscale('log')
        else:ax.set_yscale('symlog',linthresh=1e-6)
        ax.set_ylim(ymin,ymax);layers=np.arange(expected_layers[model]);ax.set_xlim(-.6,layers[-1]+.6)
        for method,(color,ls,label) in STYLE.items():
            rr=sorted([r for r in source if r['model']==model and r['site']==site and r['method']==method],key=lambda r:r['layer'])
            center=np.array([r['pooled_p99'] for r in rr]);lo=np.array([r['seed_min'] for r in rr]);hi=np.array([r['seed_max'] for r in rr])
            assert len(rr)==len(layers) and np.all(lo<=hi)
            assert np.all(center>=ymin) and np.all(hi<=ymax) and np.all(lo>=ymin)
            ax.fill_between(layers,lo,hi,color=color,alpha=.14,linewidth=0,zorder=1)
            ax.plot(layers,center,color=color,linestyle=ls,linewidth=1.05,zorder=3)
        ax.axhline(.5,color='#8A9098',ls=(0,(1.5,2.5)),lw=.75,zorder=2)
        ticks=[0,12,24,int(layers[-1])] if len(layers)>30 else [0,9,18,int(layers[-1])]
        ax.set_xticks(ticks);ax.set_xlabel('Layer',labelpad=4)
        ax.set_title('QKV input' if site=='qkv' else 'Down input',pad=7,fontsize=8.5)
        ax.annotate('abcd'[index],(0,1),xycoords='axes fraction',xytext=(-17,7),textcoords='offset points',ha='left',va='bottom',fontsize=9,fontweight='bold',annotation_clip=False)
        ax.grid(axis='y',which='major',color='#E1E6EA',linewidth=.35);ax.set_axisbelow(True)
        ax.tick_params(which='major',width=.55,length=2.5);ax.tick_params(which='minor',width=.35,length=1.3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if positive:ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
        panels.append(dict(panel='abcd'[index],model=model,site=site,layers=len(layers),methods=2,center='exact pooled P99 over all 3 seeds and all token-groups',spread='minimum to maximum of the three seed P99 values; not a CI',excluded_layer_rows=0))
    axes[0].set_ylabel('P99 offset error / step',labelpad=5)
    for start,(_,name) in zip([0,2],MODELS):
        left=axes[start].get_position().x0;right=axes[start+1].get_position().x1
        fig.text((left+right)/2,.95,name,ha='center',va='top',fontsize=10,fontweight='bold',color='#1D3557')
    handles=[Line2D([],[],color=color,ls=ls,lw=1.2,label=name) for color,ls,name in STYLE.values()]
    handles.append(Line2D([],[],color='#8A9098',ls=(0,(1.5,2.5)),lw=.8,label='Half-step threshold'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.015),ncol=3,frameon=False,handlelength=2.2,columnspacing=2.2)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=QA/'fig_offset_precision.alignment.json',overlay_svg=QA/'fig_offset_precision.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    with OUT.with_name(OUT.name+'_source.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(source[0]));writer.writeheader();writer.writerows(source)
    fig.savefig(OUT.with_suffix('.pdf'),metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(OUT.with_suffix('.svg'),metadata={'Date':None})
    svg=OUT.with_suffix('.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(OUT.with_suffix('.png'),dpi=600)
    (QA/'source_data.json').write_text(json.dumps(dict(source_rows=len(source),plotted_rows=len(source),excluded_rows=0,panels=panels,axis_scale='log' if positive else 'symlog',linthresh=None if positive else 1e-6,y_limits=[ymin,ymax],width_mm=183,height_mm=70,style_reuse='paper palette; publication-safe sans-serif typography; new statistical chart',zero_denominator_policy='documented in source statistics; P99 offset/step uses actual positive guarded steps'),indent=2)+'\n')
    caption='''**Figure E36 | fp16 offset rounding across model layers.** Panels a–b show Qwen3-4B-Base and c–d show Llama-3.2-3B; each pair separates post-RMSNorm qkv inputs from down-projection inputs. Lines show the exact pooled 99th percentile of |z−fp16(z)|/s for PrismQuant and Hadamard, measured during their Z16 passes on all 64 frozen 2048-token evaluation chunks with three paired rotation seeds, asymmetric group size 128 and k=max. Here z is the original fp32 group minimum and s is the actual guarded fp16 step. Bands span the three individual seed P99 values and are descriptive seed envelopes, not confidence intervals. The dotted reference marks half a quantization step. All layers, sites and evaluation input positions are retained; the scale transformation and limits are recorded with the source data. Per-layer P99 does not bound rare input-class maxima; BOS/massive tails and half-step exceedance counts are reported separately in the accompanying tables. The offset/step tail diagnoses storage error, while the accompanying fixed-code PPL sweep tests its loss consequence. A small PPL effect does not imply mathematically lossless offset storage.\n'''
    OUT.with_name(OUT.name+'_caption.md').write_text(caption)
    plt.close(fig)
if __name__=='__main__':main()
