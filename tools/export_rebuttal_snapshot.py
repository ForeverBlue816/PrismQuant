"""Export immutable result/entrypoint snapshots from cluster worktrees.

No model weights, dataset files or historical Git objects are copied. Harness
sample text/responses are omitted; metrics, sequence NLL and source digests remain.
This script requires the cluster worktree paths and is an archival helper.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
STAMP='2026-11-13'
DEST=ROOT/'docs/rebuttal'/STAMP
HOMEDIR=Path('/home/yanlongc')
SOURCES={key:HOMEDIR/f'prismquant-rebuttal-{key}' for key in ('e38','e39','e40','e41','e42')}
manifest=[];mapping={};source_bytes={}

def digest(data):return hashlib.sha256(data).hexdigest()
def collect(path,target):
    data=path.read_bytes()
    source_bytes[str(path.resolve())]=data
    mapping[str(path.resolve())]=target

def trim_samples(value,pointer='',omitted=None):
    if omitted is None:omitted=[]
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            where=pointer+'/'+key
            if key=='samples' and isinstance(item,(dict,list)):
                omitted.append(dict(json_pointer=where,container_items=len(item)))
            else:result[key]=trim_samples(item,where,omitted)
        return result
    if isinstance(value,list):return [trim_samples(x,pointer+'/'+str(i),omitted) for i,x in enumerate(value)]
    return value


def register():
    for experiment,repo in SOURCES.items():
        for path in sorted((repo/'results'/experiment).rglob('*')):
            if path.is_file() and path.suffix in ('.json','.md','.tex','.npz','.csv'):
                collect(path,DEST/'records'/experiment/path.relative_to(repo/'results'/experiment))
    for path in sorted((SOURCES['e38']/'results/baselines').rglob('*')):
        if path.is_file() and path.suffix in ('.json','.md','.tex'):
            collect(path,DEST/'current'/path.relative_to(SOURCES['e38']/'results/baselines'))
    collect(SOURCES['e38']/'REBUTTAL_RESULTS.md',DEST/'history.md')
    protocol=SOURCES['e38']/'experiments/rebuttal_speed_comparison_20261112.json'
    collect(protocol,DEST/'protocol.json')
    patterns={
      'e38':['slurm_rebuttal_baselines_*.sh','slurm_rebuttal_baseline_timing.sh','slurm_rebuttal_speed_8b.sh','slurm_pq_fast_rotation_8b.sh',
        'scripts/rebuttal_baseline_model_worker.sh','scripts/recover_duquant_3b_allocation.sh','scripts/resume_duquant_3b_tasks.sh',
        'scripts/baseline_timing.py','scripts/collect_requested_baselines.py','scripts/collect_native_speed_8b.py','scripts/collect_rebuttal8b.py',
        'quarot-llama3/e38/optimized_rotation.py','quarot-llama3/e38/stage_a_8b.py','quarot-llama3/e38/stage_a_8b_model.py',
        'tests/test_e38_optimized_rotation.py','tests/test_e38_stage_a_cache_timing.py','tests/test_baseline_timing.py'],
      'e39':['nar/e39_flatquant.py','nar/e39_flatquant_native.py','nar/e39_native_cache_optimization.py','slurm_e39*.sh',
        'tests/test_e39*.py'],
      'e42':['nar/e42_duquant_native.py','nar/e42_duquant_compat.py','slurm_e42_duquant_native.sh','tests/test_e42*.py'],
    }
    for key,globs in patterns.items():
        for pattern in globs:
            for path in SOURCES[key].glob(pattern):
                if path.is_file():collect(path,DEST/'entrypoints'/key/path.relative_to(SOURCES[key]))
    native=HOMEDIR/'.prismquant/flatquant-native-tools'
    for name in ('native_runtime_manifest.json','native_gpu_smoke_1736285_source_semantics.json','environment_torch26.sh','README.md'):
        path=native/name
        if path.exists():collect(path,DEST/'entrypoints/flatquant_native_runtime'/name)


def rel(target,parent):return Path(os.path.relpath(target,parent)).as_posix()

def rewrite_markdown(text,source,target):
    # Repair a known report-formatting typo without touching measurement JSON.
    text=re.sub(r'_job (?=\d)', '_job', text)
    pattern=re.compile(r'\[([^\]\n]+)\]\(([^\)\n]+)\)')
    def replace(match):
        label,link=match.groups();original=link.strip('<>')
        if original.startswith(('https://','http://','mailto:','#')):return match.group(0)
        plain,sep,anchor=original.partition('#')
        candidate=Path(plain) if plain.startswith('/') else Path(source).parent/plain
        resolved=str(candidate.resolve())
        mapped=mapping.get(resolved)
        if mapped is not None:return '['+label+']('+rel(mapped,target.parent)+(('#'+anchor) if sep else '')+')'
        # Preserve an explicit pointer to the source inventory for omitted files.
        return '['+label+']('+rel(DEST/'manifest.json',target.parent)+')'
    text=pattern.sub(replace,text)
    is_archived=('/records/' in str(target) and '/rotation_optimization/' not in str(target)) or target.name=='history.md'
    if is_archived:
        text='> Historical record snapshot. Some original job states and plans are stale. Use the [current status index]('+rel(DEST/'README.md',target.parent)+') for the active scope and observed scheduler state.\n\n'+text
    return text


def main():
    register();DEST.mkdir(parents=True,exist_ok=True)
    for source,target in mapping.items():
        data=source_bytes[source];out=data;transform=[];omitted=[]
        if target.suffix=='.json':
            value=json.loads(data)
            compact=trim_samples(value,omitted=omitted)
            if omitted:
                out=(json.dumps(compact,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
                transform.append('omit_harness_sample_payloads; aggregate metrics and sequence NLL unchanged')
        elif target.suffix=='.md':
            out=rewrite_markdown(data.decode(),source,target).encode()
            if out!=data:transform.append('repository-relative links, broken job-name link repair, historical-status banner')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(out)
        manifest.append(dict(source_path=source,source_bytes=len(data),source_sha256=digest(data),
            exported_path=target.relative_to(DEST).as_posix(),exported_bytes=len(out),exported_sha256=digest(out),
            transformations=transform,omitted_sample_payloads=omitted))
    heads={key:subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip() for key,repo in SOURCES.items()}
    snapshot=dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_heads=heads,
        notes=['Raw JSON with no transformations is byte-identical to the source.',
        'Harness sample payloads containing benchmark text/responses are omitted, with source size/SHA retained; numerical aggregates and sequence NLL are unchanged.',
        'Paths inside JSON identify original cluster artifacts; model and dataset binaries are not included.',
        'Entry-point scripts depend on the original cluster worktrees and environments; this is not a standalone reproduction package.',
        'Source worktrees may have uncommitted running records; per-file digests identify the actual snapshot.'],files=manifest)
    (DEST/'manifest.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    print(json.dumps(dict(files=len(manifest),exported_bytes=sum(x['exported_bytes'] for x in manifest),source_bytes=sum(x['source_bytes'] for x in manifest),source_heads=heads),indent=2))

if __name__=='__main__':main()
