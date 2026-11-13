#!/usr/bin/env python3
"""Release one existing speed job once both reusable 8B artifacts are complete.

No GPU, pickle loading, numerical criterion, polling, or new submission. DuQuant
writes its quantization manifest after the atomic checkpoint rename, while the
overall run status is still RUNNING; its serialized checkpoint identity is the
completion evidence. File sizes and recorded identities are checked here, not
full multi-GB content hashes (the deployment loaders retain their own checks).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/projects/nar/nar-validation')
BASE = WORK/'cache/huggingface/models--unsloth--Meta-Llama-3.1-8B/snapshots/e9a141a2091ea561b96483212645a2a05e6f99fc'
FLAT = WORK/'checkpoints/rebuttal_8b/flatquant_native_seed0'
DU = WORK/'artifacts/e42/duquant_8b_seed0_v1'
JOB = '1736351'
PARENT = '1736252'
MODEL = 'Meta-Llama-3.1-8B'


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def same_path(left, right):
    return bool(left) and Path(left).resolve() == Path(right).resolve()


def file_record(path, expected_size=None):
    require(path.is_file(), f'Missing file: {path}')
    size = path.stat().st_size
    require(size > 0, f'Empty file: {path}')
    if expected_size is not None:
        require(size == expected_size, f'File size does not match manifest: {path}')
    return dict(path=str(path), bytes=size)


def readiness(flat=FLAT, du=DU):
    results = {}
    for name, directory, filename in [('flatquant', flat, 'DONE.json'),
                                      ('duquant', du, 'quantization_manifest.json')]:
        result = results[name] = dict(ready=False, manifest=str(directory/filename))
        try:
            data = json.loads((directory/filename).read_text())
            require(data['model'] == MODEL and data['model_key'] == 'llama31_8b', 'Wrong model')
            if name == 'flatquant':
                require(data['status'] == 'COMPLETE', 'FlatQuant checkpoint is incomplete')
                require(same_path(data['base_checkpoint'], BASE), 'Wrong base checkpoint')
                require(data['official_commit'] == '9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1', 'Wrong FlatQuant source commit')
                args = data['official_args']
                require(same_path(args['model'], BASE) and args['seed'] == 0, 'Wrong FlatQuant model/seed')
                require(all(args[key] == 4 for key in ('w_bits', 'a_bits', 'k_bits', 'v_bits'))
                        and args['gptq'] is False, 'Wrong FlatQuant quantization configuration')
                checkpoint = directory/'quantized_model.pt'
                require(same_path(data['quantized_model'], checkpoint), 'Wrong FlatQuant checkpoint path')
                result['files'] = [file_record(checkpoint, data['checkpoint_bytes']),
                                   file_record(directory/'weight_quantizers.pt')]
                result['completion_evidence'] = 'COMPLETE DONE manifest and atomic checkpoint files'
            else:
                require(data['phase'] == 'quantize-ppl' and data['seed'] == 0, 'Wrong DuQuant phase/seed')
                require(data['official_commit'] == 'd56cfc6fe97c34c0eb100fec82fe439865905679', 'Wrong DuQuant source commit')
                require(data['base_revision'] == BASE.name, 'Wrong DuQuant base revision')
                args = data['official_arguments']
                require(same_path(args['model'], BASE) and args['net'] == MODEL, 'Wrong DuQuant base model')
                require(args['wbits'] == args['abits'] == 4 and args['quant_method'] == 'duquant', 'Wrong DuQuant configuration')
                require(data['checkpoint_save_seconds'] >= 0 and data['quantization_seconds'] >= 0,
                        'DuQuant serialization completion fields are missing')
                result['files'] = []
                for filename, identity in [('quantized_model.pt', data['checkpoint']),
                                           ('Rot.pkl', data['rotation_table']),
                                           ('act_scales.pt', data['smoothing']['scales']),
                                           ('calibration.pt', data['calibration']['artifact'])]:
                    path = directory/filename
                    require(same_path(identity['path'], path), f'Wrong DuQuant file identity: {filename}')
                    require(re.fullmatch(r'[0-9a-f]{64}', identity['sha256']) is not None,
                            f'Missing recorded DuQuant hash: {filename}')
                    result['files'].append(file_record(path, identity['bytes']))
                result['manifest_run_status'] = data['status']
                result['completion_evidence'] = ('Quantization manifest is written after atomic model rename; '
                    'checkpoint identity and save duration are present. Overall status RUNNING is expected '
                    'because PPL follows; no PPL or task criterion is applied.')
            result['ready'] = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            result['reason'] = str(error)
    return dict(ready=all(value['ready'] for value in results.values()), artifacts=results)


def release(record, *, apply=False, caller_job=None, run=subprocess.run):
    """Mutate only the fixed pending job and exact single expected dependency."""
    record['action'] = 'NOT_READY'
    if not record['ready']:
        return
    if not apply:
        record['action'] = 'READY_DRY_RUN'
        return
    if str(caller_job) != PARENT:
        record.update(action='SKIPPED', reason='Caller is not the authorized producer job '+PARENT)
        return
    observed = run(['scontrol', 'show', 'job', JOB, '-o'], check=True,
                   capture_output=True, text=True, timeout=15).stdout
    fields = dict(item.split('=', 1) for item in observed.split() if '=' in item)
    record['scheduler_before'] = fields
    expected = (fields.get('JobId') == JOB and fields.get('JobState') == 'PENDING'
                and fields.get('JobName') == 'pq8b-official-speed'
                and fields.get('UserId', '').endswith(f'({os.getuid()})')
                and fields.get('Command') == str(ROOT/'slurm_rebuttal_speed_8b.sh')
                and re.fullmatch(r'afterany:'+PARENT+r'(?:\((?:unfulfilled|fulfilled)\))?',
                                 fields.get('Dependency', '')) is not None)
    if not expected:
        record.update(action='SKIPPED', reason='Job state, owner, command, or exact dependency has changed')
        return
    record['release_requested_utc'] = now()
    result = run(['scontrol', 'update', f'JobId={JOB}', 'Dependency='], check=True,
                 capture_output=True, text=True, timeout=15)
    record.update(action='RELEASED', released_utc=now(), scheduler_stdout=result.stdout,
                  scheduler_stderr=result.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', action='store_true', help='Clear the fixed dependency if both artifacts are ready')
    parser.add_argument('--trigger', choices=('manual', 'e39_train-ppl', 'e42_quantize-ppl'), default='manual')
    args = parser.parse_args()
    output = ROOT/'results/baselines'
    output.mkdir(parents=True, exist_ok=True)
    lock = output/f'native_speed_release_{JOB}.lock'
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        record = dict(observed_utc=now(), job_id=JOB, producer_job_id=PARENT,
                      caller_job_id=os.getenv('SLURM_JOB_ID'), trigger=args.trigger,
                      numerical_criteria='none', **readiness())
        code = 0
        try:
            release(record, apply=args.release, caller_job=os.getenv('SLURM_JOB_ID'))
        except (OSError, subprocess.SubprocessError) as error:
            record.update(action='ERROR', error=str(error))
            code = 1
        record['finished_utc'] = now()
        history = output/f'native_speed_release_{JOB}.jsonl'
        with history.open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        path = output/f'native_speed_release_{JOB}.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(path)
        print(json.dumps(dict(action=record['action'], ready=record['ready'], record=str(path))))
        return code


if __name__ == '__main__':
    raise SystemExit(main())
