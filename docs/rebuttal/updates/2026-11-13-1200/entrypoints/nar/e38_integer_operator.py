"""Real-checkpoint E38 integer arithmetic diagnostic, seed0 only.

Weights are stored as signed INT4 nibbles and reconstructed FP32 row scales.
The execution prototype expands one 128-wide group to INT8 and calls
``torch._int_mm`` (INT32 accumulation); it is NOT a native INT4 Tensor Core
kernel or a deployment-speed claim. E14 FP16 activation metadata are retained.
BF16 per-element reconstruction residuals are deliberately measured, not hidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import torch
from torch import nn
from . import experiment as base
from . import e38_anchor as provenance

GROUP = 128
REPO = Path(__file__).resolve().parents[1]
WORK = Path('/projects/nar/nar-validation')
CAPTURE_ROOT = WORK / 'artifacts/e38/checkpoint-3b-kmax-job1732989'
SOURCE_FILES = ('nar/e38_integer_operator.py', 'nar/e38_anchor.py',
                'nar/experiment.py', 'nar/e14_w4a4kv4.py',
                'nar/e12_wy.py', 'nar/activation_experiments.py',
                'nar/extended_experiment.py', 'nar/quarot_gptq.py')


def pack_signed4(codes: torch.Tensor) -> torch.Tensor:
    if codes.ndim != 2 or codes.dtype != torch.int8 or codes.shape[1] % 2:
        raise ValueError('signed codes require a 2D int8 matrix of even width')
    if bool(((codes < -8) | (codes > 7)).any()):
        raise ValueError('signed nibble codes outside [-8, 7]')
    unsigned = codes.to(torch.uint8) & 15
    return (unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)).contiguous()


def unpack_signed4(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError('packed weights require a 2D uint8 matrix')
    values = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1).to(torch.int8)
    return torch.where(values >= 8, values - 16, values).contiguous()


def quantize_activation(value: torch.Tensor):
    """Return original-E14 QDQ and actual uint4 codes/FP16 metadata.

    Preserve input dtype (including the BF16 cast before quantization). The
    integer expression intentionally does not add BF16 reconstruction residuals.
    """
    if value.shape[-1] % GROUP or not value.is_floating_point():
        raise ValueError('activation width must be divisible by 128')
    qdq, scale, offset, codes = base.dynamic_asym_int4(value, GROUP)
    return qdq, codes.reshape(-1, value.shape[-1] // GROUP, GROUP), scale.reshape(-1, value.shape[-1] // GROUP), offset.reshape(-1, value.shape[-1] // GROUP)


class PackedSymmetric4(nn.Module):
    """One audited per-output-row symmetric weight matrix, without float W.

    ``from_weight(weight, recovered_scale)`` checks every BF16 reconstruction.
    ``forward(x)`` expects post-rotation, PRE-quantization floating inputs and
    returns the original input dtype. It applies E14 activation quantization.
    This module has no rotations, KV implementation, bias or timing machinery.
    """
    def __init__(self, packed: torch.Tensor, scale: torch.Tensor, group_sums: torch.Tensor):
        super().__init__()
        self.register_buffer('packed', packed)
        self.register_buffer('scale', scale)
        self.register_buffer('group_sums', group_sums)
        self.in_features = packed.shape[1] * 2
        self.out_features = packed.shape[0]
        self.bias = None

    @classmethod
    def from_weight(cls, weight: torch.Tensor, recovered_scale: torch.Tensor):
        if weight.ndim != 2 or weight.dtype != torch.bfloat16 or weight.shape[1] % GROUP:
            raise ValueError('expected the saved BF16 matrix with width divisible by 128')
        scale = recovered_scale.detach().to(device=weight.device, dtype=torch.float32).flatten()
        if scale.shape != (weight.shape[0],) or not bool(torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError('recovered positive finite FP32 row scales required')
        raw_codes = torch.round(weight.float() / scale[:, None])
        if not bool(((raw_codes >= -8) & (raw_codes <= 7)).all()):
            raise ValueError('recovered row scale yields out-of-range integer codes')
        codes = raw_codes.to(torch.int8)
        reconstructed = (codes.float() * scale[:, None]).to(torch.bfloat16)
        different = int((reconstructed != weight).sum())
        if different:
            raise ValueError(f'recovered representation changes {different} saved BF16 values')
        packed = pack_signed4(codes)
        if not torch.equal(unpack_signed4(packed), codes):
            raise RuntimeError('signed nibble packing failed byte-exact roundtrip')
        sums = codes.reshape(weight.shape[0], -1, GROUP).sum(-1, dtype=torch.int32)
        module = cls(packed, scale, sums)
        report = dict(shape=list(weight.shape), original_dtype=str(weight.dtype),
                      integer_code_min=int(codes.min()), integer_code_max=int(codes.max()),
                      bf16_reconstruction_unequal_elements=different,
                      pack_unpack_identity=True,
                      fp32_unrounded_weight_max_abs_error=float((raw_codes * scale[:, None] - weight.float()).abs().max()),
                      packed_bytes=packed.numel(), scale_bytes=scale.numel()*4,
                      group_sum_int32_bytes=sums.numel()*4,
                      metadata_origin='recovered from saved BF16 weights; original GPTQ codes/scales were not stored',
                      compute='expand signed4 nibbles to int8 groups; torch._int_mm -> int32; fp32 affine correction')
        return module, report

    def decode(self, dtype=torch.bfloat16):
        return (unpack_signed4(self.packed).float() * self.scale[:, None]).to(dtype)

    def integer_from_metadata(self, codes: torch.Tensor, scale: torch.Tensor,
                              offset: torch.Tensor) -> torch.Tensor:
        """2D output in FP32; explicit group sums implement the offset term."""
        m, groups, width = codes.shape
        if width != GROUP or groups * GROUP != self.in_features:
            raise ValueError('activation/weight group shape mismatch')
        if codes.device != self.packed.device:
            raise ValueError('activation and packed weights must share a device')
        # CUDA int8 GEMM alignment requirements are satisfied by padding only M;
        # the padding is excluded before scaling and has no numerical effect.
        padded_m = max(32, (m + 31) // 32 * 32)
        if self.out_features % 8:
            raise ValueError('prototype requires out_features divisible by 8')
        result = torch.zeros((m, self.out_features), dtype=torch.float32, device=codes.device)
        for group in range(groups):
            a = torch.zeros((padded_m, GROUP), dtype=torch.int8, device=codes.device)
            a[:m].copy_(codes[:, group])
            w = unpack_signed4(self.packed[:, group*(GROUP//2):(group+1)*(GROUP//2)])
            product = torch._int_mm(a, w.T.contiguous())[:m]
            if product.dtype != torch.int32:
                raise RuntimeError('integer GEMM did not return INT32')
            sa = scale[:, group].float()[:, None]
            oa = offset[:, group].float()[:, None]
            sw = self.scale[None, :]
            term = product.float() * sa * sw
            term.add_(oa * sw * self.group_sums[:, group].float()[None, :])
            result.add_(term)
        return result

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _, codes, scale, offset = quantize_activation(value)
        output = self.integer_from_metadata(codes, scale, offset)
        return output.to(value.dtype).reshape(*value.shape[:-1], self.out_features)


def difference(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    a, r = actual.float(), reference.float()
    finite = bool(torch.isfinite(a).all() & torch.isfinite(r).all())
    if not finite:
        return dict(finite=False, nonfinite_actual=int((~torch.isfinite(a)).sum()),
                    nonfinite_reference=int((~torch.isfinite(r)).sum()))
    delta = a - r
    return dict(finite=True, max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm() / r.norm().clamp_min(1e-30)),
                mean_abs=float(delta.abs().mean()), unequal_elements=int((a != r).sum()),
                elements=a.numel())


def verify_sources(commit):
    pin = provenance.git('rev-parse', commit)
    hashes = {}
    for relative in SOURCE_FILES:
        content = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{pin}:{relative}'])
        actual = provenance.sha(REPO / relative)
        if hashlib.sha256(content).hexdigest() != actual:
            raise RuntimeError(f'source differs from pinned commit: {relative}')
        hashes[relative] = actual
    return pin, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--capture-root', type=Path, default=CAPTURE_ROOT)
    parser.add_argument('--expected-source-commit', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(dict(status='STARTING', started_utc=provenance.now()), f)
    record = dict(kind='real_input_packed_integer_operator_diagnostic', status='RUNNING',
                  started_utc=provenance.now(), command_line=list(sys.orig_argv), seed=0,
                  model='llama32_3b', rotation='nar_kmax', weight_protocol='default',
                  hostname=platform.node(), python=sys.version, rows=[],
                  scope='196 real calibration-input sites; not a full-model gate or runtime benchmark',
                  implementation='4-bit weight storage; INT8 GEMM with INT32 partial sums; FP32 correction; BF16 output',
                  native_int4_tensorcore=False, slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    code = 1
    try:
        pin, hashes = verify_sources(args.expected_source_commit)
        record.update(expected_source_commit=pin, source_sha256=hashes, libraries=provenance.packages())
        probe_file = args.capture_root / 'run.json'
        probe = json.loads(probe_file.read_text())
        record['input_probe'] = dict(path=str(probe_file), sha256=provenance.sha(probe_file),
                                     source_commit=probe['expected_source_commit'])
        if probe['model'] != 'llama32_3b' or probe['rotation'] != 'nar_kmax' or probe['seed'] != 0:
            raise ValueError('this diagnostic is pinned to the 3B kmax seed0 probe')
        record['input_artifacts'] = []
        for key in ['captured_inputs', 'weight_grid_candidate_file']:
            item = probe[key]
            if provenance.sha(item['path']) != item['sha256']:
                raise RuntimeError(f'input artifact hash mismatch: {key}')
            record['input_artifacts'].append(item)
        captures = torch.load(probe['captured_inputs']['path'], map_location='cpu', weights_only=True)
        candidates = torch.load(probe['weight_grid_candidate_file']['path'], map_location='cpu', weights_only=True)
        if len(captures['values']) != 196 or set(candidates) != set(captures['values']):
            raise RuntimeError('expected 196 matching captured inputs and scale candidates')
        record['capture_semantics'] = captures['input_semantics']
        record['capture_positions'] = captures['token_positions']
        record['samples_per_site'] = 8
        torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
        if args.preflight_only:
            record.update(status='PREFLIGHT_PASS', cuda_required_for_full_run=True)
            code = 0
        else:
            if not torch.cuda.is_available():
                raise RuntimeError('GPU allocation required for the real integer operator diagnostic')
            from transformers import AutoConfig
            from . import e14_w4a4kv4 as e14
            base.seed_everything(0)
            torch.backends.cuda.matmul.allow_tf32 = False
            record['hardware'] = base.hardware_info()
            record['cuda'] = torch.version.cuda
            record['rotation_files'] = []
            for item in probe['rotation_files']:
                if provenance.sha(item['path']) != item['sha256']:
                    raise RuntimeError(f'rotation changed: {item["path"]}')
                record['rotation_files'].append(item)
            config_item = next(x for x in probe['model_files'] if Path(x['path']).name=='config.json')
            config_path = Path(config_item['path'])
            if provenance.sha(config_path) != config_item['sha256']:
                raise RuntimeError('base-model config differs from the captured-input probe')
            record['model_config'] = config_item
            config = AutoConfig.from_pretrained(config_path.parent, local_files_only=True)
            rotations = e14.RotationSet(WORK, 'llama32_3b', 'nar_kmax', 0, config, torch.device('cuda'))
            record['checkpoint_files'] = []
            with torch.inference_mode():
                for layer, item in enumerate(probe['checkpoint_files']):
                    if provenance.sha(item['path']) != item['sha256']:
                        raise RuntimeError(f'checkpoint changed: {item["path"]}')
                    weights = torch.load(item['path'], map_location='cpu', weights_only=True)
                    record['checkpoint_files'].append(item)
                    for name, cpu_weight in weights.items():
                        key = f'layer_{layer:02d}.{name}'
                        weight = cpu_weight.cuda()
                        packed, export = PackedSymmetric4.from_weight(weight, candidates[key]['fp32_scale'])
                        captured_raw = captures['values'][key]['raw_input_before_runtime'].cuda()
                        captured_linear = captures['values'][key]['linear_input_after_runtime'].cuda()
                        raw = rotations.apply('r4', layer, captured_raw).to(captured_raw.dtype) if name=='mlp.down_proj' else captured_raw
                        qdq, ac, sa, oa = quantize_activation(raw)
                        target = torch.nn.functional.linear(captured_linear, weight)
                        own_reference = torch.nn.functional.linear(qdq, weight)
                        reconstructed_weight = packed.decode()
                        decoded_control = torch.nn.functional.linear(qdq, reconstructed_weight)
                        integer = packed.integer_from_metadata(ac, sa, oa).to(raw.dtype)
                        row = dict(site=key, export=export,
                                   activation_qdq_vs_original_capture=difference(qdq, captured_linear),
                                   rotation_replayed_on_eight_tokens=(name=='mlp.down_proj'),
                                   packed_bf16_decode_control_vs_same_input_reference=difference(decoded_control, own_reference),
                                   integer_vs_same_input_e14_reference=difference(integer, own_reference),
                                   integer_vs_original_captured_reference=difference(integer, target),
                                   input_dtype=str(raw.dtype), activation_scale_dtype=str(sa.dtype), activation_offset_dtype=str(oa.dtype),
                                   integer_codes_range=[int(ac.min()), int(ac.max())])
                        if row['packed_bf16_decode_control_vs_same_input_reference']['unequal_elements'] != 0:
                            raise RuntimeError(f'packed BF16 decode control is not exact at {key}')
                        if not row['integer_vs_same_input_e14_reference']['finite']:
                            raise RuntimeError(f'nonfinite integer operator output at {key}')
                        record['rows'].append(row)
                        del weight, reconstructed_weight, packed
                    record['completed_layers'] = layer + 1
                    provenance.write(args.output, record)
                    print(f'integer operator layer {layer + 1}/28', flush=True)
            rows = record['rows']
            record['summary'] = dict(sites=len(rows),
                all_packed_weight_bf16_reconstructions_exact=all(x['export']['bf16_reconstruction_unequal_elements']==0 for x in rows),
                all_decoded_control_outputs_exact=all(x['packed_bf16_decode_control_vs_same_input_reference']['unequal_elements']==0 for x in rows),
                all_integer_outputs_finite=all(x['integer_vs_same_input_e14_reference']['finite'] for x in rows),
                maximum_integer_vs_same_input_relative_l2=max(x['integer_vs_same_input_e14_reference']['relative_l2'] for x in rows),
                maximum_integer_vs_same_input_max_abs=max(x['integer_vs_same_input_e14_reference']['max_abs'] for x in rows),
                sites_with_capture_qdq_mismatch=sum(x['activation_qdq_vs_original_capture']['unequal_elements'] != 0 for x in rows),
                full_model_equivalence='not established', runtime='not measured')
            record.update(status='COMPLETE', peak_allocated_bytes=torch.cuda.max_memory_allocated())
            code = 0
    except BaseException as exc:
        record.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        record['ended_utc'] = provenance.now()
        provenance.write(args.output, record)
        print(json.dumps(dict(status=record['status'], output=str(args.output))), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
