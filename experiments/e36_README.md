# E36 reproduction and interpretation

E36 extends the completed E34 commit `025c063`. Read `e36_preregistration.md` first: the requested code/scale invariance makes this a **fixed-code metadata reconstruction experiment**. Z16 runs the original dynamic quantizer; Z32/ZB16/S32 replay its stored activation payload at every quantized site. They do not dynamically re-encode changed upstream states.

## Reproduce

Use the same cached model/data assets and environment as E29–E34. No old experiment asset or result is rewritten.

```bash
python -m unittest discover -s tests -p 'test_e36*.py'
python nar/e36_offset_precision.py --fixed-tensor
python nar/e36_offset_precision.py --freeze
sbatch --qos=rose slurm_e36.sh qwen3_4b_base
sbatch --qos=rose slurm_e36.sh llama32_3b
# After each model's e36_DONE.json exists:
sbatch slurm_e36_analysis.sh qwen3_4b_base
sbatch slurm_e36_analysis.sh llama32_3b
# After both e36_ANALYSIS_DONE.json files exist:
python nar/e36_report.py
python figures/plot_e36_offset_precision.py
# Run source, rendered alignment, PDF text and collision audits; inspect all panels.
python nar/verify_e36.py
```

The fixed input and all Z16/Z32/ZB16 fixture hashes agree on CPU and GPU. The S32 raw fp32 scale differs by at most 1.49e-8 on this fixture because GPU division matches reciprocal multiplication; this diagnostic is recorded separately. Every PPL row uses the original GPU-computed scale from its frozen Z16 payload, and CPU scale arithmetic never enters model evaluation.

The execution manifest is frozen before the first model forward. A runner refuses changed frozen sources, factors, chunks or E34 flags. Checkpoints are written atomically after all four precision rows of a chunk finish, so a failed/interrupted chunk is repeated without altering completed measurements. The same source code and main-pipeline fp16 guards are used on every run.

Raw group summaries live under the existing project artifact store at `<model>/e36/<method>/seed<seed>/chunk<chunk>.pt`. Each contains every group's original minimum, range, stored step and rotated group mean. Codes are retained in memory only while the four matched chunk passes run; their actual per-row hashes are exported. Part B reuses the Z16 passes and adds no evaluation row.

Exact analysis first validates each raw checkpoint SHA256 and the stored fp16 metadata hashes, then assembles arrays on disk. Per-layer quantiles are exact empirical linear quantiles; the model-wide median ratio uses exact float64 radix order selection in bounded memory, not a sketch. This supports the scheduler's enforced 12 GiB memory for CPU-only jobs. Undefined 0/0 and positive infinity counts are explicit. No denominator is epsilon-clipped and no token or group is sampled away.

## Result files per model

- `e36_per_sequence.csv`, `e36_summary.csv`: all eight rows, three paired seeds and 64 chunks; PPL, seed SD, own-method Z16 deltas and 90% paired chunk/seed intervals.
- `e36_group_statistics.csv`: every layer/site and individual/pooled seed, all/anchor/other group strata, all/BOS/massive/BOS+massive/other input classes, exact ratio/error quantiles, extrema and threshold-exceedance counts.
- `e36_overall.csv`, `e36_worst_layer.csv`: weighted model-wide counts, exact median error/unaligned-step ratio, fraction-max and P99-max layer/site identities, plus token classes at each method's P99-max layer.
- `e36_threshold_layer.csv` (when exceedances exist), `e36_exceedance_classes.csv`: input-class breakdown at the maximum-exceedance layer/site and global class event counts, derived from the full group statistics without additional forwards.
- `e36_method_precision_contrast.csv`, `e36_hypotheses.json`: joint paired Hadamard-versus-PQ precision contrast and each pre-registered decision clause.
- `e36_payload_hashes.csv`: actual code and metadata hashes for every row/seed/chunk/layer/site. Codes agree across all four rows; scales agree across Z16/Z32/ZB16; offsets agree across Z16/S32.
- `e36_fixed_tensor_gate.json`, `e36_gates.csv`, `e36_baseline_replay.csv`: main-routine rounding identity, numerical residuals and bit-for-bit E29 baseline reproduction.
- `e36_raw_artifacts.csv`, `e36_metadata.json`, `e36_DONE.json`, `e36_ANALYSIS_DONE.json`: artifact hashes and completion/source provenance.

The figure source and editable exports are `figures/plot_e36_offset_precision.py` and `fig_offset_precision.{pdf,svg,png}`. The adjacent source CSV retains pooled and all three seed P99 values. PDF/alignment/source/visual checks are preserved in `figures/qa/e36/`. PNG is a 600-dpi preview; PDF/SVG are the authoritative vector exports, so no TIFF is required for this line figure.

`e36_run_manifest.json` records GPU and dependent CPU jobs, actual scheduling overrides and final states. `e36_final_verification.json` checks pairing, invariance, sampling coverage, figure-source agreement, source hashes and preservation of all old result files and the previous report prefix.

## Interpretation limits

“Within seed noise” is a predeclared baseline-SD rule, not an equivalence test. A null or small PPL change does not make fp16 offset storage mathematically lossless. bf16-offset stress sensitivity, fp16 rounding tails and the PQ-versus-Hadamard comparison are separate checks and may disagree.

At k=max all PQ groups are anchored; Hadamard has no aligned anchor groups. For PQ, the group mean is its aligned constant Walsh level; for Hadamard the same value is only incidental group DC. The comparison 2|c|/15 is the isolated sign-changing component's cost, not the net range increase of its sum with residuals. E34's fixed input-position BOS/massive flags are reused exactly, with no fabricated BOS-target loss.
