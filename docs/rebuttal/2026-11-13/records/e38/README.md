> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

# E38 working record

This directory distinguishes preflight/anchor checks from formal deployed-model evidence. `table.md` and `table.tex` remain explicitly incomplete until the same loaded deployed model has been evaluated and timed. Historical log values belong to audits and are never copied into those cells.

The initial BF16 replay uses the unchanged E14 loss function, a pinned local model snapshot and the original 141 × 2048 test tensor. Its tolerance is frozen before execution in `experiments/e38_protocol.json`. It records the difference from the legacy raw reference and separately from the prompt's rounded value. Any mismatch remains in the log and prevents treating the environment as validated.

Runner: `python -m nar.e38_anchor --run-id UNIQUE_ID --model-snapshot PINNED_LOCAL_SNAPSHOT --tokens FROZEN_TOKEN_FILE`. Append `--preflight-only` for a CPU metadata/import check; that command performs no model forward. Submit `slurm_e38_anchor.sh` from the worktree with `E38_MODEL_SNAPSHOT` set to the pinned path and `E38_EXPECTED_SOURCE_COMMIT` set to the frozen source commit. Existing run IDs are never overwritten.

First submission: job **1732842**, source `33ed3b8188fffc09728c625bfedf9081b0d549ab`, one pro6000 GPU, two-hour limit, QOS rose. Its first scheduler snapshot is `PENDING: Priority`. The job writes `raw/anchor-bf16-3b-job1732842.json`; a scheduler failure before Python starts is recorded in the Slurm `.err`/accounting output and must be retained on collection. The submitted source hashes are checked even if unrelated later commits move branch HEAD.

Completion: job 1732842 ran on `gpu-pro6000-10` from 14:58:02 to 14:59:23 SGT on 2026-11-10 and exited `0:0`. The BF16 anchor passed with PPL 7.819581289874704 and zero chunkwise deviation from the original E14 reference. `raw/anchor_completion_001.json` preserves scheduler accounting and stdout/stderr; the earlier pending snapshot is retained as history. Formal deployment measurements remain not run.

Follow-on submission at 15:50:38 SGT: **1732989** runs `slurm_e38_checkpoint_probe.sh`; **1732990** runs `slurm_e38_stage_a.sh`, both pinned to source `40330a7d796f13ebf2f056ad593af37aa74fe1f7`. The probe's output is `/projects/nar/nar-validation/artifacts/e38/checkpoint-3b-kmax-job1732989/run.json`, with immutable calibration captures and candidate-scale files alongside it. The FP16 job writes four `raw/fp16_{3b,8b}_{prefill,decode}_b1_s1_job1732990.json` records. Its 64-sequence full-logit gate and 2048-prefix decode gate run before full PPL and same-instance timing. This first job explicitly leaves eight tasks, random-weight controls and the larger timing matrix pending. Every failed/blocked process retains its result, and the shell continues the remaining planned processes.

The cluster submission plugin enforced four CPUs and **40 GiB host RAM** for the A40 job despite the requested 64 GiB; this is preserved in the submission stderr and scheduler snapshot. The runtime additionally checks selected-GPU process exclusivity, normalized host load and sampled CPU busy fraction (both at most 0.5); it retains partial timing samples without reporting a median if the environment guard fails.

Collected job1732990: all four quality paths passed, full PPL is 7.81657912796739 (3B) and 6.238432455537351 (8B), and timing produced zero samples because normalized host load exceeded 0.5. The original progress-list aliasing issue is documented in `raw/fp16_collection_001.json`; all first64 gate metrics were independently revalidated and raw logs preserved. Retry job1733011 is pinned to `bc4f66b77eda8a460274b1d5026e31be8120ba86` on gpu-a40-2; it snapshots progress lists and waits up to300seconds before timing under unchanged thresholds. It will produce new job-specific output files, never replace prior attempts.

The main E14 and packed E28 Python environments are separate and are preserved. Formal timing requires an idle exclusive GPU and an audited host; an accuracy-only anchor is not a timing measurement. No GPU-wide lock or allocation is claimed by these scripts.

Files:

- `raw/*.json`: all preflights, job submissions, attempts and measured anchor records.
- `protocol_audit.md`, `cache_audit.md`, `cluster_audit.md`: checks and limitations, with supporting JSON.
- `table.md`, `table.tex`, `summary.md`: current deliverables, including explicit `not run` cells.

No figure is generated without measured data; there is no placeholder scientific plot.
