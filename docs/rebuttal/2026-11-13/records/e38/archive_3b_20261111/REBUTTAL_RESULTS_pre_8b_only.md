> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../../README.md) for the active scope and observed scheduler state.

# PrismQuant rebuttal evidence

**Current execution policy, 2026-11-10: directly produce reviewer evidence, with no numerical execution thresholds.** Retain absolute relative PPL difference <=0.5% as the user's reporting reference; neither that comparison nor the old reference-replay0.01 check may stop later tasks. Run seed0 only, deployment accuracy uses group128 asymmetric quantization, no new timing, and no OffQ work. A failed measurement is recorded and the remaining tasks/rows are attempted. Source and artifact identities are recorded without repeating large pre-evaluation audits. See [direct-measurement policy](../../../manifest.json).

**Completed cohort, 2026-11-11 SGT:** all five jobs below completed with exit0. All four E38 accuracy rows contain full141 WikiText-2 PPL and all eight zero-shot tasks. The 3B integer prototype has PPL **8.71940160**, versus simulator **8.71802441** (absolute relative difference **0.015797%**); eight-task means are **60.4988%** and **60.0324%**, respectively. FP16 means are **62.7324%** (3B) and **68.6753%** (8B). See the [complete accuracy/per-task tables](../../../manifest.json) and [collection](../../../manifest.json). The last E38 job ended at **03:02:42 SGT on November11**; no jobs from this cohort remain queued.

E41 completed all six full141 rows: asymmetric Had/PQconstant/PQsigned PPL **6.40246 / 6.33337 / 6.36120**; symmetric **6.48862 / 6.39883 / 6.40177**. PQconstant improves on Hadamard in both formats, while the signed-minus-constant gap is **0.027827** for asymmetric versus **0.002937** for symmetric. This supports a direction-by-quantizer interaction and improvement beyond asymmetric offset alone in this activation-only setting; it is not a pure causal decomposition, because the direction swap also changes a residual coordinate. See [E41 table](../../e41/table.md).

| Completed job | Evidence | Final state |
|---|---|---|
| 1733510 | 3B k=max asymmetric integer-candidate full141 PPL + eight tasks, B8 tasks | COMPLETED, 0:0 |
| 1733511 | 3B k=max asymmetric simulator-reference full141 PPL + eight tasks, B8 tasks | COMPLETED, 0:0 |
| 1733512 | FP16 deployed3B full141 PPL + eight tasks, B2 tasks | COMPLETED, 0:0 |
| 1733513 | FP16 deployed8B full141 PPL + eight tasks, B2 tasks | COMPLETED, 0:0 |
| 1733515 | E41:8B Had/PQ constant/PQ signed × asymmetric/symmetric, six full141PPL rows | COMPLETED, 0:0 |

Jobs1733476/1733349/1733347/1733348 were still pending and were replaced by1733510/1733511/1733512/1733513 respectively; no measurements from those cancelled jobs are discarded. New E38 code is `35e76a6`; each task retains per-item results. [Submission/status snapshot](../../../manifest.json). E41 is independent on branch `experiments/e41-mechanism`, source `4ee367b`, in [/home/yanlongc/prismquant-rebuttal-e41](../../../manifest.json), with [protocol](../../../manifest.json) and [actual submission](../../e41/raw/submission_001.json). It measures activation-only qkv/down behavior with BF16 weights/KV, addressing the offset/direction mechanism separately from W4A4KV4 deployment.

The history below preserves earlier policies and job states; the current policy and schedule above supersede their thresholds and pending-job references.

Earlier accuracy-only scope update: the user narrowed the quoted deployment follow-up to asymmetric accuracy.1733476 was the retained job at that point; it has now been replaced by direct job1733510. No prototype timing is resubmitted. See [historical scope amendment](../../../manifest.json).

A40 job **1733477** already ended at **21:35:28 SGT** before this scope update, with no timing performed. Its candidate PPL **8.50487977** versus same-device reference **8.49440921** has relative difference **0.123264%**, which passes the user's0.5% rule. The separate environment check failed because that reference differs from the earlier PRO reference8.50919942 by−0.01479022, exceeding its retained absolute0.01 rule; this caused the overall `GATE_FAILED` despite the candidate tolerance passing. Decode and full141PPL were not reached. Asset hashes and numerical source files were verified, and the original logs remain intact. See [A40 collection and failure attribution](../../../manifest.json).

Latest update, **2026-11-10**: integer job **1733332** finished at **20:32:01 SGT** with the original gate-failure exit2. All196 operator sites completed with exact packed-to-BF16 controls and finite integer outputs. The complete64-sequence candidate PPL was **8.49573069** versus reference **8.50919942**, difference **−0.01346873**, absolute relative difference **0.158284%**; reference chunks reproduced the prior simulator exactly. Maximum logit difference was13.03125 and argmax agreement86.3884%. The user then explicitly changed the candidate/reference PPL rule to **absolute relative difference ≤0.5%**, under which this result passes; the original absolute0.01 failure and raw logs remain unchanged. See [diagnostic table](../../../manifest.json), [collection](../../../manifest.json), and [new protocol](../../../manifest.json).

Follow-up submission history: **1733476** (PRO6000 full141PPL plus integer-candidate eight tasks, twelve-hour cap) and **1733477** (A40 repeated64-sequence and decode gates, full141PPL and prototype timing, four-hour cap) used source `dd524c1`;1733477 subsequently ended as described above, while1733476 remains pending. The original eight-task jobs **1733347/1733348/1733349** remain `PENDING: Priority`. All work remains seed0. The candidate uses4-bit storage expanded to INT8 computation with simulated KV; its accuracy evidence does not establish a complete native INT4/packed-KV deployment. The final C1 joint-instance requirement remains open and performance work is deferred under the user's current scope. See [accuracy submission](../../../manifest.json) and [original timing submission](../../../manifest.json).

Earlier submitted batch: the user requested continuation and then restricted the current experiment batch to **seed0 only**; seeds1/2 are deferred. **1733332** was the 3B k=max integer diagnostic on one PRO6000 (four-hour cap, source `8851a69`); **1733347** and **1733348** are the 3B and8B FP16 deployed-backend eight-task evaluations on one A40 each; **1733349** is the original 3B k=max quantized-simulator eight-task reference on one PRO6000. Each eight-task job has a twelve-hour cap, source `e8d914b`, a preceding model-quality check and complete per-item outputs. See [initial submissions and scheduler snapshot](../../../manifest.json).

The integer job checks all196 real calibration-input linear sites, then64-sequence original-E14 versus integer-candidate logits/PPL, and runs full141-sequence candidate PPL only if the0.01 gate passes. Seven CPU checks and the source/data preflight passed. The candidate uses signed4 weight storage expanded to INT8 products with INT32 partial sums and FP32 affine correction, retaining E14 rotation and KV simulation; this job makes no native INT4 runtime claim. See [single-seed amendment](../../../manifest.json), [submission](../../../manifest.json), and [model-gate preflight](../../../manifest.json).

Eight-task preparation verified the frozen harness, all dataset splits, tokenizer compatibility against original E14 token IDs for both models, and the complete per-item result interface; seven additional CPU checks passed. The initial dataset-compatibility preflight failure remains recorded. The FP16 runner reads modern List[Value] metadata without changing cache files and explicitly selects original E14 tokenizers0.23.1 within the process; actual library paths and hashes are recorded. Each FP16 GPU job also gates variable-length/right-padded batch2 model calls before full-task evaluation. See [final FP16 preflight](../../../manifest.json) and [simulator-reference preflight](../../../manifest.json). The simulator reference is separate from deployment evidence. All six3B/8B seed0 Hadamard/k8/kmax checkpoints are now available, with the last restore completed at20:41:44SGT; see [restoration collection](../../../manifest.json).

Previous completed batch, **2026-11-10 18:08:30 SGT**: A40 timing retry **1733011 COMPLETED**, exit `0:0`, and all four FP16 model/phase results passed validation. At B1 with a 2048-token prefix, prefill median [IQR] is **210.955 [0.576] ms** for 3B and **417.469 [0.889] ms** for 8B; decode is **42.911 [0.134]** and **29.038 [0.011] tokens/s**, respectively. Each cohort completed10 warmups and50 measured repetitions with all52 environment checks passing. PPL and all141 chunk records match the earlier FP16 attempt exactly. See [updated table](../../../manifest.json) and [validated collection](../../../manifest.json). These are seed0/session1 FP16 baselines; quantized deployment, eight tasks and remaining timing sessions are still incomplete.

Status: work in progress, 2026-11-10 (Asia/Singapore). Slurm job **1732842** completed successfully on `gpu-pro6000-10` at **14:59:23 SGT**, exit `0:0`. The 3B BF16 anchor passed: PPL **7.819581289874704**, with zero difference from the legacy reference at both corpus PPL and every one of the 141 chunk NLLs. See [GPU result](../../../manifest.json) and [scheduler completion/logs](../../../manifest.json). This is an environment-anchor result; deployed INT4 quality and runtime rows remain not run.

At **15:50:38 SGT**, two follow-on jobs were submitted with frozen source `40330a7`: **1732989**, one PRO6000 for the actual 3B k=max E14 checkpoint replay/capture/export audit (two-hour limit), and **1732990**, one A40 for 3B/8B real FP16 backend gates, full WikiText-2 PPL and B1 prefill/decode session 1 timing (eight-hour limit). Both initial states are `PENDING: Priority`. See [run matrix](../../../manifest.json), [checkpoint submission](../../../manifest.json), [FP16 submission](../../../manifest.json), and [scheduler snapshot](../../../manifest.json). Timing is prohibited on a failed model gate or busy GPU/host; quality-only or failed results remain recorded. Eight tasks, integer deployment adapters and the remaining batch/session/seed matrix have not been submitted in this bounded first batch.

First FP16 collection: **1732990 COMPLETED at 16:04:04 SGT**, but the experiment is **PARTIAL**: both FP16 models passed the 64-sequence full-logit and decode gates and completed full141-sequence PPL (3B **7.81657913**, 8B **6.23843246**); all four timing attempts were blocked by host CPU load and produced zero samples. GPU process exclusivity passed. The two processes for each model have identical chunk NLLs and use the same seed0, so they do not supply seed uncertainty. See [partial table](../../../manifest.json) and [collection audit](../../../manifest.json).

At **16:12:15 SGT**, timing retry **1733011** was submitted for `gpu-a40-2`, again one A40 with an eight-hour limit. It repeats same-instance quality and timing, keeps the original numerical and idle thresholds, and permits at most 300 seconds of idle waiting before each timing cohort; mid-cohort guard failures still abort reporting. See [retry protocol](../../../manifest.json), [submission](../../../manifest.json), and [scheduler status](../../../manifest.json). Earlier partial/blocked logs remain intact.

Checkpoint collection: **1732989 COMPLETED at 16:32:37 SGT**, exit0, with 3B k=max original-simulator PPL **8.71896825** versus historical **8.71444608** (difference **+0.00452217**, inside the frozen0.01 tolerance, but not exact chunkwise reproduction). All28layers and196linear sites were captured/audited. The current exact plain-integer export check fails for every complete matrix: recovered FP32 scales reproduce the saved BF16 weights after BF16 rounding, but direct integer accumulation followed by one scale is a distinct computation. This is an export diagnostic, not a failure of an already-built deployed model and not an impossibility proof. See [diagnostic table](../../../manifest.json) and [raw result](../../../manifest.json). Timing retry1733011 was pending at this earlier checkpoint collection and subsequently completed as recorded above.
User scope update: proceed in stages without an urgent deadline; **single seed0 first**, with no multi-seed submissions in this batch; exclude **all OffQ experiments**, including hybrids, ladder rows and OffQ comparisons. E40 is reserved and will not be implemented.

The full research code is retained on branch `experiments/e38-joint-deploy`, based on `728e6e5`, in `/home/yanlongc/prismquant-rebuttal-e38`. The public `main` branch and old experimental results are unchanged. No remote publication is authorized or performed in this phase.

Submitted manuscript: `/home/yanlongc/elastic_kv_probe/11144_PrismQuant_Optimal_Null_.pdf`. Paper-to-log provenance discrepancies are recorded, not silently reconciled. The first run replays the **legacy E14 raw BF16 reference** as an environment check; it is not a reproduction claim for the manuscript's rounded BF16 cell or a deployed INT4 measurement.

## gL7s

- Model quality: [E38 table](../../../manifest.json) now contains complete asymmetric integer-candidate/reference PPL and eight-task means, plus both FP16 baselines. The current deployment scope is accuracy only. Historical FP16 B1 timing remains separately identified; no integer speed result is claimed.
- Matched protocol and reproducibility: [frozen protocol](../../../manifest.json), [protocol audit](../../../manifest.json), [cache audit](../../../manifest.json), [raw logs](../../../manifest.json).
- Per-task uncertainty: the new four accuracy rows each retain22419 item-level outputs. Integer-minus-reference mean is+0.466341pp, with paired task-stratified bootstrap95%CI[-0.064921,+1.002023]pp over10000 resamples. This includes zero and describes item uncertainty for two fixed runs on different GPUs; it is not seed/hardware uncertainty or an isolated backend effect. [Per-task intervals](../../../manifest.json), [raw analysis](../../../manifest.json). The broader Hadamard/k8/kmax and official-baseline comparisons still need their per-item evaluations.

## Hdp7

- Exact reviewer-point attribution needs the review text; only the experiment prompt is available.
- E41 mechanism evidence is complete: [8B seed0 six-row table](../../e41/table.md) measures Hadamard/PQconstant/PQsigned under both quantizers. E39 official FlatQuant/DuQuant comparison remains next; local official DuQuant source is available but needs its compatible environment, while FlatQuant code is not locally cached.
- All OffQ-dependent tests are excluded by the user's update.

## sDQh

- Exact reviewer-point attribution needs the review text; only the experiment prompt is available.
- E41's first8B direction-by-offset matrix is complete, with PQconstant improving PPL over Hadamard by0.069084 (asymmetric) and0.089790 (symmetric). The full factorial/objective–quality study remains additional work. E42 per-task uncertainty can use the complete per-item outputs from the direct eight-task jobs.

## NeUz

- Q2 format/rotation overhead: E38 will measure the two effects separately, with the same GPU, loaded checkpoints and timing protocol. No overhead result yet.
- C1 joint accuracy/efficiency: the existing integer backend has a different activation/weight/KV contract; [protocol audit](../../../manifest.json) records requirements before timing.
- Stage B prerequisite: [CPU arithmetic validation](../../../manifest.json) and nine unit tests cover group sums, all affine corrections, packing and E14 rounding edge cases. These are synthetic arithmetic checks, not GPU execution or model-quality evidence.

## Execution order

1. Completed: direct asymmetric candidate/reference and FP16 PPL/eight-task measurements, with every result retained and no numerical acceptance thresholds.
2. Completed: first8B E41 direction/quantizer mechanism matrix; use it to answer the offset/direction reviewer point.
3. Next: E39 official baselines, remaining seed0 Hadamard/k8/kmax accuracy rows, and E42 item-level uncertainty; extend mechanism evidence where the review requires it.
4. Learned rotations and 70B only after the corresponding 3B/8B evidence is complete.

Status records live under `results/e38/`. Long-running work will append a status record at least every 12 hours while actively monitored; no unattended recurring chat notification has been installed.
