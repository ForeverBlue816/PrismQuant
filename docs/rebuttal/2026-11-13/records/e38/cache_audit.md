> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

# E38 cache and protocol audit

Audit time (UTC): 2026-11-10T05:51:42.537170+00:00. Code: `1ead849c3d6c70267ca68d38d60fe6d8d01ee883`.

This is an input audit, not a new GPU result. All numbers below are explicitly historical; source JSON paths and SHA256 values are in [raw/cache_audit.json](raw/cache_audit.json). OffQ experiments are excluded by the user.

## Ready for an initial anchor

Both original bf16 base snapshots and the frozen WikiText-2 token streams are local. A complete 3B PrismQuant k=max default-weight checkpoint is also local, with all 28 layers and the required rotation factors. Its cached historical PPL is 8.71444608076717; the cached bf16 PPL is 7.819581289874704. These are replay targets, not the unverified prompt anchors.

```text
/projects/nar/nar-validation/cache/huggingface/models--ForeverBlue--nar-w4a4kv4-llama-3.2-3b/snapshots/25c6b2d9a04151fab7e5f37b9ce93c8f0edf2ee5/checkpoints/llama32_3b/gptq_nar_kmax_seed0
```

Use the parent `checkpoints/` directory as `--artifact-root`, seed 0, sequence length 2048, row `nar_kmax_asym_g128`, weight protocol `default`, and quantized KV. Give new evaluations an isolated results directory because E14 skips existing result paths. Root is preparing the first bf16 GPU run.

## Pinned public sources

Official Hugging Face API metadata was fetched without downloading weights:

| Model | Public repository and immutable revision | Availability |
|---|---|---|
| llama32_3b | [ForeverBlue/PrismQuant-Llama-3.2-3B](https://huggingface.co/ForeverBlue/PrismQuant-Llama-3.2-3B/tree/78e0a124e359aace82793c74ff5e3827bf18b64c) · `78e0a124e359aace82793c74ff5e3827bf18b64c` | Default, g128, g128_asym, act_order seed0 variants; approximately 5.637 GB per checkpoint |
| llama31_8b | [ForeverBlue/PrismQuant-Llama-3.1-8B](https://huggingface.co/ForeverBlue/PrismQuant-Llama-3.1-8B/tree/fc812e6687165115feee00293c953a1b521d3bd0) · `fc812e6687165115feee00293c953a1b521d3bd0` | Default-weight Hadamard/k8/kmax seeds 0/1/2; approximately 13.959 GB per checkpoint; no g128_asym weight variants |

Original artifact-root checkpoints were deliberately archived and deleted; `results/hf_uploads.json` records the operation. A `DONE.json` alone does not establish local availability. The audit records every layer path for the complete local anchor and checks its cache blob name against official LFS SHA256 metadata; large-file content hashes were not recomputed.

## Historical seed0 values

The eight-task means are recomputed solely from the historical task aggregates, using E14 task membership and metrics. They are not new evaluations and are not three-seed means.

| Model | Row | Historical seed0 PPL | Historical seed0 eight-task mean (%) |
|---|---|---:|---:|
| llama32_3b | bf16_reference | 7.8195813 | not established by audited E14 files |
| llama32_3b | hadamard_asym_g128 | 9.2090076 | 59.3776894 |
| llama32_3b | nar_k8_asym_g128 | 8.7562315 | 59.2942505 |
| llama32_3b | nar_kmax_asym_g128 | 8.7144461 | 60.1866982 |
| llama31_8b | bf16_reference | 6.2410354 | not established by audited E14 files |
| llama31_8b | hadamard_asym_g128 | 7.2063810 | 65.3659032 |
| llama31_8b | nar_k8_asym_g128 | 6.9898380 | 65.3591902 |
| llama31_8b | nar_kmax_asym_g128 | 6.9146690 | 65.4343620 |

The exact prompt 3B seed mean remains unknown: these files contain only Hadamard seeds 0/1 and PrismQuant k8/kmax seed0 PPL, with seed0-only task accuracy. Their values differ from the prompt numbers. The submitted PDF is now located at `/home/yanlongc/elastic_kv_probe/11144_PrismQuant_Optimal_Null_.pdf`; the separate manuscript audit must establish the exact table protocol. Missing seeds cannot be interpolated or inferred from a rounded table.

## Protocol and statistical gaps

- E14 default weights are symmetric per-channel GPTQ INT4 (`groupsize=-1`), whereas activations are asymmetric group-128 INT4 with fp16 scale and offset (4.25 bits). A checkpoint suffix `g128_asym` changes the weight protocol.
- Calibration uses distinct caches: GPTQ uses 128 seeded random train windows; rotation calibration uses 128 contiguous train chunks. Preserve both frozen streams. The original unsuffixed rotation manifest records calibration seed 20260902; preserve its relationship to checkpoint seed0.
- WT2 PPL uses 141 contiguous 2048-token test windows without added BOS. Preserve model/tokenizer revisions and the E14 bf16/Hugging Face loss path for the first replay.
- The pinned harness is `b954108c9baaaa934b4ad842033b31a97ee30816`. The eight-task average excludes LAMBADA and uses `acc_norm` for ARC, PIQA, HellaSwag and OpenBookQA; `acc` for BoolQ, Social-IQA and WinoGrande.
- Historical E14 explicitly used `log_samples=False`. E42 requires reruns with item-level samples and a stable paired identity before computing 10,000-resample CIs. Aggregate accuracies cannot supply these intervals.
- Historical logs lack some requested new-run provenance fields. Current file hashes establish exactly which archived logs were read; they do not manufacture missing original commands or code commits.

## Audited commands and verification scope

Exact remote metadata URLs, immutable revisions, small-artifact SHA256 values, base-model file existence/size checks, required rotation files and historical source hashes are recorded in the JSON. Inspection used `git rev-parse`, read-only `find`, Python `pathlib`, `hashlib`, and `urllib.request`; this audit launched no inference job. No OffQ experiment or reimplementation was considered.
