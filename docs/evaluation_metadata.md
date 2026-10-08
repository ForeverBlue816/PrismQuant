# Evaluation metadata for paper result indexing

Checked on 2026-10-08 for [issue #3](https://github.com/ForeverBlue816/PrismQuant/issues/3).
The seven entries returned by the [index's public API](https://paperswithcode.co/api/v1/evaluations/?paper_id=122307)
match the model names and rounded scores in [arXiv v1](https://arxiv.org/html/2609.32429v1).
This checks transcription and protocol metadata; it is not an independent benchmark
rerun or a recomputation of the paper's three-seed aggregates. The archived E22/E26
JSONs cited below are individual seed-0 records, not evidence of three independent runs.

## Verified entries

All entries use nominal W4A4KV4. Percentages below are on a 0–100 scale.

| Entry | Base model | Rank | Benchmark and metric | Published score | Paper table |
| --- | --- | --- | --- | ---: | --- |
| 33009 | Llama-3.1-70B | 8 | WikiText-2 perplexity ↓ | 3.85 | 1 |
| 33010 | Qwen3-30B-A3B-Base | max | C4 perplexity ↓ | 11.18 | 2 |
| 33011 | Llama-3.1-70B | 8 | ARC-C, zero-shot `acc_norm` ↑ | 61.86% | 1 |
| 33012 | Llama-3.1-70B | 8 | ARC-Easy, zero-shot `acc_norm` ↑ | 83.59% | 1 |
| 33013 | Llama-3.1-70B | 8 | BoolQ, zero-shot `acc` ↑ | 86.73% | 1 |
| 33014 | Qwen3-8B-Base | max | MMLU-Redux, 5-shot generative exact match ↑ | 79.51% | 4 |
| 33015 | Qwen3-8B-Base | 8 | GSM8K, 4-shot CoT exact match ↑ | 84.31% | 4 |

## Metadata to retain or clarify

- **Weight and activation groups are different settings.** The 70B uses
  per-channel GPTQ INT4 weights; group size 128 describes its activations.
  The two Qwen models use group-128 asymmetric GPTQ weights as well as
  group-128 asymmetric activations. Do not label all these settings simply
  “group size 128.”
- **Nominal bits exclude overhead.** Activations include fp16 scales and real
  offsets (4.25 effective bits/value). KIVI keys use token groups of 32,
  values use head-axis groups of 128, and the newest 32 tokens retain full
  precision. At context 2048, effective K/V widths are about 5.17/4.43 bits.
- **Perplexity protocol:** WikiText-2 uses 141 contiguous test windows;
  C4 uses 256 windows from its first validation shard. Windows contain 2048
  tokens and NLL is accumulated in fp32. Keep model/tokenizer and windowing
  details attached when comparing scores.
- **Harness metrics:** the ARC entries are normalized accuracy, not plain
  `acc`. BoolQ uses `acc`. E22 pins lm-evaluation-harness to
  `b954108c9baaaa934b4ad842033b31a97ee30816`.
  MMLU-Redux is the generative task with an 8-token answer cap and a space
  target delimiter; its archived headline is `exact_match,default`.
  GSM8K allows 512 new tokens and uses `exact_match,flexible-extract`.
- **MoE rank:** `k=max` means residual rank R1=16 here; each expert has its
  own R4 at rank 6, and R2 has rank 1. The router is retained at full precision.
  This is the Base checkpoint, not an instruction-tuned model.
- **Openness:** code, calibration/evaluation implementation and model artifacts
  are public. Code is Apache-2.0; checkpoints retain their upstream model
  licenses, including the Llama license. `is_open=true` should not imply that
  every model weight has an unrestricted OSI-approved license. The reference
  loader uses floating-point tensors containing dequantized INT4 values and
  activation/KV simulation; it is not a packed INT4 serving engine.

The API's seven `hf_model_url` fields were empty at review time. Use these
released artifact repositories, with exact revisions and variants in the
[model catalog](../prismquant/models.json):

| Indexed model | Released artifacts |
| --- | --- |
| Llama-3.1-70B | [PrismQuant-Llama-3.1-70B](https://huggingface.co/ForeverBlue/PrismQuant-Llama-3.1-70B) |
| Qwen3-30B-A3B-Base | [PrismQuant-Qwen3-30B-A3B-Base](https://huggingface.co/ForeverBlue/PrismQuant-Qwen3-30B-A3B-Base) |
| Qwen3-8B-Base | [PrismQuant-Qwen3-Base](https://huggingface.co/ForeverBlue/PrismQuant-Qwen3-Base) — select the 8B variant |

Leaderboard position is a property of the indexed comparison set. These entries
do not establish an unrestricted WikiText-2 or C4 SOTA claim across different
base models, tokenizers, precision budgets and evaluation protocols.

## Reproduction references

- [Paper, Appendix A.1](https://arxiv.org/html/2609.32429v1#A1.SS1).
- [Immutable research archive](reproduction.md), retaining the experiment code and records.
- [E22 harness configuration](https://github.com/ForeverBlue816/PrismQuant/blob/research-archive-2026-09-25/nar/e22_benchmarks.json).
- [MMLU-Redux seed-0 result](https://github.com/ForeverBlue816/PrismQuant/blob/research-archive-2026-09-25/results/qwen3_8b_base/e22_nar_kmax_asym_g128_mmlu_redux.json).
- [GSM8K seed-0 result](https://github.com/ForeverBlue816/PrismQuant/blob/research-archive-2026-09-25/results/qwen3_8b_base/e22_nar_k8_asym_g128_gsm8k.json).
- [MoE C4 seed-0 result and rank metadata](https://github.com/ForeverBlue816/PrismQuant/blob/research-archive-2026-09-25/results/qwen3_30b_a3b_base/e26_nar_kmax_asym_g128_c4.json).

This repository note supplies the verified values and requested clarifications;
it does not imply that the third-party index has applied metadata changes.
