# Released models

Use `prismquant models` to list the exact checkpoint names. The loader verifies
that every decoder layer and rotation factor is present before loading a model.
Downloads are restricted to one selected variant at an immutable Hub revision.

## Model catalog

| Model key | Base model | Reference compute | Default weight protocol |
| --- | --- | --- | --- |
| `qwen3_0.6b_base` | Qwen/Qwen3-0.6B-Base | fp32 | asymmetric group-128 GPTQ |
| `qwen3_1.7b_base` | Qwen/Qwen3-1.7B-Base | fp32 | asymmetric group-128 GPTQ |
| `qwen3_4b_base` | Qwen/Qwen3-4B-Base | fp32 | asymmetric group-128 GPTQ |
| `qwen3_8b_base` | Qwen/Qwen3-8B-Base | fp32 | asymmetric group-128 GPTQ |
| `llama32_3b` | unsloth/Llama-3.2-3B | bf16 | symmetric per-channel GPTQ |
| `llama31_8b` | unsloth/Meta-Llama-3.1-8B | bf16 | symmetric per-channel GPTQ |
| `llama31_70b` | unsloth/Meta-Llama-3.1-70B | fp32 | symmetric per-channel GPTQ |

The defaults preserve the released checkpoint/evaluation conventions. Additional
Llama-3.2-3B protocols and paired seeds are available in the catalog; not every
model has every variant. A filename containing `nar` denotes PrismQuant's original
internal naming, retained to preserve checkpoint compatibility.

## Download and select a variant

```bash
prismquant download --model qwen3_0.6b_base
prismquant generate --model qwen3_0.6b_base \
  --checkpoint gptq_nar_k8_seed0_g128_asym \
  --prompt "A neural network is"
```

```python
from prismquant import download_checkpoint, load_model

snapshot = download_checkpoint("qwen3_0.6b_base")
loaded = load_model("qwen3_0.6b_base", snapshot_path=snapshot)
print(loaded.generate("The capital of France is", max_new_tokens=32))
```

Use `cache_dir=...` for a custom Hugging Face cache. For a fully local run, provide
`snapshot_path`, `base_model_path` and `local_files_only=True`; the base directory
must include the tokenizer as well as the original model weights and config.
Public PrismQuant artifacts do not require a login. Access to any gated upstream
model still follows its upstream terms.

## What is stored and computed

`checkpoints/<model>/<variant>/layer_XX.pt` stores the seven attention/MLP linear
weight tensors after rotation folding and GPTQ. These are dequantized INT4 values
stored as floating-point tensors, with restricted `weights_only=True` loading.
They are **not** a standard Transformers checkpoint or a packed INT4 model.

`rotations/<model>/...` contains calibrated Householder factors and permutations.
The loader restores the base model's embedding, head, Q/K norms where present,
and other unquantized state; folds the residual rotation into embedding/head;
then loads the already-folded GPTQ decoder linears. It installs the original
online R2/R4 rotations, asymmetric group-128 activation QDQ, and the KIVI-style
32-token residual policy for K/V. Both metadata values use the validated fp16
rounding routine.

The reference runtime keeps its weights and cache in floating-point storage.
Budget memory according to the table's **reference compute** column, including
unquantized embeddings/head, activations and KV state. A 70B fp32 model requires
roughly 280 GB for parameter values alone, before overhead. Start with 0.6B;
`device="cpu"` is available for debugging but is slow.

For larger models, `load_model(..., device_map="balanced", max_memory={...})`
uses Accelerate placement and device-local rotation copies. Provide enough memory
for all tensors. Disk/meta offloading is not supported. Keep the default dtype
for comparisons with the released results; silently casting the model would
change the numerical protocol.

## Advanced usage

The returned object exposes `model`, `tokenizer`, `hooks` and provenance
`metadata`. Batched Transformers generation is available directly:

```python
inputs = loaded.tokenizer(["The capital of France is", "The capital of Japan is"],
                          padding=True, return_tensors="pt")
inputs = inputs.to(loaded.model.get_input_embeddings().weight.device)
output = loaded.model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(loaded.tokenizer.batch_decode(output, skip_special_tokens=True))
```

The tokenizer uses left padding and the attention hook masks padded keys before
they can affect group statistics. `quantize_activations=False` and
`quantize_kv=False` are explicit diagnostic options. `loaded.close()` removes
runtime hooks; do not use the rotated model afterward as an ordinary base model.

## Model licenses

Qwen-derived artifacts retain Apache-2.0. Llama-derived artifacts retain the
Llama 3.1/3.2 Community License and the accompanying usage policy. Built with
Llama. See the exact license and provenance in each linked Hub repository.
