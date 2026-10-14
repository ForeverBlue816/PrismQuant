# Reproduction guidance

The public release provides the PrismQuant reference implementation, pinned
checkpoint catalog, loading examples, unit tests and paper figures. Evaluation
settings and reported results are described in the
[paper](https://arxiv.org/abs/2609.32429) and [evaluation metadata](evaluation_metadata.md).

## Load an exact released variant

```bash
pip install -e .
prismquant models --json
prismquant download --model qwen3_0.6b_base
prismquant generate --model qwen3_0.6b_base \
  --prompt "The key idea behind quantization is" --max-new-tokens 64
```

The [catalog](../prismquant/models.json) specifies immutable artifact and
base-model revisions, supported variants and required files. Select an explicit
variant with `--checkpoint` or the Python `checkpoint` argument. See the
[model guide](models.md) for device placement and offline loading.

Record the model key, variant, artifact/base revisions, compute dtype, package
versions and evaluation settings when reporting a reproduced result. Preserve
tokenization, dataset split, sequence length, few-shot prompts, extraction rules
and cache mode. Reference weights are dequantized INT4 values in floating-point
containers; the loader is not a packed INT4 serving engine.

## Calibration and evaluation scope

`fit_rotation` constructs a rotation from user-provided activation samples.
The package also includes GPTQ, model folding and the activation/KV quantizers;
see the [technical guide](method.md) for conventions.

This release does not include a turnkey calibration-to-benchmark command for
all paper models, the full experiment orchestration, or the end-to-end packed
serving benchmark. Running the loading example or unit tests does not reproduce
the paper's accuracy, perplexity or speed tables. For evaluations, use the
published settings with the appropriate model variant and evaluation harness.
The [kernel guide](kernels.md) describes the separately exposed low-level kernels.

## Development checks

```bash
pip install -e '.[dev]'
pytest -q
python -m build
```

See [validation scope](validation.md) for what these checks establish and their
limits. Paper figures are available in the [figure gallery](../Figures/README.md).
