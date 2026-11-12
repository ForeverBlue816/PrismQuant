> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../../README.md) for the active scope and observed scheduler state.

**E42 — Meta-Llama-3.1-8B; DuQuant official native: official W4 per-channel affine / A4 per-token affine; official Q/K/V quantizers; FP16 simulation**

Base revision `e9a141a2091ea561b96483212645a2a05e6f99fc`. Direct paper-model run; published Llama3-8B validation is background work and imposes no execution gate. Native format is explicitly distinct from E38 GPTQ/g128.

| Method | Seed | Full141 PPL | Eight-task mean (%) | Calibration seconds | GPU hours | Peak GiB | State |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DuQuant official native | 0 | pending | pending | pending | pending | pending | Multi-GPU compute worker prepared |

The quantized checkpoint and per-sequence NLL are persisted; task evaluation reuses that checkpoint. No hyperparameters are tuned to match Table1. g128 adaptation and online-transform timing are deferred.
