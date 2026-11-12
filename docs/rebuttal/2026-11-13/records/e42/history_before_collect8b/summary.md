> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../../README.md) for the active scope and observed scheduler state.

E42 directly evaluates official DuQuant on the pinned Meta-Llama-3.1-8B checkpoint with seed0 and no Table1 or original-paper validation prerequisite.
The native configuration is official W4 per-channel affine / A4 per-token affine; official Q/K/V quantizers; FP16 simulation, and is not labelled as the E38 GPTQ/g128 reference.
The prepared worker saves the quantized checkpoint, full141 per-sequence NLL, calibration/quantization cost and peak memory, then evaluates main-table tasks from the saved checkpoint.

- Pending: GPU measurements; CPU implementation checks are not accuracy results.
- Deferred: published Llama3-8B validation and g128/runtime adaptation.
