> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../../README.md) for the active scope and observed scheduler state.

E39 directly evaluates official FlatQuant on the pinned Meta-Llama-3.1-8B checkpoint with seed0 and no Table1 or original-paper validation prerequisite.
The native configuration is W4 RTN per-channel symmetric; A4 per-token symmetric; native asymmetric K4/V4 per head128, and is not labelled as the E38 GPTQ/g128 reference.
The prepared worker saves the quantized checkpoint, full141 per-sequence NLL, calibration/quantization cost and peak memory, then evaluates main-table tasks from the saved checkpoint.

- Pending: GPU measurements; CPU implementation checks are not accuracy results.
- Deferred: published Llama3-8B validation and g128/runtime adaptation.
