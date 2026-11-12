Meta-Llama-3.1-8B has two requested official-method rows; currently FlatQuant PPL pending / tasks 0/8, DuQuant PPL pending / tasks 0/8.

Actual weight, activation and KV formats are stated per method, including floating fake-quant execution.

Timing follows complete PPL on the same loaded checkpoint and uses an exclusive GPU, real cached decode and common workload settings.

The paired PPL interval uses the same saved sequence resamples and conditions on seed0 rather than measuring calibration-seed variation.

No FP16 run or missing benchmark value is introduced; unsupported native backends remain separate from measured implementation speed.
