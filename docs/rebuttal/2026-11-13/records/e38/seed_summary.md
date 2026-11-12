> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

**E38 — Meta-Llama-3.1-8B; three-seed E14 reference GPTQ W4 / asymmetric g128 A4 / simulated KV4; Hadamard and PQ k=8**

| Rotation | Seed0 reference PPL | Seed1 PPL | Seed2 PPL | PPL mean ± sample SD | Eight-task % mean ± sample SD |
| --- | --- | --- | --- | --- | --- |
| hadamard | 7.20974 | pending | pending | pending (requires seeds0/1/2) | pending (requires seeds0/1/2) |
| nar_k8 | 6.99041 | pending | pending | pending (requires seeds0/1/2) | pending (requires seeds0/1/2) |

| Rotation | Task | Seed0 % | Seed1 % | Seed2 % | Mean ± sample SD (%) |
| --- | --- | --- | --- | --- | --- |
| hadamard | arc_easy | pending | pending | pending | pending (requires seeds0/1/2) |
| hadamard | arc_challenge | pending | pending | pending | pending (requires seeds0/1/2) |
| hadamard | boolq | pending | pending | pending | pending (requires seeds0/1/2) |
| hadamard | hellaswag | pending | pending | pending | pending (requires seeds0/1/2) |
| hadamard | openbookqa | pending | pending | pending | pending (requires seeds0/1/2) |
| hadamard | piqa | pending | pending | pending | pending (requires seeds0/1/2) |
| hadamard | social_iqa | pending | pending | pending | pending (requires seeds0/1/2) |
| hadamard | winogrande | pending | pending | pending | pending (requires seeds0/1/2) |
| nar_k8 | arc_easy | pending | pending | pending | pending (requires seeds0/1/2) |
| nar_k8 | arc_challenge | pending | pending | pending | pending (requires seeds0/1/2) |
| nar_k8 | boolq | pending | pending | pending | pending (requires seeds0/1/2) |
| nar_k8 | hellaswag | pending | pending | pending | pending (requires seeds0/1/2) |
| nar_k8 | openbookqa | pending | pending | pending | pending (requires seeds0/1/2) |
| nar_k8 | piqa | pending | pending | pending | pending (requires seeds0/1/2) |
| nar_k8 | social_iqa | pending | pending | pending | pending (requires seeds0/1/2) |
| nar_k8 | winogrande | pending | pending | pending | pending (requires seeds0/1/2) |

All three quantization seeds are required for each aggregate; task RNG remains0. Seed0 PPL comes exclusively from the reference side of the paired run, never the integer candidate. R1/R2 and GPTQ sampling vary with seed; original E11 R4 factors remain fixed with seed-dependent signs.

hadamard seed0: [integer_8b_hadamard_seed0_equivalence-timing_job1734010.json](raw/integer_8b_hadamard_seed0_equivalence-timing_job1734010.json).

hadamard seed1: pending.

hadamard seed2: pending.

nar_k8 seed0: [integer_8b_nar_k8_seed0_equivalence-timing_job1734010.json](raw/integer_8b_nar_k8_seed0_equivalence-timing_job1734010.json).

nar_k8 seed1: pending.

nar_k8 seed2: pending.
