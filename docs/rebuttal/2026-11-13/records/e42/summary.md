> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

Current requested scope (2026-11-13 Singapore): official-method seed0 PPL and eight tasks on **Llama-3.2-3B and Meta-Llama-3.1-8B**, plus inference speed. See the [current 3B/8B tables](../../current/table.md) and [same-A40 8B speed job1736351](../../current/native_8b/table.md). No new FP16 control. Earlier entries below retain their historical scope.

DuQuant on Meta-Llama-3.1-8B seed0 is submitted in two-PRO6000 job1736252, pending Priority, with no numerical prerequisite.

The configuration is official run.sh W4A4 baseline,epochs0,block128,max_rotation_step256,alpha0.6,lac0.9,swc0.8; floating fake-quant execution.

Full141 PPL and the common eight zero-shot tasks remain unmeasured; no missing metric is filled from another model or protocol.

Quantized checkpoints, calibration artifacts and per-sequence NLLs are saved for reuse without repeated calibration.

The user has restricted active work to FlatQuant and DuQuant and explicitly excluded new FP16/PrismQuant runs; seed0 and the common evaluation protocol are disclosed overrides where official defaults differ.

[Table](table.md); [current submission](../e38/raw/baselines_only_submission_20261112.json).
