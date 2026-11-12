> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

**E38 — Meta-Llama-3.1-8B, trained default-GPTQ W4, asymmetric g128 A4 / E14 simulated KV4; reference versus INT8 arithmetic prototype, seed0**

| Rotation | Task | Reference % | Integer % | Delta pp [95% paired CI] | Prediction changed | Wrong→wrong changed | Correct→wrong | Wrong→correct | Correctness flips | Paired items | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hadamard | arc_easy | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hadamard | arc_challenge | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hadamard | boolq | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hadamard | hellaswag | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hadamard | openbookqa | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hadamard | piqa | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hadamard | social_iqa | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hadamard | winogrande | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| nar_k8 | arc_easy | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| nar_k8 | arc_challenge | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| nar_k8 | boolq | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| nar_k8 | hellaswag | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| nar_k8 | openbookqa | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| nar_k8 | piqa | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| nar_k8 | social_iqa | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| nar_k8 | winogrande | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

Pairs require identical doc_id/filter/doc_hash, actual document/prompt/target inputs, tokenizer hashes, checkpoint identity, shared E14 source, recorded harness commit/init-module hash, B1 and max_length2048. HFLM causal likelihood uses right zero padding; tokenizer configuration is recorded separately. Prediction changes include wrong→wrong changes, recovered from stored filtered responses using frozen harness argmax and task doc_to_choice character normalization for acc_norm; unrecoverable predictions are unavailable with a recorded reason. Correctness flips separately count reported metric changes0↔1.

hadamard: pending; reference pending; integer RUNNING.

nar_k8: pending; reference pending; integer RUNNING.

[Paired metadata, source links, changed-item identities and intervals](task_flips.json).
