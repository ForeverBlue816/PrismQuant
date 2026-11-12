> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

Status: `superseded_by_8b_only_scope`; the partial [CPU audit](raw/audit_a3_integer_001.json) is retained unchanged, its retry was stopped, and no further 3B statistics or GPU work is planned ([scope record](raw/audit_a3_integer_scope_001.json)).

The existing 3B direct runs share recorded seed0, batch8 task settings, tokenizer, harness and requests, but integer job1733510 used RTX PRO 6000 Server whereas reference job1733511 used A40; their logs do not retain full-vocabulary logits or tokenwise argmax, and neither these cross-GPU results nor older within-GPU diagnostics should be presented as the requested 8B same-instance evidence or entered into the main table.
