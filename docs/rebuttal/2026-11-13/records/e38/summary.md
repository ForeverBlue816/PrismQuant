> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

Collected 15 actual 8B deployment-run artifacts; current row states: native/fp16: COMPLETE; tasks RUNNING; native/hadamard: COMPLETE; native/nar_k8: COMPLETE; integer/hadamard: COMPLETE; tasks RUNNING; integer/nar_k8: COMPLETE; tasks RUNNING; absent metrics remain pending.

Seed0 reference minus Table1 reported values (not a three-seed equivalence claim): hadamard: ΔPPL -0.00026, Δtask mean pendingpp; nar_k8: ΔPPL 0.00041, Δtask mean pendingpp.

Native per-token symmetric A4/per-head KV4 format results: hadamard PPL 8.56296; nar_k8 PPL 9.99501; these are measured format variants rather than Table1 g128 results.

Integer-path output diagnostics: hadamard: max |Δlogit| 27.0625, argmax agreement 85.3055%; nar_k8: max |Δlogit| 28.0000, argmax agreement 86.9130%; close corpus PPL does not establish logit or prediction equivalence.

The g128 prototype uses expanded INT8 GEMM, component profiles are non-additive, bootstrap intervals condition on fixed runs rather than calibration seeds, and no numerical difference blocks execution.

[Table and raw source links](table.md)
