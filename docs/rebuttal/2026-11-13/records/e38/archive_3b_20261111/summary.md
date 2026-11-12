> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../../README.md) for the active scope and observed scheduler state.

Four seed0 direct accuracy jobs completed all141 WikiText-2 sequences and all eight zero-shot tasks: 3B FP16 PPL7.81658 / mean62.7324%, 3B PQ simulator8.71802 /60.0324%, 3B PQ integer prototype8.71940 /60.4988%, and8B FP16 6.23843 /68.6753%.
The integer-minus-simulator PPL difference is+0.00137719, or0.015797% relative, and the eight-task mean difference is+0.466341 percentage points; these runs used PRO6000 Server and A40 respectively.
Each row retains22419 item-level outputs, and the unweighted eight-task integer-minus-reference mean has a10000-resample paired bootstrap95%CI of[-0.064921,+1.002023] percentage points, conditional on these two fixed runs.
No numerical difference blocked evaluation; the user's0.5% PPL reference is for reporting only, and no new timing was run.
Earlier A40 FP16 timing remains in the table as a separately labelled job: 3B/8B prefill210.955/417.469ms and decode42.911/29.038tokens/s at B1, prefix2048,10 warmups and50 repetitions.

- Against FP16: 3B integer prototype PPL is higher by0.90282247 and eight-task mean is lower by2.233608 percentage points.
- Mixed task differences versus simulator: integer prototype ARC-Easy is lower by0.841751pp and ARC-Challenge lower by0.511945pp; the other six task accuracies are higher. Cross-device differences are descriptive.
- Not yet complete: full seed0 Hadamard/k8/kmax accuracy comparisons across both model sizes, official E39 baselines and broader E42 paired uncertainty.
- No integer speed claim: the prototype expands signed4 storage to INT8 GEMM with original E14 rotations and simulated KV; no native INT4 timing was measured.
- Excluded/deferred by the user: all OffQ experiments, additional seeds and new prototype timing.
- Historical partial and failed attempts remain in raw logs and the root chronology; they are not prerequisites for these completed measurements.
- Submitted Table1 correspondence remains unresolved for some archived results; these measurements report their actual model/protocol identities.
