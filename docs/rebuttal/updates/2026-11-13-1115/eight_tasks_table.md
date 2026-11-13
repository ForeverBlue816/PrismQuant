**Llama-3.2-3B — seed0, zero-shot, common frozen harness, batch1; official FlatQuant/DuQuant quantization recipes, floating fake-quant execution**

| Task / metric | FlatQuant % | DuQuant % |
| --- | ---: | ---: |
| arc_easy / acc_norm,none | 69.402 | 65.699 |
| arc_challenge / acc_norm,none | 44.283 | 40.956 |
| boolq / acc,none | 73.119 | 67.370 |
| hellaswag / acc_norm,none | 71.211 | 67.128 |
| openbookqa / acc_norm,none | 40.000 | 39.200 |
| piqa / acc_norm,none | 76.170 | 73.286 |
| social_iqa / acc,none | 44.729 | 43.398 |
| winogrande / acc,none | 65.430 | 62.983 |
| Eight-task mean | **60.5431** | **57.5025** |

DuQuant7 completed task outputs were reused after verified checkpoint/protocol/file identity; only Winogrande was rerun byjob1736906. No partial mean is used.
