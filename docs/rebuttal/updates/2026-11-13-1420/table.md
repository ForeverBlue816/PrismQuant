**Llama-3.2-3B — seed0；每格为正确数／总数**

| 任务 | PrismQuant k=8（历史参考） | FlatQuant | DuQuant |
| --- | ---: | ---: | ---: |
| ARC-c | 502/1172 | 519/1172 | 480/1172 |
| ARC-e | 1622/2376 | 1649/2376 | 1561/2376 |
| BoolQ | 2357/3270 | 2391/3270 | 2203/3270 |
| HellaSwag | 7116/10042 | 7151/10042 | 6741/10042 |
| OpenBookQA | 194/500 | 200/500 | 196/500 |
| PIQA | 1391/1838 | 1400/1838 | 1347/1838 |
| SocialIQA | 888/1954 | 874/1954 | 848/1954 |
| WinoGrande | 858/1267 | 829/1267 | 798/1267 |
| 八任务均值 | 60.2106% | 60.5431% | 57.5025% |

**Llama-3.1-8B — seed0；每格为正确数／总数**

| 任务 | PrismQuant k=8（历史参考） | FlatQuant | DuQuant |
| --- | ---: | ---: | ---: |
| ARC-c | 573/1172 | 591/1172 | 549/1172 |
| ARC-e | 1877/2376 | 1888/2376 | 1752/2376 |
| BoolQ | 2620/3270 | 2659/3270 | 2500/3270 |
| HellaSwag | 7727/10042 | 7720/10042 | 7498/10042 |
| OpenBookQA | 221/500 | 207/500 | 206/500 |
| PIQA | 1436/1838 | 1443/1838 | 1388/1838 |
| SocialIQA | 923/1954 | 912/1954 | 875/1954 |
| WinoGrande | 866/1267 | 904/1267 | 856/1267 |
| 八任务均值 | 65.3592% | 65.7515% | 62.5947% |

所有48个任务单元均已完成。FlatQuant8B现已8/8完成，八任务均值65.751548%。ARC-e、ARC-c、HellaSwag、OpenBookQA、PIQA使用acc_norm；BoolQ、SocialIQA、WinoGrande使用acc。八任务均值按八个任务准确率等权平均，不是合并样本计数后求比率。

FlatQuant/DuQuant正确数由逐条0/1指标求和，已验证doc_id唯一、effective样本总数与聚合准确率一致。PrismQuant使用上一表对应的历史k8seed0结果：正确数由原始完整精度accuracy乘日志effective总数恢复（乘积均为整数至浮点误差），没有使用四舍五入的展示百分比；这些历史文件不含逐条预测，未做逐条核验。

PrismQuant3B采用asymmetric g128 GPTQ权重，8B采用symmetric per-channel GPTQ权重；两者均asymmetric g128 A4和模拟KV4。FlatQuant/DuQuant是本轮官方配置的浮点量化评测。此表不是部署内核精度或三种子均值。

[CSV：16行宽表，整数列分开](counts.csv) · [CSV：48行明细含百分比](counts_long.csv) · [来源与核验](provenance.json)
