# Rebuttal 当前实验状态

更新时间：**2026-11-13 15:18 新加坡时间**。

**OffQ仍不运行。** 以下为本机A40实测，取代此前引用其他硬件数字作为本地性能依据的做法。

## Meta-Llama-3.1-8B — 两组A40实测已完成，seed0

**job1736915：同卡FP16／g128整数原型。** 两行都在同一已加载模型上完成141段PPL和计时。

| 后端 | 完整141段PPL | Prefill ms | Decode tokens/s | 峰值allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 6.238432 | 389.488 | 33.10000 | 17.4618 |
| PQ k=8，g128整数原型 | 6.989246 | 13214.108 | 0.40628 | 7.3721 |

g128原型相对FP16的prefill慢**33.9269倍**、decode慢**81.4707倍**。这是实际延迟测量，但该实现使用W4存储展开INT8分组GEMM，并在BF16 cache中模拟KV4；**它不构成原生g128 INT4加速结果**。两条路径的框架、attention和cache实现不同，不能据此隔离格式本身的开销，也不能把1.6%乘加或3.1%存储开销解释为相应延迟开销。

**job1736351：另一张同卡A40上的FlatQuant／PQ／DuQuant。** 五行完整PPL和计时均已完成。

| 后端 | 完整141段PPL | Prefill ms | Decode tokens/s | 峰值allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| FlatQuant 官方 packed INT4 | 6.952981 | 302.482 | 9.74138 | 6.2440 |
| FlatQuant，移除无效随机分配 | 6.952981 | 294.382 | 10.40427 | 6.2440 |
| PQ k=8 原生 INT4，per-token symmetric | 9.995008 | 377.209 | 16.75236 | 7.7907 |
| FlatQuant 官方浮点 fake-quant | 7.020863 | 1070.645 | 5.55694 | 16.6105 |
| DuQuant 官方浮点 fake-quant | 8.198439 | 1163.505 | 4.84537 | 16.4019 |

PQ原生per-token变体相对FlatQuant官方packed内核，decode快**1.7197倍**，prefill延迟增加**24.7046%**；相对清理版，decode快**1.6101倍**，prefill延迟增加**28.1359%**。同时PPL为**9.9950对6.9530**，所以不能宣称同等精度下全面加速，也不能把该变体作为Table1 g128部署结果。

共同计时协议：batch1、prefix2048、decode128/drop8、token100、10次预热／20次测量、last-position logits、fresh cache，eager同步墙钟计时。每行43/43 GPU独占检查通过。两组分属不同A40，速度比只在各自同卡组内计算。FlatQuant原生prefill使用变换后的未量化KV，decode使用packed KV；PQ使用原生packed per-head KV。浮点fake-quant实现与packed内核分别列出。

本次A40浮点PPL与PRO6000校准后评测略有差异：FlatQuant7.020543→7.020863，DuQuant8.188651→8.198439。同checkpoint、同141段tokens、两次均use_cache=False；具体数值漂移原因未隔离，不能直接归因cache。各速度行保留其同次部署PPL。

[A40完整原始记录与说明](updates/2026-11-13-1515/README.md) · [g128／FP16完整表](updates/2026-11-13-1515/e38/g128_vs_fp16/job1736915/table.md) · [FlatQuant／PQ／DuQuant完整表](updates/2026-11-13-1515/native_8b/table.md)

## Llama-3.2-3B — seed0：PPL、八任务、同卡速度全部完成

| 方法 / 官方配置 | 完整141段 PPL | 八任务均值 | Prefill median | Decode median | 峰值allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| FlatQuant，W4RTN/A4/KV4，BF16 fake-quant，15 epochs | **8.716573** | **60.5431%** | **580.081 ms** | **7.5693 tokens/s** | 7.2548 GiB |
| DuQuant，官方W4A4/QK/V quantizers，FP16 fake-quant，epochs0 | **10.349871** | **57.5025%** | **606.882 ms** | **6.7625 tokens/s** | 7.1000 GiB |

**速度是官方浮点fake-quant实现的运行时间，不是原生packed INT4 kernel性能。** 两条速度行在job1736286同一张独占A40上顺序完成，各43/43独占检查通过；B1、prefix2048、decode128/drop8、token100、10次warmup/20次计时、fresh cache、last-position logits。每条路径用同一已加载模型先测完整PPL再计时，141段NLL与其校准后记录逐段相同。

[逐任务结果](updates/2026-11-13-1115/eight_tasks_table.md) · [FlatQuant八任务原始汇总](updates/2026-11-13-1115/flatquant_3b_eight_tasks.json) · [DuQuant完整八任务与复用证据](updates/2026-11-13-1115/duquant_3b_eight_tasks_complete.json) · [FlatQuant计时](updates/2026-11-13-1115/flatquant_3b_timing.json) · [DuQuant计时](updates/2026-11-13-1115/duquant_3b_timing.json)

两方法使用相同141×2048测试窗口、288,627 scored tokens。配对FlatQuant−DuQuant PPL差 **−1.633298**，95%逐段bootstrap区间 **[−1.724958,−1.544721]**（20,000次，固定seed0）。该区间不代表校准种子变化。DuQuant为官方run.sh的epochs0配置，不是DuQuant+LWC20epoch；没有调参拟合论文。

DuQuant旧作业的最后一项Winogrande曾被取消；job1736906验证并复用七个完整任务，仅补测缺失的一项，现已得到完整八任务均值。原中断记录保留并标为INTERRUPTED，不能再当作运行中的任务。

## Meta-Llama-3.1-8B — 官方浮点评测路径，seed0

| 官方方法 | 完整141段PPL | 八任务均值 | 状态 |
| --- | ---: | ---: | --- |
| FlatQuant，W4RTN/A4/KV4，BF16 fake-quant，15epochs | **7.020543** | **65.7515%** | PPL与八任务8/8完成 |
| DuQuant，官方W4A4/QK/V，FP16 fake-quant，epochs0 | **8.188651** | **62.5947%** | PPL与八任务8/8完成 |

两方法的checkpoint已经保存，job1736252已完成两方法全部八任务。FlatQuant校准耗时**59.887分钟**、**0.9981 GPU小时**，峰值allocated **20.2995 GiB**，运行于PRO6000。配对Flat−Du PPL差**−1.16811**，95%逐段bootstrap区间**[−1.23328,−1.10578]**（固定seed0，非校准种子区间）。这些精度结果和上方原生部署比较分别记录。

[两模型三方法逐任务正确数/总数](updates/2026-11-13-1420/table.md) · [可填表CSV](updates/2026-11-13-1420/counts.csv) · [FlatQuant8B完整八任务](updates/2026-11-13-1420/flatquant8b_eight_tasks_complete.json)

[本轮8B原始记录索引](updates/2026-11-13-1200/README.md) · [源文件与导出SHA](updates/2026-11-13-1200/manifest.json)

## 之前的PrismQuant部署优化

PPL **9.9950** 对应 **Meta-Llama-3.1-8B** 的per-token symmetric原生部署变体，不能与上方3B行或论文Table1 g128行混为一谈。它的原版/优化版同卡prefill625.837→346.651ms、decode13.765→18.719tokens/s，完整PPL保持不变。跨方法的8B原生部署比较已完成，以上方job1736351的fresh-cache结果为准。

[8B PQ优化表](2026-11-13/records/e38/rotation_optimization/table.md) · [00:27完整历史归档](2026-11-13/README.md)。
