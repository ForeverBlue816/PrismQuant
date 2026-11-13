# Rebuttal 当前实验状态

更新时间：**2026-11-13 13:47 新加坡时间**。

**OffQ不运行，也不进行无公开官方代码的复现。** 当前队列无OffQ，历史通用worker2已明确禁用。当前优先完成官方FlatQuant和DuQuant。

## Meta-Llama-3.1-8B — 新部署比较，seed0

**1736915 已在A40运行**，顺序测试FP16和PQ k=8的g128 asymmetric整数原型。两条路径都先测完整141段PPL，再在同一已加载模型上计时；B1、prefix2048、decode128/drop8、10warmups/20repeats、fresh cache和last-position logits，逐次检查GPU独占。

| 同一A40的部署行 | 完整141段PPL | Prefill median | Decode median | 状态 |
| --- | ---: | ---: | ---: | --- |
| FP16，原生FP16 paged KV | **6.238432** | **389.488 ms** | **33.1000 tokens/s** | 完成 |
| PQ k=8，GPTQ W4存储／asymmetric g128 A4／模拟KV4 | **6.989246** | **13214.108 ms** | 进行中 | decode未完成 |

**g128仍是展开INT8计算的整数原型，不是原生packed INT4加速后端。** 它保留BF16 KV cache并在attention中模拟KV4；FP16与其软件栈及attention/cache实现不同，结果比较的是完整后端，不能解释为纯格式开销。完成后再汇总完整速度比。

FlatQuant官方packed INT4作业 **1736351** 已于**12:43新加坡时间**自动解除checkpoint依赖，目前因`QOSMaxJobsPerUserLimit`等待两个运行名额之一释放。它将顺序测官方内核、移除无效随机分配的对照、PQ优化后的原生per-token symmetric后端以及浮点对照；各路径单独测部署PPL，使用同一张A40和相同计时协议。**现有fake-quant PPL不会冒充原生内核PPL。**

[部署记录与冻结快照](updates/2026-11-13-1200/README.md) · [历史g128/FP16同卡结果](updates/2026-11-13-1200/g128_vs_fp16_historical/table.md) · [实际解除依赖记录](updates/2026-11-13-1200/native_speed_release_1736351.json)

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
| FlatQuant，W4RTN/A4/KV4，BF16 fake-quant，15epochs | **7.020543** | 进行中 | PPL与checkpoint完成，八任务进行中 |
| DuQuant，官方W4A4/QK/V，FP16 fake-quant，epochs0 | **8.188651** | **62.5947%** | PPL与八任务8/8完成 |

两方法的checkpoint已经保存，job1736252仍在运行FlatQuant八任务。FlatQuant校准耗时**59.887分钟**、**0.9981 GPU小时**，峰值allocated **20.2995 GiB**，运行于PRO6000。配对Flat−Du PPL差**−1.16811**，95%逐段bootstrap区间**[−1.23328,−1.10578]**（固定seed0，非校准种子区间）。这些精度结果和上方原生部署比较分别记录。

[本轮8B原始记录索引](updates/2026-11-13-1200/README.md) · [源文件与导出SHA](updates/2026-11-13-1200/manifest.json)

## 之前的PrismQuant部署优化

PPL **9.9950** 对应 **Meta-Llama-3.1-8B** 的per-token symmetric原生部署变体，不能与上方3B行或论文Table1 g128行混为一谈。它的原版/优化版同卡prefill625.837→346.651ms、decode13.765→18.719tokens/s，完整PPL保持不变。跨方法的8B原生部署比较仍待完成。

[8B PQ优化表](2026-11-13/records/e38/rotation_optimization/table.md) · [00:27完整历史归档](2026-11-13/README.md)。
