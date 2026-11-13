# Rebuttal 当前实验状态

更新时间：**2026-11-13 11:23 新加坡时间**。

**OffQ不运行，也不进行无公开官方代码的复现。** 当前队列无OffQ，历史通用worker2已明确禁用。当前优先完成官方FlatQuant和DuQuant。

## Llama-3.2-3B — seed0：PPL、八任务、同卡速度全部完成

| 方法 / 官方配置 | 完整141段 PPL | 八任务均值 | Prefill median | Decode median | 峰值allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| FlatQuant，W4RTN/A4/KV4，BF16 fake-quant，15 epochs | **8.716573** | **60.5431%** | **580.081 ms** | **7.5693 tokens/s** | 7.2548 GiB |
| DuQuant，官方W4A4/QK/V quantizers，FP16 fake-quant，epochs0 | **10.349871** | **57.5025%** | **606.882 ms** | **6.7625 tokens/s** | 7.1000 GiB |

**速度是官方浮点fake-quant实现的运行时间，不是原生packed INT4 kernel性能。** 两条速度行在job1736286同一张独占A40上顺序完成，各43/43独占检查通过；B1、prefix2048、decode128/drop8、token100、10次warmup/20次计时、fresh cache、last-position logits。每条路径用同一已加载模型先测完整PPL再计时，141段NLL与其校准后记录逐段相同。

[逐任务结果](updates/2026-11-13-1115/eight_tasks_table.md) · [FlatQuant八任务原始汇总](updates/2026-11-13-1115/flatquant_3b_eight_tasks.json) · [DuQuant完整八任务与复用证据](updates/2026-11-13-1115/duquant_3b_eight_tasks_complete.json) · [FlatQuant计时](updates/2026-11-13-1115/flatquant_3b_timing.json) · [DuQuant计时](updates/2026-11-13-1115/duquant_3b_timing.json)

两方法使用相同141×2048测试窗口、288,627 scored tokens。配对FlatQuant−DuQuant PPL差 **−1.633298**，95%逐段bootstrap区间 **[−1.724958,−1.544721]**（20,000次，固定seed0）。该区间不代表校准种子变化。DuQuant为官方run.sh的epochs0配置，不是DuQuant+LWC20epoch；没有调参拟合论文。

DuQuant旧作业的最后一项Winogrande曾被取消；job1736906验证并复用七个完整任务，仅补测缺失的一项，现已得到完整八任务均值。原中断记录保留并标为INTERRUPTED，不能再当作运行中的任务。

## Meta-Llama-3.1-8B — 官方FlatQuant、DuQuant

精度作业 **1736252** 仍在Priority排队，尚无完整8B FlatQuant/DuQuant PPL或八任务结果。部署比较1736351等待对应checkpoint作业。保留原排队年龄，没有新增FP16或OffQ工作。

[本轮调度快照](updates/2026-11-13-1115/scheduler_completion.json) · [源文件与导出SHA](updates/2026-11-13-1115/manifest.json) · [当前范围/OffQ禁用](updates/2026-11-13-1115/protocol.json)

## 之前的PrismQuant部署优化

PPL **9.9950** 对应 **Meta-Llama-3.1-8B** 的per-token symmetric原生部署变体，不能与上方3B行或论文Table1 g128行混为一谈。它的原版/优化版同卡prefill625.837→346.651ms、decode13.765→18.719tokens/s，完整PPL保持不变。跨方法的8B原生部署比较仍待完成。

[8B PQ优化表](2026-11-13/records/e38/rotation_optimization/table.md) · [00:27完整历史归档](2026-11-13/README.md)。
