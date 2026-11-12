# Rebuttal 当前实验状态

更新时间：**2026-11-13 01:03 新加坡时间**。当前优先完成官方 **DuQuant、FlatQuant** 的 PPL、八任务和速度；模型为 Llama-3.2-3B 和 Meta-Llama-3.1-8B。

## Llama-3.2-3B — seed0，官方方法原生量化配置，浮点 fake-quant 执行

| 方法 | 配置 | 完整141段 PPL | 八任务 | 计时 |
| --- | --- | ---: | --- | --- |
| FlatQuant | W4 RTN / per-token symmetric A4 / per-head asymmetric KV4；BF16 fake-quant；15 epochs | **8.716573** | 0/8完成，继续运行 | checkpoint已完成，已解除对八任务结束的依赖，排队中 |
| DuQuant | 官方run.sh W4A4、QK/V quantizers；FP16 fake-quant；epochs0 | **10.349871** | 3/8完成，继续运行 | checkpoint已完成，同一个计时作业排队中 |

两行使用相同3B测试tokens，均为141×2048窗口、288,627个scored tokens。FlatQuant − DuQuant 的配对PPL差为 **−1.633298**，95%区间 **[−1.724958, −1.544721]**；20,000次逐段bootstrap，固定seed0，不代表校准种子不确定性。尚无完整八任务均值。

FlatQuant完成15轮变换训练，耗时 **71.80分钟 / 1.19663 GPU小时**，训练峰值allocated显存 **13.200 GiB**。DuQuant采用官方run.sh的epochs0配置，**不是DuQuant+LWC20epoch**。没有为拟合论文数字调整参数。

[FlatQuant完整PPL记录](updates/2026-11-13-0100/flatquant_3b_ppl.json) · [DuQuant完整PPL记录](2026-11-13/records/e42/raw/duquant_3b_seed0_quantize-ppl_retry1_job1736285.json) · [配对CI与来源](updates/2026-11-13-0100/paired_ppl_collection.json)

## Meta-Llama-3.1-8B — 官方FlatQuant、DuQuant

精度作业 **1736252** 仍为 **PENDING / Priority**；没有已完成的8B FlatQuant/DuQuant PPL或八任务数字。已检查其他合法GPU资源：重新提交的估计等待更长，保留现有队列年龄。集群拒绝修改既有作业TimeLimit，因此该作业实际仍为原来的72小时上限；这不是已经运行72小时。

## 作业推进

- **1736285**：两卡并行3B八任务，两个PPL和量化checkpoint均已完成。
- **1736286**：3B两方法同卡速度，已清除原afterany1736285依赖；现在按Priority排队，不再等八任务全部结束。
- **1736252**：8B两方法官方精度，保持原排队年龄。
- **1736351**：8B同卡跨后端速度，仍等待8B训练checkpoint作业。

[调度快照](updates/2026-11-13-0100/scheduler.json) · [调度调整及返回结果](updates/2026-11-13-0100/priority_update.json) · [本次新增记录SHA](updates/2026-11-13-0100/manifest.json)

## 之前完成的部署优化与历史记录

**PPL9.9950是8B PrismQuant部署变体，不是3B，也不是Table1 g128精度行。** 原版/优化版在同卡上保持141段NLL一致，prefill625.837→346.651ms、decode13.765→18.719tokens/s。与DuQuant/FlatQuant的完整跨方法速度比较尚未完成。

[8B PQ优化表](2026-11-13/records/e38/rotation_optimization/table.md) · [00:27完整归档索引](2026-11-13/README.md)。归档记录保留原时间，本页更新当前进度。
