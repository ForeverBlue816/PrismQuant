# PrismQuant rebuttal 实验索引 — 2026-11-13

**状态快照：2026-11-13 00:27 新加坡时间（UTC+8）。** 测量结果按各原始文件的完成时间记录；调度状态见 [scheduler_snapshot.json](scheduler_snapshot.json)。这里是当前状态入口，历史文件里最后写入的 `RUNNING` 不代表作业仍在运行。

## 当前范围与已完成结果

当前只推进 **Llama-3.2-3B / Meta-Llama-3.1-8B 的官方 FlatQuant、DuQuant**，以及 **8B PrismQuant 部署速度优化**。不新增 FP16、OffQ、E40/E41 扩展、多种子或70B作业。[当前协议](protocol.json)。

| 模型 | 方法 / 实际执行路径 | 完整141段 PPL | 八任务 | 速度状态 |
| --- | --- | ---: | --- | --- |
| **Llama-3.2-3B** | DuQuant 官方 W4A4，FP16 fake-quant | **10.349871** | 运行中，1/8完成，尚无完整均值 | 排队等待 checkpoint 作业结束 |
| **Llama-3.2-3B** | FlatQuant 官方 W4A4KV4，BF16 fake-quant | 未完成，训练中 | 未完成 | 排队 |
| **Meta-Llama-3.1-8B** | DuQuant 官方 W4A4，FP16 fake-quant | 未完成，排队 | 未完成 | 排队 |
| **Meta-Llama-3.1-8B** | FlatQuant 官方 W4A4KV4，BF16 fake-quant | 未完成，排队 | 未完成 | 排队 |
| **Meta-Llama-3.1-8B** | PrismQuant k=8，原生 packed INT4 部署变体 | **9.995008** | 本轮优化未跑 | 原版/优化版同卡对照已完成；跨方法比较排队 |

- [3B 官方方法明细](current/llama32_3b/table.md) / [summary](current/llama32_3b/summary.md)
- [8B 官方方法明细](current/llama31_8b/table.md) / [summary](current/llama31_8b/summary.md)
- [8B 跨方法速度表（尚未完成）](current/native_8b/table.md)

**fake-quant 使用浮点运算模拟量化；不能把其耗时称为原生 INT4 kernel 性能。** 每条部署计时路径单独测量 PPL，不把论文精度行直接贴到另一个运行后端上。

## 已完成：8B PrismQuant 旋转优化

模型是 **Meta-Llama-3.1-8B，k=8，seed0**；实际格式为 **GPTQ per-channel symmetric W4 / per-token symmetric A4 / native asymmetric per-head128 KV4**。这不是 Table1 的 asymmetric g128 / KIVI 格式。

| 指标 | 原版 | CUDA 旋转优化后 |
| --- | ---: | ---: |
| PPL，141段 | 9.995008261 | 9.995008261 |
| Prefill，median | 625.837 ms | 346.651 ms（1.805×） |
| Decode，median | 13.765 tokens/s | 18.719 tokens/s（1.360×） |
| 峰值 allocated 显存 | 8.115 GiB | 7.791 GiB |

两行使用同一张独占 A40、相同量化 checkpoint；141段 NLL 逐段相等，64个实际输入旋转探针的记录差值为0。该探针结果不等同于穷尽的整模型 logit 一致性检验。

[完整表、IQR与分解](records/e38/rotation_optimization/table.md) / [summary](records/e38/rotation_optimization/summary.md) / [原版记录](records/e38/raw/stage_a_8b_nar_k8_seed0_original_job1736296.json) / [优化记录](records/e38/raw/stage_a_8b_nar_k8_seed0_fast_job1736296.json)。

计时作业 **1736296** 已完成。上述两行采用匹配的预分配缓存协议；新跨方法作业 **1736351** 改为 fresh cache，prefill 包含缓存创建，decode 前缀在计时外重建。旧数字不会代替新的跨方法结果。新作业顺序测 FlatQuant 官方 packed 后端、同后端去除被立即覆盖的随机分配、PQ优化后端和两种官方浮点模拟路径；各路径先测完整 PPL，再在同一已加载对象上计时。

## 防止模型与数字混淆

| 数字 | 对应模型与配置 | 使用范围 |
| --- | --- | --- |
| **9.9950 PPL** | **8B** PQ k=8，per-token symmetric 原生 INT4 部署版 | 当前旋转优化前后对照 |
| 6.990412 → 6.989246 PPL | **8B** g128参考 → 展开到 INT8 GEMM 的整数原型 | 历史原型；不能称原生 INT4 加速，logit一致性也未建立 |
| 8.718 / 60.03 | **3B** 历史 k=max、seed0参考行 | 与论文 k=8 三种子均值不是同一配置 |
| 8.60 / 60.80 | 论文 **3B** Table1 k=8 三种子均值 | 论文锚点，不是本轮9.9950的来源 |

[本次8B身份核对](model_identity.json)：模型记录和 checkpoint manifest 都是 `llama31_8b`，共有 **32个 packed layers**，第一层 K 投影形状为 **1024×4096**。manifest 内另有一个历史3B `fp32_algebra_control` 诊断引用，它不是本次被测模型。[历史3B配置核对](records/e38/audit_a1_reference_table1.md)。

## 当前作业与中断记录

| 作业 | 内容 | 快照时状态 |
| --- | --- | --- |
| 1736285 | 两张A40：3B FlatQuant训练、DuQuant八任务 | RUNNING；DuQuant使用已完成的 checkpoint |
| 1736252 | 两张PRO6000：8B FlatQuant、DuQuant精度 | PENDING / Priority |
| 1736286 | 3B两方法同卡速度 | PENDING / afterany1736285 |
| 1736351 | 8B同卡跨后端速度，五条路径 | PENDING / afterany1736252 |
| 1736296 | 8B PQ原版/优化版 | COMPLETED |
| 1736287 | 旧8B两方法速度安排 | CANCELLED，被1736351替换 |
| 1736074 / 1736075 / 1736076 | 旧多实验/种子/FP16任务安排 | CANCELLED，当前不再推进 |

DuQuant 3B 初次运行遇到 RoPE 设备兼容问题，修复后复用保存的校准统计，完成 PPL 与 checkpoint。其 retry1 八任务在客户端中断时停止；当前为 **retry2 / step1736285.6**。失败与中断的原始记录均保留，不将其当作完整评测。[恢复记录](current/duquant_3b_tasks_recovery_20261113.json)。

## 历史与暂停实验

- [E38历史表与审计](records/e38/table.md)：包含此前3B/8B、FP16、g128整数原型与各部署变体。最新选中行可能来自不同作业，不能直接称为同卡对照。
- [E39历史记录](records/e39/table.md)、[E42历史记录](records/e42/table.md)：当前官方方法状态以本页与 current/ 表为准。
- [E41六组与bootstrap](records/e41/table.md)：已完成的是 **8B、BF16 W/KV、仅激活A4 g128** 机制实验；扩展暂停，不能当作完整W4A4KV4精度行。
- [E40历史计划](records/e40/table.md)：暂停，无新增GPU测量；旧文本里的提交计划不代表当前安排。
- [完整历史叙述](history.md)。

## 文件与来源

[manifest.json](manifest.json) 记录每个导出文件的源路径、源SHA256、发布SHA256及变换。完整测量汇总、逐段NLL、计时样本、配置与失败记录均保留；harness 的逐样本题面/响应载荷未上传，原文件仍在集群，路径、大小与SHA可供核验。JSON未标记变换的文件与源文件逐字节一致。权重和数据集不进入此记录目录。

[执行脚本索引](entrypoints/README.md) 保存所用 runner、适配器、计时和Slurm脚本。它们依赖原集群工作树与固定环境，是归档执行入口，不是独立可直接运行的复现包。源码身份以 manifest 的各文件SHA为准，提交号同时保留。
