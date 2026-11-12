> Historical record snapshot. Some original job states and plans are stale. Use the [current status index](../../README.md) for the active scope and observed scheduler state.

# A1：3B 当前 reference 与投稿 Table 1 的来源核对

核对日期：2026-11-11 UTC。仅 CPU 读取既有结果、源码、小型 calibration/rotation tensors，并列出公开仓库文件；没有提交 GPU，也没有下载权重。完整数据、文件 SHA256、逐层 rank 和搜索范围见 [audit_a1_reference_table1_001.json](raw/audit_a1_reference_table1_001.json)。

**当前 reference job 1733511 不是投稿 Table 1 的 3B k=8 配置，不能把两者差值解释成同配置复现误差。** 它实际是 k=max、seed0、对称 per-output-channel GPTQ、BF16 容器；投稿 Appendix A.1 写的是 3B asymmetric GPTQ group128、FP32 容器，Table 1 写三种子平均。

## 1. 直接核对投稿 PDF

原件：[11144_PrismQuant_Optimal_Null_.pdf](../../manifest.json)，SHA256 `e40d8ecfbf63fcf562ae01de585b3316cfd7521483d0377f5958688799db7f87`。本次直接提取的物理页 7、8、14 见 [PDF 文本证据](../../manifest.json)。下列“行号”为 PDF 自带 manuscript line number。

| 投稿 Table 1，物理第 8 页 | PPL | ARC-c | ARC-e | BoolQ | HellaSwag | OBQA | PIQA | SIQA | WinoGrande | 八任务平均 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PrismQuant k=8，行391 | 8.60 | 43.43 | 69.91 | 74.07 | 71.59 | 40.20 | 74.97 | 45.50 | 66.77 | 60.80 |
| PrismQuant k=max，行392 | 8.58 | 43.69 | 70.24 | 74.16 | 71.60 | 41.00 | 75.08 | 45.91 | 68.19 | 61.23 |

因此，用户给出的 **k=8 / 8.60 / 60.80** 与投稿版相符。第 8 页行418–419及第14页行740明确声称 PrismQuant 结果为三随机种子平均。

Appendix A.1（物理第14页）行708–710写 FP32 containers holding released BF16 weights；行711–715写 3B 权重采用 asymmetric GPTQ group128，约4.16 bits，8B/70B为 per-channel。行718–720写 K 的 token-axis group32、V 的 head group128、32-token full-precision residual。行721–729写 rotation 校准128×2048 WikiText-2 train，且与 GPTQ 用同一批 sequences。行732–740写完整141×2048 WikiText-2 PPL、八任务零样本及指标定义。这里报告的是 PDF 声明，不把它自动视为已由原始日志证实。

## 2. 当前 reference 1733511 的实际配置

主证据：[direct_reference_3b_kmax_seed0_job1733511.json](raw/direct_reference_3b_kmax_seed0_job1733511.json)；调用路径：[e38_direct_accuracy.py](../../manifest.json)，特别是行210–216实际 loader 和 RuntimeHooks。

| 项目 | 已核对值及来源 |
|---|---|
| 模型 | `unsloth/Llama-3.2-3B`，base revision `d4446454d87d51aa42e1fb174f25acc5f8762331` |
| 旋转 | `nar_kmax`；CPU 读取实际全部 factors：R1 active rank24（3072/128），28层R2均rank1，28层R4均rank64（8192/128）；不是 k=8 |
| 种子 | runner/checkpoint seed0；GPTQ calibration_seed0；既有 rotation manifest 的 seed 是 `20260902`，不能把这些不同字段合并成同一个随机数 |
| 权重 | `protocol=default`，4 bits、perchannel=true、symmetric=true、groupsize=-1、act_order=false；MSE clip norm2.4/grid100/maxshrink0.8、blocksize128、percdamp0.01、static_groups=false；embed_tokens/lm_head不做GPTQ |
| checkpoint | `.../models--ForeverBlue--nar-w4a4kv4-llama-3.2-3b/snapshots/25c6b2d9a04151fab7e5f37b9ce93c8f0edf2ee5/checkpoints/llama32_3b/gptq_nar_kmax_seed0/`，28层BF16 fake-quant state files |
| checkpoint manifest | `DONE.json` SHA256 `a94bddf636350100474c6897da91613ce66c373e30a70d4c397318b1b2e55e11`；原生成 job136794，QuaRot commit `5008669b08c1f11f9b64d52d16fddd47ca754c5a` |
| 数值容器 | loader明确 `dtype=torch.bfloat16`；不是 Appendix A.1 所述 FP32 containers。见 [experiment.py](../../manifest.json) |
| 激活 | asymmetric group128，FP16 scale+real offset，QDQ 后返回原 BF16 dtype；[experiment.py](../../manifest.json) |
| KV | 原 E14 `RuntimeHooks(...quantize_kv=True)`：asymmetric K group32沿token轴，V group128沿head维；K最新未完成chunk 1–32 tokens及V最近32 tokens保留BF16；真实浮点张量上的模拟QDQ，不是native packed KV。见 [e14_w4a4kv4.py](../../manifest.json) |
| R1/R2/R3/R4 | 原checkpoint规范：RMSNorm先融合，R1残差补偿；R2在v_proj后应用并在o_proj补偿；PrismQuant没有Hadamard的R3；R4在线应用且down_proj权重补偿。实际RuntimeHooks行682–695/750起及DONE.fold均已保留在JSON来源 |
| 执行版本 | requested和actual HEAD均 `35e76a6307602c17d0a801dcb7a4c73abd7273fb`；torch2.11.0+cu128，transformers5.16.0.dev0，tokenizers0.23.1；完整实际library paths/source SHA见原job及本audit JSON |
| harness | 记录预期commit `b954108c9baaaa934b4ad842033b31a97ee30816`；使用 `/home/yanlongc/.e13_packages/lm_eval`，原job明确只记录实际文件，不执行全树版本门槛 |
| 评测 | A40，full141 PPL B1，八任务0-shot、B8、max_length2048、seed0、log_samples=true，8项均PASS |
| 当前结果 | PPL **8.718024406077516**；八任务平均 **60.03241439144562%** |

权重文件完整性范围：1733511重新哈希了 DONE 和小资产，对28个大权重文件只记录路径/size/stat，`content_hash_checked=false`，并保留旧 probe 的历史哈希。本次 CPU 审计没有重新读取全部大权重，不能把历史哈希写成1733511重新验证所得。

校准也有可核验区别：rotation cache `llama32_3b-train-o0-n128-l2048.pt` 的 SHA256 为 `de43576731a5ebf52ede54a1400af335ca6eef21f2c7289f0e2b4c22bdc7e308`；GPTQ cache `llama32_3b-quarot-wikitext2-train-seed0-n128-l2048.pt` 为 `4e951761aceed1f85b17f2596e827e8345203eef6c60997f170ba9a03091937e`。两者均128×2048 int64，CPU逐元素比较 **不相等**，262144位置仅2334相同。源码对应连续BOS-prefixed chunks（[experiment.py 行405起](../../manifest.json)）与 seed0 random windows（[e14_w4a4kv4.py 行275起](../../manifest.json)）。这不满足 Appendix A.1 写的“相同 calibration sequences”；两者同属 WikiText-2 train 和128×2048并不足以证明相同。

## 3. 可找到的原始 seed0 结果与逐种子缺口

以下平均由原 `zero_shot.json` 的5个重叠任务加 `zero_shot_extra.json` 的3项重新算术平均，排除 LAMBADA；没有使用原六任务 `mean_accuracy`。逐任务原值、文件SHA及路径已保存在 audit JSON。

| 历史3B row | GPTQ协议 | seed | full141 PPL | 八任务平均 % |
|---|---|---:|---:|---:|
| k=8 | default，对称per-channel | 0 | 8.756231517884965 | 59.29425051167899 |
| k=max | default，对称per-channel | 0 | 8.71444608076717 | 60.18669818102309 |
| k=8 | g128_asym | 0 | **8.636331433141889** | **60.21062442254881** |
| k=max | g128_asym | 0 | **8.597017867763926** | **60.80439511426721** |
| 投稿k=8三种子的第2/3个原始run | g128_asym | 1 / 2 | **未找到 / 不可核验** | **未找到 / 不可核验** |

上述来源位于 `/projects/nar/nar-validation/results/llama32_3b/e14_nar_{k8,kmax}_asym_g128[_g128_asym]_seed0_{ppl,zero_shot,zero_shot_extra}.json`。这里的可见 seed0 也没有证据能映射为投稿“三种子平均”中的第一个run；不能据此构造原三种子数组。

**直接可验证的数值对应：投稿 k=8 的 PPL、8个任务和平均，共10个展示数字，全部等于本地 `kmax / g128_asym / seed0` 结果保留两位小数。** 这构成来源疑点，但仅凭数值相等，不能宣称已证明投稿行误标或已定位其真实生成过程。历史 [report.md 行581–585](../../manifest.json) 也分别列出 k8=8.63633/60.21、kmax=8.59702/60.80。

投稿 k=max 的8.58/61.23和逐任务数字，对应另一份 [author_reported.json](../../manifest.json) 中的作者转述：2026-09-17、另一集群三次运行；只有8个correct/total计数，明确写明不知道这些计数属于单种子还是三次聚合，PPL8.58来自另一个seed，raw harness output不在repo。该文件没有三个run的独立指标，因此同样不能核验三种子均值或方差。

搜索边界：检查 `/home/yanlongc/e29-e33-work/results/llama32_3b`、`/projects/nar/nar-validation/results/llama32_3b` 的E14结果，以及前一仓库所有refs可达Git objects中的3B结果路径。只找到 Hadamard 的seed1 PPL，没有 PrismQuant 3B seed1/2 的原PPL/accuracy日志。另只读列出公开 `ForeverBlue/PrismQuant-Llama-3.2-3B` 和 `ForeverBlue/nar-w4a4kv4-llama-3.2-3b`：两者当前revision同为 `f529f982f80a9b9a1c55f110a2205e14f933b1e8`，各577个文件，确有 **default k8 seed1 checkpoint / rotation**，但没有result文件、没有seed2文件。见 [公开文件列表证据](raw/audit_a1_public_repository_file_lists.json)。有seed1权重不能替代其原始结果日志；本结论仅限上述范围，不能排除作者另一集群仍有未收录的原日志。

因此 A1 可确认配置差异、真实可见seed0数字和展示数字的对应关系；**投稿3B k8原三种子逐一对账仍未完成，因缺少原始逐run证据而不可核验。** 不从平均数反推任何缺失种子结果。
