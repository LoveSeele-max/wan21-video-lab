# Week5 公开归档

归档日期：2026-11-06。Week5 的测量和文档已收尾，Week6–7 性能改动未启动。
本页是公开阅读索引；历史计划、日志中的“待执行”以其后追加记录及最终报告为准。

## 结果入口

| 内容 | 入口与结论 |
|---|---|
| 总览 | [复盘](REVIEW/WEEK5_REVIEW.md)、[权衡表 v1](tradeoff_v1.md) |
| SageAttention | [结论卡 v2](week5_result_attention_v2.md)：固定任务耗时下降 7.98%；质量数值门槛达到，关键记录缺失，综合结论不足 |
| 普通 FP8 | [结论卡](week5_result_fp8.md)：速度门槛未达，存在评分者明确填写的明显退化，当前任务不推荐 |
| scaled FP8 | [结论卡](week5_result_fp8_scaled.md)：速度门槛未达，存在评分者明确填写的明显退化，当前任务不推荐 |
| 质量来源 | [维度汇总](week5_quality_summary.md)、[回收说明](week5_rating_review.md)、[描述性分析验证](week5_analysis_validation.md) |
| 自动指标 | [报告](week5_metrics_fp8.md)；`week5_metrics_fp8/` 和 `week5_metrics_fp8_scaled/` 各 36 条原始指标记录 |
| VAE 定位 | [峰值报告](week5_vae_peak.md)：最后一次 decoder 内 columns 申请 7897.5 MiB；缓存成员重建 3556.08 MiB |
| CPU 缓存对照 | [审计](week5/vae-cache-cpu-01-local-audit.json)：allocated 减少 24.20%，单次诊断耗时为原来的 4.58 倍 |
| 本机探索 | [Conv3d 布局探针](week5_conv3d_layout_probe.md)：单算子测量，不代表服务器或整段 VAE 收益 |
| 规则与过程 | [预登记补充](week5_prereg_addendum.md)、[计划](../plan/WEEK5_PLAN.md)、[日志](../logs/week5.md) |
| 下一阶段约束 | [性能规则 v2](performance_rules_v2.md)仅供未来事前登记，不回判 Week4–5 |

## 公开证据

- `evidence/week5-vae/` 保留 baseline、history、CPU-cache 三次运行的 input/result JSON 原字节，源路径和哈希见根目录 `PUBLIC_SNAPSHOT.json`。
- `week5/` 保留本机原件验收 JSON、用户摘要，以及最终 v2 分析的 audit、attribution、cache epochs、44 个边界和 242 个峰值存活分配。
- [分配曲线](week5/vae-history-01-analysis-v2/allocator_timeline.png)和[阶段显存图](week5_figures/stage_allocated_peaks.png)为既有分析图；阶段图的原始 CSV 同目录保留。
- PyTorch 源码版本、原始 URL 和 SHA256 在 [attribution.json](week5/vae-history-01-analysis-v2/attribution.json) 中，完整第三方源码不随快照发布。

## 本地保留与复算边界

评分原件、补空工作簿、逐项有效答案、私人 A/B 映射和逐格授权记录继续保留本地。
公开评分报告保留汇总计数、处理依据和局限；回收说明及两张 FP8 卡删除了逐项 A/B 身份对应细节。

不发布视频、latent、权重、完整 profiler trace、allocator pickle、内嵌快照的 memory-viz HTML、
完整逐事件时间线 CSV、用户浏览器截图或上游源码副本。报告里提及这些文件表示本地证据存在，
不表示读者已获得原件；公开哈希不能替代独立核验。

VAE 的服务器 GPU 运行由用户执行，本机完成记录、哈希、媒体及历史重放分析；本次发布没有重跑 GPU。
MP4/RGB 相同不证明原始浮点输出完全一致；单输入诊断不证明全部输入质量、完整生成收益或 16G 可用。
`week5_vae_fixed_input.json` 的 `not_run` 是运行前登记时的历史状态，后续执行见三个 result 和最终审计。
其他原始 JSON 中的 pending 字段也保留产生时的状态，不为归档回写历史原件。

原始实验目录保持不变。公开 Markdown 的脱敏、附件引用调整、汇总删节和探针表述校正，
均通过 `PUBLIC_SNAPSHOT.json` 的源/公开哈希及 `documentation_edits` 记录。
