# Week 2 profiler 与瓶颈结论

整理日期：2026-10-15。数据来自 `R5-profile-20261015-221034-3813a433/run-01/`。本报告分析现有 trace，不新增 GPU 运行。

## 捕获范围

完整生成 20 步，40 次 DiT 前向。schedule 为 wait16/warmup2/active4，按前向推进。trace 中 CPU 标记和 GPU 标记各有 ProfilerStep#18–#21 共 4 个，CPU/GPU 是同一组逻辑前向的两侧，不能当作 8 次前向。目标范围对应第 10、11 个采样步的条件/无条件分支。

trace 约 48.3 MB，包含有效 CUDA kernel 事件。R5 生成保存耗时 142.502 秒，仅作诊断，不作为延迟比较。

## 为什么不直接抄 operators.txt 的百分比

`operators.txt` 同时列出了 ProfilerStep、ATen 算子、CUDA kernel 和 CUPTI overhead。例如 aten::_flash_attention_forward 与底层 flash_fwd_kernel 是同一工作不同层级，两行时间不能相加；ProfilerStep 也不是一种模型算子。

trace 中共有 17036 条 kernel，264 条 flash 内核。其中 996 条 kernel 没有关联到本次捕获的 CPU 算子 External id，全部发生在首个 GPU ProfilerStep 开始前，含 24 条 flash。它们是捕获边界前的异步 GPU 工作，不能混入 4 次正式前向的算子表。

因此采用保守的关联过滤：只统计 External id 能对应本次捕获 CPU 算子的 kernel，排除 CPU 事件、annotation、overhead、gpu_memcpy、gpu_memset。保留 16040 条 kernel，时长之和 10.582381 秒。关联后的 flash 内核 240 次，与 4 前向 × 30 层 × 2 个 attention 的数量一致。这是 GPU 内核忙碌时间的分类占比，不是整个生成墙钟占比，也不包含所有复制/等待。

| 内核分类 | 事件数 | 时长之和（秒） | 关联内核时长占比 |
|---|---:|---:|---:|
| flash attention 前向内核 | 240 | 5.703751 | 53.90% |
| 名称匹配 GEMM/GEMV 的矩阵乘 | 1228 | 2.105034 | 19.89% |
| 其他内核 | 14572 | 2.773595 | 26.21% |

矩阵乘按 kernel 名称含 gemm/gemv 分组，是线性层开销的代理，不是全部线性层耗时。其他类含逐元素、归一化、变换等内核，不声称它们都值得或都能融合。完整事件与可复查过滤代码见原始 trace、`scripts/analyze_week2.py` 和 `notes/week2_audit.json`。

## 采集干扰

trace 记录了 4283 次 Command Buffer Full、2 次 Activity Buffer Request、3 次 Buffer Flush；原表的 Command Buffer Full CPU 总时长约 5.957 秒。这提示采集存在显著缓冲/工具开销，不能把该行当作模型瓶颈，也不能直接从墙钟减掉它。未独立证实丢事件与否，不承诺本次是无损采集。

过滤后的 attention 仍是最大类，足够支持先把 attention 实现作为候选；但只抓两个中段采样步，且有工具干扰，不外推所有时间步、分辨率或提示词。后续候选是否有效以关闭 profiler 的端到端配对测量和质量检查为准。

## 瓶颈结论

固定当前 480P、81 帧、20 步配置，R4 预热后生成并保存平均 135.735 秒。R2 的去噪区间平均 113.435 秒，占该模式总时间 83.31%，是主要时间来源；T5 loader 约 10.066 秒、VAE 解码约 10.332 秒。allocated 总峰值 14698.4 MiB 出现在 VAE 解码，去噪峰值约 4183.5 MiB，时间瓶颈与显存峰值来源不同。R5 关联内核中 attention 约 53.90%、名称匹配的 GEMM/GEMV 约 19.89%，支持首先评估去噪计算路径；没有测过任何加速收益，VAE 内部显存来源也尚未定位。

## 后续两组对照的候选顺序

1. attention 实现对照优先。当前配置虽叫 torch，trace 已显示使用 pytorch_flash 内核；不能说基线是朴素 attention，也不能假定切到 flash 包就会加速。可在兼容性和安装预算内选一个替代实现，其他条件固定，先用很小样本验证，再按冻结的评测集比较端到端、显存与质量。torch/sdpa 不重复测，flash/flash2 不重复测。
2. DiT 精度/矩阵乘路径列为第二候选：先依据已核实源码选单一 FP8 配置，只改变这一变量。若选择 fp8_scaled + fp8_fast，作为一个明确组合配置记录，不把普通 fp8 存储变化等同于硬件矩阵乘加速；端到端收益和质量代价都需实测。若安装/量化路径超预算或未见收益，记录结果后再考虑 compile 等候选，不扩大扫参。

VAE 是后续本人性能代码改动的显存候选，暂不承诺改法。须进一步辨别全分辨率激活、缓存和输出 torch.cat 累积各自影响，不能看到 VAE 峰值就认定 cat 是罪因；时间维已逐 latent 帧解码，不重做时间分块。vae_cache_cpu 可作为定位缓存贡献的对照，但不预设有收益。

## 完成范围

Week 2 服务器测量出口已满足：脚本、R1–R5 原始记录、重复运行、阶段表、分段/总峰值核对、采样、插桩影响估计、profiler 分类和限定范围的瓶颈结论。保留 profiler 缓冲开销及边界过滤限制。没有完成性能优化、质量判据、训练或本机 16G 实测。正式对照前先做 Week 3 的评测判据与提示词/种子冻结，不直接开始扫参数。
