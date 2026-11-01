# wan21-video-lab

Wan2.1 T2V 1.3B 的推理性能分析、可复现测量与视频质量评测实验。

当前公开内容覆盖 Week1–Week4，Week4 实验截至 2026-10-29，发布更新于 2026-11-01。项目使用 Musubi Tuner 的已有生成能力；新增工作集中在测量、评测编排、反例探针、结果审计与实验设计。

**SageAttention 在固定任务下端到端耗时下降约 7.98%，质量分类仍为“结论不足”。** 普通/scaled FP8 的去噪阶段 allocated 峰值分别下降约 31.10%/30.27%，但未达到冻结的速度门槛，质量尚未评估。没有训练 LoRA，也没有自行修改模型架构或编写 CUDA/Triton 内核。

## Week4 结果

| 候选 | 正式性能证据 | 质量与限制 |
|---|---|---|
| SageAttention 2.2.0 | 同一独立环境 BF16 torch 135.904 秒 → Sage 125.059 秒；下降 7.98%，三次正式测量区间不重叠；profile 确认实际 varlen 路径 | 36/36 生成成功；结构和提示词符合度评分证据不完整，最终分类“结论不足” |
| 普通 FP8 | 主环境各六次正式测量：BF16 134.031 秒，FP8 132.335 秒；耗时下降 1.26%，未达 5% 门槛；去噪 allocated 峰值下降 31.10% | 36/36 生成成功；质量尚未评估 |
| scaled FP8 | 均值 134.645 秒，较同一主环境 BF16 增加 0.46%；去噪 allocated 峰值下降 30.27% | 36/36 生成成功；`fp8_fast=false`，不能声称使用专用 FP8 矩阵乘；质量尚未评估 |

三种候选整次运行 allocated 峰值均约 14698.35 MiB，最高峰仍在 VAE 解码。阶段显存收益不能直接解释为整次峰值下降。Sage 与 FP8 使用不同环境下各自的基线，不跨环境直接排名；固定任务速度测量也不代表所有提示词的速度收益。

详见 [Week4 复盘](notes/REVIEW/WEEK4_REVIEW.md)、[Sage 结论卡](notes/week4_result_attention.md)和 [FP8 报告](notes/week4_fp8.md)。本次发布没有启动 Week5。

## 已完成的工作

| 内容 | 已验证结果 |
|---|---|
| 性能测量 | 冷启动、端到端、分阶段计时、显存统计和 profiler 诊断 |
| 服务器基线 | RTX 5880 Ada 48G；R4 三次正式端到端均值 135.735 秒 |
| 瓶颈定位 | R2 去噪约占 83.31%；最高 allocated 峰值出现在 VAE 解码 |
| 评测生成 | 固定 12 条提示词 × 三个种子；双卡各 18 段，36 段视频哈希验收通过 |
| 自动指标与探针 | 49 条 SSIM/CLIP 记录；3920 个 SSIM 帧对值、1029 个 CLIP 帧值 |
| 人工试评 | 16 段匿名单视频和四对比较；原答保留，执行偏差与解盲后澄清单列 |
| 环境检查 | SageAttention 2.2.0 在独立副本编译安装，GPU 0 BF16 小张量检查通过 |
| Week4 实际路径 | Sage varlen 算子、完整模型冒烟、profile、同环境性能对照及 FP8 源码路径核对 |
| Week4 生成与审计 | Sage、普通 FP8、scaled FP8 各 36 段，共 108 段候选；公开清单与精简审计，媒体保留本地 |

Windows RTX 5060 Ti 16G 的固定配置也完成了运行和性能测量，L2 均值 637.611 秒。2026-10-26 的补充实验显示该解释器仍能分配超出物理显存的 CUDA 内存，因此不能将“跑通”解释为严格在 16GB 独立显存内运行，也不能据此声称 Linux 16G 可用。详见 [本机记录](notes/REVIEW/week2_local_16g.md)。

## 实验条件

- 模型：Wan2.1 T2V 1.3B。
- 上游：`kohya-ss/musubi-tuner`，固定 commit `f8a1b03794a49239a3539015075f5123d6c07d66`。
- 服务器环境：Python 3.11.17、PyTorch 2.9.1+cu128；两张 RTX 5880 Ada 48G 执行独立任务。
- 固定生成设置：480×832（高×宽）、81 帧、16 fps、20 步、UniPC、guidance 5、flow shift 5。
- 基线：torch attention、BF16、GPU 初始噪声；不启用量化、block swap 或 compile。

36 段双卡生成的累计任务时间为 82.951 分钟，整体任务跨度为 42.318 分钟。这些是素材生成资源记录，不能作为单卡延迟或优化收益。

## 目录

```text
scripts/                  测量、评测、探针、审计脚本与 CPU 测试
configs/                  冻结提示词、基线和已登记候选配置
doc/                      项目说明、测量规范、假设与总清单
plan/                     已有阶段计划
logs/                     历史执行日志
notes/                    报告、指标明细、清单及服务器测量记录
notes/REVIEW/             周复盘与本机实验报告
evidence/local/           本机运行的精简 JSON/CSV 证据及配置
REPRO.md                  环境、权重来源与复现记录
PUBLICATION.md            公开范围、路径约定和复现限制
PUBLIC_SNAPSHOT.json      导入文件及本地原件/公开副本的 SHA256
requirements.lock         历史服务器 pip freeze 快照
```

不分发模型、Python 环境、缓存、视频/latent 全集、私人评分映射和已填评分表。公开数据保留结果与原始视频哈希，无法在没有原视频时重新验收视频像素或文件哈希。

## 本地检查

以下 CPU 编排测试不需要下载模型或使用显卡。Python 3.11 环境安装 PyYAML 后执行：

```bash
python -m pip install PyYAML==6.0.3
python -m unittest discover -s scripts -p 'test_bench.py'
python -m unittest discover -s scripts -p 'test_gen_eval.py'
python scripts/gen_eval.py --dry-run --shard 0/2
```

指标行为测试需要 PyAV、NumPy、OpenCV、scikit-image 和 Pillow 等评测依赖，使用已有项目评测环境执行：

```bash
python -m unittest discover -s scripts -p 'test_metrics.py'
```

真实生成还需按 [复现记录](REPRO.md) 安装固定版本 Musubi Tuner、准备权重及离线分词器。配置中的 `/home/user/projects/wan21-video/` 是原实验路径；迁移时复制配置到新文件并修改路径，保留原冻结文件。`requirements.lock` 记录的是 Linux 服务器历史环境，不是跨平台一键安装清单。

不要为检查仓库直接重跑已有实验。生成、指标和匿名包脚本的前置条件与限制见 [公开说明](PUBLICATION.md)。

## 阅读入口

- [Week3 复盘](notes/REVIEW/WEEK3_REVIEW.md)：目标、实现、结果与局限。
- [Week4 复盘](notes/REVIEW/WEEK4_REVIEW.md)、[Week4 计划](plan/WEEK4_PLAN.md)与 [执行日志](logs/week4.md)。
- [Sage 性能](notes/week4_perf_attention.md)、[质量证据](notes/week4_quality_attention.md)与 [FP8 性能和生成审计](notes/week4_fp8.md)。
- [服务器性能基线](notes/week2_baseline.md)与 [profiler 分析](notes/week2_profile.md)。
- [反例探针结果](notes/week3_probe.md)：静止 SSIM 可为 1；自动指标不替代质量判断。
- [评分标准](notes/week3_rubric.md)与 [试评解盲报告](notes/week3_trial_review.md)。
- [生成说明](notes/WEEK3_GENERATION_GUIDE.md)、[指标说明](notes/WEEK3_METRICS_GUIDE.md)与 [资源成本](notes/week3_cost.md)。
- [项目原始说明](doc/README.md)与 [进度清单](plan/WEEK3_PLAN.md)。

## 结论边界与来源

单评分者、会话信息不完整及规则执行偏差均已记录，不宣称严格独立盲评或稳定检出全部退化。同种子不保证跨平台逐字节相同输出；不同维度和显存口径不合成一个总分。

上游代码、模型及依赖的权利归各自权利人，本仓库不重新分发模型或完整上游源码。新增内容尚未指定开放源码许可证；公开可见不代表已授予 MIT/Apache 等许可。来源与许可核对范围见 [THIRD_PARTY.md](THIRD_PARTY.md)。
