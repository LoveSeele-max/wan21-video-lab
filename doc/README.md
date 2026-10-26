# Wan2.1 1.3B：单卡视频生成的性能与窄域适配

> **一句话**：在单张 RTX 5880 Ada 48G 上定位 Wan2.1 T2V 1.3B 推理的时间与显存瓶颈，做对照和性能代码改动。同时训出一个单人物动作 LoRA，最终给出质量—延迟—显存的权衡表与可点的 demo。本机 16G 已完成固定配置运行与性能基线测量。

**状态（2026-10-22）**：Week 2 性能测量与 Week3 评测准备已收尾。已冻结 12 条提示词 × 三个种子，36 段基线 MP4 哈希验收通过；49 条 SSIM/CLIP 记录、匿名试评、资源成本和后续预登记已完成。SageAttention 2.2.0 在服务器独立副本安装及小张量检查通过。尚无优化收益或正式优化质量对照结论，试评的规则执行偏差保留。见 [Week3 复盘](../notes/REVIEW/WEEK3_REVIEW.md)、[复现记录](../REPRO.md) 与 [预登记](../notes/week4_prereg.md)。

本机 RTX 5060 Ti 16G 在 Windows 下固定 BF16 配置完成 L0–L2 九次生成；L2 正式均值 637.611 秒，L3 也成功。2026-10-26 实测该解释器仍可分配 24 GiB（超出物理显存），且 L3 解码期间显存读数贴顶，因此推断运行依赖 Windows 的显存超量使用；不证明 Linux 16G 可运行，详见 [本机记录](../notes/REVIEW/week2_local_16g.md)。

---

## 一、项目定位

面向 AIGC 视频方向的实习申请。核心不是"跑通了一个开源项目"，而是：

1. **能定位瓶颈并修改代码**：用阶段计时和 profiler 找出瓶颈，在视频 DiT 的推理与训练路径（attention、量化、block swap、VAE 解码、LoRA 注入）上做有依据的改动。
2. **能实测**延迟、峰值显存与质量，并解释每项优化为什么有效、在什么条件下失效。
3. **能设计公平对照**，报告收益、代价与失败案例，而不是只展示精选样本。

**不是**：从零实现的训练框架；论文等价物；纯 ComfyUI 工作流展示。

## 二、技术起点（路径与参数已对照上游源码，main 分支 2026-10-11；固定 commit 后复核）

| 项 | 取值 | 说明 |
|---|---|---|
| 主工程仓库 | [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) | 复用其缓存/训练/生成链路；不从零写 |
| 模型 | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) T2V 1.3B | 官方权重，Apache-2.0 |
| 主文档 | [docs/wan.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md) | 参考用。**与代码不一致时以源码为准**（已发现不一致处，见 EXPERIMENT_PLAN 第七节） |
| 分辨率 | 480P（固定） | 不把上调分辨率当作结构问题的补救 |
| 训练方式 | LoRA，单卡优先 | 不引入 DeepSpeed / DDP |
| 硬件 | 服务器：双 RTX 5880 Ada 48G、256G 内存｜本机：RTX 5060 Ti 16G（Windows 固定 BF16 配置已跑通并测量） | 双卡跑**独立任务**，不视为一张 96G 卡 |

**代码阅读入口**（内部实现沿导入关系继续读）：

| 关注点 | 入口 |
|---|---|
| 推理主循环 | `src/musubi_tuner/wan_generate_video.py` |
| DiT 实现 | `src/musubi_tuner/wan/modules/` |
| Attention 后端 | `src/musubi_tuner/wan/modules/attention.py`（Wan 的分发逻辑）、`src/musubi_tuner/modules/attention.py`（通用） |
| 显存/卸载 | `src/musubi_tuner/modules/custom_offloading_utils.py` |
| fp8 量化 | `src/musubi_tuner/modules/fp8_optimization_utils.py` |
| 时间步与 flow shift | `src/musubi_tuner/training/timesteps.py`、`src/musubi_tuner/modules/scheduling_flow_match_discrete.py` |
| 训练主循环 | `src/musubi_tuner/wan_train_network.py`、`src/musubi_tuner/training/trainer_base.py` |
| LoRA 注入 | `src/musubi_tuner/networks/lora_wan.py` |

## 三、研究问题

**主线（工程，瓶颈驱动）**

> 固定 480P、帧数和提示词集，回答三个问题：
> 1. 一次生成的时间和显存花在哪里？
> 2. 针对瓶颈，哪些选项或代码改动能降低延迟或峰值显存？质量代价和失效条件分别是什么？
> 3. 16G 卡固定配置已跑通，其耗时与峰值已测量；后续比较优化代价与质量，不将运行成功等同于全程独立显存驻留。

**次要线（研究，允许 null）**

> 时间步采样策略与 LoRA 配置，如何影响模型学到的是"动作"还是仅仅"外观"？

次要线不预设结论。若未检出差异，写"未检出"并说明边界，不包装成提升。

## 四、交付清单

| # | 交付物 | 状态 |
|---|---|---|
| 1 | 可复现的推理性能测量脚本（含分阶段计时；延迟 / 峰值显存 / 质量同口径） | bench.py 与 R1–R5 验收完成；评测生成、指标和人工规则已建立，完整优化对照待执行 |
| 2 | 阶段耗时分解 + 瓶颈结论 | Week 2 已完成，见 [基线](../notes/week2_baseline.md) 与 [profiler](../notes/week2_profile.md) |
| 3 | **本人性能型代码改动**（至少 1 处，降延迟或峰值显存，附前后对照；插桩不计） | 未开始 |
| 4 | 单卡服务器延迟—显存—质量权衡表；16G 实机测量 | 固定配置本机基线已完成；优化与正式质量权衡未完成，不从服务器结果外推 |
| 5 | 单人物动作 LoRA 权重（许可允许时公开） | 未开始 |
| 6 | 可点的 demo（Gradio 或 HF Spaces） | 未开始 |
| 7 | 实验报告：机理、收益、代价、失效条件 | 已有基线、探针及试评报告；优化收益与最终报告待完成 |
| 8 | 上游 PR（至少**提交** 1 个；合并状态如实记录） | 未开始 |
| 9 | 复现记录（commit hash、依赖、权重版本） | [记录已补齐至 Week3](../REPRO.md)：冻结配置、36 段基线、指标、试评与 attention 检查已索引；干净环境复现尚未验证 |

## 五、文档分工

| 文档 | 职责 |
|---|---|
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | 阶段划分、每阶段做什么与止损线、数据与划分规则 |
| [METRICS.md](METRICS.md) | 怎么测、怎么公平比较、怎么报告 |
| [ASSUMPTIONS.md](ASSUMPTIONS.md) | 所有"尚未验证但结论依赖它"的前提，逐条状态 |
| [EXECUTION_CHECKLIST.md](EXECUTION_CHECKLIST.md) | 按周的执行清单与硬性 gate |
| [WEEK3_REVIEW.md](../notes/REVIEW/WEEK3_REVIEW.md) | 本周做了什么、测到了什么、哪些尚未证明，以及 Week4 的执行顺序 |
| [week4_prereg.md](../notes/week4_prereg.md) | 已冻结的候选配置、评分量及收益判定规则 |

## 六、表述纪律

以下几条贯穿全项目，写报告与简历时一律遵守：

- 上游能力（模型、缓存、训练、生成）**不写成本人从零实现**。
- "提升"只由训练后的独立测试支持，不由训练 loss 或"更像训练集"推出。
- **不把"训练后画面更像训练集"等同于"动作能力提升"**——前者可能只是外观过拟合，甚至伴随运动退化。这是本项目最容易出现的自欺。
- 未检出差异就写未检出，限定在当前任务、配置与评价口径内。
- 装依赖、解环境报错写进附录，不作为主要交付。
- 公开仓库是可验证的工程产出，**不宣称等价于论文**。

## 七、复现入口

服务器根目录为 `/home/user/projects/wan21-video`，主环境为 `env`，attention 检查副本为 `env-sagecheck`；本机项目为 `E:\cstudy\AIGC\V`，本机解释器为 `local/env/python.exe`。上游固定 commit `f8a1b03794a49239a3539015075f5123d6c07d66`。

安装来源、权重、测量及首条生成命令见 [REPRO.md](../REPRO.md)；36 段评测生成顺序见 [生成说明](../notes/WEEK3_GENERATION_GUIDE.md)，指标协议和复算见 [指标说明](../notes/WEEK3_METRICS_GUIDE.md)。已有结果不用重复生成；性能对照必须按预登记采用当期同环境基线。
