# 第 4 周计划：attention 组完整对照 + FP8 组性能测量

**日期**：2026-10-26 编写。配套 [week4_prereg.md](../notes/week4_prereg.md)、[optimization_rules.yaml](../configs/optimization_rules.yaml)、[scoring_frozen.yaml](../configs/scoring_frozen.yaml)、[METRICS.md](../doc/METRICS.md) 第四节报告模板。
**前置**：第 3 周已收尾（见 [WEEK3_REVIEW.md](../notes/REVIEW/WEEK3_REVIEW.md)）。冻结的提示词、种子、评分量和判定规则**本周一律不改**；本计划只安排执行顺序，与预登记冲突时以预登记为准。

**第 4–5 周分工**：

| 周 | attention 组（SageAttention） | 精度组（fp8 / fp8_scaled） |
|---|---|---|
| 4 | 冒烟 → 性能 → 36 段生成 → 指标 → 评分 → **结论** | 冒烟 → 性能测量 → 72 段生成（排在空闲时段） |
| 5 | — | 指标 → 评分 → 结论；两组汇总成第一张权衡表 |

**本周要拿到的东西**：

1. SageAttention 在 Wan 完整模型上**确实被调用**的证据（profiler 内核名），以及 NaN / 黑屏检查结果；
2. 当期同环境的性能对照：torch 与 sageattn 的端到端、阶段耗时、三种显存口径；
3. sageattn 的 36 段视频、自动指标、36 对配对评分和 12 段绝对评分；
4. **attention 组的结论卡**（METRICS 第四节模板），按预登记规则给出四类结论之一；
5. 两个 FP8 配置的冒烟和性能数字，以及 72 段视频（质量评价留到第 5 周）。

**本周不做**：不改上游代码；不加 fp8_fast、compile、卸载或组合配置；不换 attention 后端去追更好的结果；本机 16G 上不装 SageAttention。

---

## 开工前先弄清的一件事：实际调用的是 `sageattn_varlen`

固定 commit 的 `wan/modules/attention.py`（归档副本 `notes/source/wan_attention.py`）：

- 第 166–176 行：定长的 `sageattention.sageattn` 分支**被注释掉了**，注释写着"sage attention with fixed length seems to cause NaN in I2V inference"；
- 第 241–253 行：`attn_mode == "sageattn"` 实际调用的是 **`sageattention.sageattn_varlen`**。q/k/v 先展平成 `(B·L, N, C)`，再用 `cu_seqlens` 传入长度。

第 3 周验证的是 `sageattn`（HND 布局小张量），**`sageattn_varlen` 还没执行过**。在 SageAttention 2.2.0 中，两者可能走的是不同的内核实现，需要在已安装的源码里核对，不能凭函数名推断。因此 Day 1 先验证 varlen 路径，不直接跑模型。

---

## 本周决策

| 编号 | 决策 | 采用 | 理由 |
|---|---|---|---|
| D1 | attention 组所有运行的环境 | 全部在 `env-sagecheck` 中运行，**包括 torch 对照基线** | 预登记要求当期同环境基线；两组环境只差一个包 |
| D2 | 环境等价性检查 | 在 `env-sagecheck` 中用 `gen_eval.py` 重新生成 baseline 的 P01/s42，比较 MP4 的 SHA256 | 若与第 3 周基线一致，说明副本环境不改变 torch 路径的输出，第 3 周的 36 段基线可以直接作为配对的一侧；若不一致，见第三节的处理 |
| D3 | 阶段耗时 | 每个配置另跑一次 stages 模式（预热 1 + 正式 1），**只用于解释机理**，不参与速度门槛 | 速度门槛按预登记只看 total 模式；报告模板要求"变化集中在哪个阶段" |
| D4 | 计时期间不并行 | 性能测量都在 GPU 0 上做，期间 GPU 1 空闲；36 段评测生成放到全部计时**结束之后**再双卡运行 | 预登记的 `no_concurrent_generation` |
| D5 | 运行顺序 | 基线 → 候选；若范围重叠，按预登记**再跑一轮**，顺序仍为基线 → 候选 | 预登记写明追加轮次两侧都跑，不只重测有利一侧 |

---

## Day 1：环境核对 + varlen 数值检查（服务器）

- [x] `nvidia-smi` 确认两卡无其他训练/推理任务（保留桌面占用记录）；`git -C musubi-tuner rev-parse HEAD` 为 `f8a1b03`；`sha256sum scripts/bench.py scripts/gen_eval.py configs/eval_*.yaml configs/prompts_frozen.yaml`，与本机记录比对
- [x] 使用 `env-sagecheck` 的 Python 绝对路径，打印 torch / triton / sageattention 版本与安装路径（等效于选定副本解释器，不依赖 shell 激活）
- [x] **读安装后的源码**：`python -c "import sageattention, inspect; print(inspect.getsourcefile(sageattention.sageattn_varlen))"`，打开该文件，记录 `sageattn_varlen` 实际调用的内核（Triton 还是 CUDA 扩展、Q/K/V 各用什么精度）。写入 `notes/week4_sage.md`。**这一条是之后"机理一句话"的依据**，不能凭函数名推测。
- [x] 写 `scripts/check_sage_varlen.py`，在 GPU 0 上用 Wan 1.3B 的**真实形状**对比 `sageattn_varlen` 与 PyTorch SDPA：
  - 自注意力：b=1，L = 21×30×52 = **32760**，12 个头，head_dim 128，BF16；
  - 交叉注意力：q 长度 32760，k/v 长度 512；
  - 输入用固定种子的随机张量，再加一组"真实激活"（可选：在冒烟时用 hook 存下第 1 步第 0 层的 q/k/v）；
  - 输出：是否有 NaN/Inf、最大绝对误差、平均绝对误差、余弦相似度；两个函数各自的耗时（同步后计时，预热 3 次，测 10 次）。
- [x] 结果写入 `notes/week4_sage.md`。**这只是算子层面的证据**，不代表端到端收益。

2026-10-26：会话 `varlen-20261026T132946Z-a1c64053` 的原始文件已传回本机并核验脚本/源码哈希及计时复算。两项检查输出均 finite，自注意力耗时下降约 18.13%，交叉注意力耗时增加约 93.60%。随后用户补齐三个候选配置并提供服务器哈希，四份 eval YAML 均与本轮本机重新读取的哈希一致；Day 1 五项已验收。GPU 上有桌面进程，不能写成完全零占用。

**止损**：varlen 报错或出现 NaN → 停止 attention 组，按预登记记为"未评估 / 结论不足"，写清错误。是否启用 xformers 备选，由你决定（预登记只允许在"安装不兼容"时启用；模型层失败算不算这种情况，需要你明确拍板并记录）。

## Day 2：模型级冒烟 + 确认内核 + 环境等价性（服务器 GPU 0）

执行路径说明（2026-10-26 源码核对）：现有 `bench.py` 的结果仍保存到 `notes/week2/<tag>-<mode>-<session>/`，latent/视频保存到 `outputs/week2/<tag>-<mode>-<session>/<run>/`；以终端打印的 Session directory 和 result.json 的 output_directory 为准。本周继续复用原脚本，不为目录命名修改其验收哈希。`notes/week4/` 用于本周算子结果和审计等独立产物。

- [x] **冒烟**：`bench.py --config configs/eval_sageattn.yaml --mode stages --steps 2 --warmup 0 --repeat 1 --tag W4-sage-smoke`（在 `env-sagecheck` 中）。检查调用计数（4 次前向）和分段之和
- [x] **NaN / 黑屏检查**：加载输出 latent，检查 `torch.isfinite(latent).all()`；抽查 MP4 的第 0、40、80 帧，看像素均值和方差是否正常

2026-10-26 冒烟状态：`W4-sage-smoke-stages-20261026-214619-9754d61f` 已传回并在本机完成内容核查。4 次前向、阶段总和及 summary 复算一致，latent 全部有限，视频完整解码 81 帧且抽帧非黑屏；三帧画面严重模糊，不能作质量通过或 SageAttention 退化归因。用户随后贴回服务器 latent / MP4 哈希，与本机本轮重算的两项哈希一致，前两项已验收。审计及抽帧见 `notes/week4/smoke-audit-W4-sage-smoke-stages-20261026-214619-9754d61f/`。

- [x] **确认 SageAttention 真的被调用了**：`--mode profile --config configs/eval_sageattn.yaml --tag W4-sage-profile`（20 步，抓 2 步）。在 trace 中：
  - 应出现 SageAttention 的内核（以 Day 1 读源码得到的名字为准）；
  - **不应再出现** `pytorch_flash` / `flash_fwd` 一类内核；第 2 周 R5 中这类内核正好 240 次，对应 4 次前向 × 30 层 × 2 种 attention；
  - 按第 2 周 `analyze_week2.py` 的关联过滤方法统计：attention 类内核耗时、占比，与 R5 的 5.70 秒、53.90% 对照。仅作机理解释，不作为速度依据。
- [x] **环境等价性检查（D2）**：在 `env-sagecheck` 中用 `gen_eval.py` 和 `eval_baseline.yaml` 生成 P01/s42，输出到独立命名空间 `baseline-envcheck`，比较 SHA256：
  - 与第 3 周 baseline 的 P01_s42 一致 → 第 3 周的 36 段基线直接作为配对一侧；
  - 不一致 → 停下来，记录原因；之后两种选择：在副本环境里重新生成 36 段基线（多约 42 分钟双卡），或者接受该差异并在报告中注明。**由你决定**，不默认选择。

2026-10-26 profile 内容核查：`W4-sage-profile-profile-20261026-222504-af72f220` 已传回；本机确认40次前向和四个捕获Step，240次Sage attention、480次Q/K量化、未发现PyTorch flash前向内核。旧External-id过滤会漏掉720条Triton事件，已用launch correlation及同名CPU/GPU Step区间补回，并用同一方法复核历史R5。审计在 `notes/week4/profile-audit-W4-sage-profile-profile-20261026-222504-af72f220/`；用户补充的服务器trace/latent/MP4哈希与本机本轮重算全部一致，内核确认已验收。20步latent有限、视频可解码且抽帧非黑屏，不等于质量通过。最后的副本环境P01/s42复现随后已运行并验收，见下文；不要求该单样本具有36条汇总覆盖。

**出口**：varlen 能跑通、无 NaN；trace 证明内核已切换；环境等价性有结论。

2026-10-26 单样本复现验收：baseline-envcheck 的 P01/s42、20步生成成功，服务器新旧MP4哈希均为 `fa59fbd653bef48711826054cbfc3463da9003b10c429c2273d9cf77eb838217`，本机重算旧基线并核对原清单一致。原始收据已传回，本机确认实际torch后端、冻结设置、runtime、identity/fingerprint及events一致，按预定MP4判据通过；新旧latent文件哈希不同，张量未比较，差异留档且不扩展判据。审计见 `notes/week4/envcheck_audit.json`。Day2四项已验收，按计划沿用历史36段质量基线；仅支持此固定样本MP4复现，不外推全部36段或完整环境等价，性能仍需当期同环境基线。

## Day 3：性能测量（服务器 GPU 0，GPU 1 空闲）

全部在 `env-sagecheck` 中，每行都是一个新进程：

| 运行 | 配置 | 模式 | 次数 | 用途 |
|---|---|---|---|---|
| A1 | eval_baseline | total + `--sample-smi` | 预热 1 + 正式 3 | 速度门槛的基线侧 |
| A2 | eval_sageattn | total + `--sample-smi` | 预热 1 + 正式 3 | 速度门槛的候选侧 |
| A3 | eval_baseline | stages | 预热 1 + 正式 1 | 阶段分解（只用于机理） |
| A4 | eval_sageattn | stages | 预热 1 + 正式 1 | 同上 |
| A5、A6 | 若 A1/A2 范围重叠：baseline、sageattn | total | 预热 1 + 正式 3 | 预登记的唯一一次追加轮次 |

> bench.py 的 total 模式固定在配置文件里的 P01/s42 上，`eval_*.yaml` 的提示词和种子正好是 P01 / 42，与预登记一致。

2026-10-26 A1验收：`W4-A1-torch-total-20261026-232448-8df71ea1` 的session/result/summary及四份CSV已传回本机核验通过，1次预热及3次正式标志正确。正式均值135.90393688033024秒、范围134.4933614092879–137.39130416186526秒；三次正式allocated/reserved/整卡采样峰值分别为14698.353515625/18666/19627 MiB。审计为 `notes/week4/A1_audit.json`；A2–A4尚未执行，以下测量总项保持未勾选，详情在 `notes/week4_perf_attention.md`。

- [x] 跑 A1–A4；计算下降率 `(基线均值 − 候选均值) / 基线均值`，以及双方的 min/max 区间是否有交集
  - 2026-10-27：A1/A2原始JSON及CSV已核验，均值135.90393688033024/125.05904630338773秒，耗时下降7.97982076596643%，范围不重叠，速度门槛通过；不触发A5/A6。allocated/reserved/整卡采样峰值均为14698.353515625/18666/19627 MiB，两侧相同。A3/A4待执行，总项不勾选，质量及最终分类未评估。审计见 `notes/week4/A1_audit.json`、`notes/week4/A2_audit.json`。
- [x] 条件已检查：A1/A2范围不重叠，A5/A6不触发、不执行；仅触发时合并六次正式值重新判定

2026-10-27 A3验收：`W4-A3-torch-stages-stages-20261027-144838-89be411c` 的session/result/summary已本机核验通过，两轮各40次前向、阶段调用及聚合一致。正式总耗时134.51247651409358秒，去噪111.18730131816119秒、占82.66%；allocated峰值位于VAE解码。审计见 `notes/week4/A3_audit.json`，阶段表见 `notes/week4_perf_attention.md`。A4尚未执行，A1–A4总项保持未勾选；阶段值不作为正式速度门槛依据。

- [x] 整理 `notes/week4_perf_attention.md`：端到端表、阶段表（重点看去噪段；T5、VAE 段理论上不应变化，若有变化就说明有噪声或其他混杂因素）、allocated / reserved / 整卡采样三种峰值、每次前向的耗时
- [x] 与 Day 1 算子级加速比对照，粗算一下：attention 内核加速了 x 倍，去噪段理论上应缩短多少，实际缩短了多少。两者对不上就写出可能的原因，不强行解释

2026-10-27 Day3收尾：A4 `W4-A4-sage-stages-stages-20261027-150717-7519d3ee` 原始记录验收通过（`notes/week4/A4_audit.json`）。去噪111.187301→100.198346秒，减少9.8833%，占stages总差值约92.37%；T5等仍有波动，不全归因于attention。三种整体峰值不下降，allocated峰值仍在VAE解码。Day1算子配对和历史kernel占比粗估已写明假设及差异；完整阶段/显存/前向/total表在 `notes/week4_perf_attention.md`。Day3完成，质量未评估，不作最终可用分类。

**预期**（现在写下，事后对照，不追认）：attention 约占去噪段内核时间的 54%。若 varlen 内核比 PyTorch flash 快 1.5–2 倍，去噪段可能缩短约 18–27%，端到端约 15–23%（T5 加载和 VAE 解码不受影响）；峰值显存仍在 VAE 解码阶段，基本不变。若 varlen 走的是较慢的实现，收益可能达不到 5% 门槛。

## Day 4：36 段生成 + 自动指标

2026-10-27本机生成验收通过：36条唯一成功记录、36段MP4哈希均匹配，身份指纹/冻结参数/覆盖率和收据一致，GPU0/1各18条；36份attempt-01均ok，无失败或重试记录，events有36 running、36 ok及P01_s1001的1次已验证跳过。latent未传回，服务器汇总已核对其哈希，本机不声称独立重算latent。审计为 `notes/week4/generation_audit.json`。

- [x] 计时全部结束后，双卡运行：`gen_eval.py --config configs/eval_sageattn.yaml --shard 0/2`、`--shard 1/2`（在 `env-sagecheck` 中，放进 tmux）
- [x] `--summarize` 得到覆盖率；失败最多重试 1 次，首次失败也要报告
- [x] 传回 MP4、JSONL 和 coverage（不传 latent）；在本机逐段核对哈希
- [x] 本机复用 `metrics.py` 计算函数，得到 36 条 SSIM / CLIP。只用于发现异常（第 3 周探针已证明它们不能代替人工），并与基线同一样本**逐条对比**，把差值大的样本记下来，评分后再对照

指标执行入口为新增 `scripts/metrics_candidate.py`，复用未修改的metrics.py计算函数，保留原脚本及第三周结果；仅将输入从基线/探针切换为候选36段，CPU FP32、同版本依赖/协议/CLIP revision，输出独立的 `notes/week4_metrics_sageattn/`。本轮36/36全部ok，保存2880个SSIM帧对值及756个CLIP帧值；四个标量、36对差值及绝对差值排序已独立复算一致，审计为 `notes/week4/metrics_sageattn_audit.json`，说明及辅助线索为 `notes/week4_metrics_sageattn.md`。不以指标判定质量，不将排名给评分者。Day4完成，人工评分尚未执行。

## Day 5：盲评包 + 评分

- [x] 复用 `blind_pack.py` 编码函数制作包：36 对 baseline vs sageattn（左右随机）+ 固定 12 段 sageattn 单视频，全部混排、编号随机；映射表单独存放
- [ ] 提醒评分者（R01）：判据 v1.0；静止本身不扣时序分；方向和程度要一致，拿不准的选"无法判断"
- [ ] 回收后原件先归档并记录哈希，再解盲；方向与程度冲突的先请评分者澄清，**不自行改分**

2026-10-27：本机已制作并核验 `outputs/week4-R01-20261027.zip`，48项混排T001–T048、固定12段不重抽、每对左右随机。旧脚本是第三周固定工作量，故新增blind_pack_candidate.py和空白工作簿入口，保留旧包和答卷。48段视频来源/格式、随机化、下拉/空白、布局及ZIP CRC/53成员哈希已核验，映射和审计仅在 `notes/blind_private/week4-R01-20261027/`。用户已返回48项评分，原件归档及哈希核验完成；初始6个编号共10项冲突经两次补充后明确了2项时序，另8项相对答案仍不完整。本轮已固定解释并解盲统计，但冲突未全部解决；会话元数据缺失保留，事前提醒未核实，后两项保持未完成。核对见 `notes/week4_rating_review.md`，统计见 `notes/week4_quality_attention.md`。

## Day 6：attention 组结论 + FP8 组启动

**attention 组**：

- [x] 新增 `analyze_candidate_ratings.py`（保留第三周旧入口）：每维报告36对、完整相对答案数、变好/变差/无差异及未解决项，分别以36和完整答案数为分母；缺失观看元数据另列，不冒充完全可评分验证
- [x] 按 `optimization_rules.yaml` 的优先顺序判定：先查硬性问题 → 再查证据是否完整 → 最后查收益与质量门槛；本轮为“结论不足”，不是质量通过
- [x] 写 `notes/week4_result_attention.md`，即结论卡：

| 字段 | 内容 |
|---|---|
| 配置名 | `eval_sageattn.yaml` |
| 相对基线改了什么 | `attn_mode: torch → sageattn`（实际调用 `sageattn_varlen`） |
| 延迟 | 端到端均值 ± 范围、下降率、是否确认 |
| 阶段耗时 | 变化集中在去噪段的程度 |
| 峰值显存 | allocated / reserved / 整卡采样 |
| 质量 | 三个维度的配对计数 + 12 段绝对评分 |
| 机理 | Day 1 源码核对 + Day 2 trace 内核对照 |
| 失效条件 | 例如：需要编译安装、只验证了 sm_89、只测了 480P / 81 帧 / 20 步、I2V 定长路径有 NaN 记录 |
| 结论 | 可用 / 有条件可用 / 不推荐 / 未评估或结论不足 |

**FP8 组**（主环境，不用 `env-sagecheck`）：

2026-10-27 attention质量本轮评估完成：先固定不含映射的答案解释再解盲明确项，结构/时序/符合度完整答案为34/36/30；时序34无明显差异、1轻微更好、1轻微更差、0明显变差。结构2项、符合度6项仍未解决，会话/观看字段缺失保留；不改分、不更改冻结规则。详情 `notes/week4_quality_attention.md`。Day5冲突澄清仍未全部完成，Day6产出结论不足卡不等于质量门槛通过。

- [x] 冒烟：`eval_fp8.yaml`、`eval_fp8_scaled.yaml` 各跑一次 2 步 stages；从日志及原始JSON确认权重路径和生效 `fp8_fast=false`，本机latent finite、81帧完整解码/非黑屏验收
- [x] 读固定commit源码并核对实际参数：普通FP8由标准Linear/BF16 autocast处理，scaled在fp8_fast=false下显式反量化后F.linear，未启用专门_scaled_mm分支；记录 `notes/week4_fp8.md`，不作逐GPU指令/激活精度实测声明。**预期**：显存下降集中在 DiT 加载和去噪阶段；整次峰值仍在 VAE 解码，可能基本不变；速度可能持平或变慢

2026-10-27普通FP8冒烟收到用户服务器日志：W4-fp8-smoke-stages-20261027-233635-3ec2be40，2步完成并保存latent/视频，权重明确cast为float8_e4m3fn，终端33.127秒、allocated峰值14698.4 MiB。原始JSON/产物未传回，scaled未跑，fp8_fast生效值及内容有限性未核验；两项仍保持未完成。详情 `notes/week4_fp8.md`。

随后scaled冒烟也收到日志：W4-fp8-scaled-smoke-stages-20261027-234012-aee81310，2步完成，300层Linear优化及前向替换，终端33.541秒、allocated峰值14698.4 MiB。静态源码初查解释None权重类型及fp8_fast控制的scaled分支；两次原始记录和产物、完整普通前向路径/固定版本核验仍待完成，复合两项不勾选。下一步传回验收，无需重跑。

2026-10-28两项本机验收完成：10份原始文件与本机tar成员哈希一致，未取得服务器tar哈希不声称跨机器比对；两次配置/bench/commit、生效参数、4次前向、阶段聚合和summary复算正确，latent均finite，两段视频均81帧完整解码且非黑屏。抽帧明显模糊，仅冒烟通过。三份保存源码与固定commit blob一致（仅归一化换行），两条路径已静态核对。审计 `notes/week4/fp8_smoke_audit.json`，说明 `notes/week4_fp8.md`。下一步主环境当期baseline total，预热1+正式3，sample-smi，GPU0独占；不复用A1。

## Day 7：FP8 性能 + 生成 + 整理

- [x] FP8 组性能（GPU 0，GPU 1 按流程不并行计算，未独立全程监控，主环境）：baseline total → fp8 total → fp8_scaled total（各为预热 1 + 正式 3），再各跑一次 stages；范围重叠按预登记追加一轮，三组各六次原件已验收。**baseline 在主环境里重新测**，不复用 A1，因为环境不同

2026-10-28已收到F0–F5共六个total会话的服务器摘要，唯一确认轮结束，各配置累计正式6次。摘要合并基线/普通FP8/scaled均值134.030760/132.335426/134.644845秒；普通FP8耗时少1.2649%、scaled多0.4582%，两者范围均与基线重叠，速度门槛未达，不再追加total。六个原始notes会话及gpu.csv待传回本机验收，三个20步stages尚未执行，复合性能项保持未完成；详见 `notes/week4_fp8.md`。

随后F0–F5原件本机验收通过：60文件、24轮（含预热）result、31,692行采样均核对，六份summary复算完全一致。三配置allocated/reserved/整卡采样每次正式峰值均14698.353515625/18666/19627 MiB；原始六次速度结果保持未达门槛。审计 `notes/week4/fp8_total_audit.json`。下一步F6/F7/F8三种20步stages、各预热1+正式1，尚未执行，复合性能项保持未完成。

2026-10-29 F6–F8阶段原件已验收：12份JSON、6轮各40次前向、阶段/summary复算一致，审计 `notes/week4/fp8_stages_audit.json`。正式去噪allocated为4183.47/2882.59/2917.08 MiB，普通/scaled少31.10%/30.27%；普通DiT加载峰值反升、scaled下降，完整表见 `notes/week4_fp8.md`。整次峰值均位于VAE解码，无下降；速度门槛仍未达。性能项完成，72段生成及质量尚未执行。
- [x] 计时结束后，双卡生成 fp8、fp8_scaled 各 36 段，共72段；2026-10-29本机视频哈希/完整解码及收据events核验通过，各配置两卡18+18、首次36成功、无失败/重试
- [x] 写 `logs/week4.md`；更新 `REPRO.md`（新环境的运行命令、新增脚本）
- [ ] 第 5 周再做 FP8 的指标、两个配置的盲评和结论，然后汇总第一张权衡表

---

## 本周出口标准

- [x] `notes/week4_sage.md`：varlen 实际内核、数值误差、算子耗时
- [x] trace 证据：attention 内核已从 PyTorch flash 换成 SageAttention
- [x] 环境等价性检查有结论（及相应处理）
- [x] attention 组性能表：total 3 次（或合并 6 次）+ stages 阶段分解 + 三种峰值
- [x] sageattn 36 段生成、覆盖率、指标；36 对 + 12 段评分已回收并完成本轮解盲统计（未解决维度另列，非质量通过）
- [x] `notes/week4_result_attention.md` 结论卡，按预登记规则分类为“结论不足”
- [x] FP8 两配置：冒烟、源码核对、性能表、72 段视频；不等于质量评估通过

**止损**：

- varlen 报错 / NaN 超过半天仍无法解释 → attention 组记为"未评估"，直接转 FP8 组，不改后端；
- 环境等价性检查不一致 → 先停，由你决定处理方式，不默认选择；
- 评分到 Day 6 还没回收 → attention 结论顺延到第 5 周，不跳过质量门槛直接下结论；
- FP8 冒烟失败 → 记录错误，第 5 周只报 attention 组。

---

## 本周产出文件一览

```
E:\cstudy\AIGC\V\
  scripts/check_sage_varlen.py
  notes/week4_sage.md                 # varlen 源码核对 + 算子数值/耗时
  notes/week4_perf_attention.md       # A1–A6 性能表
  notes/week4_result_attention.md     # attention 组结论卡
  notes/week4_fp8.md                  # FP8 源码核对 + 冒烟 + 性能
  notes/eval/sageattn.jsonl, fp8.jsonl, fp8_scaled.jsonl
  notes/week4/<session>/              # bench 原始结果
  logs/week4.md

服务器 /home/user/projects/wan21-video/
  outputs/eval/sageattn/  fp8/  fp8_scaled/   # 视频（不进 git）
```

---

## 写进报告的边界

- **速度结论只覆盖 P01/s42、480P、81 帧、20 步、单张 RTX 5880 Ada（sm_89）**，质量结论覆盖冻结的 36 段。
- **算子级加速 ≠ 端到端加速**：两者都报，用阶段表把它们联系起来。
- **SageAttention 是有损的近似 attention**：质量门槛"通过"只说明在这套评测集和单评分者下没有观察到超出阈值的退化，不等于输出与基线等价。
- **attention 组的环境是副本**：其基线与第 2 周、FP8 组的基线不是同一次测量，不同组的绝对秒数不直接比较，只比较各自组内的下降率。
- **FP8 不预设收益**：在 `fp8_fast` 关闭时，省显存与提速是两个独立问题，分别报告。
