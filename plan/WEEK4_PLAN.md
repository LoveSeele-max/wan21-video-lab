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

- [ ] `nvidia-smi` 确认两卡空闲；`git -C musubi-tuner rev-parse HEAD` 为 `f8a1b03`；`sha256sum scripts/bench.py scripts/gen_eval.py configs/eval_*.yaml configs/prompts_frozen.yaml`，与本机记录比对
- [ ] 激活 `env-sagecheck`，打印 torch / triton / sageattention 版本与安装路径
- [ ] **读安装后的源码**：`python -c "import sageattention, inspect; print(inspect.getsourcefile(sageattention.sageattn_varlen))"`，打开该文件，记录 `sageattn_varlen` 实际调用的内核（Triton 还是 CUDA 扩展、Q/K/V 各用什么精度）。写入 `notes/week4_sage.md`。**这一条是之后"机理一句话"的依据**，不能凭函数名推测。
- [ ] 写 `scripts/check_sage_varlen.py`，在 GPU 0 上用 Wan 1.3B 的**真实形状**对比 `sageattn_varlen` 与 PyTorch SDPA：
  - 自注意力：b=1，L = 21×30×52 = **32760**，12 个头，head_dim 128，BF16；
  - 交叉注意力：q 长度 32760，k/v 长度 512；
  - 输入用固定种子的随机张量，再加一组"真实激活"（可选：在冒烟时用 hook 存下第 1 步第 0 层的 q/k/v）；
  - 输出：是否有 NaN/Inf、最大绝对误差、平均绝对误差、余弦相似度；两个函数各自的耗时（同步后计时，预热 3 次，测 10 次）。
- [ ] 结果写入 `notes/week4_sage.md`。**这只是算子层面的证据**，不代表端到端收益。

**止损**：varlen 报错或出现 NaN → 停止 attention 组，按预登记记为"未评估 / 结论不足"，写清错误。是否启用 xformers 备选，由你决定（预登记只允许在"安装不兼容"时启用；模型层失败算不算这种情况，需要你明确拍板并记录）。

## Day 2：模型级冒烟 + 确认内核 + 环境等价性（服务器 GPU 0）

- [ ] **冒烟**：`bench.py --config configs/eval_sageattn.yaml --mode stages --steps 2 --warmup 0 --repeat 1 --tag W4-sage-smoke`（在 `env-sagecheck` 中）。检查调用计数（4 次前向）和分段之和
- [ ] **NaN / 黑屏检查**：加载输出 latent，检查 `torch.isfinite(latent).all()`；抽查 MP4 的第 0、40、80 帧，看像素均值和方差是否正常
- [ ] **确认 SageAttention 真的被调用了**：`--mode profile --config configs/eval_sageattn.yaml --tag W4-sage-profile`（20 步，抓 2 步）。在 trace 中：
  - 应出现 SageAttention 的内核（以 Day 1 读源码得到的名字为准）；
  - **不应再出现** `pytorch_flash` / `flash_fwd` 一类内核；第 2 周 R5 中这类内核正好 240 次，对应 4 次前向 × 30 层 × 2 种 attention；
  - 按第 2 周 `analyze_week2.py` 的关联过滤方法统计：attention 类内核耗时、占比，与 R5 的 5.70 秒、53.90% 对照。仅作机理解释，不作为速度依据。
- [ ] **环境等价性检查（D2）**：在 `env-sagecheck` 中用 `gen_eval.py` 和 `eval_baseline.yaml` 生成 P01/s42，输出到独立命名空间 `baseline-envcheck`，比较 SHA256：
  - 与第 3 周 baseline 的 P01_s42 一致 → 第 3 周的 36 段基线直接作为配对一侧；
  - 不一致 → 停下来，记录原因；之后两种选择：在副本环境里重新生成 36 段基线（多约 42 分钟双卡），或者接受该差异并在报告中注明。**由你决定**，不默认选择。

**出口**：varlen 能跑通、无 NaN；trace 证明内核已切换；环境等价性有结论。

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

- [ ] 跑 A1–A4；计算下降率 `(基线均值 − 候选均值) / 基线均值`，以及双方的 min/max 区间是否有交集
- [ ] 需要时跑 A5/A6，合并六次正式值，重新判定
- [ ] 整理 `notes/week4_perf_attention.md`：端到端表、阶段表（重点看去噪段；T5、VAE 段理论上不应变化，若有变化就说明有噪声或其他混杂因素）、allocated / reserved / 整卡采样三种峰值、每次前向的耗时
- [ ] 与 Day 1 算子级加速比对照，粗算一下：attention 内核加速了 x 倍，去噪段理论上应缩短多少，实际缩短了多少。两者对不上就写出可能的原因，不强行解释

**预期**（现在写下，事后对照，不追认）：attention 约占去噪段内核时间的 54%。若 varlen 内核比 PyTorch flash 快 1.5–2 倍，去噪段可能缩短约 18–27%，端到端约 15–23%（T5 加载和 VAE 解码不受影响）；峰值显存仍在 VAE 解码阶段，基本不变。若 varlen 走的是较慢的实现，收益可能达不到 5% 门槛。

## Day 4：36 段生成 + 自动指标

- [ ] 计时全部结束后，双卡运行：`gen_eval.py --config configs/eval_sageattn.yaml --shard 0/2`、`--shard 1/2`（在 `env-sagecheck` 中，放进 tmux）
- [ ] `--summarize` 得到覆盖率；失败最多重试 1 次，首次失败也要报告
- [ ] 传回 MP4、JSONL 和 coverage（不传 latent）；在本机逐段核对哈希
- [ ] 本机运行 `metrics.py`，得到 36 条 SSIM / CLIP。只用于发现异常（第 3 周探针已证明它们不能代替人工），并与基线同一样本**逐条对比**，把差值大的样本记下来，评分后再对照

## Day 5：盲评包 + 评分

- [ ] 用 `blind_pack.py` 制作包：36 对 baseline vs sageattn（左右随机）+ 固定 12 段 sageattn 单视频，全部混排、编号随机；映射表单独存放
- [ ] 提醒评分者（R01）：判据 v1.0；静止本身不扣时序分；方向和程度要一致，拿不准的选"无法判断"
- [ ] 回收后原件先归档并记录哈希，再解盲；方向与程度冲突的先请评分者澄清，**不自行改分**

## Day 6：attention 组结论 + FP8 组启动

**attention 组**：

- [ ] `analyze_blind_ratings.py`：每个维度报 36 对的可评分数、明显变差 / 轻微变差 / 无差异 / 变好的计数，分母分别用 36 和可评分数
- [ ] 按 `optimization_rules.yaml` 的优先顺序判定：先查硬性问题 → 再查证据是否完整 → 最后查收益与质量门槛
- [ ] 写 `notes/week4_result_attention.md`，即结论卡：

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

- [ ] 冒烟：`eval_fp8.yaml`、`eval_fp8_scaled.yaml` 各跑一次 2 步 stages；从日志确认 DiT 权重的 dtype（普通 fp8 存为 float8，scaled 路径打印的量化信息），以及 `fp8_fast` 为 False
- [ ] 读源码，确认两条路径在前向时是"存 fp8、算时反量化成 bf16"，还是用了 fp8 矩阵乘（`fp8_fast` 关闭时，预期是前者），记入 `notes/week4_fp8.md`。**预期**：显存下降集中在 DiT 加载和去噪阶段；整次峰值仍在 VAE 解码，可能基本不变；速度可能持平或变慢

## Day 7：FP8 性能 + 生成 + 整理

- [ ] FP8 组性能（GPU 0，GPU 1 空闲，主环境）：baseline total → fp8 total → fp8_scaled total（各为预热 1 + 正式 3），再各跑一次 stages；范围重叠时按预登记追加一轮。**baseline 在主环境里重新测**，不复用 A1，因为环境不同
- [ ] 计时结束后，双卡生成 fp8、fp8_scaled 各 36 段（共 72 段，约 85 分钟），可以放在夜间
- [ ] 写 `logs/week4.md`；更新 `REPRO.md`（新环境的运行命令、新增脚本）
- [ ] 第 5 周再做 FP8 的指标、两个配置的盲评和结论，然后汇总第一张权衡表

---

## 本周出口标准

- [ ] `notes/week4_sage.md`：varlen 实际内核、数值误差、算子耗时
- [ ] trace 证据：attention 内核已从 PyTorch flash 换成 SageAttention
- [ ] 环境等价性检查有结论（及相应处理）
- [ ] attention 组性能表：total 3 次（或合并 6 次）+ stages 阶段分解 + 三种峰值
- [ ] sageattn 36 段生成、覆盖率、指标；36 对 + 12 段评分已回收并解盲
- [ ] `notes/week4_result_attention.md` 结论卡，按预登记规则分类
- [ ] FP8 两配置：冒烟、源码核对、性能表、72 段视频

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
