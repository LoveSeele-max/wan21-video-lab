# 第 3 周计划：评测判据、探针、冻结评测集

**日期**：2026-10-18 编写。配套 [EXECUTION_CHECKLIST.md](../doc/EXECUTION_CHECKLIST.md) 第 3 周、[EXPERIMENT_PLAN.md](../doc/EXPERIMENT_PLAN.md) 阶段 1、[METRICS.md](../doc/METRICS.md) 第三、四、七节。
**前置**：第 2 周服务器 R1–R5 与本机 L0–L2 已完成（见 [WEEK2_PLAN.md](WEEK2_PLAN.md)、[week2_baseline.md](../notes/week2_baseline.md)、[week2_local_16g.md](../notes/REVIEW/week2_local_16g.md)）。

**收尾状态（2026-10-22）**：按已确认范围完成 Week3：冻结判据与评测集、36 段基线、49 条自动指标、16 段与四对匿名试评、资源成本和后续预登记均已留档；SageAttention 独立副本安装及小张量检查通过。用户已取消人工耗时要求。试评局限、未验证的 P02 动作反例及环境边界保留，不宣称优化或评分可靠性已经证明。回顾见 [WEEK3_REVIEW.md](../notes/REVIEW/WEEK3_REVIEW.md)。

**本周要拿到的东西**：

1. **两件第 2 周遗留事项的结论**：本机是否依赖系统内存回退；跨显卡初始噪声是否不同。
2. **人工判据定稿**：每个维度三档，每档都有可操作的文字判据。
3. **冻结评测集** `configs/prompts_frozen.yaml`：12 条提示词、3 个种子、480P 及其他全部生成设置。
4. **基线评测集视频**：12 × 3 = 36 段，服务器生成。它们是第 4–5 周所有对照的"基线"一侧。
5. **探针结论**：两个自动指标分别"能区分 / 不能区分 / 部分区分"什么。
6. **成本估算**：一套评测的累计任务用时与双卡任务跨度，并据此确定第 4–5 周每组配置的数量上限；人工时间未测，用户已取消该要求。
7. **第 4–5 周的预登记**：两组对照的配置文件与配对方式，在看到任何优化结果之前写定。

**本周不做**：

- 不做任何优化对照；attention 安装检查只放在 Day 7 独立副本，允许导入及小张量检查，不跑模型生成；
- 不改上游代码；
- 不下载训练数据；
- 不引入暂缓清单里的指标（MMD、光流、ImageReward 等）。

---

## 本周决策

| 编号 | 决策 | 采用 | 理由 |
|---|---|---|---|
| D1 | 评测集规模 | **12 条提示词 × 3 个种子 = 36 段** | 实际双卡任务跨度 42.32 分钟。后续已冻结最多三个新增单项配置，见预登记；削减时减配置个数，不减提示词集 |
| D2 | 初始噪声 | **2026-10-18 用户确认不加 `--cpu_noise`**，配置明确为 false，沿用 GPU 噪声 | 两机 CPU、GPU 完整哈希均不同，CPU 生成未提供跨平台相同噪声的保证；保持 Week 2 配置连续性，不因噪声方式追加基线重测 |
| D3 | 评测生成方式 | 新写 `scripts/gen_eval.py`：**复用 `bench.py` 的参数构建、分词器包装和单提示词调用路径**，单进程循环"提示词 × 种子" | 与已测量的基线走同一条代码路径。不用上游 `--from_file` 批量模式，那条路径要先预编码全部提示词并复用模型，与基线路径不同 |
| D4 | 评测生成可以双卡并行 | GPU 0 / GPU 1 各跑一半 | 用于质量素材生成和覆盖率，不作为延迟对照；保留物理卡编号。第 1 周只验证过同设置样例输出一致，不外推所有提示词与种子 |
| D5 | 自动指标 | 相邻帧 SSIM（无需权重）；CLIP 帧—文本相似度，报均值与最低值 | 与 METRICS 3.2 一致。CLIP 用 `openai/clip-vit-base-patch32`，下载前记录许可 |
| D6 | 盲评组织 | 匿名编号 + 随机顺序，映射表单独存放、评完再打开；对比维度用左右并排、左右随机 | 本次答卷编号 R01，用户计划由舍友评分，但会话与独立性未核实；仍为单评分者试评，不宣称严格独立盲评 |

---

## 一、第 2 周遗留：两项验证（Day 1）

### 1.1 本机是否依赖系统内存回退

背景见 [week2_local_16g.md](../notes/REVIEW/week2_local_16g.md) 最后一节：本机 reserved 峰值 18666 MiB，超过 16311 MiB 的物理容量，与服务器数值完全一致；VAE 解码比服务器慢 7.0 倍，去噪只慢 4.9 倍。

- [x] NVIDIA 控制面板 → 管理 3D 设置 → 程序设置 → 添加 `E:\cstudy\AIGC\V\local\env\python.exe` → "CUDA - 系统内存回退政策"设为"首选无系统内存回退"。用户提供截图并确认已应用；运行日志未独立查询策略生效状态。
- [x] L3：`--mode stages --warmup 0 --repeat 1 --sample-smi --tag L3`。采用 stages 模式，是为了 OOM 时能看出死在哪一段。
  - **OOM**：记录 OOM 所在段；结合策略生效和运行条件证据判断是否支持依赖回退，进入下一步。
  - **实际跑通**：2026-10-18，626.700 秒，allocated 14699.2 MiB、reserved 18666 MiB、整卡采样峰值 15718 MiB，VAE 解码 72.009 秒，全部自检通过。此前的 VAE OOM 预测未出现；不据此证明全程独立显存驻留。详见本机记录的 L3 节。
- [x] L4 条件分支已判断为不适用：L3 无 OOM，不创建配置或追加运行。原预案为：若 OOM 发生在 VAE 段，新建 `evidence/local/configs/local_16g_L4.yaml`（只加 `vae_cache_cpu: true`），按同样方式跑 L4 一次。
  - 跑通：得到"关闭回退后的可运行配置"，作为本机表的新一行；
  - 仍然 OOM：记为卡点。这正是第 6–7 周 VAE 显存改动的实机动机，**本周不继续加参数**。
- [x] 记录最后一次用户确认的策略为“偏好无系统内存回退”；尚无恢复默认的记录。
- [x] 按结果更新 16G 结论的措辞：`notes/week2_local_16g.md`、`WEEK2_PLAN.md` Day 6、`doc/README.md`、`doc/EXECUTION_CHECKLIST.md` 第 6–7 周 16G 小节。限定为该 Windows 环境固定配置跑通，不宣称全程独立显存驻留。

**预算**：本机最多 2 次运行（每次约 11 分钟），不超过半天。

### 1.2 跨显卡初始噪声

- [x] 本机完整噪声指纹已实测：torch 2.9.1+cu128、CUDA 12.8、RTX 5060 Ti，seed 42、shape `(16,21,60,104)`、float32。CUDA SHA256 为 `a9d1fa9d45e8f38fba70a60b62d95a0c3eec8984cd0a3b861ea0933f180a3eb7`；CPU SHA256 为 `2eeb4278bd34ac3d08bc72362833b5caa915cbebbb312465618dbfff8e5463ef`。
- [x] 服务器 GPU 0 上执行相同检查，对比完整张量 SHA256；用户提供 torch 2.9.1+cu128、CUDA 12.8、RTX 5880 Ada 输出。CUDA SHA256 为 `8f0c5a4100c4be1e84d6a79016a9a387cbded825dae9d6aed70ec0411843b801`；CPU SHA256 为 `d58dda95871619b9a739a5fd138a6a45ecacdfc6329feb27179411e336be985b`。两者均与本机不同。前 5 个值只用于辅助查看：

```bash
CUDA_VISIBLE_DEVICES=0 /home/user/projects/wan21-video/env/bin/python - <<'PY'
import hashlib
import json
import torch

print("torch:", torch.__version__, "CUDA:", torch.version.cuda,
      "GPU:", torch.cuda.get_device_name(0))
for device in ("cuda:0", "cpu"):
    g = torch.Generator(device=device).manual_seed(42)
    x = torch.randn((16, 21, 60, 104), dtype=torch.float32,
                    generator=g, device=device).cpu().contiguous()
    print(json.dumps({
        "device": device, "seed": 42, "shape": list(x.shape),
        "dtype": str(x.dtype), "first5": x.flatten()[:5].tolist(),
        "sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest()
    }))
PY
```

- **实际结果**：CUDA 完整哈希不同，但前 5 个值完全相同；CPU 完整哈希也不同，第 5 个值本机为 `0.6784183979034424`、服务器为 `0.6784184575080872`。
- 确认本次跨机器 GPU 噪声不同，不能据此确定差异幅度、具体实现原因或画面差异的主要原因。CPU 同种子也未实现跨平台逐字节相同噪声，因此撤回 D2 原先“加 cpu_noise 即可共享同一份噪声”的保证。
- Day 1 检查结束，不追加视频生成或底层 RNG 排查；D2 已在 Day 2 由用户确认不加 `cpu_noise`。

---

## 二、人工判据定稿（Day 2）

在 METRICS 3.1 的基础上，形成“三项单视频评分 + 逐维度配对比较”，详细标准以 [week3_rubric.md](../notes/week3_rubric.md) 为准。v1.0 已于 2026-10-18 冻结；校准样例保留用户原分及局限，不声称已证明评分者一致性。

- [x] 三项单视频判据已写出：画质与结构、时序与运动自然度、提示词符合度；每项 2/1/0 三档及“无法判断”，不合成总分。
- [x] 写出**片段内动作判据**草案（沿用 [BASELINE_PROBE.md](../doc/BASELINE_PROBE.md) 第三节）：挥手至少一次往返；行走要求有效步态与方向位移，不要求“走完”；12 条提示词冻结时再核对。
- [x] 每个维度保留“无法判断”，区分明确失败、缺少输入与无法观察；没有动作不自动扣时序分。
- [x] 配对比较逐维度选 A 更好/无明确优劣/B 更好/无法判断，并单独记程度和证据，解盲后再转换成相对基线结果。
- [x] 制作 Excel 评分表（原本地引用：`../outputs/week3-scoring-20261018/week3_scoring.xlsx`；该文件未随公开快照发布或路径已调整），含单视频、配对和说明三页；下拉选项、冻结表头、已评数量公式已检查，未预填任何分数。
- [x] 第 1 周 GPU 0 视频和本机 L0 视频各按判据试评一次，三项用户评分已写进判据文件：C01 为 2/1/2，C02 为 1/1/0。
- [x] 用户补充两条试评的文字观察，已同步到评分表与判据样例，原分数保留。
- [x] 用户补充 C02 基本定住、没有明显挥手，要求保留当前主观评分；保留原分及与正式判据不完全对应的校准局限，不再重评。问题位置、查看方式及耗时未记录，不补造。
- [x] 完成试评反馈后**冻结判据 v1.0（2026-10-18）**。之后实质改动须新建版本并统一重评，不混合版本汇总。

---

## 三、冻结评测集（Day 2）

### 3.1 提示词 v1（2026-10-18 用户确认沿用以下原文，已冻结）

覆盖两类：人物动作（与 BASELINE_PROBE 候选动作重合，基线视频可以顺带当作第 11–12 周选动作的预览）；非人物场景（用来发现优化带来的通用画质退化，比如纹理、细小运动、镜头运动）。

| ID | 类别 | 提示词（英文，与基线同风格） | 主要看什么 |
|---|---|---|---|
| P01 | 动作-低 | A person waving one hand, plain background, static camera. | 沿用第 1–2 周的提示词，保持连续性 |
| P02 | 动作-低 | A woman nodding her head slowly, plain gray background, static camera. | 小幅头部运动 |
| P03 | 动作-低 | A man turning around to face the camera, plain background, static camera. | 转身时的身份一致性 |
| P04 | 动作-中 | A person walking from left to right, full body, plain studio background, static camera. | 步态、肢体协调 |
| P05 | 动作-中 | A person jogging in place, full body, plain background, static camera. | 快速肢体运动 |
| P06 | 动作-低 | A person clapping hands in front of the chest, upper body, plain background, static camera. | 双手交互、手指结构 |
| P07 | 自然 | Ocean waves crashing on dark rocks, overcast sky, static camera. | 大面积纹理运动 |
| P08 | 细节 | Close-up of a candle flame flickering in a dark room, static camera. | 细小运动；对闪烁类退化敏感 |
| P09 | 镜头 | Slow aerial camera pan over snowy mountains at sunrise. | 镜头运动的连续性 |
| P10 | 动物 | A golden retriever running across a green lawn, side view, tracking shot. | 快速运动 + 四肢结构 |
| P11 | 城市 | A busy city street at night with cars passing, neon lights, static camera. | 多主体、高频细节 |
| P12 | 静物 | Steam rising from a cup of coffee on a wooden table, soft morning light, static camera. | 几乎静止的画面，检验"静止"是否被误判为好 |

### 3.2 其余设置

- **种子**：`[42, 1001, 2002]`，在看到任何评测视频之前写定，不再更换。
- 480 × 832（高 × 宽）、81 帧、16 fps、20 步、UniPC、guidance 5.0、flow shift 5.0、torch attention、BF16，负面提示词使用固定 commit 上游默认值；`cpu_noise: false`，已按用户决定冻结。
- 写入 `configs/prompts_frozen.yaml`，文件头注明冻结日期，以及"修改须新建版本文件"。

- [x] 提示词沿用本表原文，种子沿用 `[42, 1001, 2002]`，已写入 `configs/prompts_frozen.yaml`；每配置预定 36 段。完整生成设置已在下方 D2 确认后冻结。
- [x] 在 `notes/week3_frozen.md` 写明每条提示词的入选理由（一句话），以及**刻意没覆盖的范围**（如多人、文字、人脸特写、高分辨率）。
- [x] 确认 D2 初始噪声方式为 GPU，补齐 `cpu_noise: false` 并冻结完整生成设置；已建立 `configs/eval_baseline.yaml`。

---

## 四、生成基线评测集（Day 3）

### 4.1 `scripts/gen_eval.py`

只做这些事：

- 读取 `prompts_frozen.yaml` 和一个生成配置 yaml（本周用基线配置，第 4–5 周只换这个文件）；
- 复用 `bench.py` 的 `build_args`、分词器包装和单次调用路径；每段视频前覆盖 `args.prompt` 和 `args.seed`；
- 支持 `--shard 0/2`、`--shard 1/2`，把 36 段分给两张卡；
- 成功记录的配置身份及 MP4/latent 完整哈希匹配才跳过；单有文件或半成品不跳过，以便中断后续跑；
- 每段写一行 JSONL：提示词 ID、种子、配置名、状态（ok / OOM / error）、耗时、allocated 峰值、输出路径、MP4 的 SHA256。**失败也写**，不静默删除（METRICS 第二节第 4 条）；
- 输出目录按 `outputs/eval/<配置名>/<提示词ID>_s<种子>.mp4` 命名。

- [x] 写 `scripts/gen_eval.py`；2026-10-19 本机 14 项 CPU 模拟测试通过（分片、成功跳过、损坏/半成品恢复、OOM/失败/中断记录、两次上限及覆盖率等）。真实上游参数解析通过，不代表实际 GPU 推理已验证。操作说明见 [WEEK3_GENERATION_GUIDE.md](../notes/WEEK3_GENERATION_GUIDE.md)。
- [x] 2026-10-19 用户通过 SCP 将 gen_eval.py、test_gen_eval.py、prompts_frozen.yaml、eval_baseline.yaml 四个文件传到服务器对应目录，四项均显示 100%；服务器 dry-run 与推理检查待执行。
- [x] 2026-10-19 15:28 服务器 dry-run 通过：baseline-smoke2、2 步、分片 0/2、两个任务 P01/s42 和 P01/s2002。用户提供的 nvidia-smi 进程清单仅见桌面相关进程，未见模型计算任务；GPU 0 占用 545 MiB，GPU 1 占用 20 MiB。该检查不代替实际冒烟推理。
- [x] 2026-10-19 服务器 GPU 0 冒烟生成完成：P01/s42、P01/s2002，各 2 步，均输出 ok，latent、VAE 解码、MP4 保存成功。事件日志路径已输出；依据用户完整终端日志核对，未独立读取服务器 JSONL 内容。
- [x] **路径一致性检查**：2026-10-19，`gen_eval.py --prompt-id P01 --seed 42 --tag pathcheck` 的 20 步、GPU 噪声 MP4 与 R4 `R4-total-20261015-164943-11cd808d/run-01` 比较 SHA256，用户报告完全一致。此结论限于本次同设置样例；未在本地独立重算服务器哈希。
- [x] 正式双卡生成运行结束：2026-10-19 用户报告已完成；预定各 18 段，实际成功数量与文件完整性待汇总核对。
- [x] 执行 `--summarize`：2026-10-19 用户提供输出 `Coverage: {'ok': 36}`，36 个组合通过记录与文件哈希检查，最终清单为 `notes/eval/baseline.jsonl`，覆盖率表为 `notes/eval/baseline/coverage.json`。
- [x] 传回本机：36 段 MP4、最终 JSONL 和 coverage.json（latent 不传）。2026-10-19 本机独立核查：36 条唯一记录均为 ok，36 段 MP4 的 SHA256 全部与清单一致，GPU 0/1 各 18 条。

**出口**：36 行 JSONL，状态全部记录；覆盖率短表第一行（基线）。

实现说明：双卡各写独立事件 JSONL，全部结束后以 `--summarize` 合并为 `notes/eval/baseline.jsonl`（36 行）和 `coverage.json`，避免并行写入冲突。原始尝试、失败和中断均保留。输出重命名为预定文件名，不改变上游保存方式。

---

## 五、自动指标与反例探针（Day 4）

### 5.1 `scripts/metrics.py`（本机运行）

- 相邻帧 SSIM：灰度、全分辨率，每段输出均值与最小值（共 80 个相邻帧对）；
- CLIP 帧—文本相似度：每 4 帧取 1 帧（21 帧），输出均值与最低值；
- 输出 CSV：视频路径、提示词 ID、四个数。
- [x] 下载 CLIP 权重前，在 `notes/week3_metrics.md` 记录模型名、revision、许可依据及模型卡限制；固定 revision 为 `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`，权重与许可原文已保存到项目 E 盘目录。
- [x] 实现 `scripts/metrics.py`：固定灰度全分辨率 SSIM 与 21 帧 CLIP 余弦口径、逐帧记录和错误留痕。三项行为测试通过，本机 pip check 通过，bench.py 哈希未变。
- [x] 运行全部基线和探针指标：49 条（36 基线 + 12 像素探针 + 1 文本替换）全部成功；80/21 逐帧值数量、汇总、哈希、作用区间和有限值均核对通过，见 `notes/week3_metrics_run/`。

### 5.2 探针样本（由基线视频构造，不另外生成）

| 探针 | 构造方法 | 要回答的问题 |
|---|---|---|
| 正常 | 原样使用 P01/s42、P04/s42、P07/s42 | 好样本上的取值范围 |
| 静止 | 每段的第 0 帧重复 81 次 | SSIM 会不会**偏爱不动的视频** |
| 闪烁 | 第 30–40 帧亮度交替 ±15% | 对瞬时退化是否敏感 |
| 模糊 | 第 30–40 帧高斯模糊（σ=3） | 同上 |
| 局部变形 | 第 30–40 帧在主体区域做局部像素打乱或扭曲 | 对**局部**退化是否敏感 |
| 动作不符 | P02（点头）的视频配 P01（挥手）的文本算 CLIP | CLIP 能否区分画质与动作符合度 |

- [x] 写 `scripts/make_probes.py`，构造参数全部写死并记录；帧区间为 0-based 30–40（含端点，11 帧），区域按来源第 35 帧固定，FFV1 无损编码并逐帧核对解码像素。
- [x] 逐指标逐探针填表："能区分 / 不能区分 / 部分区分"。计算前在 `notes/week3_metrics.md` 将边缘明确为区间宽度的 5%；结果及解释见 `notes/week3_probe.md`，不事后调阈值。静止 SSIM 高端异常，其余像素退化全部在全局范围内；文本替换的 SSIM 不变，不作动作识别证据。
- [x] 四类像素探针已混入 Day5 匿名试评，16 段及四对答卷已回收、解盲并记录澄清。四类现象均有备注；静止时序评分存在规则执行偏差，不据此声称人工判据稳定检出。P02 文本替换候选未单独验明动作内容，不作为已确认动作反例。

**预期**（写下来，事后对照，不追认）：SSIM 会把"静止"判成最好；CLIP 对"动作不符"大概率区分不了；均值对局部 10 帧的退化不敏感，最低值可能部分敏感。

**出口**：`notes/week3_probe.md`，包含探针表、每个指标"可以用于 / 不可以用于"的一句话结论。

---

## 六、试评与成本（Day 5）

### 6.1 盲评工具

- [x] `scripts/blind_pack.py`：2026-10-21 已制作 `outputs/week3-R01-20261021.zip`；统一格式 MP4、匿名随机编号、空白 XLSX/CSV，映射保存在包外 `notes/blind_private/`，评完再打开。未覆盖原有评分表。
- [x] 并排模式：四对 A/B 左右随机并排视频，来源、方向、帧数及编码验收通过。

### 6.2 试评

- [x] 试评包：12 段基线（每条提示词随机取 1 个种子）+ 4 段探针（静止、闪烁、模糊、局部变形各 1 段），共 16 段，打乱顺序；抽样、顺序、左右均保存随机种子并复算核对。用户计划由舍友 R01 评分，实际是否提前看过视频由其在表内记录，仍是单评分者试评。
- [x] 2026-10-22 收到 R01 答卷：16 条均填写三项档位，四对均填写三项方向/程度；原件已按 SHA256 归档，评分表未改。
- [x] 2026-10-22 用户澄清完成：V002 时序 0 仅因静止，作为判据执行偏差保留；C004 大体无差异，后续按无明确优劣解释，保留原答。用户确认未计时且不追补，人工时间标未测；其余缺失会话信息按局限保留，不补造。
- [x] 已在保留原件后解盲，四类探针备注均有对应现象；观察与档位差异及静止规则执行偏差写入 `notes/week3_trial_review.md`。不更改冻结定义，不宣称稳定检出。
- [x] 并排模式已收到四对实际评分，已逐项解盲并保留原答；C004 按解盲后用户澄清“大体无差异”处理，不计来源胜出。

### 6.3 成本估算，写入 `notes/week3_cost.md`

- [x] 2026-10-22 已完成现有生成记录的资源成本表与可复算审计：`notes/week3_cost.md`、`notes/week3_cost_audit.json`；36 段均值 138.251 秒、累计 4977.035 秒、双卡任务整体跨度 2539.075 秒。人工时间按用户决定记未测；抽检量和配置上限尚未确定。

| 项目 | 计算 |
|---|---|
| 一套评测的 GPU 时间 | 36 段 × JSONL 实测单段均值；单卡 / 双卡分别算 |
| 一套评测的人工时间 | 用户已取消要求；本次未测，后续不计时、不作为验收条件 |
| 第 4–5 周生成资源参考 | 已冻结最多三个新增配置 × 每配置生成成本；不含 bench、传输、转码与重试，不计算人工时间 |

- [x] 2026-10-22 固定第 4–5 周评分量：每配置 36 对三项配对比较、12 段三项绝对评分；子集沿用 R01 首次试评的 12 条基线，详见 `configs/scoring_frozen.yaml`、`notes/week4_prereg.md`。不要求人工耗时。此处将历史“四个维度的绝对档位”明确为冻结判据的三项绝对评分，配对另记。
- [x] 每组配置上限已登记：attention 最多一个、精度两个，总新增最多三个；不加入 fp8_fast、compile、卸载或叠加。成本表仍为估计场景。

**出口**：判据冻结版、试评记录、成本表、N 值。

---

## 七、第 4–5 周预登记（Day 6）

依据 [week2_profile.md](../notes/week2_profile.md) 的候选排序，写 `notes/week4_prereg.md`：

- [x] **组 1：attention 实现**。当前 `torch` 后端实际调用的是 PyTorch 自带的 flash 内核；已依据独立环境安装、导入与小张量检查选定 SageAttention 2.2.0（commit eb615cf6cf4d221338033340ee2de1c37fbdba4a）。不测 `sdpa`、`flash2`、`flash3`；安装日志与 pip freeze 差异已归档验收，模型对照仍待进行。
- [x] **组 2：DiT 精度 / 矩阵乘**。登记普通 fp8 与 fp8_scaled 两个配置；固定源码确认 scaled 分支独立，第一轮不加 fp8_fast。
- [x] 建立 eval_fp8.yaml、eval_fp8_scaled.yaml、eval_sageattn.yaml，与基线各只差一个行为变量及输出身份；gen_eval 配置加载均覆盖 36 项。SageAttention 已选定并通过安装及小张量检查，完整模型实验尚未执行。
- [x] 结论规则已冻结：total 预热 1 + 3 次、当期同环境基线、≥5% 速度门槛及重叠时一轮复测；三维分别至少 33 对可评、明显变差 0、轻微变差最多 3；最终 36/36 成功。分类和局限详见 `notes/week4_prereg.md`、`configs/optimization_rules.yaml`。
- [x] 噪声变更重测分支不适用：用户决定不加 `cpu_noise`，因此不追加仅由该参数变更导致的基线重测。第 4 周正式对照仍须保证两侧参数与计时方式一致。

**出口**：预登记文件 + 配置文件。登记之后再新增的配置，标注"事后追加"。

---

## 八、Day 7：整理与缓冲

- [x] 写 `logs/week3.md`，保留过程记录并追加收尾状态。
- [x] 更新 `REPRO.md`：脚本与已执行范围、CLIP 来源与许可、冻结配置、试评与归档、成本和预登记索引。
- [x] 回看出口标准并同步总清单、README 与阶段说明；局限和未验证项明确保留。
- [x] 写 [Week3 复盘](../notes/REVIEW/WEEK3_REVIEW.md)，用于复习本周目标、实现、结果与 Week4 衔接。
- [x] 可选：在服务器的**独立副本环境**里试装 attention：env-sagecheck 中 SageAttention 2.2.0 编译安装、API 导入和 sm_89 GPU 0 BF16 小张量检查通过，**未跑生成**。pip freeze 安装前后差异与构建日志已传回核对，记录见 `notes/week3_attention_check.md`。

---

## 每日安排

| 天 | 服务器 | 本机 |
|---|---|---|
| Day 1 | 噪声对照（2 条命令） | 回退验证 L3 / L4；更新 16G 措辞 |
| Day 2 | — | 判据定稿；提示词与种子冻结 |
| Day 3 | gen_eval 冒烟、路径一致性检查、36 段生成 | 写 gen_eval.py 与模拟测试（上午） |
| Day 4 | — | metrics.py、make_probes.py、探针表 |
| Day 5 | — | 盲评工具、试评、成本表 |
| Day 6 | — | 第 4–5 周预登记 |
| Day 7 | 可选：后端安装检查 | 日志、REPRO、缓冲 |

---

## 本周出口标准（周末自查）

- [x] 系统内存回退对照已执行：设置“偏好无回退”后 L3 仍成功；16G 结论措辞已同步到 4 个文件。未直接测量内存驻留或独立读取策略生效状态，不宣称完全排除系统内存参与。
- [x] 跨机器噪声对照完成：CUDA、CPU 完整哈希均不同，结果已记录。
- [x] `cpu_noise` 已由用户确认不加，配置为 false，不以跨平台噪声必然一致为理由。
- [x] `notes/week3_rubric.md` 判据 v1.0 冻结，含两条用户校准评分及适用边界
- [x] `configs/prompts_frozen.yaml` 中提示词与种子已冻结（12 条 × 3 个种子）
- [x] 完整生成设置冻结，评测基线配置已建立；脚本实现、服务器冒烟与路径一致性检查均已通过。
- [x] 基线 36 段生成完成，JSONL 与覆盖率短表齐全：服务器汇总为 `Coverage: {'ok': 36}`；已传回本机，36 段 MP4 的 SHA256 均独立核对通过。
- [x] gen_eval 与 bench 的路径一致性检查通过：用户报告 P01/s42 同设置 MP4 的完整 SHA256 一致。
- [x] 自动指标探针表与人工试评均已收尾：范围规则的负面结果、评分执行偏差和用户澄清分别记录，不宣称稳定检出所有退化。
- [x] 成本表 + 抽检量 N：每配置 36 对三维配对、固定 12 段三维绝对评分；人工时间未测且已取消要求。
- [x] 第 4–5 周预登记文件 + 配置文件：SageAttention、普通 FP8、scaled FP8 最多三个新增单项配置，收益和质量门槛已冻结。
- [x] 日志、REPRO、总清单及 Week3 复盘完成；本周不生成优化配置视频。

**止损**：

- 回退验证超过半天仍无法确定内存驻留 → 标为未直接验证，不以 reserved 数值替代驻留证据，不挤占 Day 2；
- CLIP 权重下载或许可有问题 → 本周只用 SSIM，探针表里 CLIP 一栏记"未测"；
- 36 段生成中出现失败 → 照实记录、重跑一次；仍失败就保留失败记录，不替换提示词或种子；
- 人工时间要求已由用户取消，不触发按评分用时合并判据或重跑；实质修改冻结判据仍须新版本，不追认原始评分。

---

## 本周产出文件一览

```
E:\cstudy\AIGC\V\
  configs/prompts_frozen.yaml
  configs/eval_baseline.yaml
  configs/eval_*.yaml                  # 第 4–5 周预登记的配置
  evidence/local/configs/local_16g_L4.yaml      # 视 L3 结果决定是否需要
  scripts/gen_eval.py
  scripts/metrics.py
  scripts/make_probes.py
  scripts/blind_pack.py
  notes/week3_rubric.md
  notes/week3_frozen.md
  notes/week3_metrics.md
  notes/week3_probe.md
  notes/week3_cost.md
  notes/week4_prereg.md
  notes/WEEK3_REVIEW.md
  notes/sagecheck/archive_audit.json
  configs/scoring_frozen.yaml
  configs/optimization_rules.yaml
  logs/week3.md

服务器 /home/user/projects/wan21-video/
  outputs/eval/baseline/               # 36 段视频（不进 git）
  notes/eval/baseline.jsonl
```

---

## 写进报告的边界

- **评分者只有一人，且知道研究目的**：匿名化只挡住"知道是哪个配置"，挡不住个人偏好；不做评分者间一致性统计（METRICS 暂缓清单）。
- **12 条提示词 × 3 个种子是小样本**：结论限定在这套评测集内。
- **探针是人为构造的退化**，不等于真实优化会产生的退化；它只说明指标"有没有能力"看到某类问题。
- **评测集视频的耗时是并行生成测的**，不作为延迟依据；延迟一律以 `bench.py` 为准。
- **本周的基线视频不是 LoRA 的独立测试证据**（BASELINE_PROBE 第八节）。
