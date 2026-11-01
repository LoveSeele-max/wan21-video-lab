# Week4 Day 1: SageAttention varlen 源码核对与算子诊断

日期：2026-10-26，Asia/Shanghai。

## 当前状态和证据来源

- 已收到用户在服务器执行环境检查、源码读取及算子脚本后的输出；下述服务器状态来自用户提供的输出，不是本机远程重跑。
- 上游 HEAD 为 `f8a1b03794a49239a3539015075f5123d6c07d66`。
- 服务器 `bench.py`、`gen_eval.py`、`eval_baseline.yaml`、`prompts_frozen.yaml` 的 SHA256 与本轮本机读取的对应文件一致。
- 服务器初次检查仅有 baseline。用户随后同步三个候选配置并贴回哈希，四份 eval YAML 均与本轮本机重新计算的 SHA256 一致；配置核对完成。
- 用户提供的 nvidia-smi 显示未见训练或推理任务；GPU 0 有桌面相关进程，约 545 MiB 占用，GPU 1 约 20 MiB。不是零占用；性能测量前需重新检查并记录。
- 两项 varlen 算子诊断完成，记录的检查输出均 finite；原始结果目录已传回，2026-10-26 本机验收通过。验收不包含重新执行 GPU 算子或由原始张量重算误差。

## 副本环境

用户输出确认：

| 项目 | 值 |
|---|---|
| Python | `/home/user/projects/wan21-video/env-sagecheck/bin/python` |
| torch | `2.9.1+cu128` |
| torch CUDA build | `12.8` |
| Triton | `3.5.1` |
| SageAttention | `2.2.0` |
| 安装位置 | 均来自 `env-sagecheck/lib/python3.11/site-packages/` |

本轮未重新执行依赖全量差异审计，不把版本核对表述成系统环境完整等价。

## 实际 varlen 路径

依据用户提供的已安装 Python 源码：

1. `sageattention/core.py::sageattn_varlen` 接收 `(sum_L, heads, head_dim)`。
2. 本次计划 BF16 输入时，V 显式转换为 FP16；`smooth_k=True` 时 K 减去序列维均值。
3. `triton/quant_per_block_varlen.py::per_block_int8` 分配 INT8 Q/K 和 FP32 缩放系数；Q 块 128 token，K 块 64 token。
4. `quant_per_block_int8_kernel` 将输入块转换成 FP32 进行缩放和量化，再存为 INT8。Q 的缩放融合 `sm_scale * 1.44269504`，K 使用 1.0。
5. 非因果路径调用 `triton/attn_qk_int8_block_varlen.py::forward`，启动 `_attn_fwd`；其内部调用 `_attn_fwd_inner`。
6. `_attn_fwd_inner` 对 INT8 Q/K 调用 `tl.dot`，结果转换为 FP32 后应用缩放。该语句没有显式写 INT32 累加器，不将源码推断包装成已检查硬件指令。
7. softmax 使用 `exp2`；运行最大值、归一化量及外层 `acc` 初始化为 FP32。
8. P 转为 FP16，V 为 FP16；`tl.dot(p, v, out_dtype=tl.float16)` 明确指定分块乘法输出为 FP16，再加到外层 FP32 `acc`。不能概括为 P×V 全程 FP32 累加。
9. 归一化结果写入原输入 dtype 的输出，本次为 BF16。

本路径的量化和 attention 来自 Triton 实现，不是根据 `sageattn` 定长入口或依赖包的编译情况推断。profiler 可首先检索 `quant_per_block_int8_kernel` 和 `_attn_fwd`；最终内核名及实际执行情况仍待模型 trace 验证。`_attn_fwd_inner` 是内核内部函数，不要求在 trace 中独立出现。

## 检查脚本口径

- batch=1，12 heads，head_dim=128，BF16。
- 自注意力：Q/K/V 长度均为 32760；交叉注意力：Q 长 32760，K/V 长 512。
- 固定种子 42，独立标准正态随机输入；并非真实模型激活，不替代后续完整模型验证。
- 参考使用 PyTorch SDPA，明确限制为 `FLASH_ATTENTION`；不静默回退到其他后端。依据 PyTorch 2.9 官方接口：
  <https://docs.pytorch.org/docs/2.9/generated/torch.nn.attention.sdpa_kernel.html>
- 两侧输入数值相同；SDPA 使用转置视图并将结果恢复成连续 `(L,H,D)`，Sage 使用同一 Q/K/V 与预建的 int32 累计长度。
- 同步墙钟计时：首次调用及编译不计时，另外预热 3 次、正式 10 次；记录每次、均值和范围。
- Sage 计时含其函数内部平滑、量化、V 转换和临时分配；SDPA 计时含输出布局恢复。不含输入创建、哈希、长度张量创建、误差计算或模型外围工作。
- 对每个算子的一次未计时输出检查 finite、最大/平均绝对误差及全张量展平余弦相似度；不声称每次计时输出都做了有限性检查。
- 不新增数值误差通过阈值，不以余弦相似度代替质量门槛；NaN/Inf 或运行错误时保留失败记录并停止。
- 结果独立存到 `notes/week4/varlen-<UTC时间>-<随机后缀>/`，含 result.json、实际安装模块副本及哈希、nvidia-smi 快照。编译缓存放项目 `.cache/`。
- GPU 默认物理索引 0；脚本仅在进程内设置可见设备，不安装依赖，不改主环境或上游。

## 本机脚本检查

2026-10-26 使用项目 `local/env/python.exe` 完成语法编译检查和 `--help`。
另用 CPU 小张量核对相同输出误差为零、余弦接近 1、NaN 分支不计算误差、dtype 不一致会报错，以及两项固定形状。
这些检查通过；未在本机安装或调用 SageAttention，未执行本机 CUDA 算子测试，不构成服务器运行证据。

## 首次数值与计时结果（用户服务器输出）

会话：`varlen-20261026T132946Z-a1c64053`。
原始目录：`/home/user/projects/wan21-video/notes/week4/varlen-20261026T132946Z-a1c64053/`。
终端最终状态为 `completed_operator_check_only`，两项 case 均为 `completed`。

| 项目 | 自注意力 | 交叉注意力 |
|---|---:|---:|
| SDPA / Sage 检查输出均 finite | 是 | 是 |
| 最大绝对误差 | 0.00146484375 | 0.015625 |
| 平均绝对误差 | 0.00009717637294670567 | 0.0007500409847125411 |
| 全张量展平余弦相似度 | 0.9999088048934937 | 0.9999092817306519 |
| SDPA 均值 ms | 40.66572189331055 | 0.5412466824054718 |
| SDPA 范围 ms | 39.8336723446846–41.394446045160294 | 0.5239211022853851–0.657194759696722 |
| Sage 均值 ms | 33.29109037294984 | 1.047834474593401 |
| Sage 范围 ms | 31.947071198374033–35.47416813671589 | 0.9288503788411617–1.3228286989033222 |
| SDPA / Sage 均值比 | 1.2215196750164978 | 0.5165383421990346 |

按均值，自注意力耗时下降约 18.13%，交叉注意力耗时增加约 93.60%（约为 SDPA 的 1.94 倍）。保留相反方向的结果，不因交叉注意力变慢更换后端或选择性重测。
这是固定形状、随机输入和本脚本计时边界内的单次诊断。未设置误差接受阈值，不将 finite 或高余弦解释成质量通过，也不由两项算子数字外推去噪或端到端收益。

## 原始文件本机验收

2026-10-26 检查已传回的完整会话目录，结果通过。审计侧记见
`week4/varlen-20261026T132946Z-a1c64053/local_audit.json`，不修改服务器原始文件。

- `result.json` 状态及两项 case 完成状态正确，固定形状、BF16、种子 42、预热 3 和正式 10 均符合本次脚本口径。
- 记录的脚本 SHA256 与本机实际脚本一致：`cf3b82fba270893e4f012f3a124794cd392ab0658ef73fa70cff27e71a5d1b85`。
- 三份源码副本均可解析，实际文件 SHA256 均与 `source_sha256` 一致。
- 两项 case、两种算子各有十个正且有限的毫秒计时；均值、最小值、最大值和加速比均独立复算一致。
- 环境元数据为副本解释器、torch 2.9.1+cu128、Triton 3.5.1、SageAttention 2.2.0，GPU 名称为 RTX 5880 Ada、capability 8.9，记录的选择索引为 0。
- 2026-10-26 21:29:47 的 nvidia-smi 快照包含副本 Python 和桌面进程；GPU 1 未列出训练或生成进程。该快照不是全程后台占用监控。
- 六个输入哈希格式正确，同种子同形状的两项 Q 哈希一致。输入及输出张量未归档，不能独立重算输入哈希、finite 或误差；这些数值仍是服务器脚本记录。

`result.json` 本机 SHA256：`91553bd39b668eeaa99fe1c11aa17278779ddb08a14a0c1897d28731da572327`。
候选配置随后已同步并核验，Day 1 文件核对完成；下一步进入 Day 2，不要求重跑本次算子诊断。

## 候选配置同步验收

2026-10-26 用户提供服务器 `sha256sum configs/eval_*.yaml` 输出，本机重新计算对应文件哈希并逐项比对，全部一致：

| 文件 | SHA256 |
|---|---|
| eval_baseline.yaml | `25c0ac2d0e431b3f08910be32930099c03e9c12680c23b359d391b2b48626874` |
| eval_fp8_scaled.yaml | `e950a4e01e3894744aff17403a194193ad77932120fe76592460c372c7ec5ff3` |
| eval_fp8.yaml | `23fc82eab585987a5d3d09a5f531dbb2f28cda2a4c5b9afb0b7380ab2d08bbca` |
| eval_sageattn.yaml | `f7705c8978172f5735dc571ecc28ff488e05929bb2cac82e388c1d2a96b2c731` |

服务器哈希来自用户执行结果；本机文件哈希由本轮独立读取计算，不表述为自行远程查询。
Day 1 五项已验收，不包含真实激活（可选）或完整模型。
Day 2 的两步冒烟运行记录与产物已传回，本机内容检查通过；用户随后补充服务器产物哈希，与本机重新计算的两项哈希一致，两步冒烟和 NaN/黑屏检查已验收。
现有 bench.py 实际将结果写入 `notes/week2/`、视频和 latent 写入 `outputs/week2/` 下的独立会话目录，以终端路径及 result.json 为准，不修改 bench.py 来调整目录名称。

## Day 2 冒烟：用户提供的服务器输出

会话：`W4-sage-smoke-stages-20261026-214619-9754d61f`。
运行：副本 Python，GPU 0，eval_sageattn，stages，2 步，预热 0、正式 1。

- 日志显示 SageAttention 导入成功，DiT / 权重 / VAE 精度为 BF16，FP8 未启用。
- 日志显示两步采样结束，latent 已保存，VAE 解码完成，MP4 已保存，result.json 和 summary.json 已写出。
- 日志显示解码前 latent 形状为 `[1, 16, 21, 60, 104]`，本次端到端记录为 34.142 秒，allocated 峰值 14698.4 MiB。
- 以上是两步冒烟的用户服务器输出，不与正式 20 步基线比较，不作为性能收益或质量结果。
- 记录目录：`/home/user/projects/wan21-video/notes/week2/W4-sage-smoke-stages-20261026-214619-9754d61f/`。
- 产物目录：`/home/user/projects/wan21-video/outputs/week2/W4-sage-smoke-stages-20261026-214619-9754d61f/run-01/`。
- session/result/summary、latent 和 MP4 已传回；本机已完成配置、四次前向、阶段计时、latent 全部有限性、视频完整解码和抽帧检查，详见下一节。服务器/本机两项产物哈希已比对一致。
- 当前没有模型 profiler trace，不把导入成功或采样结束视为已独立确认实际内核切换。

## Day 2 冒烟：本机内容检查

审计文件：`week4/smoke-audit-W4-sage-smoke-stages-20261026-214619-9754d61f/audit.json`。
抽帧图：同目录 `frames_0_40_80.png`。

- session/result 均为 ok；固定 commit、配置文件及 bench.py 哈希与本机一致。生效设置符合 sageattn、2 步、种子 42、BF16 日志及关闭 FP8/compile/block swap 的配置。
- 前向调用 4 次，前向耗时列表有 4 项；八项阶段函数各调用一次。用原始区间复算阶段总和为 34.14155129669234 秒，总耗时 34.142102516721934 秒，差为 0.0016144876529573435%，通过原脚本 <2% 的校验。
- 从 result 重新调用未修改 bench.py 的 make_summary，与传回 summary.json 完全一致。
- 保存的 latent 键为 `latent`，形状 `[16,21,60,104]`、dtype FP32，所有元素 finite，最小 -2.5399763584136963、最大 2.556401252746582。保存 latent 的 dtype 不等同于 DiT 权重精度。
- CPU 完整解码 81 帧，480×832，16 fps；没有全零帧。

| 视频帧 | RGB 像素均值（0–255） | 像素方差 | 最小/最大 |
|---|---:|---:|---|
| 0 | 90.50057508680555 | 783.1747642926726 | 25 / 199 |
| 40 | 90.42319628071581 | 770.6320552485331 | 25 / 197 |
| 80 | 90.86310513488247 | 775.807547689868 | 27 / 196 |

本机查看三帧图：不是黑屏，但画面严重模糊、主体难以辨认。两步冒烟不是正式质量样本，没有对应两步 torch 对照，不能断定模糊来自 SageAttention，也不能写成质量通过或按正式质量门槛分类。后续仍需模型 trace、20 步运行和冻结质量评价。

本机已计算 latent 和 MP4 哈希：

| 文件 | SHA256 |
|---|---|
| 20261026-214643-312_42_latent.safetensors | `b66144c365093efca7a833c578cc58329836b4fdfe25c49efc162e29f4867c44` |
| 20261026-214654-024_42_.mp4 | `739db333a662c598c9139369de912889a08c4326b87fde5b56d1481d95297355` |

用户随后提供上述两项服务器 sha256sum 输出，本机重新计算 latent 和 MP4 哈希，全部一致。证据来源仍区分用户服务器输出与本机独立读取，不声称自行远程查询。
Day 2 前两项“两步冒烟”和“NaN/黑屏检查”已验收；这不构成正式质量通过或模型内核 trace 验证。

随后用户执行 GPU 0、副本环境的 20 步 profile，标签 `W4-sage-profile`。现有 bench.py 在 profile 模式强制不做额外整段预热、正式运行 1 次；模型仍完整采样 20 步，预期 40 次前向，profiler 内部调度 wait=16、warmup=2、active=4 次前向，trace 捕获其中 4 次前向（2 个 CFG 步）。检查实际内核及关联耗时，只用于机理解释，不能作为正式延迟收益依据。

## Day 2 profile：用户提供的服务器输出

- 会话：`W4-sage-profile-profile-20261026-222504-af72f220`。
- 终端确认 profile、实际 20 步、预热 0、正式 1，BF16 DiT/权重/VAE 日志，FP8 未启用。
- 用户输出显示 20/20 采样结束、latent 保存、VAE 解码和 MP4 保存结束，并写出 result.json 与 summary.json。
- 记录耗时 138.178 秒、allocated 峰值 14698.4 MiB，包含 profiler 开销，不与 Week2 历史基线直接作正式速度对照。
- 记录目录：`/home/user/projects/wan21-video/notes/week2/W4-sage-profile-profile-20261026-222504-af72f220/`。
- 产物目录：`/home/user/projects/wan21-video/outputs/week2/W4-sage-profile-profile-20261026-222504-af72f220/run-01/`。
- 原始 trace、operators、session/result/summary、20 步 latent 和视频已传回，本机内容检查完成；用户随后提供服务器 trace、latent、MP4 哈希，与本机本轮重新计算的三个哈希全部一致。
- 运行状态、40 次前向、profiler.trace_ready、捕获范围及实际内核均已本机核对，三项传输哈希比对通过，模型内核确认已验收；不据这次耗时给出速度结论。

## Week4 内容为什么仍在 week2 目录

现有 `scripts/bench.py` 是 Week2 建立并验收的通用测量脚本，`run_once` 将产物写到 `outputs/week2/`，`main` 将记录写到 `notes/week2/`，目录名在源码中固定。本周沿用该脚本及其验收哈希，不为目录名称修改生成或测量程序。

目录名称不决定实验所属周次：本次以 `W4-sage-profile` 标签、完整会话 ID、有效配置和日志登记为 Week4；会话 ID 唯一，不覆盖旧 R1–R5 数据。
本周新增分析、审计和抽帧保存于 `notes/week4/`，解释及清单位于 `notes/week4_sage.md`、`plan/WEEK4_PLAN.md`，日志为 `logs/week4.md`。不为整理名称移动已经被 session/result 引用的原始文件。

## Day 2 profile：本机内容核查与关联统计

可复算脚本：`scripts/analyze_week4_profile.py`。
本机执行命令：

```powershell
& 'E:\cstudy\AIGC\V\local\env\python.exe' 'E:\cstudy\AIGC\V\scripts\analyze_week4_profile.py' --session W4-sage-profile-profile-20261026-222504-af72f220
```

输出目录：`notes/week4/profile-audit-W4-sage-profile-profile-20261026-222504-af72f220/`，含 audit.json 和 frames_0_40_80.png。脚本首次执行已通过真实 trace 的全部断言；输出目录存在时拒绝覆盖，复核已有结果可直接读取审计文件。未运行本机 GPU 推理，未修改 bench.py 或上游。

### 捕获与筛选口径

- session/result 状态均为 ok，bench 和配置哈希与本机一致，固定 commit 正确，40 次模型前向，trace_ready=true，summary 复算完全一致。
- CPU/GPU 各有 ProfilerStep#18–#21 四个标记，是同四次前向的两侧，不计成八次。
- 直接照搬 Week2 的 CPU op External-id 过滤，只能得到 21800 条内核，会漏掉没有 External id 的 240 次 `_attn_fwd` 和 480 次 `quant_per_block_int8_kernel`，不能据此声称 attention 占比为零。
- 本轮增加关联规则：kernel 的 correlation 对应 CUDA runtime/driver 启动事件；启动事件须在同 PID/TID 的捕获 CPU Step 内，kernel 须在同名 GPU Step 内（边界容差 1 微秒）。保留原 External-id 关联，并用该规则补回 720 条 Triton 内核。
- 最终保留 22520 条内核，剩余 27 条全部结束于首个 GPU Step 前；每个捕获 Step 恰有 60 次 Sage attention、120 次 Q/K 量化。
- 用同一补充规则重新分析历史 R5：仍保留 16040 条，无需补回内核；排除的 996 条全部位于捕获边界前，与原 Week2 审计一致。未运行旧 analyze_week2.py 的全目录扫描入口，避免其 CSV 假设及报告覆盖。

### 关联内核分类

| 内核分类 | 事件数 | 时长之和（秒） | 关联内核时长占比 |
|---|---:|---:|---:|
| Sage `_attn_fwd` | 240 | 4.236092 | 44.41% |
| Sage Q/K INT8 量化 | 480 | 0.063730 | 0.67% |
| 名称匹配 GEMM/GEMV | 1228 | 1.952186 | 20.47% |
| 其他内核 | 20572 | 3.286772 | 34.46% |
| 合计 | 22520 | 9.538781 | 100% |

当前 trace 全部 kernel 事件中未发现 flash 前向内核，捕获区间内 240 次 Sage attention 与 4 前向 ×30 层 ×2 attention 对应，支持实际切换到 varlen Triton 内核的结论。

历史 R5 flash 为 240 次、5.703751 秒、53.90%，只作机理参考。Sage attention 与量化的两行合计约 4.299823 秒、45.08%，不包含所有平滑、dtype 转换、分配及主机工作，不能称为整个 attention 包装函数开销或端到端收益。内核忙碌时长占比不等于总生成墙钟占比。

本次 trace 诊断事件为 Activity Buffer Request 3 次、Buffer Flush 4 次，未匹配到 Command Buffer Full；这不独立证明没有丢事件。捕获仅覆盖中间两个 CFG 步。

### 20 步产物

- latent：`[16,21,60,104]`，FP32，全部 finite，最小 -2.779804229736328、最大 3.0707859992980957。
- 视频：完整解码 81 帧、480×832、16 fps，没有全零帧。第 0/40/80 帧 RGB 均值约 91.48/90.56/92.01，方差约 2061.63/2055.85/2018.45。
- 已查看三帧：人物和举起的手可辨认、非黑屏；没有逐帧动作评价，不构成动作自然或质量保持结论，也不由两步和二十步差异判定 Sage 的相对质量。

本机 SHA256（用户提供的服务器三项哈希已于本轮比对一致）：

| 文件 | SHA256 |
|---|---|
| trace.json | `16f236727fc69053aa8ece18eed434d2fe35e8cda179fb9a075f389ca61b10a0` |
| 20261026-222711-900_42_latent.safetensors | `72981a9b7e10090cffd1c208cc6e4dc14c7678457503db5841ee0e49995012d5` |
| 20261026-222722-933_42_.mp4 | `c52a344e4a33c6bc3fd5eaaa4db5c85ddadfffb1c129d1c318af1a7e4365a6dd` |

## Day 2 最后一项：副本环境单样本复现（按 MP4 判据已验收）

使用 `gen_eval.py --config configs/eval_baseline.yaml --prompt-id P01 --seed 42 --gpu-index 0 --tag envcheck`，仍由副本 Python 执行。配置为 torch attention，20 步；`--tag envcheck` 自动组成 `baseline-envcheck` 命名空间，不应写成 `--tag baseline-envcheck`。

本机 dry-run 曾确认仅选 P01_s42 一条、20 步、Dataset baseline-envcheck。用户随后提供服务器运行输出，显示20步生成及保存结束、`P01_s42: ok`，事件日志为 `events-0-of-1-a35d671ac4ed4259817d9fbd6ba983ff.jsonl`。运行收据及 dataset/events 已传回，本轮本机核验通过，审计见 `week4/envcheck_audit.json`。

该脚本的路径与 bench.py 不同：

- 视频及 latent：`/home/user/projects/wan21-video/outputs/eval/baseline-envcheck/`。
- 原始收据：`/home/user/projects/wan21-video/notes/eval/baseline-envcheck/attempts/P01_s42/attempt-01/result.json`（首次尝试）。
- 独立命名空间，不覆盖原 baseline。已有成功收据会校验并跳过，不应为重跑删除旧收据或修改文件。

本轮重新计算本机原基线 `outputs/eval/baseline/P01_s42.mp4` 的 SHA256，与原 baseline.jsonl 一致，为 `fa59fbd653bef48711826054cbfc3463da9003b10c429c2273d9cf77eb838217`。
用户提供的服务器原 baseline 与 baseline-envcheck 两项 MP4 哈希均为上述 `fa59...8217`。本机本轮重新计算历史 baseline MP4 哈希，并与原 baseline.jsonl 核对，同样一致；本机未重新执行服务器生成，也未读取新生成的视频。
此证据支持服务器上该固定样本 MP4 复现一致，不证明整个环境或全部36段输出等价。按预定计划沿用36段历史基线作为配对侧，保留单样本环境检查范围的说明；性能仍需副本环境中的当期基线。

本机收据核查：

- dataset、attempt-01、历史 baseline 清单的 identity/fingerprint 完全一致，四项脚本/配置/提示词哈希与本轮本机文件一致。
- status=ok、attempt=1、P01/s42、20步，effective_args.attn_mode=torch；与历史收据相比生效参数仅 save_path 不同，负面提示词完全相同。
- runtime 的 torch/CUDA/GPU/物理索引/Python/commit/分词器字段与旧收据一致，events 仅 running→ok，无第二次尝试。
- 收据视频哈希与用户提供的两项服务器视频哈希、本机旧视频哈希及旧清单一致。日志中的 SageAttention import 仅说明依赖可导入，不能据此判断本次模型使用了 Sage 后端。
- 新旧 latent 文件哈希不同：旧 `45b483abe74e4cc29943b15c40931aa11ff265d652abf964132636303f0def16`，新 `7d4a03a4110388c685d0567a9f8b2d3e0b41b2f3e7f45a3e3d71c8a0a0da95a9`。本次新 latent 未传回，没有比较张量内容，不能推断张量一致或不同，也不编造差异原因。当前预定检查对象为 MP4，不新增 latent 一致作为门槛；保留此差异和范围。

本次只选择一条诊断样本，不运行 `--summarize` 来要求36段覆盖；该汇总函数遍历完整冻结集，会将其余35条记为 missing，这不是本次单样本检查的失败条件。

Day 2 四项已验收。下一步为 Day 3 A1：副本环境、GPU 0、torch baseline、total + sample-smi，预热1、正式3，GPU 1 不并行生成。该测量尚未执行，不能复用本次复现耗时或历史 Week2 均值作为当期性能基线。

## 后续验收

| 项目 | 状态 |
|---|---|
| GPU 0 自注意力 varlen finite / 误差 / 耗时 | 服务器执行完成，原始文件本机验收通过 |
| GPU 0 交叉注意力 varlen finite / 误差 / 耗时 | 服务器执行完成，原始文件本机验收通过 |
| 服务器候选配置文件哈希 | 已同步，四份 eval YAML 与本机哈希一致，Day 1 第一项已勾选 |
| 真实激活（可选） | 未采集 |
| Wan 两步完整链路冒烟 | 内容检查及两项服务器/本机产物哈希比对通过，已验收 |
| latent finite / 视频非黑屏 | 已在本机验证；抽帧严重模糊，不构成质量通过 |
| 模型 trace | 内核、捕获范围和三项服务器/本机哈希全部核对通过，已验收 |
| 环境 P01/s42 复现 | 收据配置、identity、runtime及MP4哈希核对通过，按单样本MP4判据已验收；latent文件哈希差异另记 |
| 正式速度和质量 | 未验证 |

本记录不构成正式端到端速度收益、完整评测集质量保持或其他配置兼容结论。
