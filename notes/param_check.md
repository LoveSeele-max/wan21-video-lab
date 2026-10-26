# 固定版本参数核对记录

更新：2026-10-14。固定 commit：
`f8a1b03794a49239a3539015075f5123d6c07d66`。

状态：已完成 12 项的静态源码/文件清单核对。2026-10-14 用户再次提供 HEAD 为固定 commit，传回 `git ls-files` 清单。依据用户从固定 checkout 传回的副本，未另逐文件与 Git 对象校验；未启用的优化后端没有运行验证。不能将静态核对写成性能实测。本轮阅读结束。

## 已归档的帮助文本

2026-10-13 已将服务器的 help_generate.txt（原本地引用：`help_generate.txt`；该文件未随公开快照发布或路径已调整） 传回本机并读取。以下仅是 CLI 帮助中的声明，不能替代源码行为验证。

| 参数 | 帮助文本确认的信息 | 本次使用 |
|---|---|---|
| `--task` | 支持 `t2v-1.3B` | `t2v-1.3B` |
| `--video_size` | 顺序为 height、width | `480 832` |
| `--video_length` | 帧数，默认依任务而定 | 显式指定 81 |
| `--fps` | 默认 16 | 显式指定 16 |
| `--infer_steps` | 步数，默认依任务而定 | 显式指定 20 |
| `--seed` | 生成种子 | 显式指定 42 |
| `--negative_prompt` | 未指定时使用默认负面提示词 | 未显式指定；共享配置第 20 行原文已归档到配置与任务配置笔记 |
| `--sample_solver` | 可选 `unipc`、`dpm++`、`vanilla` | 未显式指定；源码第 74 行确认默认 unipc |
| `--device` | 未指定时优先 CUDA，不可用则 CPU | 未显式指定；日志为 cuda |
| `--attn_mode` | 包含 `torch` 等后端 | 显式指定 torch |
| `--vae_dtype` | 声明默认 bfloat16 | 未显式指定；日志确认 bf16 |
| `--cpu_noise` | 声明默认 False | 未启用 |
| `--cfg_skip_mode` | 声明默认 none，不跳过 | 未显式指定；实现待核对 |
| `--cfg_apply_ratio` | 声明默认 None，所有步应用 CFG | 未显式指定；实现待核对 |
| `--vae_cache_cpu` | VAE 特征缓存放在 CPU | 未启用；实际作用范围待源码核对 |
| `--fp8_fast` | 帮助声明仅用于 fp8_scaled | 未启用；条件分支与校验待源码核对 |

帮助把 `--save_path` 描述为保存路径，没有明确写目录；目录行为由本次执行结果支持，之后仍需保存源码位置。帮助没有证明 attention 别名、block swap 上限或 VAE 内部实现。

## 已有执行证据

- `wan_generate_video.py --help` 成功，列出 `t2v-1.3B`、`--device`、`--seed`、`--attn_mode` 等选项。
- 实际使用 `--attn_mode torch`，两次生成均正常退出并产出 MP4。
- 实际 `--video_size 480 832` 产出宽 832、高 480 的视频。
- 实际 `--save_path` 使用目录，产出自动命名文件。
- `--output_type both` 已执行；MP4 已检查，日志报告两份 latent 保存路径；2026-10-14 用户 `ls -lh` 确认两份存在，均为 8.0M，未加载或比较 latent 内容。
- 完整日志确认两次的 `flow_shift=5.0`、`guidance_scale=5.0`、DiT/权重/VAE 精度为 bf16；日志显示 T5 转换为 bf16。

会话中曾查阅固定版本生成源码以确认高宽顺序和保存目录，但尚未保存带行号的源码片段。需要补齐引用，不能当作已完成全部源码审查。

## 12 项待核对

| # | 断言 | 当前证据状态 | 下一步 |
|---|---|---|---|
| 1 | `--device` 存在 | 帮助及本机源码副本第 210 行确认；第 2228 行为自动设备选择 | 指定 `cuda:N` 的行为未单独测试；源码副本未另与 Git 对象校验 |
| 2 | `sdpa` 与 `torch` 同分支 | 2026-10-14：源码副本 `wan_attention.py` 第 92 行同分支，第 101 行调用 PyTorch SDPA | 第 4 项已追到实际调用；未分别运行别名对照 |
| 3 | `flash` 与 `flash2` 同分支 | 2026-10-14：源码副本 `wan_attention.py` 第 117 行同分支，第 125 行调用 `flash_attn.flash_attn_func` | 未安装或执行 flash 后端，不声称实测可用 |
| 4 | DiT attention 调用及 `attn_mode` 传递 | 已静态核对：入口第 652 行 → `wan_model.py` 第 1033、661、383、385、192 行；第 21 行导入 Wan 的 `flash_attention`，第 237、276 行实际调用 | self/cross attention 使用 Wan 分发；通用 attention 文件不是本次 DiT 的直接分发入口 |
| 5 | `fp8_fast` 只在 scaled 分支生效且无校验 | 结论需精确化：入口第 659 行无条件传 `use_scaled_mm`；模型第 1059 行仅在 fp8_scaled 下安装补丁；工具第 378、409 行才选 scaled_mm | 没有显式依赖检查；单独开启不激活该优化路径。未启用 LoRA/lycoris 的普通路径已核对，未运行 FP8 对照 |
| 6 | inactive DiT/lazy loading 只在多 DiT 生效 | 已静态核对：入口第 574、575、581、582 行依赖 high model；第 1417 行切换依赖多个模型 | 单 DiT 无收益；两个开关仍有全局互斥 assert（第 2220 行） |
| 7 | 1.3B 层数与 block swap 上限 | 配置 30 层；模型第 750、753 行用实际 block 数检查 `<= num_blocks - 1`，即最多 29 | 纠正旧文档“代码不校验上限”；未运行 block swap，assert 是静态证据 |
| 8 | VAE 逐 latent 帧解码、缓存、无空间分块、输出累积 | 已静态核对：`wan_vae.py` 第 570、577、578、581、584、586、587 行 | conv2 先处理整个 latent；decoder 逐 latent 时间片且带缓存，输出逐次 cat；已读编码器/解码器路径未见空间 tile 入口 |
| 9 | `vae_cache_cpu` 影响的缓存位置 | 入口第 504、505 行 → VAE 第 713、714 行 → 第 518 至 522、419 至 425 行传播 → 第 428、433 行缓存转移 | 输出卸载分支第 588 至 600 行被注释，不能把缓存卸载当作输出卸载或全阶段 CPU 解码 |
| 10 | 生成入口没有阶段计时 | 已读入口流程，未见阶段计时输出或 perf_counter/time.time/CUDA Event；存在同步、卸载等待 sleep 和文件名时间戳 | tqdm 只有采样进度粗略计时，不是阶段 benchmark；同步本身不是计时 |
| 11 | `compile_transformer` 所在位置 | `utils/model_utils.py` 对应副本第 309 行定义；入口第 728 行调用 | 第 332 至 341 行逐 block torch.compile；未实际启用或测收益 |
| 12 | README 阅读入口路径都存在 | 已核对：传回的 `key_files.txt` 包含全部入口文件及 Wan modules 目录；用户再次确认 HEAD 为固定 commit | README 中部分缩写路径已改为完整仓库相对路径；不据文件存在推导内部行为 |

以上行号引用 `notes/source/` 的归档副本，原始路径见下一节。逐文件 Git 对象校验未额外执行，保留该证据边界，不继续扩展本轮阅读。

### 文件清单核对结果

2026-10-14 对传回的 key_files.txt（原本地引用：`key_files.txt`；该文件未随公开快照发布或路径已调整） 做精确路径匹配，以下文件全部存在：

```text
src/musubi_tuner/wan_generate_video.py
src/musubi_tuner/wan_train_network.py
src/musubi_tuner/wan/modules/attention.py
src/musubi_tuner/modules/attention.py
src/musubi_tuner/modules/custom_offloading_utils.py
src/musubi_tuner/modules/fp8_optimization_utils.py
src/musubi_tuner/training/timesteps.py
src/musubi_tuner/modules/scheduling_flow_match_discrete.py
src/musubi_tuner/training/trainer_base.py
src/musubi_tuner/networks/lora_wan.py
src/musubi_tuner/utils/model_utils.py
```

清单包含 `src/musubi_tuner/wan/modules/` 下的文件。用户本次提供的 HEAD 为 `f8a1b03794a49239a3539015075f5123d6c07d66`，未提供单独的状态文本，故不额外声称本次远程工作树由助手独立检查为干净。

## 第一轮生成入口阅读

2026-10-13 已读取服务器传回的 `source/wan_generate_video.py`，建立 [调用链](call_graph.md)。确认 solver 默认 unipc（第 74 行）、高宽顺序（第 114、448、449 行）、输出目录行为（第 1667、1668、1708、1709 行）及 `offload_inactive_dit` 与 `lazy_loading` 的 assert 互斥（第 2220 行）。负面提示词回退到配置文件（第 855 行），具体文本仍待读取配置。未将内部模块相关条目标记完成。

## 任务配置阅读

2026-10-14 已读取 `wan_t2v_1_3B.py` 与 `shared_config.py`。确认默认步数 50、shift 5.0、guidance 5.0、DiT 30 层及默认负面提示词原文。此次命令显式指定 20 步，覆盖默认 50。详见 [任务配置笔记](task_config.md)。block swap 上限仍未核实，不因层数确认而整项勾选完成。

## Attention 分发阅读

2026-10-14 已读取 `source/wan_attention.py`。`flash_attention()` 内第 92 行将 torch/sdpa 放在同一分支，第 117 行将 flash/flash2 放在同一分支，因此在该分发函数中两对名字分别是别名。尚未追踪 DiT 实际调用，不能因为看到文件中有函数就认定运行一定经过它。

## 2026-10-14 批量源码核对

已读取三组归档文件，当前没有新增的大范围阅读要求：

| 副本 | 上游相对路径 |
|---|---|
| `wan_model.py` | `src/musubi_tuner/wan/modules/model.py` |
| `wan_attention.py` | `src/musubi_tuner/wan/modules/attention.py` |
| `common_attention.py` | `src/musubi_tuner/modules/attention.py` |
| `wan_vae.py` | `src/musubi_tuner/wan/modules/vae.py` |
| `custom_offloading_utils.py` | `src/musubi_tuner/modules/custom_offloading_utils.py` |
| `model_utils.py` | `src/musubi_tuner/utils/model_utils.py` |
| `fp8_optimization_utils.py` | `src/musubi_tuner/modules/fp8_optimization_utils.py` |

关键片段：

```python
# wan_model.py:752
assert (
    self.blocks_to_swap <= self.num_blocks - 1
)

# wan_model.py:1059
if fp8_scaled:
    apply_fp8_monkey_patch(model, sd, use_scaled_mm=use_scaled_mm)

# wan_vae.py:581
for i in range(iter_):
    # decoder receives x[:, :, i : i + 1, :, :]
    # later iterations accumulate out using torch.cat([out, out_], 2)
    ...
```

块交换通用工具中的 `create_offloader`（第 161 行）接收架构侧参数；推理设置 `supports_backward=False`，选择普通 ModelOffloader。实际 block 调度见模型第 895、901 行和工具第 538 至 605 行。上限校验在模型内部，不应只搜索生成脚本后声称不存在校验。

读源码不证明量化、卸载或编译一定加速。VAE 内部 `conv2(z)` 仍先处理整个 latent，不能把逐时间片 decoder 夸大为所有计算完全流式。没有继续扩大到训练实现阅读。

## 原定阅读范围（已执行）

将剩余源码核对收拢为三组，仅检查第一周 12 项及普通 T2V 路径，不要求逐行学习仓库：

1. 模型与 attention 调用：`wan/modules/model.py`、`modules/attention.py`，追踪实际后端、block swap、fp8_fast 参数传递。
2. 显存与 VAE：`modules/custom_offloading_utils.py`、`wan/modules/vae.py`，确认卸载限制和缓存/解码行为；必要时沿直接调用补读。
3. 工具与文件位置：`utils/model_utils.py`、`modules/fp8_optimization_utils.py`，核对编译定义、FP8 实现与入口条件；用固定 commit 文件清单核对文档路径。

部分项目可直接用已取得的生成入口完成，如双 DiT 开关条件和阶段计时检查。文件数量是当前预计，不把内部依赖已全部核实作为既成事实。后续源码可批量归档，教学每次只讲一个结论。
