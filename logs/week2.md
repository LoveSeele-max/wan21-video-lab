# Week 2 执行记录

## 2026-10-15：Day 1

### 执行范围

- 先做服务器测量脚本、基线与瓶颈定位。
- 本机 RTX 5060 Ti 16G 推理继续暂缓，不作为当前完成门槛。
- 尚未启动 Week 2 测量，没有新的性能结果。

### 服务器状态确认

用户通过 VS Code Remote SSH 执行并提供输出：

- 当前目录：`/home/user/projects/wan21-video/musubi-tuner`。
- HEAD：`f8a1b03794a49239a3539015075f5123d6c07d66`，与 Week 1 一致。
- 用户提供的输出中没有 `git status --short` 条目。
- 检查时间：服务器显示 `2026-10-15 13:50:54`。
- GPU 0：816 MiB / 49140 MiB，利用率 0%。
- GPU 1：20 MiB / 49140 MiB，利用率 0%。
- 列出的进程为桌面相关进程；本次检查未发现生成计算任务。
- 这是运行前的状态快照，不是生成峰值显存。

### 外部包装的静态核对

依据 Week 1 保存的 `notes/source/wan_generate_video.py`，以下函数在目标调用位置使用模块全局名直接调用，可作为外部包装入口：

| 函数 | 目标调用行 |
|---|---|
| `prepare_t2v_inputs` | 1555、1569 |
| `load_text_encoder` | 868 |
| `load_dit_models` | 1572 |
| `run_sampling` | 1586 |
| `save_latent` | 1764 |
| `decode_latent` | 1768 |
| `load_vae` | 1633 |
| `save_video` | 1770 |

这是已有源码副本的静态检查；包装是否实际触发，留给 Day 2 的调用计数冒烟测试验证。

### 脚本说明与语法检查

- 用户已在服务器创建 `scripts/bench.py` 并保存测量口径文档字符串。
- 激活项目环境后，`python --version` 输出 `Python 3.11.17`。
- `python -m py_compile /home/user/projects/wan21-video/scripts/bench.py` 无报错返回。
- Day 1 服务器准备完成；这仅验证了文件语法，尚未验证生成模块导入或测量逻辑。

### 待继续

- Day 2：先验证从服务器 checkout 导入生成模块，再逐步实现测量功能。
- 实现时遵守说明中的边界：端到端包含视频保存；冷启动覆盖导入与初始化；独立运行的两种峰值不要求严格相等。

## 2026-10-15：Day 2，逐步实现与联网冒烟

- 用户提供的导入测试确认生成模块来自服务器项目 checkout 的 `musubi-tuner/src/musubi_tuner/wan_generate_video.py`。
- 用户创建 `configs/baseline_bf16.yaml` 并成功读取：480×832（高×宽）、81 帧、20 步、seed 42。
- 上游参数解析及尺寸检查通过：torch 后端，FP8 False、blocks_to_swap 0、compile False。
- `bench.py` 增加最小 `run_once`，调用 generate、清理、save_output，使用前后 CUDA 同步与 perf_counter 计时。
- 本次仅在内存中覆盖为 2 步，输出到独立的 `outputs/week2/smoke`；正式 YAML 保留 20 步。

### 联网冒烟结果

- 原始粘贴日志归档：`logs/week2_smoke_online.log`。
- 2/2 采样完成，显示约 10 秒；不能把进度条时间当作整次生成时间。
- DiT 权重匹配成功，VAE 解码完成，latent 和 MP4 均记录保存成功。
- latent：`/home/user/projects/wan21-video/outputs/week2/smoke/20261015-142842-587_42_latent.safetensors`。
- 视频：`/home/user/projects/wan21-video/outputs/week2/smoke/20261015-142852-525_42_.mp4`。
- 生成和保存计时：38.82 秒；不含进程启动、模块导入及 get_generation_settings 等前置准备。
- 日志顶部 Steps 20 来自配置打印，实际执行前覆盖为 2，采样进度条证实 2/2。
- 日志存在 hf-mirror.com 向 huggingface.co 的跳转、tokenizer 相关请求与若干 404；这些没有阻止本次成功。
- SageAttention 导入失败未阻止 torch 后端运行。

这不是 20 步正式基线，也没有阶段峰值、调用计数或分段之和证据；不能标为测量脚本完整验收或计划 R0 的完整 20 步预热已完成。下一步验证 2 步离线冒烟。

### 首次离线冒烟失败

- 原始日志：`logs/week2_smoke_offline_failed.log`。
- 命令设置 `HF_HUB_OFFLINE=1`，沿用项目内 HF_HOME。
- 在 T5 初始化中的 `HuggingfaceTokenizer` / `AutoTokenizer.from_pretrained` 失败，尚未进入 DiT 去噪。
- 异常包含 `LocalEntryNotFoundError`，配置回退加载最终报 OSError。
- 尚不能确定是 tokenizer 文件缺失、缓存路径或修订引用不一致，不能直接认定三个大型模型权重损坏或缺失。
- 下一步只查询实际缓存路径及 tokenizer 相关文件的命中情况，不重复运行 GPU 生成。

### 分词器缓存与本地加载验证

- 用户查询确认有效缓存目录为项目内 `.hf-home/hub`。
- `tokenizer_config.json`、`special_tokens_map.json`、`spiece.model`、`tokenizer.json` 均返回快照内路径，revision 为 `a064a6c71f5be440641209c07bf2a5ce7a2ff5e4`。
- `config.json` 返回特殊对象；按缓存 API 语义，可能为已缓存的不存在标记，本次没有直接进行标记身份比较。
- 用户使用同一快照的分词器目录执行 `AutoTokenizer.from_pretrained(..., local_files_only=True)` 成功，类型为 T5Tokenizer。
- 证据支持：该本地目录可以加载；尚不支持断言具体是哪一处 Transformers 离线路径逻辑导致原错误。
- 下一步在 bench 内外部包装 T5EncoderModel 的构造调用，明确传入此本地 tokenizer_path；保持上游文件及权重不变，再验证离线完整生成。

### 指定本地路径后的第二次离线失败

- 用户执行外部包装后，仍在 T5 分词器初始化失败，没有进入去噪。
- 本次堆栈不同：Transformers 的配置加载路径执行 `os.path.join(subfolder, pretrained_model_name_or_path)` 时收到 None，抛出 TypeError。
- 仅覆盖 tokenizer_path 的修复不完整；下一步在 bench 的外部包装内删除值为 None 的 subfolder 参数，使其使用默认值。
- 修复增加实际 HuggingfaceTokenizer 包装器的 CPU 预检查，先验证分词器构造，再进入大模型生成。
- 原始离线查找失败是否也由此参数触发尚未直接验证，不将两种异常等同。

### 外部包装修正后，离线完整生成通过

- 用户提供的运行日志显示 `Tokenizer precheck passed.`。
- 使用本地分词器快照，并在外部包装中删除值为 None 的 subfolder 参数后，离线完成 T5、DiT 2/2 采样、VAE 解码及保存；日志没有显示 HTTP 请求。
- latent：`/home/user/projects/wan21-video/outputs/week2/smoke/20261015-144117-656_42_latent.safetensors`。
- MP4：`/home/user/projects/wan21-video/outputs/week2/smoke/20261015-144127-573_42_.mp4`。
- 生成与保存耗时：33.16 秒，采样进度约 10 秒。
- 与联网冒烟的 38.82 秒条件不同且均为单次运行，不据此计算或声称优化收益。
- 当前仍为 2 步冒烟，尚未完成阶段插桩、调用计数验收、总峰值测量或正式基线重复。
- 后续基线与优化测量应共用这套分词器路径及兼容处理，避免改变测量条件。

### 首次分段冒烟通过（2 步）

用户提供完整输出：8 个模块函数均调用 1 次；分段之和 33.278 秒，整体显示 33.28 秒，差距按两位小数显示 0.00%。该一致性验证了当前记录区间的时间覆盖，不证明阶段语义或所有显存口径均已完整验收。

| 区间 | 秒 | allocated 峰值 MiB |
|---|---:|---:|
| T5 加载 | 10.646 | 10836.5 |
| 文本准备（加载后部分，含编码、释放与噪声准备） | 0.733 | 11092.6 |
| DiT 加载 | 0.495 | 2731.6 |
| 去噪（2 步） | 10.025 | 4183.5 |
| 保存 latent | 0.004 | 9.1 |
| VAE 加载 | 0.067 | 251.4 |
| VAE 解码（加载后部分） | 9.850 | 14698.4 |
| 保存视频 | 1.088 | 9.1 |

- 本次已打印区间中，VAE 解码 allocated 峰值最高，约 14.35 GiB；不是 nvidia-smi 整卡占用，也不是 16G 本机可运行证据。
- 文本准备和解码函数的嵌套加载区间已拆出，同名区间是互斥片段，汇总时相加，不能重复包含加载时间。
- 输出 MP4：`outputs/week2/smoke/20261015-151735-354_42_.mp4`。
- 尚未完成 DiT forward 次数验证、逐前向计时、结构化 JSON 归档、正式 20 步重复及独立总峰值测量。

### 逐前向与 JSON 冒烟通过，改为集中整理脚本

- 用户随后提供的服务器运行输出确认：2 步、4 次 DiT 前向、8 个阶段函数各 1 次。
- 分段之和 33.528 秒，整体 33.53 秒，显示计时差距 0.00%。
- 原版结果保存到服务器 `notes/week2/smoke_stages.json`；尚未传回该 JSON，本机依据用户贴出的终端输出记录。
- 用户指出反复零碎改脚本影响教学节奏，决定一次整理剩余功能，后续用命令切换。
- 用户已传回 `scripts/bench.py`，本机将原件备份为 `scripts/bench_smoke_archive.py`。

### 整理版准备完成，待服务器验证

- `scripts/bench.py` 集中补齐 total / stages / cold / profile 模式、命令行步数覆盖、预热和重复。
- 默认沿用服务器原配置，离线、GPU 0、20 步、预热 1 次和正式 3 次；cold/profile 固定单次。
- 所有模式使用已验证的本地分词器兼容处理；包装均按上下文恢复。
- 每次命令创建独立结果目录，记录有效参数、版本/哈希、失败、每轮结果和排除预热后的汇总。
- 视频/latent 位于 `outputs/week2/<session>/<run>`，JSON 与采样数据位于 `notes/week2/<session>`。
- 可选 nvidia-smi 100ms 采样；profile 按前向 wait16/warmup2/active4 导出 trace 与算子表。
- 另建本机配置副本 `configs/baseline_bf16.yaml`，内容与此前指导用户在服务器创建的配置一致；本次不要求覆盖服务器原配置。
- 使用说明：`notes/WEEK2_BENCH_GUIDE.md`；Week 2 计划顶部已写入本机暂缓及计时边界覆盖说明。
- 本机检查：帮助命令、语法检查通过；`scripts/test_bench.py` 的 10 项 CPU 模拟测试通过。
- 模拟测试包括主入口预热 1 + 正式 3、每轮恢复 seed、区间覆盖、前向次数、异常保存/恢复、分词器兼容/恢复、profiler 调度/导出、冷启动子进程参数、采样解析与汇总。
- 这些不是 CUDA、CUPTI 或真实模型性能测试。整理版尚未在服务器实际运行，不能标为正式基线完成。
- 下一步：只传回 bench.py，执行 `--mode stages --steps 2 --warmup 1 --repeat 1 --sample-smi --tag verify`，一次验证同进程两轮及保存/采样。

### 整理版首次服务器验证：缓存预检查失败及修正

- session：`verify-stages-20261015-154633-ec02eac3`。
- 在 install_tokenizer 的缓存检查处报错，未加载 T5/DiT/VAE，未生成视频。
- 原整理版使用 `os.environ.setdefault("HF_HOME", ...)`，且查询未显式指定 cache_dir；终端已有 HF_HOME/HF_HUB_CACHE 时可能偏离此前成功使用的项目缓存。用户尚未提供实际环境变量，故此解释仍待修正版实机验证。
- 修正为仅在脚本进程内明确设置项目 HF_HOME、HF_HUB_CACHE，并在缓存查询中显式传 cache_dir，不改 shell 配置。
- 扩充已有测试：模拟外部错误缓存环境，验证主入口覆盖路径及缓存查询显式参数。仍需重新上传脚本并实机验证。

### 整理版服务器分段验证通过

- 原始终端日志已归档：`logs/week2_verify_stages.log`。
- session：`verify-stages-20261015-154836-9c9bf1ac`。
- 实际参数：stages、2 步、预热 1 次、正式 1 次、启用外部显存采样。
- 分词器预检查成功，同一进程顺序生成两次，两次都保存 latent 和 MP4。
- 预热：33.410 秒，allocated 峰值 14698.4 MiB。
- 正式验证轮：31.876 秒，allocated 峰值 14698.4 MiB。
- 两轮 result.json 和 session summary.json 均提示已保存，流程正常结束。根据此版本脚本的成功路径，两轮阶段/前向/时间覆盖检查均通过；尚未直接读取 JSON 逐项核对。
- 没有正式 20 步结果；不能将以上 2 步数字当作基线。
- 终端未打印采样有效样本数；尚不能确认 nvidia-smi 外部采样的数据质量，待传回结果文件检查。
- 下一步开始 R1：cold 模式、20 步、单次生成，关闭分段和逐前向包装。

### R1：20 步 cold 模式运行完成，待读取原始 JSON

- 用户提供完整终端输出，session：`R1-cold-20261015-155339-81958c93`。
- mode cold、实际 20 步、预热 0、执行 1；分词器预检查通过，采样 20/20 完成，视频/latent/result/summary 均提示保存成功。
- 终端 `run-01` 显示 132.020 秒，这是 generate 到视频保存的 elapsed_seconds，不是包含导入和启动的冷启动数值。
- allocated 峰值显示 14698.4 MiB；reserved 峰值和外部采样尚待读取 JSON。
- 冷启动字段为 `launch_to_video_saved_seconds`，本轮终端未打印，不能用文件名时间差或 132.020 秒代替。
- 视频：`outputs/week2/R1-cold-20261015-155339-81958c93/run-01/20261015-155551-920_42_.mp4`。
- 去噪进度约 108 秒（5.42 s/it），仅作日志粗略值，不冒充同步分阶段测量。
- 接下来传回 verify 与 R1 的结果目录，核对冷启动及采样有效性；不要求重复生成。

### verify 与 R1 原始文件传回并核对

- 用户已将两个 session 目录传回本机 `notes/week2/`。
- R1：launch_to_video_saved_seconds 136.69256233796477，elapsed_seconds 132.01988840987906。
- R1 allocated 15412340736 bytes（14698.4 MiB），reserved 19572719616 bytes（18666 MiB）。
- R1 整卡采样峰值 19897 MiB；CSV 实读 1319 条，和 JSON 的数量与峰值一致，没有采样 warning。
- verify 预热/测量 CSV 分别 335/318 条，峰值均 19897 MiB；正式验证轮的调用计数和时间覆盖检查直接读取确认通过。
- R1 session 确认固定环境、配置和本地分词器 revision；记录的脚本 SHA256 与当前本机脚本一致。
- 基线表已写入 `notes/week2_baseline.md`。当前只有一条正式 R1，不将它称作稳态结果或最终瓶颈结论。
- 下一步 R2：20 步，预热 1 次、分段正式测量 3 次。

### R2：20 步分段测量运行完成

- 原始终端日志：`logs/week2_R2.log`。
- session：`R2-stages-20261015-160317-d581c4df`。
- 预热 131.828 秒；正式三轮分别 136.075、136.828、135.596 秒，均打印 allocated 峰值 14698.4 MiB。
- 按终端舍入值计算，正式均值约 136.166 秒，范围 135.596–136.828 秒。正式汇总以传回原始 JSON 为准。
- 四轮均完成 20/20 采样与保存，末尾 summary 保存成功；依据脚本成功路径，阶段验收没有触发失败。
- 当前不能仅凭终端输出填写精确阶段表、reserved 或外部采样质量；JSON/CSV 后续与 R3 一起传回核对。
- 下一步 R3：同卡新进程，重复 R2，检查跨进程计时稳定性。

### R3：新进程分段复测运行完成

- 原始终端日志：`logs/week2_R3.log`。
- session：`R3-stages-20261015-163353-6833f29b`。
- 预热 130.004 秒，正式三轮 136.048、135.275、135.944 秒，allocated 峰值均显示 14698.4 MiB。
- 按终端舍入值，正式均值约 135.756 秒，范围 135.275–136.048 秒；与 R2 均值差约 0.30%，没有触发计划中超过 20% 的调查线。
- 全部完成生成和保存，summary 提示保存成功。阶段计数、reserved、外部采样仍需原始 JSON/CSV 核对。
- R2/R3 同配置两次测量的微小差异不算性能优化收益。
- 下一步 R4：关闭分段与逐前向包装，预热 1 次、正式 3 次，测端到端与独立总峰值。

### R4：total 模式运行完成

- 原始终端日志：`logs/week2_R4.log`。
- session：`R4-total-20261015-164943-11cd808d`。
- 20 步，预热 131.241 秒，正式三轮 135.025、135.251、136.928 秒。
- 按终端舍入值，正式均值约 135.735 秒，范围 135.025–136.928 秒；allocated 峰值均显示 14698.4 MiB。
- 四次生成及保存完成，summary 提示保存成功。reserved、采样与精确汇总仍待原始文件核对。
- R2 均值比 R4 高约 0.431 秒（约 0.32%）；只是独立组间净差估计，运行噪声与时间顺序可能混杂，不能将全部差值归因于插桩。
- 下一步 R5 profiler 单次运行，之后统一传回 R2–R5。

### R5：profiler 运行完成，待核对导出文件

- 用户提供终端输出，session：`R5-profile-20261015-221034-3813a433`。
- profile 模式，20 步，预热 0、执行 1；采样、latent 保存、VAE 解码和视频保存均完成。
- result.json 与 summary.json 提示保存成功；按当前脚本成功路径，profiler 的 trace_ready 检查通过。实际导出内容、设备时间是否有效仍需读取文件确认。
- 生成保存耗时 142.502 秒，allocated 峰值显示 14698.4 MiB；profiler 耗时不作为正式端到端基线或加速对照。
- 计划捕获 wait16/warmup2/active4 次前向（20 步内 2 个 CFG 步），须结合原始结果/trace 核对，不仅凭日志宣称算子占比有效。
- R1–R5 运行均已结束，下一步统一传回 R2–R5 JSON、CSV、算子表与 trace，分析后再确认 Week 2 出口。

### 全部数据传回、核对与 Week 2 收口

- R2–R5 已传回 `notes/week2/`，与先前 verify/R1 一起解析核对。
- session/result 状态、正式次数、各阶段调用计数、40 次前向、时间覆盖、分段峰值及阶段时间汇总检查通过。
- CSV 实读数量和采样峰值与 JSON 一致；全部 session 使用相同 bench 哈希。
- 完成 `scripts/analyze_week2.py`，可在无 torch 的本机重复解析；生成 `notes/week2_audit.json` 和两份报告。
- `notes/week2_baseline.md`：R4 主基线 135.734687 秒，范围 135.024600–136.928295；R2 去噪 113.434795 秒，占 83.31%；T5 loader 10.065527 秒、VAE 解码 10.331807 秒。
- R2/R3 分段最大值与 R4 独立总峰值恰好一致：allocated 14698.4 MiB、reserved 18666 MiB；R1–R4 正式整卡采样峰值均 19897 MiB。allocated 峰值来源为 VAE 解码，不能据此认定其内部具体分配。
- R2 减 R4 均值 0.431514 秒，约 0.318%；独立组间差值，不作插桩纯开销的强归因。
- R5 trace 有有效 CUDA 内核与 4 个逻辑前向的 CPU/GPU 标记；原表混有 ATen/内核/annotation/overhead，不能直接相加。
- 发现 996 条无捕获 CPU 算子关联的 kernel，均在首个 GPU ProfilerStep 前，含 24 条 flash。按 External id 过滤后，16040 条关联 kernel 时长约 10.582381 秒，attention 240 次，时长占比 53.90%，GEMM/GEMV 名称匹配类 19.89%，其他 26.21%。
- R5 存在 4283 次 Command Buffer Full，采集干扰明确；报告保留局限，不把 overhead 当成模型瓶颈或从墙钟直接扣除。未另行证明事件是否有丢失，不称无损采集。
- `notes/week2_profile.md` 给出限定范围的瓶颈结论：先把 attention 实现、DiT 精度/矩阵乘路径列为两组候选；VAE 内部显存定位留作后续本人代码改动候选，不预设改法或收益。
- Week 2 调整后的服务器出口完成，计划和执行清单已更新。没有开始优化对照、质量评测或本机推理；后续先做 Week 3 评测准备，等用户继续。

### 按用户要求逐项勾选每日计划

- 更新 `plan/WEEK2_PLAN.md` 的 Day 1–5、Day 7 服务器相关复选框，逐组附证据和文件入口，不仅勾选末尾出口标准。
- Day 1 本机 Python 3.11.17 检查沿用 Week 1 已有输出，注明本周未重做；本机安装、权重、余量检查和 Day 6 实机验证继续暂缓。
- 如实记录 R0 实际为 2 步联网准备及 2 步离线冒烟，20 步完整离线链路由 R1 等后续运行验证。
- 区分用户的服务器执行/文件传输与助手的脚本实现、审计分析和文档整理。

### 用户恢复本机实验

- 用户明确解除暂缓，恢复本机 16G 执行范围。
- 助手只读检查确认项目 Python 3.11.17、pip 26.2.1，尚无 torch；E 盘约剩 169.7 GiB。
- 本机 GPU 为 RTX 5060 Ti 16311 MiB，检查时占用 1234 MiB，驱动 591.86。
- 记录入口 `notes/week2_local_16g.md` 已建立，计划顶部已写最新范围决定；历史暂缓记录保留，不视为继续暂停。
- 当前仅前置检查完成，下一步安装项目 PyTorch 后验证 CUDA；没有启动本机生成。

### 本机 PyTorch 安装与 GPU 检查通过

- 用户安装 torch 2.9.1+cu128、torchvision 0.24.1+cu128 成功。
- 助手直接调用项目 python.exe 确认版本、CUDA build 12.8、GPU available True、RTX 5060 Ti、capability (12, 0)。
- 小型 CUDA 整数张量求和返回 45，同步完成；pip check 无破损依赖。
- 计划内本机 PyTorch/GPU 前置项已勾选，完整代码与模型推理仍未完成。下一步获取固定版本本机代码。

### 本机仓库固定完成

- local/musubi-tuner 完整克隆已确认，工作区干净，detached HEAD 固定到服务器同一 commit f8a1b03794a49239a3539015075f5123d6c07d66。
- 计划内代码版本前置项已勾选；新增 local/constraints.txt 固定已安装的 torch/torchvision 和服务器 bitsandbytes 版本，等待安装其余运行依赖。

### 2026-10-16 本机运行依赖检查通过

- 用户完成 editable 安装；助手复核项目环境 pip check 无冲突，生成模块 --help 退出码为 0。
- 未启动模型推理；下一步准备本机模型权重和分词器。

### 2026-10-17 本机 L0 完整生成成功

- 三个权重已传回且 SHA256 全部匹配，分词器离线检查通过。
- 本机独立脚本/配置完成适配；L0 stages 实测成功，耗时 627.740 秒，去噪 547.180 秒，解码 68.977 秒。
- allocated 峰值 14699.2 MiB，整卡采样峰值 15929 MiB；reserved 18666 MiB 单独记录，不解释为物理显存占用。
- 首次生成已勾选；内容检查与 stages / total 重复组尚未完成。

### 2026-10-18 本机 L1 分段重复组完成

- 已读取 L1-stages-20261018-144206-60e8ffc4 原始记录；预热及正式三次均成功，阶段检查全部通过。
- 正式平均 645.834 秒，范围 637.306–655.344 秒；allocated 峰值均为 14699.2 MiB，整卡采样峰值最大 15896 MiB。
- 计划勾选 L1，保留 L2 total 与最终汇总待完成。

### 2026-10-18 本机 L2 与 Week 2 汇总完成

- 读取 L2-total-20261018-154702-30f6e37f 的 session/summary/四轮原始结果；四次状态 ok，视频和 latent 存在。
- 正式三次均值 637.611 秒，范围 635.399–641.259 秒；allocated 14699.2 MiB、reserved 18666 MiB，正式轮整卡采样峰值最大 15759 MiB。
- 已完成 L1/L2 对照、本机独立最终表、计划出口与 REPRO 更新。L1-L2 均值差 8.224 秒不视为纯插桩开销。
- 运行前进程清单未留存，明确作为证据缺口，不补勾也不因补历史记录重跑。Week 2 完成结论仅覆盖记录中的配置、少量重复和测量边界。
