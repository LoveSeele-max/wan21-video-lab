# 第 2 周计划：测量脚本、基线、瓶颈定位（含本机 16G 实机）

**最新执行决定（2026-10-15）**：用户明确恢复本机实验。下文历史“暂缓”状态自此解除，本机相关未完成项重新进入当前执行范围；服务器已完成的 R1–R5 不重复。本机先检查专用环境、安装匹配版本 PyTorch 并验证 GPU，再准备代码/权重和运行，结果单独记入 [week2_local_16g.md](../notes/REVIEW/week2_local_16g.md)。旧勾选反映当时证据，不把恢复执行等同于已完成。

**当前执行覆盖（2026-10-18 更新）**：服务器 R1–R5、本机 L0/L1/L2 及最终汇总均完成。运行前独立进程清单未留存，作为证据缺口记录，不补勾该项、不为此重复长实验。服务器测量使用 `scripts/bench.py`，本机使用独立副本 `local/scripts/bench.py`。命令和准确计时边界见 [WEEK2_BENCH_GUIDE.md](../notes/WEEK2_BENCH_GUIDE.md)。端到端统一截止视频保存完成；“扣加载”仅扣三个 loader 函数区间，不代表常驻模型计算时间；独立运行的两种峰值只交叉核对，不要求严格相等。本文以下与之冲突的原文字句以本条为准。

**完成记录（2026-10-15）**：服务器 R1–R5 已运行、原始结果已归档并核对；调整后 Week 2 出口完成。见 [基线与阶段表](../notes/week2_baseline.md)、[profiler 与瓶颈结论](../notes/week2_profile.md)、[执行日志](../logs/week2.md)。profiler 有缓冲开销与捕获边界前异步事件，已按关联内核重新分类，局限见报告；不要求为追求无瑕疵 trace 继续重跑。本机项保持未完成/暂缓。

**日期**：2026-10-14 编写；同日按审阅修订（V2）。配套 [EXECUTION_CHECKLIST.md](../doc/EXECUTION_CHECKLIST.md) 第 2 周、[EXPERIMENT_PLAN.md](../doc/EXPERIMENT_PLAN.md) 阶段 1、[METRICS.md](../doc/METRICS.md) 第一节。
**前置**：第 1 周已完成（见 [WEEK1_PLAN.md](WEEK1_PLAN.md)、[REPRO.md](../REPRO.md)）。

**V2 修订要点**（依据 `notes/source/wan_generate_video.py`，固定 commit `f8a1b03`）：

1. 峰值统计改为**平铺、互不嵌套的分段**：`load_text_encoder` 在 `prepare_t2v_inputs` 内部调用（第 868 行），`load_vae` 在 `decode_latent` 内部调用（第 1633 行）。如果嵌套的函数各自重置峰值，外层的峰值会被内层清掉。
2. 新增 **VAE 加载**阶段：T2V 路径下 `args._vae` 为 None，每次 `decode_latent` 都会重新加载 VAE。
3. 正式测量设 **`HF_HUB_OFFLINE=1`**：tokenizer 路径为 None（第 521 行），每次都可能联网拉取 `google/umt5-xxl` 的配置，第 1 周的日志可以印证。
4. CFG 已核实为**两次独立前向**（条件分支第 1456 行，无条件分支第 1488 行），Day 1 不再查这一项。
5. 跨进程重复改为**同一张卡**；计时运行期间**不让两张卡同时跑**。
6. 阶段计时与分阶段峰值合并为同一次运行；profiler 在 **20 步配置内**只抓 2 步。
7. 本机 16G：准备工作挪到 Day 1–3 后台并行；加优化的顺序改为**按 OOM 所在阶段决定**，并补上 `--vae_cache_cpu`。

---

**本周要拿到的东西**：

1. `scripts/bench.py`：能输出延迟、阶段耗时、峰值显存的测量脚本；
2. 服务器**基线数字表**（bf16、torch 后端、未优化）；
3. **阶段耗时与峰值分解表** + profiler 算子占比表 + 一段**瓶颈结论**；
4. **本机 RTX 5060 Ti 16G 的实机基线**：能否跑通；跑不通时卡在哪个阶段；跑通时的耗时、峰值与剩余余量。

**本周不做**：

- 不改上游代码（所有插桩都通过外部包装完成，见 D1）；
- 不做任何优化实验（attention 后端、fp8、步数扫描都留到第 4 周以后）；本机为了"能跑起来"所加的参数除外，见 Day 6；
- 不写通用 benchmark 框架；
- 不做质量判据（探针与盲评在第 3 周及以后）。

---

## 本周决策

| 编号 | 决策 | 采用 | 理由 |
|---|---|---|---|
| D1 | 插桩方式 | **外部包装**：`bench.py` 导入 `wan_generate_video` 模块，替换模块级函数与 `WanModel.forward`，不改上游源码 | 基线与后续优化共用同一套插桩；将来提 PR 时再把插桩整理成可选开关 |
| D2 | 基线步数 | 沿用第 1 周的 **20 步** | 能和第 1 周的视频直接对应 |
| D3 | 本机平台 | **Windows 原生**（`local/env`） | 环境已建好；装不上再考虑 WSL，并记录原因 |
| D4 | 本机加参数的顺序 | **按 OOM 所在阶段决定**，每次只加一项（见 Day 6） | 固定顺序会出错：比如 OOM 出在 T5 阶段时，`--fp8` 只作用于 DiT，加了也没用 |
| D5 | 网络 | 先做一次预热运行填满 HF 缓存，正式测量设 `HF_HUB_OFFLINE=1` | 去掉每次可能出现的 tokenizer 网络请求；如果离线就报错，说明依赖网络，这一点本身也要记录 |

---

## 一、计时口径（写进 `bench.py` 注释，与 METRICS 1.1 一致）

| 名称 | 起点 → 终点 | 怎么得到 |
|---|---|---|
| 冷启动 | 进程启动 → 视频写盘 | 新进程首次运行，`bench.py` 整体计时 |
| 稳态端到端 | 预热后，一次"生成 + 保存"的开始 → 视频写盘 | 同一进程内，预热 1 次后再计 3 次 |
| 阶段耗时 | 见下方分段表 | 在边界事件处 `torch.cuda.synchronize()` + `perf_counter()` |

**两个"冷/热"必须说清**：

- **冷启动 ≠ 磁盘冷**：服务器的系统文件缓存在预热运行后已经变热，清掉它需要 sudo（`drop_caches`），本周不做。所以本周的"冷启动"指**进程冷**。
- **稳态 ≠ 模型常驻**：每次 `generate` 都会重新加载 T5、DiT、VAE（加载后释放）。因此稳态端到端必须报两列：**含加载**、**不含加载**（扣掉 T5 加载、DiT 加载、VAE 加载三段）。

**`bench.py` 要复刻的调用顺序**（对应 `main` 的单提示词分支，第 2300–2323 行）：

```text
parse_args → device → setup_args → check_inputs → get_generation_settings
  → generate(args, gen_settings)            # 每次稳态运行重复这一段及之后的部分
  → gc.collect + clean_memory_on_device
  → save_output(latent, args, cfg, h, w)    # 内含 save_latent → decode_latent(→ load_vae) → save_video
```

每次重复前恢复 `args.seed = 42`。`generate` 会改写 `args.seed`，也会设置 `args._vae`。

---

## 二、分段方式：平铺，不嵌套

**做法**：包装下列函数的**进入和退出**。每发生一次进入或退出，就是一个"边界事件"。在每个边界事件处：同步 → 记录时间 → 读取上一段的峰值 → `reset_peak_memory_stats()` → 开始下一段。这样各段首尾相接、互不重叠，嵌套调用也不会让内层清掉外层的峰值。

| 包装的函数 | 源码位置 |
|---|---|
| `prepare_t2v_inputs` | 定义第 825 行，`generate` 内调用 |
| `load_text_encoder` | 定义第 509 行，在 `prepare_t2v_inputs` 内第 868 行调用 |
| `load_dit_models` | 定义第 561 行，`generate` 内第 1572 行调用 |
| `run_sampling` | 定义第 1305 行，`generate` 内第 1586 行调用 |
| `save_latent` | 定义第 1655 行 |
| `decode_latent` | 定义第 1615 行 |
| `load_vae` | 定义第 489 行，在 `decode_latent` 内第 1633 行调用 |
| `save_video` | 定义第 1697 行 |

由此得到的段（按发生顺序）：

| 段 | 内容 | 备注 |
|---|---|---|
| S0 | `generate` 开头 → 进入 `prepare_t2v_inputs` | Accelerator 初始化等，应很小 |
| S1 | 进入 `prepare_t2v_inputs` → 进入 `load_text_encoder` | 尺寸计算、种子 |
| **S2 T5 加载** | `load_text_encoder` 进入 → 退出 | |
| **S3 文本编码** | 退出 `load_text_encoder` → 退出 `prepare_t2v_inputs` | 含第 869 行 `text_encoder.model.to(device)`、两次编码、释放 T5、生成噪声。T5 搬上 GPU 落在这一段，报告时注明 |
| **S4 DiT 加载** | `load_dit_models` 进入 → 退出 | |
| S5 | 退出 `load_dit_models` → 进入 `run_sampling` | `setup_scheduler` 等 |
| **S6 去噪采样** | `run_sampling` 进入 → 退出 | 20 步 = **40 次 DiT 前向**（CFG 两次独立前向） |
| S7 | 退出 `run_sampling` → 进入 `save_latent` | 释放 DiT、`gc`、清理显存 |
| S8 保存 latent | `save_latent` 进入 → 退出 | |
| S9 | 进入 `decode_latent` → 进入 `load_vae` | 应很小 |
| **S10 VAE 加载** | `load_vae` 进入 → 退出 | |
| **S11 VAE 解码** | 退出 `load_vae` → 退出 `decode_latent` | 含 `vae.decode` 与结果搬回 CPU |
| **S12 保存视频** | `save_video` 进入 → 退出 | MP4 编码 |

粗体是报告主表的行；其余段合并为"其他"，但原始数据保留。各段相加应约等于稳态端到端时间。**这个等式就是插桩正确性的自检。**

**去噪单步**：包装 `WanModel.forward`（类级别），每次前向前后同步并计时。报告两种单位：**每次前向**（40 次的均值 ± 范围）和**每步 = 2 次前向**。每次前向都同步会带来额外开销，所以这些数字只用于定位，不进入最终对照。

**包装是否生效，靠冒烟测试证明**：打印每个包装函数的调用次数。20 步时 `WanModel.forward` 应为 40 次，其余函数各 1 次。

---

## 三、两种峰值口径（METRICS 1.1）

| 口径 | 做法 | 回答的问题 |
|---|---|---|
| A. 分段峰值 | 第二节的平铺分段，每段读 `max_memory_allocated` 与 `max_memory_reserved` | 哪一段最占显存 |
| B. 总峰值 | **另跑一次**，全程不重置、不分段同步，结束后读总峰值 | 这张卡会不会 OOM |

口径 A 的段如果真正首尾相接，它们的最大值理论上应等于口径 B。两者对得上，说明分段没有漏掉区间；对不上，就要查原因。

外部采样：`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop-ms=100`，日志里注明采样间隔。它的口径和分配器不同，两个数都报。

---

## 四、运行矩阵（服务器）

**统一配置**（与第 1 周一致，本周冻结）：

```yaml
# configs/baseline_bf16.yaml
commit: f8a1b03794a49239a3539015075f5123d6c07d66
task: t2v-1.3B
video_size: [480, 832]        # 高, 宽
video_length: 81
fps: 16
infer_steps: 20
seed: 42
prompt: "A person waving one hand, plain background, static camera."
negative_prompt: null         # null = 使用共享配置的默认负面提示词，原文见 notes/task_config.md
attn_mode: torch
output_type: both
# 不加 fp8 / blocks_to_swap / compile
env:
  HF_HUB_OFFLINE: "1"         # R0 之后的所有运行
  CUDA_VISIBLE_DEVICES: "0"
```

| 运行 | 卡 | 进程 | 内容 | 目的 |
|---|---|---|---|---|
| R0 | GPU 0 | 新进程 | 联网跑 1 次，不计入结果；随后用 `HF_HUB_OFFLINE=1` 冒烟一次 | 填满缓存，确认离线可用 |
| R1 | GPU 0 | 新进程 | 冷启动 × 1，只计端到端 | 冷启动口径（进程冷） |
| R2 | GPU 0 | 同一进程 | 预热 1 + 稳态 3；**分段计时 + 口径 A 峰值 + 每次前向计时** | 阶段分解、分段峰值、单步 |
| R3 | GPU 0 | 新进程 | 与 R2 相同 | 跨进程稳定性（同卡，只变进程） |
| R4 | GPU 0 | 同一进程 | 预热 1 + 稳态 3；**不插桩**，只计端到端 + 口径 B 总峰值 | 干净的端到端数字与总峰值；与 R2 对比，可估出插桩开销 |
| R5 | GPU 0 | 新进程 | 20 步配置，`torch.profiler` 只抓 2 步 | 算子占比 |

**R5 的抓取方式**：用 `torch.profiler.schedule(wait=16, warmup=2, active=4, repeat=1)`，在每次 `WanModel.forward` 结束时调用 `prof.step()`。以前向为单位计数：跳过前 8 步（16 次前向），预热 1 步，正式抓 2 步（4 次前向）。导出按 CUDA 时间排序的算子表，以及一份 Chrome trace。

**运行纪律**：

- 每次运行前看一眼 `nvidia-smi`，确认卡空闲；被占用就作废重跑。
- **计时运行期间 GPU 1 不跑其他任务**：两个进程同时从同一块 NVMe 读十几 G 权重，加载时间会互相拖慢。GPU 1 可以在计时间隙做与计时无关的工作。
- 长任务放在 tmux 里：`tmux new -s wan21-w2`。

---

## 五、每日计划

**勾选说明（2026-10-18 更新）**：以下按实际证据更新，不代表按自然日花了七天。你负责执行生成命令和提供结果；我负责脚本、核对与文档。服务器任务已完成，本机任务已恢复，未完成项继续保留。

### Day 1：口径注释 + 本机准备（后台）

服务器：

- [x] `git rev-parse HEAD` 确认仍在固定 commit；`nvidia-smi` 确认卡空闲
- [x] 确认第二节表中 8 个函数在 `generate` / `save_output` / `decode_latent` 内部都是**按模块全局名调用**的，这样替换模块属性后包装才会生效。已知 `generate` 内直接调用 `prepare_t2v_inputs`、`load_dit_models`、`run_sampling`
- [x] 先在 `bench.py` 文件头写好口径注释（第一至三节），再写代码
- [x] 把 CFG 两次前向的结论同步到 `notes/call_graph.md`（已完成，见该文件）

证据：用户提供的 HEAD/GPU 输出及 [执行日志](../logs/week2.md)；源码核对及实际阶段调用计数均通过。

本机（后台进行，不占服务器时间）：

- [x] 直接用 `E:\cstudy\AIGC\V\local\env\python.exe` 调用，确认版本为 3.11.17（Week 1 已有用户输出，本周未重做；不表示本机推理通过）
- [x] 恢复后完成：安装 PyTorch cu128（`torch==2.9.1 torchvision==0.24.1`）；本机实测 CUDA 可用、capability `(12, 0)`，小型 CUDA 运算及 pip check 通过
- [x] 获取同一 commit 的本机 musubi-tuner 完整 checkout；已确认 local/musubi-tuner 工作区干净，并固定到 f8a1b03794a49239a3539015075f5123d6c07d66（依赖安装另行完成）

### Day 2：bench.py 与冒烟

- [x] 实现 `scripts/bench.py`，只做这些事：
  - 第二节的边界事件机制（平铺分段：计时 + 口径 A）；
  - `WanModel.forward` 计时（可开关）；
  - 口径 B 模式（不插桩，只计端到端 + 总峰值）；
  - profiler 模式（第四节 R5 的 schedule）；
  - 冷启动 / 预热 / 稳态的次数控制，每次重复前恢复 `args.seed`；
  - 每次运行输出 JSON：有效配置、模式、各段耗时、各段峰值、调用计数；配置名在 session 内，轮次由独立目录名标识。
- [x] R0 的联网准备与离线冒烟：实际先联网 2 步，再离线 2 步；完整 20 步离线链路由 R1 及后续正式运行补充验证，没有另跑一条 20 步联网预热
- [x] 检查调用计数：`WanModel.forward` 为 4 次（2 步 × 2），其余函数各 1 次
- [x] 检查各段耗时之和 ≈ 端到端（差距 < 2%）

证据：`verify-stages-20261015-154836-9c9bf1ac` 的原始 JSON/CSV，以及 [测量脚本说明](../notes/WEEK2_BENCH_GUIDE.md)。额外完成离线分词器兼容处理、10 项 CPU 模拟测试和服务器实机验收；模拟测试没有代替 GPU 测量。

本机（后台）：

- [x] 为本机推理准备三份权重并与服务器逐一核对哈希（2026-10-17 全部匹配）
- [x] 本机 E 盘余量已检查：准备阶段约 169.7 GiB；2026-10-18 安装与传输后约 148.3 GiB 可用

**出口**：冒烟通过，有调用计数和分段之和两项自检作为证据；本机准备完成或已知卡点。

### Day 3：服务器基线 R1–R4

- [x] R1：冷启动 × 1
- [x] R2：分段计时 + 口径 A + 每次前向计时，稳态 × 3
- [x] R3：新进程，重复 R2
- [x] R4：关闭分段和逐前向包装，端到端 + 口径 B，稳态 × 3
- [x] 从 R2 与 R4 的端到端之差估算插桩影响，并记录独立组间噪声的限制

证据：[基线表](../notes/week2_baseline.md)及 `notes/week2/R1-*` 至 `R4-*` 原始目录。R1 冷启动 136.693 秒；R4 预热后生成保存均值 135.735 秒；R2 与 R4 净差约 0.432 秒，不直接等同于插桩纯开销。

**出口**：四组原始 JSON。R1 与第 1 周的 157.61 秒量级接近即可（第 1 周含联网和 `/usr/bin/time`，不要求一致）。

### Day 4：profiler 与峰值核对

- [x] R5：profiler，导出算子表与 trace
- [x] 口径 A 各段最大值与口径 B 总峰值对照（本次一致，不作为不同运行必须相等的规则）
- [x] `nvidia-smi` 外部采样与分配器数字并列记录

证据：[profiler 分析](../notes/week2_profile.md)、[基线显存表](../notes/week2_baseline.md)。R5 捕获 4 次逻辑前向；原表存在不同层级重复统计与采集 overhead，已利用 trace 的 External id 关联过滤生成分类表。allocated 峰值约 14.35 GiB，reserved 约 18.23 GiB，整卡采样约 19.43 GiB，三者不能相加。

**出口**：分段峰值表 + 总峰值 + 算子占比表。

### Day 5：阶段分解与瓶颈结论

- [x] 阶段耗时表：R2 三次的均值与最小/最大范围，含加载 / 扣三个 loader 区间分别列出；R3 作为跨进程对照列
- [x] 去噪：每次前向、相邻两次前向之和各自的均值与范围；另报去噪总时长除以 20，说明前向对不含调度器等单步开销
- [x] 阶段占比与 profiler 占比对照：先由阶段表判断"哪段最耗时"，profiler 只解释去噪段**内部**的分布
- [x] **瓶颈结论**（一段话）：
  - 时间主要花在哪一段（数字 + 占比）；加载类的段单独说明；
  - 峰值来自哪一段（口径 A），总峰值是多少（口径 B）；
  - 证据是什么，**不能证明什么**（如：只有 3 次重复；profiler 只抓了 2 步；尚未测任何优化收益）
- [x] 据此排出第 4–5 周两组对照的候选顺序（只排序，不实施）

证据：[阶段表](../notes/week2_baseline.md)、[瓶颈与候选排序](../notes/week2_profile.md)。去噪占 R2 总耗时 83.31%，VAE 解码产生 allocated 峰值。后续先评估 attention 实现，再评估 DiT 精度/矩阵乘路径；尚未测优化收益。VAE 内部显存分配原因留待后续定位。

**出口**：阶段表 + 峰值表 + profiler 表 + 瓶颈结论 + 候选排序。

### Day 6：本机 RTX 5060 Ti 16G 实机基线

**状态：本机基线实验与汇总完成。** L0/L1/L2 均成功，L2 正式端到端均值 637.611 秒；详见本机实验记录。视频质量问题与运行前进程清单缺口已明确保留。

**Week 3 Day 1 补充（2026-10-18）**：用户设置项目 Python“偏好无系统内存回退”并确认应用后，L3 单次 stages 完整成功，626.700 秒，allocated 14699.2 MiB、reserved 18666 MiB，未触发 L4。该结果不支持“改为偏好无回退就会在 VAE OOM”的预测，也不证明全程独立显存驻留或 Linux 16G 可运行。L2 仍为正式性能基线，原始数据保留。

**前置检查**（Day 1–2 的后台准备结果，全部通过才进入运行）：

- [x] torch 识别 GPU，capability 为 (12, 0)；额外执行小型 CUDA 运算通过
- [x] 本机代码在 commit `f8a1b03`
- [x] 三个权重的哈希与服务器一致（2026-10-17，本机 SHA256 实测全部匹配）
- [ ] 证据缺口：未留存 L1/L2 运行前独立 `nvidia-smi` 进程清单，不能严格确认无其他 GPU 任务；保留外部采样，不补勾或重跑历史实验

**L0：bf16，不加任何参数**，用 `bench.py` 的分段模式跑一次冷启动。即使 OOM，也能从最后一个完成的段看出死在哪一阶段。

- [x] L0 首次完整生成成功（2026-10-17）：627.740 秒，allocated 峰值 14699.2 MiB，整卡采样峰值 15929 MiB；40 次 forward 校验通过。此处为首次 stages 运行，不是 cold 模式启动计时。
- [x] 本机 stages / total 各完成预热 1 次、测量 3 次，并汇总本组重复范围与峰值；L0 视频已复核，动作与提示词符合度不足已记录。
- [x] L1 stages：预热 1 次 + 正式 3 次全部成功，平均 645.834 秒，范围 637.306–655.344 秒；调用计数及分段和检查通过。
- [x] L2 total：预热 1 次 + 正式 3 次全部成功，均值 637.611 秒，范围 635.399–641.259 秒；已与 L1 对照并完成最终本机汇总。

- **跑通了**：再按服务器的 R2、R4 各跑一组稳态 × 3。分别记录 allocated、reserved 和整卡采样峰值；reserved 不直接代表物理显存占用，不能用物理容量减 reserved 推算余量。用整卡容量减整卡采样峰值仅报告“采样时剩余容量”，注明可能漏掉短峰，不能保证安全余量。本机部分到此结束。
  这时结论是"**不加任何显存优化即可在 16G 上运行**，峰值 X、余量 Y"，同样是一条有价值的结果。
- **OOM 了**：按出事的阶段，每次只加一项，每一级单独存一个配置文件 `configs/local_16g_LN.yaml`：

| OOM 所在段 | 先加 | 仍 OOM 再加 |
|---|---|---|
| S2/S3（T5 加载 / 文本编码） | `--fp8_t5` | 记录为卡点（T5 是独立阶段，DiT 侧的参数无效） |
| S4/S6（DiT 加载 / 去噪采样） | `--fp8` | `--blocks_to_swap`（从 10 起，按需增加，上限 29） |
| S10/S11（VAE 加载 / 解码） | `--vae_cache_cpu` | 记录为卡点。**这正好是第 6–7 周 VAE 代码改动的实机动机** |

每加一级都要重新跑，看 OOM 是否移到了下一个阶段。

**记录与措辞**：

- 只跑通一个配置：写"找到可运行配置"，不写"最小代价"；
- 比较过多个可运行配置后，才可以写"已测配置中代价最小"；
- Windows 上不用 `/usr/bin/time`，数字一律由 `bench.py` 产出；
- WDDM 驱动下 `nvidia-smi` 的显存读数口径与 Linux 不同，以分配器数字为主，外部数字作参考；
- 本机输出**不要求**与服务器逐帧一致（操作系统、驱动、数值路径都不同），不做像素比对；
- 服务器数字**不外推**到本机，本机单独成表。

**出口**：本机表至少一行（每行：配置、结果状态、OOM 阶段或耗时/峰值/余量）。

### Day 7：整理与缓冲

- [x] 写 `logs/week2.md`：执行步骤、数字、踩的坑、修复和文档变更已记录
- [x] 更新 `REPRO.md`：补齐服务器测量信息及本机环境、权重、分词器、L0 命令和已验证边界
- [x] `scripts/bench.py`、`configs/baseline_bf16.yaml` 归档，在 `REPRO.md` 中注明运行命令；旧冒烟脚本保留备份
- [x] 最终对照出口标准：服务器及本机实验与汇总完成；profiler 局限、运行前进程清单缺口和画面质量问题已记录

证据：[执行日志](../logs/week2.md)、[复现文档](../REPRO.md)、[数据审计](../notes/week2_audit.json)。`scripts/analyze_week2.py` 可以重算审计和两份报告，不修改原始实验数据，不启动 GPU。

---

## 本周出口标准（周末自查）

- [x] `scripts/bench.py` 存在，口径注释与 METRICS 1.1 一致；两项自检通过（调用计数正确、分段之和 ≈ 端到端）
- [x] 服务器 R1–R5 的原始 JSON 已归档
- [x] 阶段耗时表（含加载 / 扣 loader 区间两列；跨进程对照）
- [x] 峰值：口径 A 分段表 + 口径 B 总峰值，两者已对照
- [x] 插桩开销估计（R2 与 R4 的端到端之差；未作纯因果归因）
- [x] profiler 算子/内核表，注明"20 步配置内只抓 2 步"，保留缓冲开销与边界过滤说明
- [x] 瓶颈结论一段，写明证据与不能证明的内容，并给后续两组候选排序
- [x] 本机 16G 独立结果表、L0/L1/L2 原始记录、重复组与最终汇总完成；正式端到端采用 L2，结论边界已注明
- [x] `logs/week2.md` 与 `REPRO.md` 已更新

**止损**：

- 平铺分段的包装超过 2 天仍不能稳定触发 → 退回三段粗分（DiT 之前 / `run_sampling` / `save_output`），先拿到数字；
- 本机环境超过 2 天仍无法让 torch 识别 GPU → 记录卡点，本机结果标为"未完成"，不挤占第 3 周；
- R2 与 R3 的稳态差异超过 20% → 先查原因（卡占用、系统负载、缓存），不下结论。

---

## 本周产出文件一览

```
E:\cstudy\AIGC\V\  (本机)
  scripts/bench.py                 # 服务器与本机共用
  configs/baseline_bf16.yaml
  configs/local_16g_L*.yaml        # 本机每一级一个文件
  logs/week2.md
  notes/call_graph.md              # 已补 CFG 两次前向的结论
  notes/week2_baseline.md          # 服务器基线 + 阶段表 + 峰值表
  notes/week2_profile.md           # profiler 占比与瓶颈结论
  notes/week2_local_16g.md         # 本机实机记录
  REPRO.md                         # 补充本机环境

服务器 /home/user/projects/wan21-video/
  outputs/week2/                   # 测量期间产生的视频（不进 git）
  logs/week2_*.log
  notes/week2_*.json               # bench 原始输出
  notes/week2_trace.json           # profiler trace
```

---

## 写进报告的边界

- **稳态不是常驻**：每次生成都会重新加载三个模型，稳态数字包含加载，必须同时报两列。
- **冷启动只是进程冷**：系统文件缓存没有清。
- **插桩有开销**：分段与前向计时只用于定位；干净的端到端数字以 R4 为准。
- **profiler 只抓了 2 步**，不作为延迟依据。
- **3 次不是统计显著性**：只能说明"这次的范围大约是这些"。
- **本机与服务器不可直接比较**：硬件、驱动、操作系统都不同。
- **16G 的结果是"能不能跑"的事实，不是"最优"**：本周不调参。
