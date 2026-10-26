# Week 2 测量脚本使用说明

整理日期：2026-10-15。当前仅做服务器；本机 16G 实测继续暂缓。

脚本位置为服务器 `/home/user/projects/wan21-video/scripts/bench.py`。它读取 `configs/baseline_bf16.yaml`；默认 20 步，GPU 0，离线模式，预热 1 次、正式测量 3 次。通过命令切换模式，不再逐段编辑。

脚本进程内明确使用项目 `models/.hf-home` 和其中的 `hub` 缓存，不沿用终端的其他 HF_HOME/HF_HUB_CACHE 值；不修改 shell 配置。分词器缓存查询还显式传入此 hub 路径。

## 先验证整理后的脚本

在本机 PowerShell 传输脚本，服务器配置文件使用你之前创建的原文件：

```powershell
scp "E:\cstudy\AIGC\V\scripts\bench.py" user@workstation02:/home/user/projects/wan21-video/scripts/bench.py
```

在已激活项目环境的服务器终端运行：

```bash
python /home/user/projects/wan21-video/scripts/bench.py \
  --mode stages --steps 2 --warmup 1 --repeat 1 \
  --sample-smi --tag verify
```

这条命令实际生成两次：一次预热、一次测量，均为 2 步。它验证同进程重复、阶段计数、前向计数、JSON 保存及外部显存采样。预热不参与汇总。实际检查结果在每次 `result.json` 的 `checks` 字段；只要检查失败，进程会非零退出，保存失败记录。

这不代表全部模式在 CUDA 上已验证；正式测量依次验证 total、cold、profile 模式。profiler 可能受服务器驱动或 CUPTI 可用性影响，失败时保留错误，不把空 trace 算作成功。

## 正式运行矩阵

下面是后续使用参考；初次验证后再逐组执行，不需要现在全部运行。每次开始前确认服务器没有其他计算任务；计时期间另一张 GPU 也不安排读取同一磁盘权重的任务。

```bash
# R1：进程冷启动，自动启动子进程，固定只生成一次。
python scripts/bench.py --mode cold --sample-smi --tag R1

# R2：阶段定位，预热 1 次、正式 3 次，含逐前向同步计时。
python scripts/bench.py --mode stages --sample-smi --tag R2

# R3：新进程、同一张卡，重复 R2。
python scripts/bench.py --mode stages --sample-smi --tag R3

# R4：关闭阶段和前向包装，预热 1 次、正式 3 次。
python scripts/bench.py --mode total --sample-smi --tag R4

# R5：20 步，仅执行一次；前向 schedule 抓其中 2 步。
python scripts/bench.py --mode profile --sample-smi --tag R5
```

这些相对路径命令在 `/home/user/projects/wan21-video` 执行。`--gpu-index 0` 是默认值，脚本在导入 torch 之前设置单卡可见；若 shell 已设置不一致的 CUDA_VISIBLE_DEVICES，会报错，避免悄悄选错卡。

`--steps 2` 仅用于冒烟，正式测量省略。脚本不会修改 YAML。`cold` 和 `profile` 自动使用 0 次预热、1 次执行。

## 保存在哪里

每次命令生成独立 session，名称含 tag、模式、时间和随机后缀，避免覆盖旧结果。

```text
notes/week2/<session>/
  session.json          版本、有效配置、文件哈希、分词器路径及完成状态
  warmup-01/
    result.json
    gpu.csv             使用 --sample-smi 时生成
  run-01/
    result.json
    gpu.csv
    trace.json          仅 profile
    operators.txt       仅 profile
    operators.json      仅 profile
  run-02/               按重复次数出现
  run-03/
  summary.json          正式成功运行的均值、最小值、最大值

outputs/week2/<session>/<warmup-or-run>/   latent 和视频
```

如果生成失败，通常仍会保存该轮 `result.json`，包含错误、已完成区间和当时阶段，并在 session 标记失败后退出，不继续以残留状态测下一轮。导入、配置解析、版本检查等早于 session 建立的错误由终端堆栈报告。

## 怎么读数字

- `elapsed_seconds`：从 generate 前到生成、清理和 latent/视频保存结束。包含每轮重新加载模型，不含导入、分词器预检查及 settings 准备。
- `launch_to_video_saved_seconds`：仅 cold 有。父进程在创建子进程前启动计时，截止子进程保存完成；覆盖子进程启动、导入、环境和配置检查。使用同主机共享的 perf_counter 时钟；不是磁盘冷启动，且不包含退出和报告写盘。
- `stage_summary`：同名非重叠区间耗时相加，显存峰值取最大值。文本准备在 T5 loader 返回后仍包含编码、模型搬移/释放、噪声准备；解码区间不重复包含 VAE loader。
- `excluding_load_intervals_seconds`：只扣除三个 loader 函数区间的算术结果，不能称为模型常驻的纯计算耗时；loader 之外仍可能存在搬移与清理。
- `forward_seconds`：逐次前向的同步墙钟计时，带额外开销；每两个相加仅表示两次前向的和，不包含 scheduler 等循环开销。
- `peak_allocated_bytes` / `peak_reserved_bytes`：PyTorch 分配器峰值，单位 byte。stages 模式取覆盖全程的区间最大值，total 模式独立统计全程。
- `gpu.csv`：100 毫秒采样整张卡，包含桌面及其他进程；采样在计时前启动，报告结束后停止，可能包括短暂的计时外区间，也可能漏掉瞬时峰值。
- `operators.json`：按 self device time 排序，不重复累加父算子的 inclusive 时间。需要结合 trace 解读；异步或重叠设备工作时，算子时间和不等于端到端墙钟。

R2/R3 用来定位，R4 用于正式端到端基线；R5 不用于计算延迟收益。R2 与 R4 只能估算插桩影响，需保留运行间噪声的限制。独立运行的分段峰值与总峰值不必严格相等。

## 当前验证范围

- 原版分段脚本已经在服务器 2 步离线运行通过，前向计数为 4。
- 整理版在本机进行了语法、帮助命令与 CPU 模拟控制流测试；未在本机安装/运行 PyTorch，未做本机 GPU 推理。
- 模拟测试验证的是包装恢复、区间覆盖、失败保存、预热排除及调度/导出控制流程，不能代替真实 CUDA 和 profiler 验证。
- 上游仍固定 commit `f8a1b03794a49239a3539015075f5123d6c07d66`；脚本检查版本及工作区状态，不修改上游文件。
- 分词器固定为缓存 revision `a064a6c71f5be440641209c07bf2a5ce7a2ff5e4`，沿用已验证的本地路径与去除 None subfolder 的兼容处理。所有模式使用相同处理。

旧版分段冒烟脚本保留在本机 `scripts/bench_smoke_archive.py`，不再用于正式测量。
