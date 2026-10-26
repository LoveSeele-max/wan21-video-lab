# Week 3 评测生成脚本

2026-10-19。实现：[gen_eval.py](../scripts/gen_eval.py)，CPU 测试：[test_gen_eval.py](../scripts/test_gen_eval.py)。

收尾状态（2026-10-22）：服务器冒烟、路径核对、正式生成及本机 36 段 MP4 哈希验收均已完成。下方分阶段记录和首次运行命令用于复习，不表示需要重新执行；路径核对仍以用户提供的服务器比较结果为证据。

## 当前验收范围

- [x] 14 项 CPU 模拟测试通过：两分片互斥且覆盖 36 个组合、哈希校验后跳过、残缺文件不跳过、损坏重试、两次尝试上限、OOM/普通失败/中断留痕、锁冲突、配置或环境变化拒绝混用、冒烟隔离、缺失覆盖率、单次调用顺序及参数副本。
- [x] 用本机项目解释器实际导入固定版本上游模块，经 `bench.build_args` 和 `wan.check_inputs` 解析评测基线；得到 `(480,832,81)`、`cpu_noise=False`、torch attention、output_type=both。只验证参数，不加载权重或生成。
- [x] dry-run 列出 GPU 0 分片的两个冒烟任务：P01/s42、P01/s2002。
- [x] 2026-10-19 用户提供 SCP 输出，四个脚本/配置文件传输均为 100%；后续 dry-run 已完成，见下一项。
- [x] 2026-10-19 服务器 dry-run 实际通过，输出 P01/s42、P01/s2002，2 步，baseline-smoke2。15:28 的进程清单未见模型计算任务；实际推理在后续冒烟与正式组验收。
- [x] 2026-10-19 服务器实际冒烟成功：P01/s42、P01/s2002 均为 ok，终端显示 latent/MP4 保存和 VAE 解码完成。事件日志为 `notes/eval/baseline-smoke2/events-0-of-2-fd0ee6d64d39492bbad624d3a2db353a.jsonl`。目前依据完整终端输出，未独立读取服务器 JSONL 或观看两步视频，不作画质结论。
- [x] 2026-10-19 用户报告 P01/s42 的 20 步 pathcheck MP4 与 Week 2 R4 run-01 的完整 SHA256 一致；本次同设置样例复现检查通过。未独立在本地读取服务器视频重算哈希。
- [x] 正式双卡生成运行结束：2026-10-19 用户报告已完成；随后已汇总并核对文件，见下两项。
- [x] 正式 36 段汇总：2026-10-19 用户提供 `Coverage: {'ok': 36}`，全部组合有效成功，最终清单为 `notes/eval/baseline.jsonl`。
- [x] 传回本机并独立核查：36 段 MP4、最终清单及 coverage.json 齐全；36 条唯一记录均为 ok，全部 MP4 的 SHA256 与清单一致，GPU 0/1 各 18 条。未传 latent，本机未验证 latent 内容。

本机哈希检查验证收到的视频与清单一致；服务器实际生成过程依据用户提供的终端输出和结果记录，不声称已独立远程重跑。

## 调用路径

`scripts/bench.py` 保持不变。脚本导入它的 `build_args`、`install_tokenizer`、固定 commit 和分词器 revision，使用同一离线缓存、移除 tokenizer 的 `subfolder=None` 包装。

没有直接调用 `bench.run_once`，因为它会把输出强制放到 Week 2 目录。新脚本采用同一条核心调用顺序：

1. 为每条提示词/种子复制参数。
2. `wan.check_inputs`、`wan.get_generation_settings`。
3. 同步、reset allocator 峰值。
4. `wan.generate`。
5. `gc.collect`、`wan.clean_memory_on_device`。
6. `wan.save_output`，继续使用 output_type=both。
7. 同步并读取耗时、allocated/reserved 峰值。

每段重新加载模型，不使用 `--from_file` 或常驻模型复用。保存 MP4 后才复制成固定文件名，不改变上游编码参数。记录有效参数及实际默认负面提示词。

并行生成的时间只作诊断，不能作为性能收益。调用路径的静态复用与单样例输出核对分别记录，不将一次相同结果外推到所有设置。

## 服务器准备与干运行

需传输以下四个文件到相应项目目录：

```text
scripts/gen_eval.py
scripts/test_gen_eval.py
configs/prompts_frozen.yaml
configs/eval_baseline.yaml
```

现有 `scripts/bench.py`、环境、上游仓库、权重和缓存继续使用，不覆盖。服务器执行：

```bash
cd /home/user/projects/wan21-video
/home/user/projects/wan21-video/env/bin/python scripts/gen_eval.py \
  --dry-run --shard 0/2 --steps 2 --limit 2
```

dry-run 不导入 torch/Musubi，不写结果，不分配模型显存。正式运行会检查固定 commit、上游工作区干净、三份权重存在、CUDA 和离线分词器。

## 先冒烟，再核对，最后正式生成

先确认 GPU 没有其他计算任务，再跑两段、每段 2 步：

```bash
CUDA_VISIBLE_DEVICES=0 /home/user/projects/wan21-video/env/bin/python \
  scripts/gen_eval.py --gpu-index 0 --shard 0/2 --steps 2 --limit 2
```

输出单独放到 `outputs/eval/baseline-smoke2`，不会占用正式样本记录。冒烟只检查链路，不判断画质。

20 步路径核对用独立命名空间：

```bash
CUDA_VISIBLE_DEVICES=0 /home/user/projects/wan21-video/env/bin/python \
  scripts/gen_eval.py --gpu-index 0 --prompt-id P01 --seed 42 --tag pathcheck
```

对 `outputs/eval/baseline-pathcheck/P01_s42.mp4` 与 Week 2 R4 参考 MP4 计算完整 SHA256。相同是复现证据；若不同，先检查有效参数、latent 张量和解码帧，不能只凭 MP4 哈希不同就判定调用路径不同。未完成核对不直接进入正式组。

正式两张卡在不同 tmux 窗口分别执行：

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 /home/user/projects/wan21-video/env/bin/python \
  scripts/gen_eval.py --gpu-index 0 --shard 0/2
```

```bash
# GPU 1
CUDA_VISIBLE_DEVICES=1 /home/user/projects/wan21-video/env/bin/python \
  scripts/gen_eval.py --gpu-index 1 --shard 1/2
```

两边都结束后再汇总：

```bash
/home/user/projects/wan21-video/env/bin/python scripts/gen_eval.py --summarize
```

## 输出和断点恢复

正式输出：

```text
outputs/eval/baseline/P01_s42.mp4
outputs/eval/baseline/P01_s42_latent.safetensors
outputs/eval/baseline/.attempts/P01_s42/attempt-01/...
notes/eval/baseline/dataset.json
notes/eval/baseline/attempts/P01_s42/attempt-01/result.json
notes/eval/baseline/events-0-of-2-<运行ID>.jsonl
notes/eval/baseline/events-1-of-2-<运行ID>.jsonl
notes/eval/baseline.jsonl
notes/eval/baseline/coverage.json
```

每次 worker 写独立事件日志，不让双卡同时追加同一个文件。事件日志含开始及结束事件；`--summarize` 从持久化结果生成每个预定组合一行的最终 JSONL，共 36 行，并报告成功、缺失、失败或输出失效等状态。评分可判断性以后由人工表记录，不在这里假定。

重新执行相同 worker 命令即可续跑。只有成功记录的配置身份和 MP4/latent 文件哈希均匹配才跳过；孤立 MP4 或半成品不算成功。失败不在一次 worker 中自动重试；下次执行最多再尝试一次，总计最多两次。两次都失败则保留错误，不换种子或提示词。

保留每次尝试的原始文件和完整错误；stable 文件是其副本，因此会多占少量磁盘。配置、脚本或 bench 哈希变化会拒绝混入同一个数据集。已有尝试的 torch/CUDA/GPU 型号/Python/commit/分词器 revision 变化也会拒绝复用；两张同型号 GPU 的物理编号可不同。

单样本使用互斥锁，避免重叠分片同时写同一样本。若进程被强杀留下锁，脚本会停下并显示路径；先核对进程已经退出，再处理该锁，不自动抢锁。

`--summarize` 在任一样本不是有效成功时返回非零退出码，这是覆盖率提示。冒烟或只跑部分样本也会有其余样本 missing，不能当成 36 段完成。
