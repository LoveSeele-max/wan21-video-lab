# 复现记录

更新日期：2026-10-22（Asia/Shanghai）。

当前状态：Week 1–3 的已执行工作与复现入口已记录。服务器 R1–R5、本机固定配置 L0–L3、36 段评测基线、49 条自动指标、匿名试评、资源成本与后续预登记已完成各自验收；SageAttention 独立副本安装及 GPU 0 小张量检查通过。未执行优化配置完整模型对照、训练 LoRA 或干净环境独立复现。详见 [Week3 复盘](notes/REVIEW/WEEK3_REVIEW.md)。

第一周历史范围：环境和双卡顺序推理跑通，12 项静态源码/路径核对及调用链建立，视频与 latent 文件记录；2026-10-14 曾决定暂缓本机推理，后在 Week 2 解除并完成固定配置测量。以下历史记录保留当时语境，最新验收状态见后续分周小节，不将服务器结果外推为本机任意配置可用。

## 1. 工作目录与隔离方式

- 操作方式：本机 VS Code Remote SSH 连接服务器，命令在服务器终端执行。
- SSH 主机：`workstation02`；用户名：`user`。
- 服务器项目根目录：`/home/user/projects/wan21-video`。
- 上游代码：`/home/user/projects/wan21-video/musubi-tuner`。
- 独立 Conda 环境：`/home/user/projects/wan21-video/env`。
- 模型目录：`/home/user/projects/wan21-video/models/wan21`。
- Hugging Face 缓存：下载和生成命令指定 `HF_HOME=/home/user/projects/wan21-video/models/.hf-home`。
- 本机文档目录：`E:\cstudy\AIGC\V`。
- 本机视频目录：`E:\cstudy\AIGC\V\video`。

环境激活方式：

```bash
conda activate /home/user/projects/wan21-video/env
cd /home/user/projects/wan21-video/musubi-tuner
```

进入 `env` 文件夹不等于激活环境。新终端中需确认提示符不再是 `(base)`。

## 2. 硬件与系统

| 项目 | 已记录值 | 依据与边界 |
|---|---|---|
| GPU | NVIDIA RTX 5880 Ada，2 张 | 2026-10-12 用户提供的 `nvidia-smi` |
| 每卡显存 | 49140 MiB，约 48 GiB | 同上；两张卡分别运行，不合并显存 |
| 驱动 | 580.126.20 | 同上 |
| `nvidia-smi` 显示的 CUDA | 13.0 | 驱动报告值，不等于 PyTorch CUDA 构建 |
| 项目所在磁盘 | `/dev/nvme0n1p4`，总计 2.5T，可用 2.0T | 2026-10-12 `df -h .`；空间是当时的快照 |
| 操作系统 | Ubuntu 24.04.3 LTS (Noble Numbat) | 2026-10-13 用户提供的 `/etc/os-release` |
| 内核 | `6.17.0-35-generic` | 同日 `uname -r` |
| 系统内存 | 总计 251 GiB；当时 available 239 GiB | 同日 `free -h`；空闲状态仅为采集时快照 |
| Swap | 0 B | 同日 `free -h` |
| CPU | AMD EPYC 7C13 64-Core Processor，2 个 socket，每 socket 64 核，每核 2 线程；共 128 物理核心、256 逻辑线程 | 2026-10-14 用户提供的 `lscpu` 输出；架构 x86_64 |
| NUMA | 2 个节点；节点 0：`0-63,128-191`，节点 1：`64-127,192-255` | 同次 `lscpu`；未做 CPU 绑核或 GPU/NUMA 亲和性测试 |
| 本机 GPU | NVIDIA GeForce RTX 5060 Ti，16311 MiB，WDDM | 2026-10-14 13:26:42 本机 `nvidia-smi`；推理尚未尝试 |
| 本机驱动 | 591.86；`nvidia-smi` CUDA 显示 13.1 | 同次检查，不能当作本机 PyTorch 构建版本 |
| 本机 GPU 占用快照 | 1114 MiB，利用率 1% | 同次检查，存在桌面程序占用；不当作推理显存测量 |
| 本机 Conda | 26.3.2 | 用户提供 `conda --version` |
| 本机 Python Launcher | 列出 Python 3.13 (64-bit) | `py --list`；不是项目独立环境版本 |

本机已创建独立 Conda 环境 `E:\cstudy\AIGC\V\local\env`，用户直接执行环境内 `python.exe --version` 确认 Python 3.11.17。`conda activate` 后当前 PowerShell 的 `python` 仍指向 `D:\PYTHON313`（3.13.7，pip 25.2），激活问题尚未验证修复；未在该项目环境安装或检查 PyTorch，未运行模型。

2026-10-14 用户决定暂缓本机推理：不继续安装或排查，本机环境保留，不删除文件。Conda 缓存配置第一项为 `D:\miniforge3\pkgs`，另有两个 C 盘候选，未实测缓存空间。当前结论限定在服务器单张 RTX 5880 Ada 48G；不能将本机环境存在写成 16G 推理成功，也不能从服务器显存数字推导本机可运行。

## 3. 软件与代码版本

| 项目 | 值 |
|---|---|
| musubi-tuner commit | `f8a1b03794a49239a3539015075f5123d6c07d66` |
| commit 时间 | `2026-09-27 21:25:26 +0900` |
| commit 说明 | `Merge pull request #1142 from kohya-ss/dev` |
| 版本固定方式 | `git checkout --detach <上述完整 hash>` |
| 固定后状态 | `git status --short --branch` 显示非分支 HEAD，没有文件修改条目 |
| 项目版本 | `pyproject.toml`：`0.3.6` |
| Git | 2.43.0 |
| Conda 版本 | 25.7.0 |
| 项目 Python | 3.11.17 |
| 项目 pip | 26.2.1 |
| PyTorch | `2.9.1+cu128` |
| PyTorch CUDA 构建 | 12.8 |
| torchvision | `0.24.1+cu128`，已由归档依赖清单确认 |
| CUDA 可用 | `True` |
| PyTorch 检测 GPU 数 | 2 |
| tmux | 3.4 |
| 依赖检查 | `python -m pip check`：`No broken requirements found.` |

安装记录：

```bash
http_proxy=http://127.0.0.1:7897 \
https_proxy=http://127.0.0.1:7897 \
conda create --prefix /home/user/projects/wan21-video/env python=3.11 pip

conda activate /home/user/projects/wan21-video/env

python -m pip install --proxy http://127.0.0.1:7897 \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128

cd /home/user/projects/wan21-video/musubi-tuner
python -m pip install --proxy http://127.0.0.1:7897 -e .
python -m pip check
```

环境中的原始 Python 曾显示 3.13.5；项目最终使用独立环境中的 3.11.17。未在 Conda base 环境安装本项目依赖。

服务器已保存、尚未传回本机的文件：

- `/home/user/projects/wan21-video/notes/musubi_commit.txt`
- `/home/user/projects/wan21-video/notes/weights.sha256`

生成帮助已归档到本机 `notes/help_generate.txt` 并核对；CLI 说明见 [参数核对记录](notes/param_check.md)。生成入口源码已传回 `notes/source/wan_generate_video.py`，第 74 行确认采样器默认 `unipc`，第 1271 行为实际分支。2026-10-14 已读取任务与共享配置，确认默认负面提示词原文及任务默认 50 步；本次显式 20 步覆盖默认值，详见 [任务配置笔记](notes/task_config.md)。已建立 [入口调用链](notes/call_graph.md)，内部模块追踪未完成。

两份完整生成日志已传回本机：`logs/week1_first_run.log`、`logs/week1_gpu1_run.log`，命令及结果已与日志核对。

服务器的 `requirements.lock` 已传回项目根目录并检查，原始文件保留不改。关键版本：

| 包 | 归档清单中的版本 |
|---|---|
| torch | `2.9.1+cu128` |
| torchvision | `0.24.1+cu128` |
| accelerate | `1.15.0` |
| diffusers | `0.40.0` |
| transformers | `5.17.0` |
| huggingface_hub | `1.32.0` |
| av | `17.1.0` |
| bitsandbytes | `0.50.2` |
| safetensors | `0.8.0` |
| numpy | `2.4.6` |
| triton | `3.5.1` |

清单的 Musubi Tuner 可编辑安装条目为：

```text
-e git+https://github.com/kohya-ss/musubi-tuner.git@f8a1b03794a49239a3539015075f5123d6c07d66#egg=musubi_tuner
```

该条目与已固定的 commit 一致。这里的 `requirements.lock` 是 `pip freeze` 版本快照，不包含完整安装源、Python/Conda/驱动版本或全部包的校验值；复现仍需结合本文件的安装顺序和 PyTorch cu128 来源，尚未验证直接用它在干净环境安装。该快照为服务器环境，不作为 Windows 本机环境已经兼容的证据。

权重来源仓库 revision、Conda 环境导出、训练入口帮助尚未归档。

## 4. 网络记录

- 服务器直连 GitHub 克隆失败，报连接 443 超时。
- `ss -lntp` 显示 `verge-mihomo` 监听 `127.0.0.1:7897`。
- 通过该 HTTP 代理访问 GitHub 和 Hugging Face，均获得 `HTTP/2 200`。
- Git 使用单次命令的 `-c http.proxy=http://127.0.0.1:7897`，未要求修改全局 Git 代理。
- pip 使用单次命令的 `--proxy`；HF 下载使用命令级环境变量。
- 生成原始日志显示两次都先访问 `hf-mirror.com`，再重定向到 `huggingface.co`，用于获取或检查 T5 tokenizer 配置。因此此前“未使用镜像站”的记录不成立，现按日志修正。
- 2026-10-13 用户执行 `printenv HF_ENDPOINT`，当前服务器终端返回 `https://hf-mirror.com`，与两次生成的访问日志一致。生成命令本身没有显式设置 `HF_ENDPOINT`，因此复现时还需记录终端继承的镜像环境。
- 该环境变量最初由哪个配置文件或命令设置仍未追踪；当前查询不能单独证明历史进程环境，但历史日志已确认实际访问镜像。不修改现有镜像设置。
- 第一次有 tokenizer 文件 GET 请求，第二次请求更少，存在缓存状态差异；不能将总耗时差异全部归因于缓存或镜像。
- 其他环境复现时需要自行提供可用网络，不默认拥有该代理。

## 5. 权重

以下指纹来自服务器 `sha256sum` 输出，已记录，但本次未与来源仓库的校验值独立比对。

| 组件 | 来源仓库 | 相对于 `models/wan21` 的文件 | 终端显示大小 |
|---|---|---|---|
| DiT | `Comfy-Org/Wan_2.1_ComfyUI_repackaged` | `split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors` | 2.7G |
| VAE | `Comfy-Org/Wan_2.1_ComfyUI_repackaged` | `split_files/vae/wan_2.1_vae.safetensors` | 243M |
| T5 | `Wan-AI/Wan2.1-I2V-14B-720P` | `models_t5_umt5-xxl-enc-bf16.pth` | 11G |

```text
DiT: 6f999b0d6cb9a72b3d98ac386ed96f57f8cecae13994a69232514ea4974ad5fd
VAE: 2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b
T5:  7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d
```

固定 commit 的 `docs/wan.md`：
<https://github.com/kohya-ss/musubi-tuner/blob/f8a1b03794a49239a3539015075f5123d6c07d66/docs/wan.md>

许可核对待单独归档，不能以本记录代替发布前的许可检查。

## 6. 第一条生成命令

用户确认第一条使用物理 GPU 0，第二条使用物理 GPU 1。以下命令已与本机归档日志中的 `Command being timed` 核对。该日志仅打印 Python 命令，不打印外层环境变量，物理 GPU 归属依据用户执行记录。

```bash
conda activate /home/user/projects/wan21-video/env
cd /home/user/projects/wan21-video/musubi-tuner
mkdir -p /home/user/projects/wan21-video/outputs/week1

CUDA_VISIBLE_DEVICES=0 \
http_proxy=http://127.0.0.1:7897 \
https_proxy=http://127.0.0.1:7897 \
HF_HOME=/home/user/projects/wan21-video/models/.hf-home \
/usr/bin/time -v python src/musubi_tuner/wan_generate_video.py \
  --task t2v-1.3B \
  --video_size 480 832 \
  --video_length 81 \
  --fps 16 \
  --infer_steps 20 \
  --prompt "A person waving one hand, plain background, static camera." \
  --seed 42 \
  --dit /home/user/projects/wan21-video/models/wan21/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors \
  --vae /home/user/projects/wan21-video/models/wan21/split_files/vae/wan_2.1_vae.safetensors \
  --t5 /home/user/projects/wan21-video/models/wan21/models_t5_umt5-xxl-enc-bf16.pth \
  --attn_mode torch \
  --output_type both \
  --save_path /home/user/projects/wan21-video/outputs/week1 \
  2>&1 | tee /home/user/projects/wan21-video/logs/week1_first_run.log
```

第二条保持模型与生成参数一致，仅将：

- `CUDA_VISIBLE_DEVICES` 改为 `1`；
- 输出目录改为 `/home/user/projects/wan21-video/outputs/week1/gpu1`（先创建）；
- 日志改为 `/home/user/projects/wan21-video/logs/week1_gpu1_run.log`。

`--video_size` 按高度、宽度填写，最终视频宽 832、高 480。`--save_path` 是目录。两份日志都确认 `flow_shift=5.0`、`guidance_scale=5.0`；DiT、权重和 VAE 精度为 `torch.bfloat16`，T5 日志显示迁移到 CUDA 并转换为 `torch.bfloat16`。生成入口源码确认未指定 solver 时默认 `unipc`；其余默认值仍需继续核对，不能声称已冻结正式评测配置。

未开启 fp8、block swap、compile 等优化参数。20 步仅用于第一周跑通，不代表最佳质量设置。

## 7. 结果与结论边界

| 字段 | GPU 0，第一次 | GPU 1，第二次 |
|---|---|---|
| 用户提供的命令 Exit status | 0 | 0 |
| 总墙钟时间 | 157.61 秒 | 139.70 秒 |
| CPU user 时间 | 192.81 秒 | 181.07 秒 |
| CPU system 时间 | 67.54 秒 | 66.05 秒 |
| CPU 内存峰值，time 报告 kbytes | 12308148 | 12283684 |
| 稳态生成耗时 | 未测 | 未测 |
| 去噪进度条报告耗时（粗略） | 约 108 秒，20/20，最终平均 5.42 秒/步 | 约 107 秒，20/20，最终平均 5.38 秒/步 |
| GPU 峰值显存 | 未测 | 未测 |

两个命令各执行一次，没有预热、阶段计时或正式性能重复。157.61 秒与 139.70 秒均含加载、生成和保存；不能据此认定 GPU 1 更快。去噪值来自 tqdm 进度条的取整时间，不是同步 GPU 测量；两次去噪时间接近，总时间差异主要出现在去噪之外，但日志没有足够阶段时间戳将其细分到加载、文本编码或保存。CPU user/system 时间不能当作生成延迟，CPU 内存峰值不能当作显存峰值。

两份日志均报告 DiT 权重 `<All keys matched successfully>`，单模型采样、`high noise: False`、`timestep boundary: None`，latent shape 为 `[1, 16, 21, 60, 104]`，完成 VAE 解码并保存视频。2026-10-14 用户提供 `ls -lh` 输出确认下列文件存在，两份均显示 8.0M；没有重新加载或比较 latent 内容：

- GPU 0：`/home/user/projects/wan21-video/outputs/week1/20261013-165305-667_42_latent.safetensors`
- GPU 1：`/home/user/projects/wan21-video/outputs/week1/gpu1/20261013-172200-617_42_latent.safetensors`

日志中的部分 tokenizer 可选文件请求返回 404，但程序继续并正常完成；不是一次生成失败。完整摘要见 [原始日志解读](notes/run_log_summary.md)。

本机收到的两份 MP4 均为 2526841 字节、832×480、81 帧、16 FPS。文件 SHA256 完全一致，81 帧解码为 RGB 后也逐帧一致。完整检查记录见 [视频比较](notes/video_comparison.md)。

用户连续播放观察：挥手发生，但手腕、手掌和手指的运动关系不符合人体规律。记录为动作与时序结构缺陷；不能把“手在动”当作质量达标。抽帧只用于定位可见形态，不能替代连续运动评价。

目前支持的结论：双卡分别顺序推理成功；本次同参数、同种子输出一致；当前样本有手部动作质量缺陷。尚不支持普遍确定性、质量改善或性能优化结论。

## 8. 完成项与后续补充

第一周调整后的出口已满足。以下未勾选项属于后续证据完善或发布/训练阶段任务，不因材料收集继续推迟第二周。

- [x] 两份完整生成日志传回本机归档，并核对命令、参数和生成结果。
- [x] 服务器依赖快照传回本机，关键版本及可编辑安装 commit 已核对。
- [x] 将服务器生成帮助文本传回本机归档，并核对实际使用参数的说明。
- [x] torchvision 实际版本确认：`0.24.1+cu128`。
- [x] 核对 OS、内核、系统内存和 Conda 实际版本。
- [x] 核对 CPU 型号及核心信息：双路 EPYC 7C13，128 物理核心、256 逻辑线程，2 个 NUMA 节点。
- [ ] 记录权重来源 revision，分别核对许可。
- [x] 完成 12 项静态源码/文件清单核对及本次普通 T2V 调用链，见 `notes/param_check.md` 与 `notes/call_graph.md`。
- 用户再次确认固定 HEAD，归档源码未另逐文件与 Git 对象校验。模型卸载上限已确认 29；旧文档“代码不校验”已纠正。优化参数的运行效果仍待实测。
- [x] 从日志提取粗略去噪耗时；显存没有采样，仍保留未测状态。
- [x] 确认当前终端 `HF_ENDPOINT=https://hf-mirror.com`，记录日志中的实际镜像访问。
- [ ] 记录 tokenizer 实际配置文件与缓存状态；需要时追踪 HF_ENDPOINT 的原始设置位置。
- [x] 核对 `both` 生成的 latent 文件存在并记录路径：两份均显示 8.0M；文件内容未独立加载验证。
- 暂缓：本机 16G 推理，2026-10-14 用户决定移出当前门槛。环境已创建但未运行模型，16G 可用性未验证。
- [ ] 干净环境独立复现（后续发布阶段）。

## Week 2 测量脚本（2026-10-15）

入口为项目自有 `scripts/bench.py`，与上游 checkout 分开；不改上游文件。完整模式、结果路径、操作定义和验证状态见 [WEEK2_BENCH_GUIDE.md](notes/WEEK2_BENCH_GUIDE.md)。

- 服务器已有 `configs/baseline_bf16.yaml`，沿用第一周 20 步配置。
- 默认设置 `HF_HUB_OFFLINE=1`、项目内 HF_HOME、单卡 GPU 0；无需修改 shell 全局配置。
- 分词器使用 `Wan-AI/Wan2.1-T2V-14B` 的缓存 revision `a064a6c71f5be440641209c07bf2a5ce7a2ff5e4`，子目录 `google/umt5-xxl`。
- 实测直接从本地目录加载成功；完整推理还需要外部包装删除值为 None 的 subfolder 参数。此兼容处理在各模式保持一致，不计作性能优化。
- 原版脚本在服务器离线 2 步通过：前向 4 次、各阶段函数 1 次，JSON 保存成功。
- 整理版已通过本机语法和 10 项 CPU 模拟测试；实际 CUDA/profiler 仍待服务器验证。本机 16G 推理继续暂缓。

整理版初次服务器验证，在激活项目环境后执行：

```bash
python /home/user/projects/wan21-video/scripts/bench.py \
  --mode stages --steps 2 --warmup 1 --repeat 1 \
  --sample-smi --tag verify
```

它生成两次（预热 1 + 测量 1），不修改正式 YAML。每次调用新建独立 session 目录；预热记录保留但不进入汇总。冷启动以父进程创建子进程前至子进程保存完成为边界；正式端到端以 generate 前至视频保存完成为边界。

### Week 2 实机验收与结果

2026-10-15：整理版在服务器完成 verify 及 R1–R5，原始结果已传回本机 `notes/week2/`，其状态、时间汇总、阶段检查和外部采样均已解析核对。前文“待服务器验证”为当时状态，至本段已完成。profiler 有有效 CUDA kernel 和预定 4 次逻辑前向的标记，同时存在缓冲开销及边界前异步事件，处理方式和局限见报告。

- [基线与阶段/显存表](notes/week2_baseline.md)：冷启动 136.693 秒；R4 生成保存均值 135.735 秒，3 次范围 135.025–136.928 秒。
- [profiler 与瓶颈结论](notes/week2_profile.md)：去噪约占 R2 的 83.31%；VAE 解码产生 allocated 最高峰；没有已测优化收益。
- [机器可读审计](notes/week2_audit.json)：脚本哈希、原始统计、采样核对及内核关联过滤。
- 本机仅运行解析脚本，不需安装 torch；重算命令：`python scripts/analyze_week2.py`。它重写上述审计和两份报告，不修改原始 session 数据，不启动 GPU。
- Week 2 调整后服务器范围完成；本机 16G 可运行性、质量评测和性能优化仍未完成。

### 本机恢复执行与 L0 验证（2026-10-18 更新）

以上本机“暂缓”和“未验证”是历史状态。当前本机环境、权重和离线分词器已就绪，L0 已成功生成；本机重复测量尚未完成。

- 解释器：`E:\cstudy\AIGC\V\local\env\python.exe`，Python 3.11.17，torch 2.9.1+cu128，torchvision 0.24.1+cu128。
- GPU：RTX 5060 Ti 16GB，驱动 591.86；小型 CUDA 运算、pip check 和推理模块导入通过。
- 本机仓库：`local/musubi-tuner`，commit `f8a1b03794a49239a3539015075f5123d6c07d66`。
- 三份权重位于 `local/models/wan21`，SHA256 与服务器全部匹配；四个分词器文件位于 `local/models`，离线加载及编码通过。
- 本机脚本：`local/scripts/bench.py`，配置：`evidence/local/configs/baseline_bf16.yaml`。原服务器脚本和结果保持独立。

```powershell
nvidia-smi
& "E:\cstudy\AIGC\V\local\env\python.exe" `
  "E:\cstudy\AIGC\V\local\scripts\bench.py" `
  --mode stages --warmup 0 --repeat 1 --sample-smi --tag L0
```

L0 生成保存 627.740 秒，allocated 峰值 14699.2 MiB，reserved 18666 MiB，整卡采样峰值 15929 MiB。三种口径分开报告，reserved 不解释为物理显存占用。调用计数、分段和自检全部通过；视频动作和提示词符合度不足。记录见 [本机实验文档](notes/REVIEW/week2_local_16g.md)。本机 stages / total 各预热 1 次、测量 3 次及最终汇总仍待完成；没有由一次成功推断稳定性。

### 本机重复测量完成（2026-10-18）

上述待完成项已由 L1/L2 补齐，运行命令：

```powershell
& "E:\cstudy\AIGC\V\local\env\python.exe" `
  "E:\cstudy\AIGC\V\local\scripts\bench.py" `
  --mode stages --warmup 1 --repeat 3 --sample-smi --tag L1
& "E:\cstudy\AIGC\V\local\env\python.exe" `
  "E:\cstudy\AIGC\V\local\scripts\bench.py" `
  --mode total --warmup 1 --repeat 3 --sample-smi --tag L2
```

L1 会话 `L1-stages-20261018-144206-60e8ffc4`，L2 会话 `L2-total-20261018-154702-30f6e37f`，均在 `evidence/local/week2`；每组预热和正式三次均成功，输出视频及 latent 已确认存在。L1 所有调用计数与分段和检查通过。

正式端到端采用 L2：均值 637.611 秒，范围 635.399–641.259 秒；allocated 峰值 14699.2 MiB，reserved 18666 MiB，正式轮整卡采样峰值最大 15759 MiB。L1 均值 645.834 秒，其峰值与 L2 分配器峰值一致。时间含加载与保存，不是模型常驻或进程冷启动时间。

本机没有留存运行前独立进程清单，不能严格排除其他进程影响；少量固定配置重复不保证长时稳定。质量问题不视为性能实验失败，未据此证明跨平台差异原因。Week 2 服务器与本机实验及汇总完成，详细表和边界见本机实验文档。

## Week 3 评测生成准备（2026-10-19）

冻结清单为 `configs/prompts_frozen.yaml`，基线配置为 `configs/eval_baseline.yaml`：12 条原计划提示词、种子 42/1001/2002、每配置 36 段，继续使用 GPU 噪声。人工判据见 `notes/week3_rubric.md` v1.0。

新增 `scripts/gen_eval.py` 和 `scripts/test_gen_eval.py`；14 项 CPU 编排测试、干运行和真实上游参数解析通过。服务器推理与数据集验收随后完成，见 Day3 小节。原 `scripts/bench.py` 的 SHA256 仍为 `ffc9af27120d7fd0f903b1482122a393c39a372aa0be0f4eed67873851ea775c`，未修改。

新脚本复用 bench 参数/分词器路径及单次生成/清理/保存顺序，支持双卡分片和成功哈希校验后的续跑；保留原始尝试与失败，最多尝试两次。两卡结束后用 `--summarize` 输出完整 36 行 manifest 和覆盖率，缺失或失败不隐藏。并行生成耗时不作为延迟基线。完整命令、输出目录及局限见 [Week 3 生成说明](notes/WEEK3_GENERATION_GUIDE.md)。

### Week 3 Day 3 验收与 Day 4 指标实现（2026-10-19）

服务器冒烟、20 步路径核对已通过；用户报告 P01/s42 MP4 与 Week 2 R4 参考 SHA256 相同。正式双卡各 18 段，汇总 `ok=36`；36 段视频和清单已传回，本机逐段核对 MP4 SHA256 全部匹配。未传 latent。

新增 `scripts/prepare_clip.py`、`metrics.py`、`make_probes.py`、`test_metrics.py`。本机新增 scikit-image 0.25.2 及依赖，pip check 通过；原 bench.py 哈希未变。CLIP 固定 revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`，权重、官方模型卡、MIT 许可原文与来源记录保存在 `local/models/clip-vit-base-patch32`。许可依据与口径在下载前写入 [指标协议](notes/week3_metrics.md)。

探针按固定区间和区域构造，FFV1 无损保存并逐帧核对 RGB 像素。指标默认 CPU FP32；记录输入、协议、脚本哈希与依赖版本。完整顺序、复算方式、结果路径见 [Day 4 说明](notes/WEEK3_METRICS_GUIDE.md)。实际计算完成状态以 `notes/week3_metrics_run/run.json` 为准；自动指标与人工判据分别报告。

### Week 3 Day 5 匿名包准备（2026-10-21）

新增 `scripts/blind_pack.py`、`scripts/week3_scoring/blind_workbook.mjs`、`scripts/verify_blind_pack.py`。匿名包 `outputs/week3-R01-20261021.zip` 包含 16 段单视频、四对左右随机配对和空白 XLSX/CSV；抽样随机种子及来源映射在 `notes/blind_private/week3-R01-20261021/`，不进入评分者压缩包。原评分表未改。

制作时随机化复算、视频格式/来源/方向、表格空白和下拉/公式、ZIP 完整性已验证。所有待评视频统一有损编码，原始视频和 Day4 指标不变；代理编码和单评分者边界见 [试评包说明](notes/week3_blind.md)。后续评分、解盲及资源成本已完成，见收尾小节；原 ZIP 当前不在原路径，解压目录及私人映射仍在，不将历史 ZIP 验收冒充本次重新检查。

### Week 3 attention 安装检查（2026-10-22）

服务器主环境通过 `conda create --clone ... --copy` 复制为项目内 `env-sagecheck`。在副本安装 ninja 1.13.2，获取 SageAttention v2.2.0 / commit `eb615cf6cf4d221338033340ee2de1c37fbdba4a`，使用 CUDA Toolkit 12.8 及现有工具以 `--no-build-isolation --no-deps` 编译安装。具体工具版本、构建隔离限制及 wheel 日志信息见 [检查记录](notes/week3_attention_check.md)。

依据用户服务器输出，安装后 pip check、site-packages 导入和 GPU 0 BF16 小张量运算通过；torch 2.9.1+cu128、Triton 3.5.1 保持不变。安装日志和 pip freeze 已传回验收：原 71 条依赖记录均未变，仅新增 SageAttention；三份原始文件的 SHA256 见 `notes/sagecheck/archive_audit.json`。快照不覆盖 Conda 或系统库完整状态，源码 commit 依据用户克隆输出，完整源码及 wheel 未传回。此步未生成视频，不能作为完整模型兼容性、GPU 1 运行、加速或质量证据；后续 attention 对照需在副本内重测 torch 基线。

### Week 3 收尾、评分与预算（2026-10-22）

收尾复习入口为 [WEEK3_REVIEW.md](notes/REVIEW/WEEK3_REVIEW.md)，逐项验收见 [WEEK3_PLAN.md](plan/WEEK3_PLAN.md)。配置冻结和实际结果分别引用，不把计划文件当成运行成功证据。

| 内容 | 复现或检查入口 | 已验收范围 |
|---|---|---|
| 冻结生成设置 | `configs/prompts_frozen.yaml`、`configs/eval_baseline.yaml` | 12×3；480×832、81 帧、16 fps、20 步；GPU 噪声 |
| 评测生成 | `scripts/gen_eval.py`、`notes/WEEK3_GENERATION_GUIDE.md` | 36 个唯一成功组合；GPU 各 18 段；本机 MP4 哈希匹配 |
| 自动指标 | `scripts/metrics.py`、`notes/week3_metrics_run/run.json` | 49 条成功，3920 个 SSIM 帧对与 1029 个 CLIP 帧值；CPU FP32 |
| CLIP 来源与许可 | `scripts/prepare_clip.py`、`local/models/clip-vit-base-patch32/`、`notes/week3_metrics.md` | 固定 revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`，来源、哈希及许可文件已存 |
| 探针构造 | `scripts/make_probes.py`、`notes/week3_probe.md` | 12 段无损探针与一个文本替换候选；结论限定预先规则 |
| 匿名包 | `scripts/blind_pack.py`、`scripts/verify_blind_pack.py`、`notes/week3_blind.md` | 首次包制作验收；收尾按私人映射复核解压后 20 段视频，原 ZIP 不在原路径 |
| 答卷与解盲 | `scripts/analyze_blind_ratings.py`、`notes/week3_trial_review.md` | 16 段三维评分与四对；原件 SHA256 `e3153c55afbbe3d7e908596b9bee9da2e1d6a5f50b234f8c24a1be9e20109d21` 不变 |
| 资源成本 | `scripts/analyze_week3_cost.py`、`notes/week3_cost.md` | 现有记录复算；累计 82.951 分钟、双卡任务跨度 42.318 分钟，非优化延迟 |
| 未来评分量 | `configs/scoring_frozen.yaml` | 每配置 36 对三维配对、固定 12 段三维绝对评分；不要求人工计时 |
| 候选与判定规则 | `configs/optimization_rules.yaml`、`notes/week4_prereg.md` | 最多三个新增单项配置；当前无优化收益结论 |

本次答卷独立性与部分会话信息未记录，静止误扣时序分及模糊配对冲突按用户澄清单列。原始工作簿和分数不改，不将试评替代正式优化质量对照。P02 文本替换候选未单独验明动作，保留为未验证事项。干净环境复现、GPU 1 SageAttention 运算、完整模型及性能质量对照留后续，不要求重新生成本周基线。
