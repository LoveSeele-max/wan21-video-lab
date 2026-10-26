# 第 1 周计划：环境、首段视频、参数核对

**日期**：2026-10-11 编写，2026-10-13 修正示例命令与链接。配套 [EXECUTION_CHECKLIST.md](../doc/EXECUTION_CHECKLIST.md) 第 1 周、[EXPERIMENT_PLAN.md](../doc/EXPERIMENT_PLAN.md) 阶段 0。

**执行进度（2026-10-14）**：第一周按调整后的服务器范围完成。双卡顺序生成、12 项静态源码/文件清单核对、调用链与复现日志已整理，latent 文件存在已确认。用户决定暂缓本机推理，不能标记为已尝试。后续训练帮助等补充项不阻塞第二周。实际配置和结果以 [REPRO.md](../REPRO.md) 与 [第一周日志](../logs/week1.md) 为准，下方按天清单保留原计划。

**本周只做三件事**：

1. 在服务器上跑出第一段 480P 视频；
2. 把用到的参数和代码位置对照**固定 commit 的源码**核实一遍；
3. 原定本机 16G 最小推理暂缓；服务器阶段已完成，转入第二周测量。

**本周不做**：

- 不写测量脚本（第 2 周）；
- 不装 xformers / sageattention / flash-attn；
- 不调任何优化参数；
- 不下载训练数据。

**命令来源**：下面的安装与推理命令取自上游 README 和 `docs/wan.md`（main 分支，2026-10-11 查阅）。**路径和文件名都是占位，执行时以实际为准**，以固定 commit 下的文档为最终依据。

---

## 出口标准（周末自查）

- [x] 服务器上产出一段能播放的 480P 视频，两张卡顺序运行均完成
- [x] `REPRO.md` 已记录环境、权重、命令及两次结果；干净环境复现仍待后续验证，补充项见该文件
- [x] `notes/param_check.md` 12 项静态源码/文件清单核对已记录，证据边界写明
- 暂缓：本机 16G 推理，由用户决定移出当前门槛
- [x] `logs/week1.md` 已完成第一周记录，保留质量缺陷与未测项

**止损**：服务器环境卡超过 3 天，改用官方 Wan2.1 仓库的推理入口先出视频；本机卡住则不影响本周出口，记录卡点即可。

---

## Day 1：服务器环境

### 1.1 先记录硬件与驱动

```bash
nvidia-smi                      # 驱动版本、CUDA 版本上限、两张卡的编号
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python3 --version
df -h <放权重的盘>               # 1.3B 全套权重 + 输出，预留 ≥ 50G
free -g
```

全部原样贴进 `REPRO.md`（模板见文末）。

### 1.2 克隆并固定 commit

```bash
git clone https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
git log -1 --format="%H %cd"    # 记录 hash 和日期
```

**本项目整个周期都钉在这个 commit 上**。不要中途 `git pull`；需要升级时，单独记一次变更，并重新核对参数。

### 1.3 独立虚拟环境

上游要求：Python ≥ 3.10、PyTorch ≥ 2.6。

```bash
python3 -m venv ~/venvs/musubi        # 或 conda，二选一，记进 REPRO.md
source ~/venvs/musubi/bin/activate
pip install --upgrade pip

# PyTorch：按驱动支持的 CUDA 版本选 index-url。上游示例为 cu124
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -e .
pip freeze > requirements.lock
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**决策点**：本机 5060 Ti 属于 Blackwell 架构，需要较新的 CUDA 构建（通常 cu128 起）。如果希望两台机器的 PyTorch 版本一致，服务器上也直接装同一 cu12x 版本。驱动允许的话，**推荐两边统一**，这样第 6–7 周对比两台机器时少一个变量。选了哪个、为什么，记进 `REPRO.md`。

---

## Day 2：权重下载

`docs/wan.md` 给出的来源：

| 组件 | 文件 | 来源 |
|---|---|---|
| DiT（1.3B T2V） | `wan2.1_t2v_1.3B_bf16.safetensors` 一类（**以仓库内实际文件名为准**） | `Comfy-Org/Wan_2.1_ComfyUI_repackaged` → `split_files/diffusion_models` |
| VAE | `wan_2.1_vae.safetensors`（或官方 `Wan2.1_VAE.pth`） | 同上 → `split_files/vae` |
| T5 | `models_t5_umt5-xxl-enc-bf16.pth` | Wan-AI 官方仓库（文档链接的是 `Wan-AI/Wan2.1-I2V-14B-720P`） |
| CLIP | 不需要 | 仅 I2V 使用 |

```bash
pip install -U "huggingface_hub[cli]"
# 若直连 HF 不通，可设镜像：export HF_ENDPOINT=https://hf-mirror.com（记进 REPRO.md）

mkdir -p ~/models/wan21
# 先列出文件名，确认后再下载对应文件，不要整个仓库全下
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  --include "split_files/diffusion_models/*1.3B*" "split_files/vae/*" \
  --local-dir ~/models/wan21
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \
  --include "models_t5_umt5-xxl-enc-bf16.pth" \
  --local-dir ~/models/wan21
```

下载完成后：

```bash
sha256sum ~/models/wan21/**/*.safetensors ~/models/wan21/*.pth > weights.sha256
```

把文件名、来源仓库、sha256 写进 `REPRO.md`。**许可**：Wan2.1 为 Apache-2.0；Comfy-Org 重打包仓库的许可声明也看一眼，记一笔。

---

## Day 3：第一段视频

### 3.1 最朴素的命令

**不加任何优化参数**（这就是之后的"未优化基线"雏形）：

```bash
CUDA_VISIBLE_DEVICES=0 python src/musubi_tuner/wan_generate_video.py \
  --task t2v-1.3B \
  --video_size 480 832 --video_length 81 --infer_steps 20 \
  --prompt "a person waving hand, plain background, static camera" \
  --seed 42 \
  --save_path outputs/week1 --output_type both \
  --dit ~/models/wan21/split_files/diffusion_models/<1.3B文件名> \
  --vae ~/models/wan21/split_files/vae/wan_2.1_vae.safetensors \
  --t5  ~/models/wan21/models_t5_umt5-xxl-enc-bf16.pth \
  --attn_mode torch \
  2>&1 | tee logs/week1_first_run.log
```

几点说明：

- 上游文档示例带 `--fp8`。这里**故意去掉**，让基线是 bf16。48G 卡对 1.3B 应该够用；如果 OOM，就记录下来再加 `--fp8`。这本身就是一条数据。
- `--seed` 的参数名以 `--help` 为准。
- `video_length 81` 满足 4n+1。
- `--video_size` 顺序为高度、宽度；`480 832` 输出宽 832、高 480。
- `--save_path` 指定目录，文件名由程序生成；运行前创建目录。当前实际执行的提示词、fps、绝对路径见 `REPRO.md`，与本节原示例有差异，复跑应使用实际命令。

### 3.2 粗略计时（本周只求有数，不求精确）

```bash
/usr/bin/time -v <上面整条命令>  2>&1 | tee logs/week1_time.log
```

记三个数：总墙钟时间、日志里显示的去噪进度条耗时、`nvidia-smi` 目测峰值。**含加载 / 不含加载分开写**，不含加载的部分从日志时间戳推算。

### 3.3 再跑一条

换卡 1（`CUDA_VISIBLE_DEVICES=1`），用同一命令、同一种子再跑一次。确认两张卡都能用，并对比两次输出是否逐帧一致（目测即可）。不一致也记下来，这关系到第 2 周的复现性设定。

---

## Day 4：参数与代码位置核对

**这是本周最重要的一天。** V2.0/V2.1 有两次错误，都是只看文档或只看调用处造成的。本周要在**固定 commit**上把它们全部落实。

### 4.1 拿到权威清单

```bash
python src/musubi_tuner/wan_generate_video.py --help > notes/help_generate.txt
python src/musubi_tuner/wan_train_network.py --help > notes/help_train.txt
git ls-files | grep -E "attention|offload|fp8|timesteps|scheduling|lora_wan|model_utils|vae" > notes/key_files.txt
```

### 4.2 逐条核对（填进 `notes/param_check.md`）

| # | 待核对的断言（EXPERIMENT_PLAN 第七节） | 怎么核 |
|---|---|---|
| 1 | `--device` 存在 | `--help` 里搜 |
| 2 | `sdpa` 与 `torch` 走同一分支 | 读 `wan/modules/attention.py` 的分发 |
| 3 | `flash` 与 `flash2` 同一分支 | 同上 |
| 4 | Wan 的 DiT 实际调用哪个 attention 函数、`attn_mode` 怎么传进去 | 从 `wan/modules/model.py`（或实际 DiT 文件）追到 attention 调用处 |
| 5 | `--fp8_fast` 只在 `--fp8_scaled` 下生效，且无校验 | `grep -n fp8_fast` 生成脚本 |
| 6 | `--offload_inactive_dit` / `--lazy_loading` 只在多 DiT 时生效 | `grep -n` 两个参数，确认判断条件 |
| 7 | 1.3B 的 DiT 层数 → `--blocks_to_swap` 上限 | 读模型配置（`num_layers`），对照文档的"29" |
| 8 | VAE 解码：逐 latent 帧 + 特征缓存，无空间分块，输出 `torch.cat` 累积 | 读 `wan/modules/vae.py` 的 `WanVAE_.decode` |
| 9 | `--vae_cache_cpu` 只影响缓存特征的位置 | 追 `cache_device` 的用法 |
| 10 | 生成脚本内无阶段计时 | `grep -n "perf_counter\|time.time"` |
| 11 | `compile_transformer` 位于 `utils/model_utils.py` | `key_files.txt` + `grep -n` |
| 12 | README 第二节"代码阅读入口"表的每个路径都存在 | 对照 `key_files.txt` |

每行写成：

```
#N  结论：符合 / 不符  依据：<文件>:<行号>（commit <短hash>）  备注：
```

**发现不符就改文档**：ASSUMPTIONS 对应条目、EXPERIMENT_PLAN 第七节、README。改完在 `logs/week1.md` 记一笔。

### 4.3 顺便画一张调用链草图

只画推理一条：

```
main → 加载 T5 / DiT / VAE → 文本编码 → 去噪循环（scheduler step × N）→ VAE decode → 保存
```

每个箭头标出文件和函数名。**这张图就是第 2 周插桩的位置清单**，写进 `notes/call_graph.md`。

---

## Day 5：本机 5060 Ti 16G 尝试

**2026-10-14 用户决定延后执行，计划保留（建议第 8 周里程碑前完成，待确认）。当前不执行，也不作为第一周门槛。**

**目标**：只求知道能不能跑、卡在哪。本周不调优。

1. 装环境（Windows 或 WSL 二选一，记录选择）。PyTorch 要选支持 Blackwell 的 CUDA 构建；`torch.cuda.get_device_capability()` 应返回 (12, 0)。
2. 用 Day 3 的同一条命令，`--attn_mode torch`，**先不加任何优化**跑一次。
3. 如果 OOM，按以下顺序**只加一项**再试，每次都记录：
   1. `--fp8`
   2. `--fp8 --fp8_t5`

   本周到此为止，`blocks_to_swap` 留给第 6–7 周。
4. 记录：成功 / OOM（在哪个阶段，看报错前的最后一条日志）/ 其他报错。

---

## Day 6–7：整理与缓冲

- [x] 填完 `REPRO.md`
- [x] 写 `logs/week1.md`：跑了什么、数字、踩的坑、改了哪些文档
- [x] 回头检查出口标准，没完成的项用这两天补（本机推理已延后，见 Day 5）
- [ ] **预读第 2 周**：根据 `notes/call_graph.md`，列出要插桩的 5 个位置（T5 编码前后、去噪循环前后、单步、VAE 解码前后、保存前后）

---

## 本周产出文件一览

```
musubi-tuner/                 # 上游仓库（钉在固定 commit）
  requirements.lock
  weights.sha256
E:\cstudy\AIGC\V\
  REPRO.md                    # 新建
  notes/help_generate.txt
  notes/help_train.txt
  notes/key_files.txt
  notes/param_check.md        # 12 项核对
  notes/call_graph.md         # 推理调用链
  logs/week1.md
  logs/week1_first_run.log
  video/20261013-165316-160_42_.mp4        # 服务器 GPU 0 产出，本机收到（不进 git）
  video/gpu1/20261013-172211-153_42_.mp4   # 服务器 GPU 1 产出，本机收到（不进 git）
```

---

## 附：`REPRO.md` 模板

```markdown
# 复现记录

## 硬件
- 服务器：GPU ×2 型号 / 显存 / 驱动版本 / nvidia-smi 显示的 CUDA 版本
- 本机：GPU 型号 / 显存 / 驱动版本 / OS（Windows / WSL）

## 软件
- OS 与内核：
- Python：
- PyTorch / CUDA 构建：（及选择理由：为何选 cu12x）
- musubi-tuner commit：<hash>（<日期>）
- 依赖锁定：requirements.lock
- HF 镜像：是否使用，地址

## 权重
| 组件 | 文件名 | 来源仓库 | sha256 | 许可 |
|---|---|---|---|---|
| DiT | | | | |
| VAE | | | | |
| T5  | | | | |

## 第一条命令
（原样粘贴，包括 CUDA_VISIBLE_DEVICES）

## 第一次结果
- 输出路径：
- 总墙钟：     含加载：     不含加载（估）：
- 峰值显存（nvidia-smi 目测）：
- 卡 0 / 卡 1 输出是否一致：
- 本机 16G：成功 / OOM（阶段）/ 报错

## 偏离与已知问题
（与文档不一致处、临时绕过的问题）
```
