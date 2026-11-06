# Week5 Day7：VAE 输入与服务器预检

日期：2026-11-06，Asia/Shanghai。状态：**原版仅解码本机验收完成；用户已回传探针快照生成摘要，待取回原件分析**。服务器 GPU 运行由用户完成，本机未独立读取 latent 或重跑 GPU 测量。

## 已有依据

本机上游 HEAD 为 `f8a1b03794a49239a3539015075f5123d6c07d66`。固定版本 `wan_generate_video.py` 中：

- `--latent_path` 接受路径列表，但只支持一个输入，且 output_type 只能是 video/images。
- `main` 的 latent 分支读取 safetensors 的 `latent` 张量及 metadata，再调用 `save_output`，不运行 T5 或 DiT。
- `decode_latent` 默认 VAE 为 BF16，加载后在 autocast/no_grad 中解码，随后转 float32 并移到 CPU。
- 只解码与完整生成的模型驻留、缓存和 allocator 上下文不同。峰值不一致时须核对上下文，不能先认定一致，也不能把只解码时间当整次生成加速。

来源：本机固定源码约 233、1615、2225 行；服务器执行前仍要核对版本。

`notes/eval/baseline.jsonl` 中 P01/s42 的历史成功记录给出：

| 字段 | 历史清单值 |
|---|---|
| 路径 | `/home/user/projects/wan21-video/outputs/eval/baseline/P01_s42_latent.safetensors` |
| SHA256 | `45b483abe74e4cc29943b15c40931aa11ff265d652abf964132636303f0def16` |
| task | t2v-1.3B |
| 输出范围 | 480×832，81 帧，16 fps，20 步 |
| VAE | `/home/user/projects/wan21-video/models/wan21/split_files/vae/wan_2.1_vae.safetensors` |
| VAE/cache | 默认 BF16，vae_cache_cpu=false |
| latent dtype/shape | 用户回传：torch.float32，`[16, 21, 60, 104]` |

路径与哈希已与本次用户回传的服务器输出核对一致；本机未收到该 latent，不声称已独立核验其张量内容。

## 第一步：仅检查

在服务器 Bash 中运行以下短命令，不启动推理、不修改文件：

```bash
cd /home/user/projects/wan21-video
nvidia-smi
git -C musubi-tuner rev-parse HEAD
ls -lh outputs/eval/baseline/P01_s42_latent.safetensors
sha256sum outputs/eval/baseline/P01_s42_latent.safetensors
```

返回输出后核对 GPU0 当前占用、版本、文件存在性与上表哈希。不存在或不匹配时先查现有收据与路径，不立即重新生成。

## 第一步回收结果（2026-11-06）

用户粘贴的 `nvidia-smi` 时间为 `Fri Nov 6 11:27:18 2026`；该服务器显示时间未单独核实其时区，不作为 Asia/Shanghai 的精确时间记录。

- 上游 commit 与固定版本一致；P01/s42 latent 存在，ls 显示约 8.0M；SHA256 与上表历史记录完全一致。
- GPU0 当时显存 600/49140 MiB、GPU-Util 0%，进程列表为远程桌面、gnome-shell 和 Xwayland，未见计算推理任务；这是当时快照，正式运行前仍检查占用。
- GPU1 当时 20 MiB、GPU-Util 0%；本次依旧按计划用 GPU0，不据快照改变设备。
- 驱动 580.126.20，nvidia-smi 显示 CUDA Version 13.0；此输出不证明当前 Python 的 torch/CUDA build，后续测量单独记录运行环境。
- 尚未读取 latent dtype/shape/finite，未启动 VAE 解码，Day7 定位未完成。

## 第二步：只读 latent 内容

在服务器 Bash 执行，张量只加载至 CPU：

```bash
cd /home/user/projects/wan21-video
./env/bin/python - <<'PY'
import json
import torch
from safetensors import safe_open

path = "outputs/eval/baseline/P01_s42_latent.safetensors"
with safe_open(path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    metadata = f.metadata()
    latent = f.get_tensor("latent")

print(json.dumps({
    "keys": keys,
    "dtype": str(latent.dtype),
    "shape": list(latent.shape),
    "finite": bool(torch.isfinite(latent).all().item()),
    "min": float(latent.min().item()),
    "max": float(latent.max().item()),
    "metadata": metadata,
}, ensure_ascii=False, indent=2))
PY
```

回传结果后登记实际 dtype/shape 与 metadata，再准备仅解码测量；此步没有修改输入文件或上游代码。

## 第二步回收结果与固定输入登记

用户回传：dtype `torch.float32`、shape `[16, 21, 60, 104]`、finite=true、最小值 -2.86393404006958、最大值 3.138010263442993。metadata 的 seed=42、P01 提示词、guidance=5、480×832、81 帧和 20 步均与历史 manifest 一致。

实际值及来源已保存至 [固定输入登记](week5_vae_fixed_input.json)。latent 的存储 dtype 为 FP32；本次上游 VAE 设置为 BF16，二者分别记录，不手动将输入改存为 BF16。

## 第三步：原版仅解码测量

新入口为 `scripts/vae_decode_measure.py`。仅包装固定上游的 `decode_latent` 并调用 `wan.main()` 的 latent-only 分支，不修改上游或原 bench.py；当前不打开 memory history。

- 固定物理 GPU0、主环境 torch 2.9.1+cu128/CUDA build 12.8、干净的固定上游 commit、同一 latent 哈希/内容和 BF16 VAE。运行时重新核对，不满足则停止。
- 单个新进程、无预热、只解码一次，用于诊断，不是正式速度收益实验。解码区间前后同步 GPU，在区间开始重置峰值统计。
- 区间包含 VAE 加载、GPU 传输、解码和输出转 FP32/CPU，排除 MP4 编码；因此与 Week4 嵌套 load_vae 分段的耗时不直接等同。阶段峰值比较也需检查加载/缓存上下文。
- allocated/reserved 是当前进程 allocator 的绝对峰值字节及 MiB，不减去桌面占用，也不等于整卡峰值。
- 先保存 input.json，结束或异常时保存 result.json；保留失败，不自动重试，不覆盖已存在的输出目录。视频核验仅包括输出张量形状、非空 MP4 和哈希，完整文件解码/质量验证另行进行。
- 本机已通过无 GPU 导入的 dry-run、语法检查，以及模拟的同步/峰值重置顺序、绝对峰值/单位、单次调用限制与 cache 参数对照检查；未执行真实 GPU 解码。

本机 PowerShell 上传脚本：

```powershell
scp E:/cstudy/AIGC/V/scripts/vae_decode_measure.py <SSH_HOST>:/home/user/projects/wan21-video/scripts/
```

服务器 Bash 进入持久会话：

```bash
tmux new-session -A -s w5-vae
```

在会话中运行：

```bash
cd /home/user/projects/wan21-video
nvidia-smi
./env/bin/python scripts/vae_decode_measure.py \
  --output outputs/week5-vae/decode-baseline-01
```

若出现其他推理计算任务，先不启动本次测量。Ctrl+B、D 分离；重新进入使用同一 tmux 命令。把末尾的 status/decode_records/video_path 和错误（如有）回传，result.json 路径也会打印。下一步核对峰值，再准备快照，不直接开始改法或 cache 对照。

## 第三步回收结果（2026-11-06）

用户回传 `decode-baseline-01/result.json` 的终端摘要，status=ok：

| 字段 | 本轮用户回传 | Week4 已验收的基线 VAE 阶段 |
|---|---:|---:|
| 耗时（秒） | 10.993825 | 10.321187 |
| allocated 峰值（MiB） | 14698.970703125 | 14698.353515625 |
| reserved 峰值（MiB） | 18674 | 18666 |

本机依据原阶段审计复算 allocated 差值为 +0.6171875 MiB（+0.004199%），reserved +8 MiB。峰值接近，未触发“只解码峰值与完整生成差很多”的排查分支；可以继续准备仅解码内部分配诊断。但两次耗时边界、预热和上下文不同，不把秒数差当作性能回归或加速结论，也不称精确等价。

输入为 FP32 `[1,16,21,60,104]`，输出为 CPU FP32 `[3,81,480,832]`。before allocated/reserved=0；after allocated=8519680 字节、reserved=19581108224 字节。单次退出区间的数值不能用于认定泄漏、全部缓存占用或峰值来源。

原摘要与复算来源另存 [用户回传记录](week5/vae-baseline-01-user-return.json)，不是原服务器 result.json。视频路径为 `outputs/week5-vae/decode-baseline-01/20261106-113527-240_42__P01_s42_latent.mp4`；目前未在本机核验原 JSON 的脚本/输入哈希、视频哈希或完整 MP4 解码，尚未独立播放验收。

本机尝试 BatchMode scp 读取原目录，但 SSH 返回 `Permission denied (publickey,password)`，未取得文件，没有启动服务器实验。当前本机已建立空的 `outputs/week5-vae/` 父目录，子目录尚不存在。下一步由用户在本机 PowerShell 传回整个小目录：

```powershell
scp -r <SSH_HOST>:/home/user/projects/wan21-video/outputs/week5-vae/decode-baseline-01 E:/cstudy/AIGC/V/outputs/week5-vae/
```

传回后先检查 input.json/result.json、固定输入/脚本/源码身份和视频哈希、完整解码，再进入 memory-history 探针；不重跑这次仅解码来代替记录回收。

## 第三步本机验收完成

用户已传回原目录的三个文件。新增 `scripts/audit_vae_decode.py`，核验结果为 [本机审计](week5/vae-baseline-01-local-audit.json)：

- input/result 的固定字段一致，状态由 running 变为 ok；原 JSON 的输入、环境、设备和参数与登记一致，摘要与先前用户粘贴内容完全一致。
- 测量脚本 SHA256 与本机相同。首次核验因 Windows CRLF 与服务器 LF 原始字节不同而中止；已检查两份源码的服务器哈希与固定 commit 的 Git blob 完全相同，本地仅换行差异。未修改任何上游源码或服务器原件。
- MP4 哈希与服务器 result.json 一致，并与原基线 P01/s42.mp4 完全相同：`fa59fbd653bef48711826054cbfc3463da9003b10c429c2273d9cf77eb838217`。
- 本机完整解码 81 帧，832×480、16 fps、H.264/yuv420p，无全黑帧；所有 RGB 帧的顺序摘要保存在审计中。未将字节一致外推为其他样本、质量门槛或峰值来源。
- latent/VAE 权重本体仍未传回，其哈希和内容登记来源是服务器记录。两个原 JSON 没有单独的服务器 SHA256 回传，本机记录了文件哈希和交叉一致性，不补称跨机原 JSON 哈希独立一致。

首次验收命令如下，已有输出禁止覆盖：

```powershell
& E:/cstudy/AIGC/V/local/env/python.exe E:/cstudy/AIGC/V/scripts/audit_vae_decode.py --directory E:/cstudy/AIGC/V/outputs/week5-vae/decode-baseline-01 --output E:/cstudy/AIGC/V/notes/week5/vae-baseline-01-local-audit.json
```

## 第四步：allocator 历史与边界观察

新增 `scripts/vae_memory_probe.py`，复用已验收且哈希固定的基本测量脚本，不改原脚本或 VAE 数值代码。当前同样使用 GPU0、原 FP32 latent、BF16 VAE、cache_cpu=false。

- 使用当前本机同版 torch 2.9.1+cu128 的实际函数签名核对 `_record_memory_history` 和 `_snapshot`。历史上限 200000 条，保留分配与释放调用栈；只在本次解码区间开启，结束或异常时导出同格式 pickle 并关闭。
- 只在 conv2 和 decoder 入口/出口记录观察，不在 hook 内额外同步或重置峰值。预期 21 个 decoder 调用、44 个边界记录；记录形状、dtype、storage 地址及各边界 feat_cache 的去重 storage 字节。
- 缓存数据是边界时刻的完整底层 storage 大小，不等于 allocator block 大小，不证明正好在全局峰值瞬间的缓存占用。临时张量可在两个 hook 之间出现/释放，必须结合历史调用栈，证据不足时补针对性测量。
- 数据记录仅保存 Python 标量和列表，不保留 GPU 张量引用；hook 在异常和成功时都移除。快照可能影响运行开销，本次耗时不与无探针运行合成正式速度结论。
- 新目录保存 input/result、视频、`probe.json` 与 `vae_snapshot.pickle`；result 明确标记 memory_history=true，并记录基础/探针脚本哈希、原基线 result 哈希、快照哈希、事件数与可能截断标志。视频是否与无探针基线字节相同单列。
- 本机已通过参数 dry-run、语法、缓存 storage 去重、成功/异常快照保存与关闭、异常保留、21/44 边界计数、返回原对象和 hook 清理的模拟检查。未在本机或服务器执行探针 GPU 实验。

本机 PowerShell 上传新探针（原 `vae_decode_measure.py` 已在服务器，保持原字节）：

```powershell
scp E:/cstudy/AIGC/V/scripts/vae_memory_probe.py <SSH_HOST>:/home/user/projects/wan21-video/scripts/
```

服务器 Bash：

```bash
tmux new-session -A -s w5-vae
```

在会话中执行：

```bash
cd /home/user/projects/wan21-video
nvidia-smi
./env/bin/python scripts/vae_memory_probe.py \
  --output outputs/week5-vae/history-baseline-01
```

GPU0 若有其他推理任务先不启动。Ctrl+B、D 分离，同名 tmux 命令重新进入。完成后先回传末尾 `status`、`decode_records` 和 `PROBE_SUMMARY`；不要求立即传大快照，不重跑已有 baseline 或修改 VAE。错误保留原输出目录并回传，不自动重试。

## 第四步摘要回收与传输

用户已回传 PROBE_SUMMARY：快照 7781984 字节、SHA256 `c8682923be50a74c6884acc3e5bc1bdf5443ae33c364763092a9ceecbdd62b63`，事件 29327 条，未触及上限；decoder=21，边界记录=44。摘要保存于 `notes/week5/vae-history-01-user-summary.json`。目前只有摘要，result 中的运行状态、峰值/耗时及输出视频尚未本机验收。

快照约 7.42 MiB，可直接传整个小目录，不必再运行探针。在本机 PowerShell 执行：

```powershell
scp -r <SSH_HOST>:/home/user/projects/wan21-video/outputs/week5-vae/history-baseline-01 E:/cstudy/AIGC/V/outputs/week5-vae/
```

传回后本机核对快照大小/哈希、脚本/输入/环境、事件和边界计数、原视频身份，再分析峰值存活分配及调用栈。不从事件数量推断中间激活、输出累积或 feat_cache 的因果贡献。

## 随后的顺序

1. 使用服务器主 Python 只读检查同一 latent 的 dtype、shape、finite 及 metadata，登记固定输入。实际值回传后才记完成。
2. 准备独立测量包装与新输出目录，核对同步和峰值重置边界，再跑仅解码。长任务先进入 `tmux new-session -A -s w5-vae`；Ctrl+B、D 分离，同命令重新进入。
3. 记录未插 memory-history 的仅解码耗时与峰值，再另作分配快照及必要的形状/缓存测量；工具开销单列。
4. 同一输入只改变 `vae_cache_cpu` 做对照。内部分配归属须有测量支持，allocator 调用栈本身不证明张量形状或全部缓存占用。
5. 保存 VAE 定位报告，证据不足则写尚未定位及下一步。完成前不勾 Day7，也不进入性能改法实施。

## 第四步原件验收与内部分配分析完成

用户已传回 `history-baseline-01/`。本机核对 snapshot SHA256/大小/全部事件、input/result/probe、脚本与固定 Git blob、基线身份；history 已关闭、status=ok。81 帧完整解码、视频哈希均通过，与原版仅解码及冻结 P01/s42 相同。

新增 `analyze_vae_history.py` 按本次默认 native allocator 规则回放 29327 个事件，匹配所有最终块及全部 44 个边界观测。峰值 15412987904 字节逐字节匹配记录；最大存活分配来自第 21 次 decoder 内 native slow 3D 卷积 columns，requested 7897.5 MiB。`attribute_vae_peak.py` 结合 storage epoch 和源码赋值顺序重建缓存成员 3556.083984375 MiB，另有尚未赋值 clone 292.5 MiB。临时形状未直接 hook，详见 [定位报告](week5_vae_peak.md)。

最终机器产物在 `notes/week5/vae-history-01-analysis-v2/`；早期 `vae-history-01-analysis/` 因绘图解释器不匹配中止而保留，不作最终入口。静态曲线已查看。PyTorch memory viz HTML 已生成，工具浏览器因不允许 file: 协议而拒绝打开，未做交互截图，未绕过策略。

## 第五步：同输入 CPU 缓存对照（执行与验收记录）

继续复用服务器已有、哈希未改的 `scripts/vae_decode_measure.py`，无需上传新脚本。只增加测量包装参数 `--vae-cache-cpu`，它转为上游 `--vae_cache_cpu`。本机 dry-run 已检查，使用原 latent、BF16 VAE、GPU0、无探针、无预热的新进程。已有输出目录会拒绝运行。

先进入服务器 tmux 并检查当前 GPU0：

```bash
tmux new-session -A -s w5-vae
cd /home/user/projects/wan21-video
nvidia-smi
```

当前占用回传确认后，在同一会话执行以下待执行命令：

```bash
./env/bin/python scripts/vae_decode_measure.py \
  --vae-cache-cpu \
  --output outputs/week5-vae/decode-cache-cpu-01
```

Ctrl+B、D 分离；`tmux new-session -A -s w5-vae` 重新进入。完成后先回传 `RESULT_JSON` 与摘要；若出错保留原目录，不自动重试。比较无探针 baseline 与无探针 cache_cpu 的 allocated/reserved/耗时诊断、视频哈希和完整解码，不将一次耗时差写成正式速度收益，不预估峰值恰好减去缓存字节。

截至此处未执行 CPU 缓存对照，没有修改上游 VAE 算法，Week5 尚未收尾。

### 第五步用户摘要已回收

用户确认运行前无占用，随后按上述命令运行并回传部分结果。allocated 11142.07421875 MiB、reserved 16078 MiB、50.383249560996774 秒；形状/dtype 与基线摘要一致。相对无探针 baseline，allocated 减少 3556.896484375 MiB / 24.198269%，耗时为 4.582868 倍；仅为单次区间诊断。片段没有 status 字段，原 input/result/视频待验收。另存 `notes/week5/vae-cache-cpu-01-user-summary.json`，保留来源边界，不重跑。

本机 Windows PowerShell 传回：

```powershell
scp -r <SSH_HOST>:/home/user/projects/wan21-video/outputs/week5-vae/decode-cache-cpu-01 E:/cstudy/AIGC/V/outputs/week5-vae/
```

传回后核对开关/身份、最终状态、视频哈希及 81 帧完整解码，再更新最终对照结论。此时不能把运行摘要当作原件验收通过。

### 第五步原件验收完成

用户已传回 `decode-cache-cpu-01/`。新入口 `scripts/audit_vae_cache_cpu.py` 核验成功，审计另存 [vae-cache-cpu-01-local-audit.json](week5/vae-cache-cpu-01-local-audit.json)。status=ok，cache_cpu=true；与原基线除保存路径外只增加 `--vae_cache_cpu`，同输入/模型/环境/源码记录，基础测量脚本哈希不变，旧基线原件哈希未变。

MP4 与无探针及冻结 P01/s42 基线字节相同；81 帧完整解码、RGB 帧摘要相同，无全黑帧。allocated 减少 3556.896484375 MiB / 24.198269%，reserved 减少 2596 MiB，单次区间耗时约 4.582868 倍。本次不检查原始浮点输出张量相等，不外推完整生成性能或 16G 可用性。

首次验收命令（已有输出不可覆盖）：

```powershell
& E:/cstudy/AIGC/V/local/env/python.exe E:/cstudy/AIGC/V/scripts/audit_vae_cache_cpu.py --directory E:/cstudy/AIGC/V/outputs/week5-vae/decode-cache-cpu-01 --output E:/cstudy/AIGC/V/notes/week5/vae-cache-cpu-01-local-audit.json
```

### 第六步：memory viz 交互查看（已完成）

无需再跑 GPU 实验。在本机普通浏览器手动打开已生成的 `notes/week5/vae-history-01-analysis-v2/memory_viz.html`。该页面需要获取 PyTorch CDN 的 JavaScript 模块，快照已嵌入本地 HTML，不要求将快照上传外部服务。查看 Active Memory Timeline 并截图；先回传实际页面或报错，不假设渲染成功。工具浏览器此前拒绝 file: 协议，不能代替用户声称完成截图。

用户随后回传正常页面截图：GPU0，Active Memory Timeline，9764/9764 条分配记录；下方有选中分配及栈，当前选中 292.5 MiB，不是最大的 7897.5 MiB 分配。截图已原样保存至 `notes/week5/vae-history-01-analysis-v2/memory_viz-user-timeline.png`，305377 字节，复制前后 SHA256 均为 `4d4ac320b02198decd7131266ab1111e0aad921c162e23cc42814d0477d8e15c`。来源/观察边界另存同目录 `memory-viz-user-screenshot.json`。

用户侧查看/截图已完成，精确峰值归属仍以本机快照重放和源码审计为准。报告、计划、日志及 REPRO 已同步，Week5 测量和文档完成收尾，无需追加 GPU 实验。第 6–7 周后端诊断及性能改动需下一阶段单独预登记。
