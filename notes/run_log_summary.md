# 第一周原始生成日志解读

记录日期：2026-10-13。依据本机归档的原始文件：

- GPU 0 日志（原本地引用：`../logs/week1_first_run.log`；该文件未随公开快照发布或路径已调整）
- GPU 1 日志（原本地引用：`../logs/week1_gpu1_run.log`；该文件未随公开快照发布或路径已调整）

原始日志包含终端进度条的回车刷新字符，阅读时不要将同一进度条的多次刷新当作多次生成。

## 1. 已确认的配置

两份日志的 Python 命令与记录一致，均为 `t2v-1.3B`、高宽 `480 832`、81 帧、16 FPS、20 步、seed 42、torch attention、相同提示词及三份权重路径。输出目录不同。

运行时打印：

| 项目 | 两次共同值 |
|---|---|
| flow shift | 5.0 |
| guidance scale | 5.0 |
| DiT / weight / VAE precision | `torch.bfloat16` |
| T5 | 加载指定 pth，迁移到 CUDA 并转换为 bf16 |
| fp8 scaled | False |
| state dict | without FP8 optimization |
| 模型数量 | 1 |
| high noise | False |
| timestep boundary | None |
| latent shape | `[1, 16, 21, 60, 104]` |

日志里的 `Using device: cuda` 没有证明物理 GPU 编号；GPU 0 / 1 的归属仍依赖启动时的 `CUDA_VISIBLE_DEVICES` 与用户执行记录。其余未打印的默认参数需要固定 commit 源码核对。

## 2. 生成流程

从日志可确认的先后顺序：

1. 确定视频尺寸、参数和精度。
2. 加载 T5，访问或检查 tokenizer 相关文件。
3. 创建并加载 DiT；`All keys matched successfully`。
4. 执行 20 步采样。
5. 保存 latent。
6. 加载 VAE，对 latent 解码。
7. 报告解码完成、保存 MP4、`Done!`。

这只是日志级流程，不代替后续带函数名和源码位置的调用链。文本编码没有独立计时。

## 3. 时间

| 项目 | GPU 0 | GPU 1 |
|---|---|---|
| time 总墙钟 | 157.61 秒 | 139.70 秒 |
| tqdm 最终去噪累计时间 | 约 108 秒 | 约 107 秒 |
| tqdm 最终平均步耗时 | 5.42 秒 | 5.38 秒 |
| Exit status | 0 | 0 |

总墙钟含 Python 启动、模型加载、网络请求、文本编码、采样、解码与保存。tqdm 累计时间精度有限，未按项目 METRICS 执行 CUDA 同步计时，不作为正式性能结果。

两次总时间相差 17.91 秒，而进度条去噪时间很接近，差异主要位于去噪之外。第一条请求下载了 tokenizer 配置、模型等文件，第二条请求减少，缓存状态并不一致。不能把差异全部归于 GPU、缓存或网络，也不能从总墙钟减去去噪时间得到“加载耗时”。

`Maximum resident set size` 是 CPU 内存峰值。日志没有 GPU 峰值显存、独立 T5 编码时间或 VAE 解码时间。

## 4. 网络新发现

两份日志都先向 `hf-mirror.com` 发出请求，再重定向到 `huggingface.co`。

- tokenizer 来源：`Wan-AI/Wan2.1-T2V-14B`，子目录 `google/umt5-xxl`。
- 请求中出现的 revision：`a064a6c71f5be440641209c07bf2a5ce7a2ff5e4`，只是日志观察值，启动命令没有显式锁定 tokenizer revision。
- 部分请求 404 后程序继续，随后加载和生成成功。这些请求不能被当作本次生成失败或手部缺陷的原因。
- 2026-10-13 用户执行 `printenv HF_ENDPOINT`，当前终端值为 `https://hf-mirror.com`，与生成日志一致。
- 原命令没有显式 HF_ENDPOINT，复现需要记录该环境变量；最初在哪个配置文件或命令中设置尚未追踪。当前值不单独证明历史进程环境，但历史日志已经确认镜像访问。

## 5. 输出

| 运行 | 日志报告的 latent 路径 |
|---|---|
| GPU 0 | `/home/user/projects/wan21-video/outputs/week1/20261013-165305-667_42_latent.safetensors` |
| GPU 1 | `/home/user/projects/wan21-video/outputs/week1/gpu1/20261013-172200-617_42_latent.safetensors` |

2026-10-14 用户提供 `ls -lh` 输出，确认两份 latent 文件存在且均显示 8.0M，内容未重新加载或比较。MP4 已在本机完整解码并比较，见 [视频比较](video_comparison.md)；两份文件一致，但存在用户指出的手部连续运动缺陷。
