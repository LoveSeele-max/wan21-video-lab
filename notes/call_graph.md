# 推理调用链：第一轮入口阅读

更新：2026-10-14。依据从服务器传回的源码；用户再次提供 HEAD 为 `f8a1b03794a49239a3539015075f5123d6c07d66`，文件清单已核对。未另逐文件与 Git 对象校验。

范围：本项目第一次运行的普通 T2V、单提示词、未启用控制视频、未加载 LoRA 路径。入口与 DiT attention、VAE decode 路径及文件清单已核对；不是整个框架的完整调用图。

## 从终端命令到生成结果

同一文件中的调用顺序：

```text
__main__ (2328) → main (2216)
  → parse_args (66)：读取命令行
  → 选择 device (2228)：未指定时优先 CUDA
  → 单提示词分支 (2299)
    → setup_args (394)：补未指定的任务默认值
      → get_task_defaults (359)：从 WAN_CONFIGS 读取任务配置
    → check_inputs (439)：检查尺寸等输入
    → get_generation_settings (2180)：确定配置、设备和精度
    → generate (1510)
      → prepare_t2v_inputs (825)
        → load_text_encoder (509)：加载 T5
        → 编码正面和负面提示词 (878)
        → 删除 T5 对象并清理内存 (882)
        → 生成初始噪声 (904)
      → load_dit_models (561)：加载 DiT
      → setup_scheduler (1260)：建立采样器与时间步
      → run_sampling (1305)：执行去噪采样
      → 删除模型和 scheduler，清理内存 (1594)
      → 返回 latent (1612)
    → 清理内存 (2314)
    → save_output (1749)
      → save_latent (1655)：output_type=both 时保存 latent
      → decode_latent (1615)：加载/使用 VAE 解码
      → save_video (1697)：保存 MP4
```

这里的“删除对象并清理内存”描述源码操作，不意味着已实测该时刻所有 GPU 分配都消失，也不能替代显存测量。

## 参数怎样进入程序

以 `--infer_steps 20` 为例：

1. `parse_args()` 由 argparse 读取命令，将数值存入 `args.infer_steps`。
2. `setup_args()` 第 409 行仅在该值为 `None` 时填任务默认值，因此不会覆盖显式指定的 20。
3. `setup_scheduler()` 第 1273 行将 `args.infer_steps` 传给 scheduler，设置采样时间步。
4. 后续 `run_sampling()` 使用这些时间步执行采样。

以 `--sample_solver` 为例：

- 命令没有显式传入。
- 第 74 行参数定义的 `default="unipc"` 会填入 `args.sample_solver`。
- 第 1271 行进入 unipc 分支，创建 `FlowUniPCMultistepScheduler`。

有些默认值直接写在参数定义中；另一些先设为 `None`，再根据任务配置补齐。这就是不能只凭帮助文本推断全部实际参数的原因。

## 已定位的源码证据

| 项目 | 行号 | 结论 |
|---|---|---|
| solver 默认值 | 74；1271 | 默认 unipc，随后选择 UniPC 分支 |
| 高宽顺序 | 114；448；449；849 | 第一个值是高，第二个值是宽 |
| 显式步数保留 | 409 | 仅在 None 时使用任务默认步数 |
| 自动设备选择 | 210；2228 | 未指定时，有 CUDA 则 CUDA，否则 CPU |
| 负面提示词回退 | 855 | 未提供非空值时用 `config.sample_neg_prompt`；2026-10-14 原文已在任务配置笔记归档 |
| 输出目录 | 1667；1668；1708；1709 | 使用 `os.makedirs`，保存路径作为目录 |
| 两个开关互斥 | 2220 | offload_inactive_dit 与 lazy_loading 有 assert |

## 2026-10-14 内部路径补充

- `load_dit_model`（入口第 648 行）→ `load_wan_model`（模型第 979 行），将 attention 模式和任务结构传入 WanModel（第 1021 行）。
- WanModel 创建 block（第 661 行）；block 创建 self/cross attention（第 383、385 行）；attention 在第 237、276 行调用 Wan 的 `flash_attention`（第 21 行导入），模式为 torch 时进入 `wan_attention.py` 第 92 行的 SDPA 分支。
- block swap 初始化由入口第 713 行调用模型第 748 行，模型第 753 行检查上限；forward 中第 895、901 行等待和提交 block 迁移。
- `decode_latent` → WanVAE.decode（VAE 第 758 行）→ WanVAE_.decode（第 570 行）→ 全 latent 的 conv2（第 578 行）→ decoder 逐时间片（第 581 至 587 行）→ 输出累积。CPU cache 设置传入 decoder 的特征缓存；输出卸载分支被注释。
- 编译入口第 728 行 → model_utils 第 309 行 → 第 332 至 341 行逐 block `torch.compile`。

## 2026-10-14 第二周准备补充

- **CFG 是两次独立前向，不是批量**：`run_sampling` 中，条件分支在第 1456 行（`model(..., **arg_c)`），普通引导的无条件分支在第 1488 行（`model(..., **arg_null)`）；第 1462、1477 行是 SLG 分支，本次未启用。因此 20 步 = 40 次 DiT 前向。未启用 `--cfg_skip_mode`，所以每步都应用 CFG。
- **嵌套调用**：`load_text_encoder` 在 `prepare_t2v_inputs` 内部调用（第 868 行），T5 随后在第 869 行搬上设备；`load_vae` 在 `decode_latent` 内部调用（第 1633 行）。T2V 路径下 `args._vae` 为 None，因此每次解码都会重新加载 VAE。
- **tokenizer 路径**：`load_text_encoder` 中 `ckpt_dir` 为 None 时，tokenizer 路径为 None（第 521 行）；任务配置中的 tokenizer 名为 `google/umt5-xxl`（`wan_t2v_1_3B.py` 第 17 行），会经 HF 解析。测量时用 `HF_HUB_OFFLINE=1` 排除网络因素。
- **单提示词主流程**：`main` 第 2300–2323 行：`setup_args` → `check_inputs` → `get_generation_settings` → `generate` → `gc` + 清理显存 → `save_output`（内含 `save_latent` → `decode_latent` → `save_video`）。`generate` 会改写 `args.seed`，并设置 `args._vae`。

以上来自归档源码的静态阅读，未运行验证。

已完成本轮三组阅读，不要求继续读完整训练框架。源码结果与运行/性能证据分开登记。
