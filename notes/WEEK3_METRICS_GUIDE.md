# Day 4 运行说明

在本机执行，使用 `local/env/python.exe`，不需服务器生成新视频。协议见 [week3_metrics.md](week3_metrics.md)。

## 文件

- `scripts/prepare_clip.py`：下载固定 revision 的 CLIP，保存许可及权重来源与哈希。
- `scripts/make_probes.py`：由三个固定来源生成 12 段无损探针；再登记一个 P02 配 P01 文本的候选反例。
- `scripts/metrics.py`：36 段基线 + 13 条探针记录，共 49 条指标。文本替换使用已有 P02 文件，不多生成一段视频。
- `scripts/test_metrics.py`：验证静止 SSIM=1、变换只影响预定范围、判定规则的边界行为。

## 首次运行顺序

```powershell
& 'E:\cstudy\AIGC\V\local\env\python.exe' -m pip install `
  scikit-image==0.25.2 --cache-dir 'E:\cstudy\AIGC\V\local\pip-cache'

& 'E:\cstudy\AIGC\V\local\env\python.exe' `
  'E:\cstudy\AIGC\V\scripts\prepare_clip.py'

& 'E:\cstudy\AIGC\V\local\env\python.exe' `
  'E:\cstudy\AIGC\V\scripts\test_metrics.py'

& 'E:\cstudy\AIGC\V\local\env\python.exe' `
  'E:\cstudy\AIGC\V\scripts\make_probes.py'

& 'E:\cstudy\AIGC\V\local\env\python.exe' `
  'E:\cstudy\AIGC\V\scripts\metrics.py'
```

已完成的步骤无需重复。探针目录或指标结果目录存在时脚本主动停止，保护既有结果。若要独立复算指标，传入新的 `--output`，不要删除原始实验：

```powershell
& 'E:\cstudy\AIGC\V\local\env\python.exe' `
  'E:\cstudy\AIGC\V\scripts\metrics.py' `
  --output 'E:\cstudy\AIGC\V\notes\week3_metrics_recheck'
```

默认 CPU FP32。这里的耗时是离线指标计算时间，不是视频生成延迟。

## 结果

| 路径 | 用途 |
|---|---|
| `outputs/eval/probes/*.mkv` | 12 段无损反例 |
| `notes/eval/probes/construction.json` | 构造参数、来源区域与脚本/协议哈希 |
| `notes/eval/probes/manifest.jsonl` | 13 条探针记录，含原始来源和视频哈希 |
| `notes/week3_metrics_run/metrics.csv` | 每条视频/文本记录的四个标量 |
| `notes/week3_metrics_run/per_video.jsonl` | 同时保留 80 个 SSIM 值和 21 个 CLIP 值 |
| `notes/week3_metrics_run/baseline_ranges.json` | 36 段原始基线的各指标范围 |
| `notes/week3_metrics_run/probe_comparison.csv` | 每个探针、每个指标的范围判定和来源差值 |
| `notes/week3_metrics_run/run.json` | 依赖版本、设备、输入/协议/脚本哈希、完成状态 |

CSV 为 UTF-8 BOM，可用 Excel 查看。正常对照直接使用原始基线；“baseline”只是未经本次人工退化处理，不代表画面质量已经合格。

探针表的“能区分”只表示超出本次基线范围；高分异常、低分异常分别记录，不等价于能自动判定质量。Day5 已收到四类像素退化的匿名试评并记录执行偏差，见 `notes/week3_trial_review.md`；P02 文本替换是否确实构成动作反例仍未单独验明，不冒充已确认结论。
