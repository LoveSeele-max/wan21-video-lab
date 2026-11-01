# Week4 SageAttention生成验收与自动指标

日期：2026-10-27，Asia/Shanghai。

## 生成与传输验收

读取 `notes/eval/sageattn.jsonl`、dataset/coverage、36份尝试收据、3份events和本机36段MP4：

- 冻结12条提示词×3种子完整覆盖，36条唯一ok，视频哈希全部匹配清单。
- 两张物理GPU各18段；每条均为attempt-01、ok，无失败和重试记录。
- events合计36 running、36 ok、1 skipped：GPU1的P01_s1001先检查成功，再由完整分片续跑核验后跳过，不重复计数。
- 身份指纹、配置、20步、sageattn后端、提示词和种子等与冻结内容一致，未残留样本锁。
- latent按计划不传回。本机没有独立核对latent，服务器summarize已按脚本检查其哈希；此边界保留。

审计：`notes/week4/generation_audit.json`。
本机清单SHA256：`cb657ad95dfaeea388de33883f4b2c5a92c901e3ada7ecdf587afb601bf4675c`。

## 指标计算与验证

新增 `scripts/metrics_candidate.py`，复用原 `metrics.py` 的视频解码、SSIM、CLIP及CSV函数。原metrics.py、冻结协议及第三周输出未修改，未重新生成视频、未安装依赖或下载模型。
入口检查第三周协议/计算脚本哈希、baseline清单、逐视频参考哈希、36对提示词及输入视频哈希、依赖版本一致后才运行。

本机实际执行：

```powershell
& 'E:\cstudy\AIGC\V\local\env\python.exe' `
  'E:\cstudy\AIGC\V\scripts\metrics_candidate.py' `
  --candidate sageattn `
  --output 'E:\cstudy\AIGC\V\notes\week4_metrics_sageattn'
```

已有输出目录禁止覆盖；本次已运行完毕，无需重复执行。如确需独立复算，明确使用新的output目录。

实际结果：

- CPU FP32，CLIP revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`，与第三周相同计算口径和依赖版本。
- 36条全部ok、0 error。解码函数严格检查每段81帧、480×832、16 fps。
- 保存2880个相邻帧SSIM值（36×80）、756个CLIP帧值（36×21），采样帧0/4/…/80。
- 本次离线指标计算约150.414秒，不是生成延迟，不计入性能优化收益。
- 本机从逐帧值复算四个标量，并逐条复算36对差值和绝对差值排序，全部一致；未重复进行第二次模型计算。

输出目录：`notes/week4_metrics_sageattn/`。

| 文件 | 内容 |
|---|---|
| run.json | 协议、函数脚本/入口哈希、依赖版本、输入和参考文件哈希、完成状态 |
| per_video.jsonl | 36条逐视频明细，含80个SSIM和21个CLIP值 |
| metrics.csv | 36条四个标量 |
| paired_comparison.csv | 同提示词同种子的baseline/candidate数值和candidate减baseline差值 |
| delta_rankings.json | 每指标全部36对的绝对差值排序，无新阈值、不筛掉样本 |

输出复算审计：`notes/week4/metrics_sageattn_audit.json`。

## 辅助异常线索

以下是对应指标的绝对差值前三名，用于评分结束后复查，不作为质量好坏标签；全部36对仍按既定计划评分，不因这些排名改样本、阈值或分数。

| 指标 | 样本 | candidate减baseline |
|---|---|---:|
| SSIM均值 | P01_s1001 | +0.009000 |
| SSIM均值 | P11_s2002 | -0.007926 |
| SSIM均值 | P02_s1001 | -0.007167 |
| SSIM最低值 | P11_s2002 | -0.028280 |
| SSIM最低值 | P06_s42 | +0.012701 |
| SSIM最低值 | P06_s1001 | -0.011715 |
| CLIP均值 | P02_s1001 | +0.016762 |
| CLIP均值 | P08_s42 | +0.007723 |
| CLIP均值 | P04_s2002 | +0.007339 |
| CLIP最低值 | P06_s1001 | +0.027601 |
| CLIP最低值 | P01_s1001 | +0.020123 |
| CLIP最低值 | P10_s42 | -0.014932 |

SSIM升高可能对应更少运动，CLIP不是动作正确性的充分判据；这些指标不能替代人工质量门槛。未将全局均值变化解释为质量改善或无损，也没有事后新增数值通过阈值。
匿名评分包不得包含配置身份、这些自动指标或异常排名，保留评分者独立判断。评分回收并归档后再解盲、对照线索，不自行改原答。

## 当前状态

Day4四项已验收；正式性能已通过本项目速度门槛，36/36生成覆盖及自动指标完成。
随后人工36对三维配对、固定12段三维绝对评分已回收，原件已归档；两次补充说明后，本轮解盲统计完成。
当前分类为结论不足：结构/符合度仍有未解决的相对答案和缺失元数据，不能用自动指标补齐。
详见 `notes/week4_quality_attention.md`、`notes/week4_result_attention.md`；不要求人工耗时。
