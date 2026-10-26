"""Summarize existing generation receipts; no model execution or rating changes."""
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = ROOT / "notes/eval/baseline.jsonl"
    raw = manifest.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    expected = {f"P{i:02d}_s{s}" for i in range(1, 13) for s in (42, 1001, 2002)}
    if len(rows) != 36 or {r["key"] for r in rows} != expected:
        raise ValueError("Expected the complete frozen 36-sample baseline")
    groups = defaultdict(list)
    intervals = []
    for row in rows:
        if row["status"] != "ok" or row["attempts"] != 1 or row["actual_steps"] != 20:
            raise ValueError("Budget assumes all samples succeeded on the first attempt")
        seconds = row["elapsed_seconds"]
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid duration")
        start, finish = (datetime.fromisoformat(row[k]) for k in ("started_at", "finished_at"))
        if start.utcoffset() is None or finish <= start or (finish-start).total_seconds() < seconds:
            raise ValueError("Invalid receipt interval")
        groups[row["runtime"]["physical_gpu"]].append((row, start, finish))
        intervals.append((start, finish))
    if set(groups) != {0, 1} or any(len(g) != 18 for g in groups.values()):
        raise ValueError("Expected 18 samples on each of two GPUs")
    devices = []
    for gpu, records in sorted(groups.items()):
        records.sort(key=lambda r: r[1])
        if any(a[2] > b[1] for a, b in zip(records, records[1:])):
            raise ValueError("Overlapping tasks on the same GPU")
        start, finish = min(r[1] for r in records), max(r[2] for r in records)
        devices.append({
            "gpu": gpu, "count": len(records), "first_start": start.isoformat(),
            "last_finish": finish.isoformat(), "span_seconds": (finish-start).total_seconds(),
            "generation_seconds_sum": sum(r[0]["elapsed_seconds"] for r in records),
            "receipt_seconds_sum": sum((r[2]-r[1]).total_seconds() for r in records),
        })
    durations = [r["elapsed_seconds"] for r in rows]
    total = sum(durations)
    overall_start = min(i[0] for i in intervals)
    overall_finish = max(i[1] for i in intervals)
    wall = (overall_finish - overall_start).total_seconds()
    result = {
        "analysis_date": "2026-10-22",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "planned": 36, "ok": 36, "total_attempts": sum(r["attempts"] for r in rows),
        "generation_mean_seconds": mean(durations),
        "generation_min_seconds": min(durations), "generation_max_seconds": max(durations),
        "generation_seconds_sum": total,
        "single_gpu_serial_estimate_seconds": total,
        "two_gpu_ideal_estimate_seconds": total / 2,
        "overall_first_start": overall_start.isoformat(),
        "overall_last_finish": overall_finish.isoformat(), "observed_span_seconds": wall,
        "per_gpu": devices, "human_time": "not measured; user removed human timing requirement",
        "note": "Wall time includes host loading, saving and receipt overhead; not active GPU time, energy, or benchmark latency.",
        "scenarios": [{"new_configs": n, "videos": 36*n,
                       "generation_seconds_sum_estimate": total*n,
                       "ideal_two_gpu_seconds_estimate": total*n/2,
                       "observed_span_scaled_seconds_reference": wall*n}
                      for n in (1, 2, 3)],
    }
    audit = ROOT / "notes/week3_cost_audit.json"
    audit.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Week 3 生成资源成本表", "",
        "2026-10-22。仅统计已有基线记录；评分量引用已冻结清单，不运行生成或决定配置上限。", "",
        "## 数据与口径", "",
        "- 来源：`notes/eval/baseline.jsonl`，36 个固定组合均为 ok，各仅一次尝试；GPU 0/1 各 18 段。",
        f"- 清单 SHA256：`{result['manifest_sha256']}`。",
        "- `elapsed_seconds` 包含每段模型加载、生成、清理及 latent/视频保存；不包括后续稳定文件复制、哈希和记录写盘。",
        "- `started_at` 到 `finished_at` 是每个任务的记录区间，包含生成及发布/校验开销。各卡首任务开始到末任务结束的跨度还包含任务间间隙。",
        "- 整体跨度由两卡最早任务开始到最晚任务结束计算；不包含进程启动前的准备、模型/环境预检查、传输或人工评分。",
        "- 累计任务用时是资源规划代理，包含 CPU 与磁盘时间，不是 CUDA 活跃时长、整卡租用时长、电量或费用。",
        "- 这些是双卡并发生成记录，不能作为单卡串行性能实测或优化加速结论。性能仍以相同配置的 bench.py 测量为准。", "",
        "## 本次基线实测", "",
        "| 项目 | 秒 | 分钟 |", "|---|---:|---:|",
        f"| 单段均值 | {mean(durations):.3f} | {mean(durations)/60:.3f} |",
        f"| 单段最短 | {min(durations):.3f} | {min(durations)/60:.3f} |",
        f"| 单段最长 | {max(durations):.3f} | {max(durations)/60:.3f} |",
        f"| 36 段累计生成保存用时 | {total:.3f} | {total/60:.3f} |",
        f"| 双卡整体任务跨度 | {wall:.3f} | {wall/60:.3f} |", "",
        f"整体区间：{overall_start.isoformat()} 至 {overall_finish.isoformat()}（+08:00）。", "",
        "| 卡 | 段数 | 累计生成保存（分钟） | 首任务至末任务跨度（分钟） |",
        "|---|---:|---:|---:|",
    ]
    for d in devices:
        lines.append(f"| GPU {d['gpu']} | {d['count']} | {d['generation_seconds_sum']/60:.3f} | {d['span_seconds']/60:.3f} |")
    lines += [
        "", "## 后续整套生成的参考量", "",
        f"- 单卡串行参考：36 段累计约 **{total/60:.2f} 分钟**。这是将本次并发下任务用时相加的估计，未做同批单卡串行实测。",
        f"- 两卡理想均分参考：约 **{total/120:.2f} 分钟**，假设同时启动、任务完全均衡且无额外间隙；本次实际任务跨度为 **{wall/60:.2f} 分钟**。",
        "- 下表仅用基线成本作场景估算，不预设优化更快；新配置的量化、卸载、失败重试、编译或后端行为可能增加成本。",
        "- 已有基线 36 段作为质量对照保留，下表只算新增配置，不再加算已有基线。当前配置上限见 `configs/optimization_rules.yaml` 与 `notes/week4_prereg.md`；本脚本仅计算场景，不决定或修改上限。", "",
        "| 新增配置数（场景） | 新视频数 | 累计任务用时参考（小时） | 理想双卡（分钟） | 按本次实际跨度线性外推（分钟） |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in result["scenarios"]:
        lines.append(f"| {s['new_configs']} | {s['videos']} | {s['generation_seconds_sum_estimate']/3600:.3f} | {s['ideal_two_gpu_seconds_estimate']/60:.2f} | {s['observed_span_scaled_seconds_reference']/60:.2f} |")
    lines += [
        "", "上述外推不含正式 bench 预热/重复计时、安装检查、质量指标、视频转码、传输和重试；因此不是完整项目总成本。", "",
        "## 评分量与预算边界", "",
        "- 人工评分时间：**未测，后续不要求**。2026-10-22 用户明确取消人工耗时要求，不追补、不设为验收条件，不估造人工费用。",
        "- 评分量已冻结：每配置 36 对配对比较、12 段三项绝对评分，固定子集沿用首次试评抽样；见 `configs/scoring_frozen.yaml` 和 `notes/week4_prereg.md`。",
        "- 配置上限与完整判定规则已另行冻结，见 `configs/optimization_rules.yaml` 和 `notes/week4_prereg.md`；本文件的 1/2/3 行仅为成本场景，不能用来扩大已登记范围。",
        "- 电价、整卡计费与设备功率没有测量或价格依据，不计算金额。", "",
        "## 复算", "",
        "用任一 Python 3.11+ 运行 `scripts/analyze_week3_cost.py`；仅标准库，输出本文件与 `notes/week3_cost_audit.json`，不修改清单或评分原件。",
        "复算检查：36 唯一预定组合、全部成功且仅一次尝试、有效计时、两卡各 18 段、各卡任务区间不重叠。", "",
    ]
    (ROOT / "notes/week3_cost.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
