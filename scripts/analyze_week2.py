"""Audit saved Week 2 JSON/CSV and inspect profiler trace without running models."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "notes" / "week2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    audit = {"sessions": {}}
    for folder in sorted(DATA.iterdir()):
        if not folder.is_dir() or not (folder / "session.json").is_file():
            continue
        session = read(folder / "session.json")
        saved_summary = read(folder / "summary.json")
        rows = []
        for path in sorted(folder.glob("*/result.json")):
            result = read(path)
            smi = result["external_gpu_sampling"]
            with (path.parent / "gpu.csv").open(encoding="utf-8") as file:
                samples = list(csv.reader(file))
            used = [float(row[2]) for row in samples]
            assert len(samples) == smi["samples"], path
            assert max(used) == smi["sampled_peak_used_mib"], path
            assert smi["warning"] is None, path
            assert result["status"] == "ok", path
            if result["mode"] == "stages":
                assert all(result["checks"][name] for name in (
                    "each_stage_called_once", "forward_count_matches",
                    "timing_gap_below_2_percent")), path
                assert result["forward_calls"] == 2 * result["effective_args"]["infer_steps"]
                assert len(result["forward_seconds"]) == result["forward_calls"]
                for metric in ("peak_allocated_bytes", "peak_reserved_bytes"):
                    assert max(item[metric] for item in result["stages"]) == result[metric]
                for name, saved in result["stage_summary"].items():
                    intervals = [item for item in result["stages"] if item["stage"] == name]
                    assert abs(sum(item["seconds"] for item in intervals) - saved["seconds"]) < 1e-8
                    for metric in ("peak_allocated_bytes", "peak_reserved_bytes"):
                        assert max(item[metric] for item in intervals) == saved[metric]
            rows.append({
                "run": path.parent.name, "warmup": result["warmup"],
                "seconds": result["elapsed_seconds"],
                "allocated_mib": result["peak_allocated_bytes"] / 2**20,
                "reserved_mib": result["peak_reserved_bytes"] / 2**20,
                "smi_peak_mib": max(used), "sample_count": len(samples),
                "forward_calls": result.get("forward_calls"),
                "timing_gap_percent": result.get("timing_gap_percent"),
            })
        measured = [row for row in rows if not row["warmup"]]
        assert session["status"] == "ok"
        assert len(measured) == saved_summary["measured_runs"]
        assert abs(statistics.mean(row["seconds"] for row in measured)
                   - saved_summary["elapsed_seconds"]["mean"]) < 1e-8
        raw_measured = [read(path) for path in sorted(folder.glob("*/result.json"))
                        if not read(path)["warmup"]]
        for stage, saved in saved_summary["stage_seconds"].items():
            values = [item["stage_summary"][stage]["seconds"] for item in raw_measured]
            assert abs(statistics.mean(values) - saved["mean"]) < 1e-8
            assert min(values) == saved["min"] and max(values) == saved["max"]
        audit["sessions"][folder.name] = {
            "bench_sha256": session["bench_sha256"],
            "config_sha256": session["config_sha256"],
            "mode": session["options"]["mode"], "runs": rows,
            "summary": saved_summary,
        }

    trace_path = next(DATA.glob("R5-*/run-01/trace.json"))
    trace = read(trace_path)
    events = trace["traceEvents"]
    categories = Counter(event.get("cat", "") for event in events)
    kernels = [event for event in events if event.get("cat") == "kernel"
               and event.get("ph") == "X"]
    by_name = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
    for event in kernels:
        group = by_name[event["name"]]
        group["count"] += 1
        group["duration_us"] += event["dur"]
    steps = [event for event in events if event.get("name", "").startswith("ProfilerStep#")]
    diagnostics = [
        {key: event.get(key) for key in ("name", "cat", "ph", "ts", "dur", "args")}
        for event in events
        if any(term in event.get("name", "") for term in
               ("Command Buffer Full", "Buffer Flush", "Activity Buffer Request",
                "CUPTI_ERROR", "dropped"))
    ]
    kernel_total = sum(event["dur"] for event in kernels)
    ordered = sorted(kernels, key=lambda event: event["ts"])
    union = 0
    previous_end = None
    for event in ordered:
        start, end = event["ts"], event["ts"] + event["dur"]
        union += max(0, end - max(start, previous_end if previous_end is not None else start))
        previous_end = max(end, previous_end if previous_end is not None else end)
    group_times = defaultdict(float)
    group_counts = Counter()
    for event in kernels:
        name = event["name"].lower()
        if "flash" in name and "fwd" in name:
            group = "flash_attention"
        elif "gemm" in name or "gemv" in name:
            group = "gemm_gemv"
        else:
            group = "other_kernels"
        group_times[group] += event["dur"]
        group_counts[group] += 1
    cpu_ids = {event.get("args", {}).get("External id") for event in events
               if event.get("cat") == "cpu_op"} - {None}
    linked = [event for event in kernels
              if event.get("args", {}).get("External id") in cpu_ids]
    linked_groups = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
    linked_total = sum(event["dur"] for event in linked)
    for event in linked:
        name = event["name"].lower()
        group = ("flash_attention" if "flash" in name and "fwd" in name
                 else "gemm_gemv" if "gemm" in name or "gemv" in name
                 else "other_kernels")
        linked_groups[group]["count"] += 1
        linked_groups[group]["duration_us"] += event["dur"]
    for group in linked_groups.values():
        group["percent_of_linked_kernel_sum"] = group["duration_us"] / linked_total * 100
    cpu_steps = [event for event in steps if event.get("cat") == "user_annotation"]
    gpu_steps = [event for event in steps if event.get("cat") == "gpu_user_annotation"]
    assert len(cpu_steps) == len(gpu_steps) == 4
    assert linked_groups["flash_attention"]["count"] == 240
    unlinked = [event for event in kernels
                if event.get("args", {}).get("External id") not in cpu_ids]
    first_gpu_step = min(event["ts"] for event in gpu_steps)
    assert all(event["ts"] + event["dur"] <= first_gpu_step for event in unlinked)
    audit["profiler"] = {
        "trace_path": str(trace_path), "categories": dict(categories),
        "steps": steps, "kernel_count": len(kernels), "kernel_duration_sum_us": kernel_total,
        "kernel_union_us": union,
        "linked_kernel_count": len(linked), "linked_kernel_duration_sum_us": linked_total,
        "linked_kernel_groups": dict(linked_groups),
        "unlinked_kernel_count": len(unlinked),
        "unlinked_kernel_duration_sum_us": sum(event["dur"] for event in unlinked),
        "unlinked_all_before_first_gpu_step": True,
        "cpu_steps": cpu_steps, "gpu_steps": gpu_steps,
        "kernel_groups": {
            group: {"duration_us": duration, "count": group_counts[group],
                    "percent_of_kernel_sum": duration / kernel_total * 100}
            for group, duration in group_times.items()},
        "top_kernels": sorted(
            [{"name": name, **value} for name, value in by_name.items()],
            key=lambda item: item["duration_us"], reverse=True)[:12],
        "diagnostic_counts": dict(Counter(event["name"] for event in diagnostics)),
        "diagnostic_examples": diagnostics[:3],
    }
    output = ROOT / "notes" / "week2_audit.json"
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print("All run status, sampling, stage checks and time summary audits passed.")
    print("Bench hashes:", sorted(set(item["bench_sha256"] for item in audit["sessions"].values())))
    print("Trace categories:", dict(categories))
    print("Steps:", [(item["name"], item.get("dur")) for item in steps])
    print("Kernel groups:", audit["profiler"]["kernel_groups"])
    print("Linked kernel groups:", dict(linked_groups))
    print("Kernel sum/union seconds:", kernel_total / 1e6, union / 1e6)
    print("Diagnostics:", audit["profiler"]["diagnostic_counts"])
    print("Diagnostic examples:", diagnostics[:1])
    print("Saved:", output)
    write_reports(audit)


def write_reports(audit):
    def session(prefix):
        return next(value for name, value in audit["sessions"].items()
                    if name.startswith(prefix + "-"))
    r1, r2, r3, r4, r5 = [session(name) for name in ("R1", "R2", "R3", "R4", "R5")]
    s2, s3, s4 = [item["summary"] for item in (r2, r3, r4)]
    total = s2["elapsed_seconds"]["mean"]
    def fmt(value, digits=3):
        return f"{value['mean']:.{digits}f} [{value['min']:.{digits}f}, {value['max']:.{digits}f}]"
    names = [
        ("load_text_encoder", "T5 加载"),
        ("prepare_t2v_inputs", "文本准备（不含 T5 loader）"),
        ("load_dit_models", "DiT 加载"),
        ("run_sampling", "去噪"),
        ("load_vae", "VAE 加载"),
        ("decode_latent", "VAE 解码（不含 loader）"),
        ("save_latent", "保存 latent"),
        ("save_video", "保存视频"),
        ("other", "其他"),
    ]
    stage_lines = [
        "| 阶段 | R2 秒：均值 [最小, 最大] | R2 占比 | R3 秒：均值 [最小, 最大] |",
        "|---|---:|---:|---:|",
    ]
    memory_lines = [
        "| 阶段 | allocated 峰值 MiB | reserved 峰值 MiB |",
        "|---|---:|---:|",
    ]
    for name, label in names:
        value = s2["stage_seconds"][name]
        stage_lines.append(f"| {label} | {fmt(value)} | {value['mean']/total*100:.2f}% | {fmt(s3['stage_seconds'][name])} |")
        a = s2["stage_peak_allocated_bytes"][name]
        r = s2["stage_peak_reserved_bytes"][name]
        memory_lines.append(f"| {label} | {a['mean']/2**20:.1f} | {r['mean']/2**20:.1f} |")
    overhead = total - s4["elapsed_seconds"]["mean"]
    baseline = [
        "# Week 2 服务器基线与阶段分解",
        "",
        "整理日期：2026-10-15。R1–R5 原始文件已传回并核对；本机 16G 实测继续暂缓。"
        "本文的阶段表使用 R2，R3 作新进程同卡复测；最终端到端比较用 R4。",
        "",
        "## 固定条件",
        "",
        "Wan2.1 T2V 1.3B，BF16，torch attention，UniPC，480×832（高×宽），81 帧，16 FPS，"
        "20 步，seed 42，guidance 5，flow shift 5。不开 FP8、卸载、编译和 LoRA。",
        "",
        "提示词：`A person waving one hand, plain background, static camera.`"
        " 负面提示词沿用上游默认值。",
        "",
        "服务器 GPU 0，RTX 5880 Ada 49140 MiB；torch 2.9.1+cu128、Python 3.11.17，"
        "离线模式、native allocator。上游 commit "
        "`f8a1b03794a49239a3539015075f5123d6c07d66`。分词器使用已验证的本地快照及 subfolder 兼容包装。",
        "",
        "全部 session 的 bench SHA256 一致："
        "`ffc9af27120d7fd0f903b1482122a393c39a372aa0be0f4eed67873851ea775c`。"
        "有效配置以各 session.json 和 result.json 为准；2 步 verify 与 20 步正式配置不同。",
        "",
        "## 端到端基线",
        "",
        "| 运行 | 正式次数 | 秒：均值 [最小, 最大] | 含义 |",
        "|---|---:|---:|---|",
        f"| R1 | 1 | {r1['summary']['launch_to_video_saved_seconds']['mean']:.3f} | 父进程启动子进程前至保存，含导入和初始化 |",
        f"| R1 生成部分 | 1 | {r1['summary']['elapsed_seconds']['mean']:.3f} | generate 至 latent/视频保存 |",
        f"| R2 | 3 | {fmt(s2['elapsed_seconds'])} | 分段与逐前向计时 |",
        f"| R3 | 3 | {fmt(s3['elapsed_seconds'])} | 新进程重复 R2 |",
        f"| R4 | 3 | {fmt(s4['elapsed_seconds'])} | 无分段和逐前向包装，端到端主基线 |",
        "",
        "R2/R3/R4 各预热 1 次，均不纳入正式均值。模型每次重新加载，"
        "这里的预热后结果不代表模型常驻；也没有清系统文件缓存，R1 不是磁盘冷启动。"
        "R4 仍包含本地分词器兼容处理及 100ms 外部采样。",
        "",
        f"R2 与 R3 均值相差 {abs(total-s3['elapsed_seconds']['mean'])/total*100:.3f}%。"
        f"R2 减 R4 为 {overhead:.3f} 秒（相对 R4 {overhead/s4['elapsed_seconds']['mean']*100:.3f}%），"
        "只作插桩影响的净差估计，不能把它全部归因于插桩。"
        "R2/R3/R4 三组范围重叠，顺序运行、温度、频率及系统负载未做受控交错实验。",
        "",
        "## 阶段时间",
        "",
        *stage_lines,
        "",
        "各区间平铺且互不重叠，嵌套 loader 已单独扣出。同名区间耗时求和，"
        "显存取最大值。文本准备含编码、T5 搬移/释放和噪声准备，不应称为纯文本编码。",
        "",
        "| 算术口径 | R2 秒：均值 [最小, 最大] | R3 秒：均值 [最小, 最大] |",
        "|---|---:|---:|",
        f"| 含加载 | {fmt(s2['elapsed_seconds'])} | {fmt(s3['elapsed_seconds'])} |",
        f"| 扣三个 loader 区间 | {fmt(s2['excluding_load_intervals_seconds'])} | {fmt(s3['excluding_load_intervals_seconds'])} |",
        "",
        "第二行只是扣除 load_text_encoder、load_dit_models、load_vae 区间，"
        "仍可能包含搬移/清理，不能当作模型常驻性能。R4 没有分段，不能直接填它的扣加载数字。",
        "",
        "## 前向时间",
        "",
        f"- R2：每轮 40 次前向，三轮共 120 次；均值 [最小, 最大] {fmt(s2['forward_seconds'])} 秒。",
        f"- 相邻两次前向之和：共 60 对，{fmt(s2['paired_forward_seconds'])} 秒。",
        f"- 去噪阶段总耗时除以 20：R2 均值 {s2['stage_seconds']['run_sampling']['mean']/20:.3f} 秒/步。",
        "",
        "两次前向之和不包含 scheduler、数据准备和其他循环操作，"
        "不等于完整单步。逐前向数字有同步计时开销，仅用于定位。"
        "这些 120 次前向不是 120 次独立生成，不能据此夸大统计样本量。",
        "",
        "## 显存",
        "",
        *memory_lines,
        "",
        "表内是 R2 三次正式运行各阶段的绝对峰值；每个阶段的三次值均相同。"
        "reserved 是保留缓存，解码后的其他/保存阶段仍能维持相同 reserved；"
        "不能据此说保存视频创建了相同大小的新分配，也不能将 reserved 与 allocated 相加。",
        "",
        "| 对照 | allocated MiB | reserved MiB | 整卡采样峰值 MiB |",
        "|---|---:|---:|---:|",
    ]
    for label, sess in (("R1", r1), ("R2 三轮", r2), ("R3 三轮", r3), ("R4 三轮", r4)):
        rows = [row for row in sess["runs"] if not row["warmup"]]
        baseline.append(f"| {label} | {rows[0]['allocated_mib']:.1f} | {rows[0]['reserved_mib']:.1f} | "
                        f"{min(row['smi_peak_mib'] for row in rows):.0f}–{max(row['smi_peak_mib'] for row in rows):.0f} |")
    baseline.extend([
        "",
        "R2/R3 各阶段最大值与 R4 独立总峰值本次恰好一致：allocated 约 14.35 GiB，"
        "reserved 约 18.23 GiB。allocated 最高区间是 VAE 解码，尚未定位到其内部具体分配。"
        "nvidia-smi 为整卡口径，包含桌面、CUDA/驱动等，不把与 allocator 的差额全部算作桌面。"
        "采样每 100ms，可能包括计时外短暂区间，也可能漏瞬时峰值。"
        "这些数字不能推出本机 16G 的可运行性。",
        "",
        "## 数据验收与范围",
        "",
        "已解析并核对全部 verify、R1–R5 的 session/result、CSV 和时间汇总。"
        "阶段模式的模块调用计数、每步两次前向、计时覆盖检查均通过；"
        "CSV 的行数及峰值与 JSON 相符。检查脚本为 `scripts/analyze_week2.py`，"
        "机器可读核对结果为 [week2_audit.json](week2_audit.json)。",
        "",
        "本次单提示词、单生成种子、单配置，R2–R4 各 3 次正式生成，"
        "没有性能优化或独立质量评估。视频保存成功不等于动作自然。"
        "R5 诊断和后续候选见 [week2_profile.md](week2_profile.md)。",
    ])
    (ROOT / "notes/week2_baseline.md").write_text("\n".join(baseline) + "\n", encoding="utf-8")

    prof = audit["profiler"]
    groups = prof["linked_kernel_groups"]
    profile = [
        "# Week 2 profiler 与瓶颈结论",
        "",
        "整理日期：2026-10-15。数据来自 `R5-profile-20261015-221034-3813a433/run-01/`。"
        "本报告分析现有 trace，不新增 GPU 运行。",
        "",
        "## 捕获范围",
        "",
        "完整生成 20 步，40 次 DiT 前向。schedule 为 wait16/warmup2/active4，"
        "按前向推进。trace 中 CPU 标记和 GPU 标记各有 ProfilerStep#18–#21 共 4 个，"
        "CPU/GPU 是同一组逻辑前向的两侧，不能当作 8 次前向。"
        "目标范围对应第 10、11 个采样步的条件/无条件分支。",
        "",
        "trace 约 48.3 MB，包含有效 CUDA kernel 事件。"
        "R5 生成保存耗时 142.502 秒，仅作诊断，不作为延迟比较。",
        "",
        "## 为什么不直接抄 operators.txt 的百分比",
        "",
        "`operators.txt` 同时列出了 ProfilerStep、ATen 算子、CUDA kernel 和 CUPTI overhead。"
        "例如 aten::_flash_attention_forward 与底层 flash_fwd_kernel 是同一工作不同层级，"
        "两行时间不能相加；ProfilerStep 也不是一种模型算子。",
        "",
        "trace 中共有 17036 条 kernel，264 条 flash 内核。"
        "其中 996 条 kernel 没有关联到本次捕获的 CPU 算子 External id，"
        "全部发生在首个 GPU ProfilerStep 开始前，含 24 条 flash。"
        "它们是捕获边界前的异步 GPU 工作，不能混入 4 次正式前向的算子表。",
        "",
        "因此采用保守的关联过滤：只统计 External id 能对应本次捕获 CPU 算子的 kernel，"
        "排除 CPU 事件、annotation、overhead、gpu_memcpy、gpu_memset。"
        f"保留 {prof['linked_kernel_count']} 条 kernel，时长之和 "
        f"{prof['linked_kernel_duration_sum_us']/1e6:.6f} 秒。"
        "关联后的 flash 内核 240 次，与 4 前向 × 30 层 × 2 个 attention 的数量一致。"
        "这是 GPU 内核忙碌时间的分类占比，不是整个生成墙钟占比，也不包含所有复制/等待。",
        "",
        "| 内核分类 | 事件数 | 时长之和（秒） | 关联内核时长占比 |",
        "|---|---:|---:|---:|",
    ]
    labels = {"flash_attention": "flash attention 前向内核",
              "gemm_gemv": "名称匹配 GEMM/GEMV 的矩阵乘",
              "other_kernels": "其他内核"}
    for group in ("flash_attention", "gemm_gemv", "other_kernels"):
        value = groups[group]
        profile.append(f"| {labels[group]} | {value['count']} | {value['duration_us']/1e6:.6f} | "
                       f"{value['percent_of_linked_kernel_sum']:.2f}% |")
    profile.extend([
        "",
        "矩阵乘按 kernel 名称含 gemm/gemv 分组，是线性层开销的代理，不是全部线性层耗时。"
        "其他类含逐元素、归一化、变换等内核，不声称它们都值得或都能融合。"
        "完整事件与可复查过滤代码见原始 trace、`scripts/analyze_week2.py` 和 `notes/week2_audit.json`。",
        "",
        "## 采集干扰",
        "",
        "trace 记录了 4283 次 Command Buffer Full、2 次 Activity Buffer Request、3 次 Buffer Flush；"
        "原表的 Command Buffer Full CPU 总时长约 5.957 秒。"
        "这提示采集存在显著缓冲/工具开销，不能把该行当作模型瓶颈，"
        "也不能直接从墙钟减掉它。未独立证实丢事件与否，不承诺本次是无损采集。",
        "",
        "过滤后的 attention 仍是最大类，足够支持先把 attention 实现作为候选；"
        "但只抓两个中段采样步，且有工具干扰，不外推所有时间步、分辨率或提示词。"
        "后续候选是否有效以关闭 profiler 的端到端配对测量和质量检查为准。",
        "",
        "## 瓶颈结论",
        "",
        f"固定当前 480P、81 帧、20 步配置，R4 预热后生成并保存平均 {s4['elapsed_seconds']['mean']:.3f} 秒。"
        f"R2 的去噪区间平均 {s2['stage_seconds']['run_sampling']['mean']:.3f} 秒，"
        f"占该模式总时间 {s2['stage_seconds']['run_sampling']['mean']/total*100:.2f}%，是主要时间来源；"
        f"T5 loader 约 {s2['stage_seconds']['load_text_encoder']['mean']:.3f} 秒、"
        f"VAE 解码约 {s2['stage_seconds']['decode_latent']['mean']:.3f} 秒。"
        "allocated 总峰值 14698.4 MiB 出现在 VAE 解码，去噪峰值约 4183.5 MiB，"
        "时间瓶颈与显存峰值来源不同。R5 关联内核中 attention 约 53.90%、"
        "名称匹配的 GEMM/GEMV 约 19.89%，支持首先评估去噪计算路径；"
        "没有测过任何加速收益，VAE 内部显存来源也尚未定位。",
        "",
        "## 后续两组对照的候选顺序",
        "",
        "1. attention 实现对照优先。当前配置虽叫 torch，trace 已显示使用 pytorch_flash 内核；"
        "不能说基线是朴素 attention，也不能假定切到 flash 包就会加速。"
        "可在兼容性和安装预算内选一个替代实现，其他条件固定，"
        "先用很小样本验证，再按冻结的评测集比较端到端、显存与质量。"
        "torch/sdpa 不重复测，flash/flash2 不重复测。",
        "2. DiT 精度/矩阵乘路径列为第二候选：先依据已核实源码选单一 FP8 配置，"
        "只改变这一变量。若选择 fp8_scaled + fp8_fast，作为一个明确组合配置记录，"
        "不把普通 fp8 存储变化等同于硬件矩阵乘加速；端到端收益和质量代价都需实测。"
        "若安装/量化路径超预算或未见收益，记录结果后再考虑 compile 等候选，不扩大扫参。",
        "",
        "VAE 是后续本人性能代码改动的显存候选，暂不承诺改法。"
        "须进一步辨别全分辨率激活、缓存和输出 torch.cat 累积各自影响，"
        "不能看到 VAE 峰值就认定 cat 是罪因；时间维已逐 latent 帧解码，不重做时间分块。"
        "vae_cache_cpu 可作为定位缓存贡献的对照，但不预设有收益。",
        "",
        "## 完成范围",
        "",
        "Week 2 服务器测量出口已满足：脚本、R1–R5 原始记录、重复运行、"
        "阶段表、分段/总峰值核对、采样、插桩影响估计、profiler 分类和限定范围的瓶颈结论。"
        "保留 profiler 缓冲开销及边界过滤限制。没有完成性能优化、质量判据、训练或本机 16G 实测。"
        "正式对照前先做 Week 3 的评测判据与提示词/种子冻结，不直接开始扫参数。",
    ])
    (ROOT / "notes/week2_profile.md").write_text("\n".join(profile) + "\n", encoding="utf-8")
    print("Reports saved: notes/week2_baseline.md, notes/week2_profile.md")


if __name__ == "__main__":
    main()
