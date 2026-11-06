"""Build Week5 reports from immutable unblinded answers and prior audits."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "notes/blind_private/week5-analysis-20261106"
LABELS = {"structure": "结构", "temporal_motion": "时序与运动", "prompt_alignment": "提示词符合度"}
PACKS = ("week5-sage-followup-R01", "week5-fp8-R01", "week5-fp8s-R01")
NAMES = ("SageAttention", "普通 FP8", "scaled FP8")
KEYS = ("sageattn", "fp8", "fp8_scaled")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, text):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def quality_table(result):
    lines = [
        "| 维度 | 完整相对答案/36 | 明显更差 | 轻微更差 | 无明显差异 | 有差异难判优劣 | 轻微/明显更好 | 更差/更好但程度未定 | 无法判断 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dim, s in result["summary"].items():
        c = s["counts"]
        lines.append(f"| {LABELS[dim]} | {s['scorable_complete_answers']}/36 | "
                     f"{c['marked_worse']} | {c['mild_worse']} | {c['no_visible_difference']} | "
                     f"{c['different_no_clear_advantage']} | {c['mild_better']}/{c['marked_better']} | "
                     f"{c['worse_severity_unspecified']}/{c['better_severity_unspecified']} | {c['unjudgeable']} |")
    return "\n".join(lines)


def independent_check(results):
    checks = []
    for result in results:
        snapshot_path = RUN / f"{result['pack']}-effective-answers.json"
        assert digest(snapshot_path) == result["effective_answer_snapshot_sha256"]
        mapping = load(ROOT / "notes/blind_private" / result["pack"] / "mapping.json")
        identity = {t["id"]: t for t in mapping["tasks"]}
        if result["candidate"] == "sageattn":
            assert len(result["replacements"]) == 8
            assert result["unchanged_week4_dimensions"] == 100
            assert len(result["control_consistency"]) == 12
            assert all(not c["opposite_direction"] and not c["degree_gap_at_least_two"]
                       and not c["comparison_unavailable"] for c in result["control_consistency"])
        else:
            for row in result["pairs_by_dimension"]:
                direction, degree = row["effective_direction"], row["effective_severity"]
                if direction in ("A更好", "B更好"):
                    candidate_wins = identity[row["id"]][direction[0]]["kind"] == result["candidate"]
                    expected_suffix = "better" if candidate_wins else "worse"
                    assert row["category"].endswith(expected_suffix) or (
                        row["category"] == f"{expected_suffix}_severity_unspecified")
                    if degree in ("明显", "轻微"):
                        assert row["category"].startswith("marked" if degree == "明显" else "mild")
                elif direction == "无法判断":
                    assert row["category"] == "unjudgeable"
                else:
                    assert row["category"] in ("no_visible_difference", "different_no_clear_advantage")
        for dim, s in result["summary"].items():
            rows = [r for r in result["pairs_by_dimension"] if r["dimension"] == dim]
            counts = Counter(r["category"] for r in rows)
            assert len(rows) == len({r["source_key"] for r in rows}) == 36
            assert sum(s["counts"].values()) == 36
            assert all(counts[k] == v for k, v in s["counts"].items())
            complete = [r for r in rows if r["category"] not in (
                "unjudgeable", "unresolved", "worse_severity_unspecified", "better_severity_unspecified")]
            assert s["scorable_complete_answers"] == len(complete)
            # Even if every unknown degree were mild, four worse directions cannot pass.
            could_pass_with_unknown_degrees = (
                counts["marked_worse"] == 0 and
                counts["mild_worse"] + counts["worse_severity_unspecified"] <= 3)
            checks.append({"candidate": result["candidate"], "dimension": dim,
                           "counts_independently_matched": True,
                           "could_meet_worse_limits_if_all_unknown_degrees_mild": could_pass_with_unknown_degrees})
    audit = load(RUN / "analysis_audit.json")
    for relative, expected in audit["pre_unblind_inputs_sha256"].items():
        path = ROOT / relative
        if Path(relative).name == "analyze_week5_ratings.py":
            path = RUN / "analyzer-before-source-hash-fix.py"
        assert digest(path) == expected
    for relative, expected in audit["mapping_hashes"].items():
        assert digest(ROOT / relative) == expected
    return checks


def main():
    results = [load(RUN / f"{pack}-quality-analysis.json") for pack in PACKS]
    verification = independent_check(results)
    perf = load(ROOT / "notes/week4/fp8_total_audit.json")["pooled"]
    stages = load(ROOT / "notes/week4/fp8_stages_audit.json")["cases"]
    sage_stages = load(ROOT / "notes/week4/A4_audit.json")["formal_stage_table"]
    # Final cards use the frozen priority: reported marked regressions precede missing degree.
    classifications = {
        "sageattn": "结论不足（质量数值门槛达到，关键观看/会话记录不完整）",
        "fp8": "不推荐（当前任务/目标；有明确填写的明显退化）",
        "fp8_scaled": "不推荐（当前任务/目标；有明确填写的明显退化）",
    }
    card_paths = ("week5_result_attention_v2.md", "week5_result_fp8.md", "week5_result_fp8_scaled.md")
    machine_verdicts = {}
    for result, title, card_path in zip(results, NAMES, card_paths, strict=True):
        key = result["candidate"]
        machine_verdicts[key] = {
            "classification": classifications[key],
            "quality_numerical_thresholds_all_met": all(
                s["quality_gate"] == "numerical_thresholds_met_metadata_limits_remain"
                for s in result["summary"].values()),
            "has_reported_marked_worse": any(
                s["counts"]["marked_worse"] for s in result["summary"].values()),
            "critical_viewing_metadata_complete": False,
        }
        if key == "sageattn":
            assert not machine_verdicts[key]["has_reported_marked_worse"]
            assert machine_verdicts[key]["quality_numerical_thresholds_all_met"]
            header = """| 字段 | 结果 |
|---|---|
| 配置/路径 | `eval_sageattn.yaml`；SageAttention 2.2.0，`sageattn_varlen` Triton |
| 环境/范围 | 独立 env-sagecheck，RTX 5880 Ada；P01/s42，480×832，81帧，16fps，20步 |
| 同环境正式 total | torch 135.903937秒 → Sage 125.059046秒；各3次，预热各1次 |
| min–max | torch 134.493361–137.391304秒；Sage 123.918723–125.742168秒 |
| 速度门槛 | 耗时下降7.9798%，达到5%且闭区间不重叠 |
| 去噪阶段诊断 | 111.187301 → 100.198346秒；allocated峰值约4183.45 MiB |
| 整次显存 | allocated 14698.3535 MiB；reserved 18666 MiB；整卡100ms采样19627 MiB；均未下降 |
| 可靠性/指标 | 36/36首次生成成功；36条指标成功；不作为质量保持证据 |
| 本轮变化 | 只替换八个原未解决维度，另外100个Week4维度答案原样保留 |
| 对照检查 | 四个对照对、十二个维度均与原相对类别一致；方向相反和程度差≥2档均为0 |
"""
            extra = """## 补评与判定

补评包十对的原始评分格均为空，本次用户确认其含义为“无差异”，据此补齐六十格。
只并入六个争议视频对上的八项原未解决维度：结构两项、符合度六项。
这不是新增三十六对，也不替换其他已解决答案。四个对照只检查一致性，不改旧分。

三个维度现均为36/36完整相对答案，明显变差均0、轻微变差分别0/1/0，达到原数值门槛。
时序的轻微改善1项不抵消轻微退化1项。提示词符合度另有1项“有差异但难判优劣”，不合并为无明显差异。

最终仍记“结论不足”：原“可用”规则要求关键记录完整，但实际播放器、正常速度观看、
查看方式、具体会话日期和问题时间等未补齐，原绝对样本的分数/文字疑点也未被本次替换。
可以报告“速度收益已确认、质量数值计数达标”；不能报告严格独立盲评、无损或质量等价。
本次不是确认明显退化，也不要求为了追求可用标签重跑性能或补造观看记录。
"""
        else:
            v = perf[key]
            baseline = perf["baseline"]["elapsed_seconds"]
            elapsed = v["elapsed_seconds"]
            stage = stages[key]["stage_table"]
            base_mem = stages["baseline"]["stage_table"]["run_sampling"]["allocated_mib"]
            mem = stage["run_sampling"]["allocated_mib"]
            reduction = (base_mem - mem) / base_mem * 100
            mechanism = ("DiT直接cast为float8_e4m3fn；标准Linear/BF16 autocast，未启用专用FP8乘法分支"
                         if key == "fp8" else
                         "目标Linear保存FP8权重及scale；显式反量化为x.dtype后F.linear；日志300层")
            header = f"""| 字段 | 结果 |
|---|---|
| 配置/路径 | `eval_{key}.yaml`；fp8_fast=false；torch attention |
| 实际机理 | {mechanism} |
| 环境/范围 | 主环境env，RTX 5880 Ada；P01/s42，480×832，81帧，16fps，20步 |
| 正式total | BF16 {baseline['mean']:.6f}秒 → {title} {elapsed['mean']:.6f}秒；累计各6次 |
| min–max | BF16 {baseline['min']:.6f}–{baseline['max']:.6f}秒；候选 {elapsed['min']:.6f}–{elapsed['max']:.6f}秒 |
| 耗时变化 | 候选相对基线 {(elapsed['mean']/baseline['mean']-1)*100:+.4f}%；5%速度门槛未达，不再追加total |
| 去噪allocated峰值 | {base_mem:.4f} → {mem:.4f} MiB，下降{reduction:.2f}%；单次阶段诊断 |
| DiT加载allocated峰值 | {stage['load_dit_models']['allocated_mib']:.4f} MiB |
| 整次显存 | allocated 14698.3535 MiB；reserved 18666 MiB；整卡100ms采样19627 MiB；均未下降 |
| 可靠性/指标 | 36/36首次生成成功，无重试；36条指标成功；不作为质量通过证据 |
| 评分间隔 | 用户确认普通/scaled两包相隔两天；具体日期未提供 |
"""
            if key == "fp8":
                extra = """## 退化记录与未定程度

T027（P01_s1001）原填写三维度均“A更好／明显”，解盲后A为基线、B为普通FP8。
因此三维度各有1个评分者明确填写的明显变差，来自同一个视频对，不能称为三个独立失败视频。
原证据为“A手抽搐，B人物背对和提示词不一致”；它对符合度的说明比结构/时序更直接。
保留三个原评分，不用一句文字补造三个维度的独立证据；助手没有重新观看并验证这些判断。

T004、T005、T025、T046时序按用户确认保留方向，但轻微/明显未提供，
解盲后为更差程度未定2项、更好程度未定2项。T022按已填方向记“无法判断”。
时序完整相对答案31/36；另外四个方向不因程度缺失而消失，也不合并为轻微。

按冻结优先级，明确填写的明显变差已触发“不推荐”，不因尚有未定程度而掩盖该负面记录。
普通FP8的DiT加载峰值较BF16反升约51.48%，原因尚未定位；不将重叠存储假说写成实证。
"""
            else:
                extra = """## 退化记录与未定程度

T012（P04_s1001）时序原填写“B更好／明显”；T043（P01_s1001）时序和符合度
原填写“B更好／明显”。解盲后这两对均A为scaled FP8、B为基线。
因此时序明确填写的明显变差为2对、符合度为1对；这些记录涉及两个唯一视频对。

T012证据“A出现了两个人”更直接说明结构/符合度，T043证据“A只有手指晃动”
更直接说明目标动作是否发生，不能仅凭它们替评分者补造时序退化的独立解释。
原方向/程度保留，报告为评分者的判断；助手没有重新观看并验证。

T007的结构、时序均按已填“B更好”保留方向，程度仍未定；解盲后两项均为scaled更差。
结构已有轻微更差3项，加这1项确定退化方向，即使未定程度最终是轻微也成为4项，
若是明显又触发明显退化门槛。因此结构也不可能凭缺失程度被判通过；不需要猜档位。

时序和符合度已有明确填写的明显变差，按冻结优先级最终“不推荐”。
不能因为12段单视频全部2分而抵消36对配对中的退化；两者覆盖范围和用途不同。
"""
        absolute = "\n".join(
            f"- {LABELS[dim]}：" + "，".join(f"{score} {count}段" for score, count in dist.items())
            for dim, dist in result["absolute_distributions"].items())
        text = f"""# Week5 {title}结论卡{' v2' if key == 'sageattn' else ''}

日期：2026-11-06，Asia/Shanghai。最终分类：**{classifications[key]}**。

{header}
## 逐维质量计数

{quality_table(result)}

每维预定36对，不能跨维相加为样本数；也不以改善抵消退化。
“完整相对答案”要求方向及有效程度齐全，不等于证实完成正常速度观看。
各类别以36和该维完整可评分数为分母的比例，见[质量汇总](week5_quality_summary.md)；
未定程度不进入完整可评分分母。

{extra}
## 固定十二段绝对评分

{absolute}

绝对评分描述候选本身，不能自动证明相对基线新增退化或没有退化。
空白最高分的来源为评分者2026-11-05确认，原始表及逐格补填清单保留。

## 证据与适用边界

单评分者、有损评分代理、历史接触及缺失观看/会话元数据均限制推断。
速度只覆盖上述固定P01/s42；Sage与FP8使用不同环境，仅比较各自同环境变化率。
阶段显存不等于整次显存；D4允许阶段收益但不豁免质量门槛，组合配置未验证。
本轮不新增生成，不改Week4原件、原预登记和原卡。

- [评分回收与澄清](week5_rating_review.md)
- [原预登记](week4_prereg.md)、[Week5补充](week5_prereg_addendum.md)
- 性能原审计：`notes/week4/{'A1_audit.json、A2_audit.json、A4_audit.json' if key == 'sageattn' else 'fp8_total_audit.json、fp8_stages_audit.json、fp8_generation_audit.json'}`
- 私人机器明细：`notes/blind_private/week5-analysis-20261106/{result['pack']}-quality-analysis.json`
"""
        save_new(ROOT / "notes" / card_path, text)

    # Pair every preregistered top-five metric anomaly with every rated dimension.
    crosscheck_rows = []
    for result in results[1:]:
        key = result["candidate"]
        rankings = load(ROOT / f"notes/week5_metrics_{key}/delta_rankings.json")
        by_source = {}
        for row in result["pairs_by_dimension"]:
            by_source.setdefault(row["source_key"], {})[row["dimension"]] = row["category"]
        for metric, ranked in rankings.items():
            for rank, sample in enumerate(ranked[:5], 1):
                assert sample["id"] in by_source
                crosscheck_rows.append({
                    "candidate": key, "metric": metric, "rank": rank, "source_key": sample["id"],
                    "delta": sample["delta"], **by_source[sample["id"]],
                })
    target = RUN / "metrics_quality_crosscheck.csv"
    if target.exists():
        raise FileExistsError(target)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(crosscheck_rows[0]))
        writer.writeheader()
        writer.writerows(crosscheck_rows)
    assert len(crosscheck_rows) == 40

    metric_notes = []
    for key in ("fp8", "fp8_scaled"):
        relevant = [r for r in crosscheck_rows if r["candidate"] == key]
        marked_keys = {r["source_key"] for r in next(
            x for x in results if x["candidate"] == key)["pairs_by_dimension"]
            if r["category"] == "marked_worse"}
        top_keys = {r["source_key"] for r in relevant}
        metric_notes.append(
            f"- {key}：四项指标各Top5合计覆盖{len(top_keys)}个唯一样本；"
            f"与明确填写的明显变差样本交集为{len(top_keys & marked_keys)}个"
            f"（{', '.join(sorted(top_keys & marked_keys)) or '无'}）。")

    ratio_lines = [
        "| 配置 | 维度 | 完整答案n | 明显变差 n/36；n/完整 | 轻微变差 n/36；n/完整 | 无明显差异 n/36；n/完整 |",
        "|---|---|---:|---|---|---|",
    ]
    for result, name in zip(results, NAMES, strict=True):
        for dim, summary in result["summary"].items():
            n = summary["scorable_complete_answers"]
            cells = []
            for category in ("marked_worse", "mild_worse", "no_visible_difference"):
                count = summary["counts"][category]
                cells.append(f"{count}/36 ({count/36:.2%})；{count}/{n} ({count/n:.2%})")
            ratio_lines.append(f"| {name} | {LABELS[dim]} | {n} | " + " | ".join(cells) + " |")
    save_new(ROOT / "notes/week5_quality_summary.md", f"""# Week5 三候选质量汇总

日期：2026-11-06，Asia/Shanghai。本表是固定评分与原规则的描述性分析，不是统计等价检验。

{chr(10).join(ratio_lines)}

以上三个类别之外的计数分别见三张结论卡；全部十类的双分母比率保存在私人机器JSON。
未定程度、无法判断和未解决状态都以预定36报告，不使用“完整可评分”分母稀释缺失。

## 一次澄清的处理

原话“按照已填方向为准，评分相隔两天”在解盲前另存。
T022最终记无法判断，其他六项A/B方向确定而程度未定，不代填轻微/明显。
评分间隔按用户确认的两天记录为满足要求，具体日期、播放器和观看方式不推造。
三份有效答案快照完成后才打开Week5映射，源视频与评分代理SHA256全部核对。
Sage仅替换八个原缺口，保留另外100个维度；四个对照十二个维度类别均一致。

## 自动指标与人工答案对照

{chr(10).join(metric_notes)}

每配置按事先保留的四指标Top5逐项对照，共40行，全部来源和三个维度类别见
`notes/blind_private/week5-analysis-20261106/metrics_quality_crosscheck.csv`。
不据交集比例筛除样本，不设置事后指标阈值。SSIM是相邻帧结构指标，
CLIP是文本图像代理相似度，均不能代替配对评分或证明质量等价。

## 原文字证据的局限

普通FP8 T027的文字主要解释提示词符合度，却给三个维度均明显档；
scaled T012的“出现两个人”、T043的“只有手指晃动”也未分别解释全部维度。
原评分完整保留，明确标记为评分者报告的退化，不宣称助手已独立复看确认。
既有绝对缺陷与相对增量不同，单视频最高分不能抵消其他来源的配对退化。

## 结论入口

- [Sage v2](week5_result_attention_v2.md)：速度已确认、质量数值门槛达到，关键记录不完整，综合结论不足。
- [普通FP8](week5_result_fp8.md)：已有评分者明确填写的明显退化，当前任务不推荐。
- [scaled FP8](week5_result_fp8_scaled.md)：已有评分者明确填写的明显退化，当前任务不推荐。
- [权衡表](tradeoff_v1.md)、[描述性分析验证](week5_analysis_validation.md)。
""")

    # Publication-ready plot, based on stage-specific allocated peaks in MiB.
    chart_dir = ROOT / "notes/week5_figures"
    chart_dir.mkdir(exist_ok=True)
    chart = chart_dir / "stage_allocated_peaks.png"
    if chart.exists():
        raise FileExistsError(chart)
    stage_keys = ("load_text_encoder", "prepare_t2v_inputs", "load_dit_models", "run_sampling", "decode_latent")
    stage_labels = ("T5 load", "Text inputs", "DiT load", "Denoising", "VAE decode")
    values = [[stages[key]["stage_table"][stage]["allocated_mib"] for stage in stage_keys]
              for key in ("baseline", "fp8", "fp8_scaled")]
    colors = ("#4B5563", "#0F766E", "#BE185D")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(10.5, 4.6), layout="constrained")
    x = np.arange(len(stage_keys))
    for i, (label, row) in enumerate(zip(("BF16", "FP8", "Scaled FP8"), values, strict=True)):
        bars = ax.bar(x + (i - 1) * .24, row, .23, label=label, color=colors[i])
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in row], fontsize=8, padding=3)
    ax.set_xticks(x, stage_labels)
    ax.set_ylabel("Peak allocated memory (MiB)")
    ax.set_ylim(0, 17200)
    ax.set_title("Lower denoising peaks; unchanged VAE peak", loc="left", pad=14)
    ax.legend(loc="upper left", frameon=False, ncols=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    fig.savefig(chart, dpi=180)
    plt.close(fig)
    save_new(chart_dir / "stage_allocated_peaks.csv",
             "stage,baseline_mib,fp8_mib,scaled_fp8_mib\n" + "\n".join(
                 f"{stage},{values[0][i]},{values[1][i]},{values[2][i]}"
                 for i, stage in enumerate(stage_keys)))
    stage_table = [
        "| 阶段 | 主环境BF16 | 普通FP8 | scaled FP8 | Sage副本A4（独立环境） |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, stage in zip(stage_labels, stage_keys, strict=True):
        stage_table.append(f"| {label} | " + " | ".join(
            f"{stages[key]['stage_table'][stage]['allocated_mib']:.2f}"
            for key in ("baseline", "fp8", "fp8_scaled")) + f" | {sage_stages[stage]['allocated_mib']:.2f} |")
    save_new(ROOT / "notes/tradeoff_v1.md", f"""# 延迟、显存与质量权衡表 v1

日期：2026-11-06，Asia/Shanghai。仅覆盖Wan2.1 T2V 1.3B、480×832、81帧、16fps、20步。
未启用fp8_fast、compile、block swap、FP8 T5或组合优化。

| 配置 | 同环境基线 | 端到端耗时变化 | 5%速度门槛 | 去噪allocated峰值MiB | 整次allocated峰值MiB | 质量/生成 | 综合分类 |
|---|---|---:|---|---:|---:|---|---|
| 主环境BF16 | 自身参照 | 0 | 参照 | 4183.47 | 14698.35 | 36段参照 | 参照 |
| SageAttention | 副本A1 torch | −7.9798% | 达到，范围不重叠 | 4183.45 | 14698.35 | 三维数值计数达标；36/36 | 结论不足：关键观看/会话记录不完整 |
| 普通FP8 | 主环境F0+F3 | −1.2649% | 未达到 | 2882.59 | 14698.35 | 三维均有明确填写的明显退化；36/36 | 不推荐（当前任务/目标） |
| scaled FP8 | 主环境F0+F3 | +0.4582% | 未达到 | 2917.08 | 14698.35 | 时序、符合度有明确填写的明显退化；36/36 | 不推荐（当前任务/目标） |

速度仅覆盖P01/s42；质量覆盖P01–P12各三个种子。Sage与FP8来自不同环境，
只比较各自组内变化率，不能按跨环境绝对秒数排名。
“明显退化”为保留的评分者方向/程度判断，部分维度证据不完整，助手未独立复看。
缺失程度和观看元数据均保留，三种质量维度不合成为总分。

## 阶段显存

单位均为MiB，各阶段独立峰值，不能相加。F6/F7/F8和A4均为一次正式阶段诊断，
不是多轮平均；不同时间的峰值差不等于可同时回收的空间。

{chr(10).join(stage_table)}

![阶段allocated峰值](week5_figures/stage_allocated_peaks.png)

去噪阶段普通/scaled分别下降31.10%/30.27%，但整次峰值仍在VAE解码，约14698 MiB
（14.35 GiB，不是14.7 GiB）；文本输入阶段约11094 MiB（10.83 GiB）。
不能将阶段收益外推为当前整次省显存、更大batch、更长视频或避免OOM。
D4=A允许描述有条件阶段收益，但两个FP8已有负面质量记录，不能据此归为有条件可用。

## 机理与失效条件

| 配置 | 实际机制 | 本次局限/失效条件 |
|---|---|---|
| Sage | 替换attention为实际执行过的sageattn_varlen Triton路径 | 依赖编译与兼容版本；仅验证当前Ada环境；不是权重FP8；关键质量记录不完整 |
| 普通FP8 | DiT权重直接FP8存储，标准Linear/BF16 autocast | 加载峰值反升51.48%，原因未定位；本次未达速度/质量门槛 |
| scaled FP8 | 目标Linear权重FP8+scale，显式反量化后F.linear | fp8_fast=false，不能称硬件FP8乘法加速；本次未达速度/质量门槛 |

后续仍应先测量VAE峰值来源，再决定性能代码改动。当前阶段峰值已经确定在解码，
但尚未确定其内部是中间激活、输出累积还是特征缓存；此图不能替代分配快照。
本周未执行VAE算法改动，也未验证任何组合配置。

## 证据

- [Sage v2](week5_result_attention_v2.md)、[普通FP8](week5_result_fp8.md)、[scaled FP8](week5_result_fp8_scaled.md)
- [人工质量汇总](week5_quality_summary.md)、[回收与澄清](week5_rating_review.md)
- `notes/week4/fp8_total_audit.json`、`fp8_stages_audit.json`；`notes/week4/A1_audit.json`至`A4_audit.json`
- 图的原始数值：`notes/week5_figures/stage_allocated_peaks.csv`。
""")

    # A bounded validation report avoids converting project thresholds into inference.
    checks = [
        ("Simpson's paradox", "CAUTION", "三个维度和两套环境分别报告，不合成总体胜率；未验证提示词子组推广。"),
        ("Ecological fallacy", "NOTE", "结果只描述固定视频对，不外推所有人物、提示词或模型。"),
        ("Berkson's paradox", "CAUTION", "固定12提示词是窄任务集，非总体随机抽样；不声称无选择偏差。"),
        ("Collider bias", "NOTE", "未根据质量或自动指标筛除样本，未拟合协变量模型。"),
        ("Base rate neglect", "NOTE", "无诊断敏感度/阳性预测值推断；每维保留预定36分母。"),
        ("Regression to the mean", "CAUTION", "Sage补评针对旧缺口，已看过视频；不把补全记为质量提升。"),
        ("Survivorship bias", "NOTE", "全36样本保留，生成失败/无法判断单列；72个FP8视频均首次成功。"),
        ("Look-elsewhere effect", "NOTE", "三配置三维全报告，Top5仅作异常对照；无显著性检验。"),
        ("Garden of forking paths", "CAUTION", "保留原预登记、补充、原答卷和解盲前快照；程度未定不补档。"),
        ("Correlation versus causation", "CAUTION", "速度只比较组内；绝对人物缺陷不自动归因于优化新增。"),
        ("Reverse causality", "NOTE", "无横断面回归或方向性因果模型；不作反向因果解释。"),
    ]
    scan = "\n".join(f"| {name} | {level} | {detail} |" for name, level, detail in checks)
    save_new(ROOT / "notes/week5_analysis_validation.md", f"""## Material Passport

- Origin Skill: experiment-agent / academic-research-suite
- Origin Mode: validate
- Origin Date: 2026-11-06
- Verification Status: ANALYZED
- Version Label: week5_validation_v1

## Validation Report

来源：评分者原表、两次用户说明、现有Week4性能/生成审计。总体适用性：CAUTION。
本次只复算确定性评分统计，不重新执行GPU生成，也不称完整实验已独立复现。

### 数值与来源核对

- 三包原件/补空副本哈希与上轮最终验收一致；旧Week4输入未改。
- 三份有效答案快照先完成，再打开Week5映射；本次修复了Sage来源哈希检查目标：
  包内记录对应quality_dimensions.csv，现核对该CSV并与原JSON的108项逐项比对。
  首次失败未产生结论文件，原脚本及原快照保留；修复没有更改答案或规则。
- 三个候选全部36对逐维重算；Sage仅替换8项，100项原样保留；四个对照十二维均一致。
- 原视频和106个评分代理哈希已重核；36对来源唯一、固定12单视频子集未变。
- 独立核对脚本再次由方向/身份复核FP8更好/更差类别、各维计数与分母。
- 结果没有p值、置信区间、统计显著性或等价性结论；min/max与5%是项目操作规则。
- 未定程度、无法判断、元数据缺失分开；不能以最高分填充真实未评样本，
  本次空白转换依据评分者对已完成评分的明确说明，并保留逐格来源。

### 方法检查

覆盖：11/11，以下为适用性检查，不是声称所有偏差都不存在。

| 类型 | 级别 | 核对结果与边界 |
|---|---|---|
{scan}

### 复现范围

验证状态ANALYZED；评分计数已本机重复核算，源视频仅核对哈希，未由助手重新观看代评。
GPU性能实验未重跑，干净环境独立复现和VAE峰值内部归属仍未验证。
完整审计入口为`notes/blind_private/week5-analysis-20261106/analysis_audit.json`和
`report_verification.json`；私人映射、原评分及细节不自动公开。
""")
    result_audit = {
        "date_asia_shanghai": "2026-11-06", "classification": machine_verdicts,
        "independent_count_checks": verification, "fallacy_checks": 11,
        "metrics_crosscheck_rows": len(crosscheck_rows),
        "report_script_sha256": digest(Path(__file__)),
        "reported_source_videos_hashed": len(load(RUN / "analysis_audit.json")["source_video_hashes"]),
        "output_hashes": {str(path.relative_to(ROOT)): digest(path) for path in (
            *(ROOT / "notes" / name for name in card_paths),
            ROOT / "notes/week5_quality_summary.md", ROOT / "notes/tradeoff_v1.md",
            ROOT / "notes/week5_analysis_validation.md", chart,
        )},
    }
    save_new(RUN / "report_verification.json", json.dumps(result_audit, ensure_ascii=False, indent=2))
    print(json.dumps({"classifications": machine_verdicts, "metric_notes": metric_notes,
                      "verified_source_videos": result_audit["reported_source_videos_hashed"],
                      "report_count": len(result_audit["output_hashes"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
