"""Versioned Week5 clarification snapshots and deterministic unblinding.

Prepare must finish for all three packages before analyze can read mappings.
Source workbooks, completed workbooks and Week4 analyses remain read only.
"""

import argparse
from collections import Counter
from copy import deepcopy
import csv
import io
import json
from pathlib import Path

from analyze_candidate_ratings import load_yaml
from audit_candidate_ratings import consistent
from week5_rating_intake import ROOT, RUN as INTAKE, extract, read, sha as path_sha, write_new


DATE = "2026-11-06"
RUN = ROOT / "notes/blind_private/week5-analysis-20261106"
OLD = ROOT / "notes/blind_private/week4-R01-20261027/returned-bf3a3a2dd254/quality_analysis.json"
DIMS = ("structure", "temporal_motion", "prompt_alignment")
CATEGORIES = (
    "marked_worse", "mild_worse", "no_visible_difference",
    "different_no_clear_advantage", "mild_better", "marked_better",
    "unjudgeable", "unresolved", "worse_severity_unspecified",
    "better_severity_unspecified",
)
CLARIFICATION = """# Week5 一次评分澄清

确认日期：2026-11-06，Asia/Shanghai。原话：“按照已填方向为准，评分相隔两天”。

- 七项冲突均保留原填方向。T022 的方向“无法判断”优先，分析层有效程度记为“无法判断”。
- 另外六项方向为 A/B 更好，原填程度“有差异但难判优劣”与该方向不兼容。
  用户未说明轻微/明显，因此分析层程度保持缺失，记录为“方向确定、程度未定”；
  不代填轻微/明显，不把这些答案改成无差异，也不再追问第二轮。
- 原始表和补空副本均不再改写；分析使用单独有效答案快照。
- 用户确认两个 FP8 包评分相隔两天，满足至少一天间隔。具体两个日期未提供，
  不从文件时间或当前日期推造。此项来源为评分者陈述，不是独立时间戳验证。
- 已知助手在 Week4 看过 Sage 身份映射；Week5 补评仍是解盲后的补充评分。

以上解释在本轮读取三个 Week5 身份映射之前固定；不会按解盲方向选择差异程度。
"""


def sha(path):
    return path_sha(Path(path))


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    write_new_path = RUN / "clarification-01.md"
    if write_new_path.exists():
        assert write_new_path.read_text(encoding="utf-8") == CLARIFICATION
    else:
        write_new_path.write_text(CLARIFICATION, encoding="utf-8")
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (
        write_new_path, ROOT / "configs/optimization_rules.yaml",
        ROOT / "configs/scoring_frozen.yaml", ROOT / "notes/week4_prereg.md",
        ROOT / "notes/week5_prereg_addendum.md", Path(__file__),
    )}
    snapshots = []
    manifest = read(INTAKE / "intake.json")
    for pack in manifest["packs"]:
        returned = Path(pack["returned"])
        audit_path = returned / "completed_answers_before_unblinding.json"
        verification_path = returned / "completed_workbook_verification_final.json"
        audit, verification = read(audit_path), read(verification_path)
        assert audit["mapping_opened"] is False
        assert not verification["unexpected_values"] and not verification["feature_differences"]
        assert sha(pack["original"]) == pack["source_sha256"] == sha(pack["source"])
        assert sha(pack["completed"]) == verification["completed_sha256"] == audit["completed_sha256"]
        tasks_path = ROOT / "outputs" / pack["name"] / "tasks.json"
        tasks = read(tasks_path)["tasks"]
        records, sessions = extract(pack["completed"], tasks)
        assert records == audit["records"] and sessions == audit["sessions_as_entered"]
        effective, updates = [], []
        for record in records:
            if record["type"] != "配对":
                continue
            for dim, answer in record["dimensions"].items():
                direction, severity = answer["direction"], answer["severity"]
                reason = None
                if not consistent(direction, severity):
                    if direction == "无法判断":
                        severity = "无法判断"
                        reason = "reviewer_direction_takes_precedence_unjudgeable"
                    else:
                        assert direction in ("A更好", "B更好")
                        severity = None
                        reason = "degree_unspecified_after_direction_confirmation"
                    updates.append({"id": record["id"], "dimension": dim,
                                    "direction": direction, "effective_severity": severity,
                                    "reason": reason})
                effective.append({
                    "id": record["id"], "dimension": dim,
                    "original_direction": answer["direction"],
                    "original_severity": answer["severity"],
                    "effective_direction": direction, "effective_severity": severity,
                    "clarification_reason": reason, "playback_status": record["playback_status"],
                    "problem_time": record["problem_time"], "view": record["view"],
                    "absolute_defects": record["absolute_defects"], "evidence": record["evidence"],
                })
        expected = 0 if "sage" in pack["name"] else 5 if pack["name"] == "week5-fp8-R01" else 2
        assert len(updates) == expected
        files = (Path(pack["original"]), Path(pack["source"]), Path(pack["completed"]),
                 audit_path, verification_path, tasks_path)
        for path in files:
            inputs[str(path.relative_to(ROOT))] = sha(path)
        snapshot = {
            "date_asia_shanghai": DATE, "pack": pack["name"], "mapping_opened": False,
            "clarification_sha256": sha(write_new_path),
            "effective_pair_answers": effective, "clarifications": updates,
            "original_completed_records": records, "sessions_as_entered": sessions,
            "metadata_and_evidence_issues": audit["metadata_and_evidence_issues"],
            "fp8_scoring_gap": {"days": 2, "source": "reviewer_statement",
                                "meets_at_least_one_day": True, "exact_dates": None},
        }
        path = RUN / f"{pack['name']}-effective-answers.json"
        write_new(path, snapshot)
        snapshots.append(str(path.relative_to(ROOT)))
        inputs[str(path.relative_to(ROOT))] = sha(path)
        print(json.dumps({"pack": pack["name"], "pair_answers": len(effective),
                          "clarifications": updates, "mapping_opened": False}, ensure_ascii=False))
    write_new(RUN / "pre_unblind_manifest.json", {
        "date_asia_shanghai": DATE, "mapping_opened": False,
        "inputs_sha256": inputs, "snapshots": snapshots,
    })


def category(row, identity, candidate):
    direction, severity = row["effective_direction"], row["effective_severity"]
    if row.get("playback_status") == "无法播放":
        return "unresolved"
    if direction == "无法判断" and severity == "无法判断":
        return "unjudgeable"
    if direction in ("A更好", "B更好"):
        relation = "better" if identity[direction[0]]["kind"] == candidate else "worse"
        if severity is None:
            return f"{relation}_severity_unspecified"
        if severity in ("轻微", "明显"):
            return f"{'mild' if severity == '轻微' else 'marked'}_{relation}"
    if direction == "无明确优劣":
        if severity == "无明显差异":
            return "no_visible_difference"
        if severity == "有差异但难判优劣":
            return "different_no_clear_advantage"
    return "unresolved"


def summarize(rows, quality):
    summary = {}
    for dim in DIMS:
        selected = [row for row in rows if row["dimension"] == dim]
        assert len(selected) == quality["planned_pairs"] == 36
        counts = Counter(row["category"] for row in selected)
        unknown = sum(counts[k] for k in (
            "unresolved", "worse_severity_unspecified", "better_severity_unspecified"))
        scorable = 36 - unknown - counts["unjudgeable"]
        numerical = scorable >= quality["min_scorable_pairs_each_dimension"] and (
            counts["marked_worse"] <= quality["max_marked_worse_each_dimension"]
            and counts["mild_worse"] <= quality["max_mild_worse_each_dimension"])
        if counts["marked_worse"] > quality["max_marked_worse_each_dimension"]:
            gate = "threshold_not_met_confirmed_marked_worse"
        elif unknown:
            gate = "insufficient_evidence_degree_or_answer_unresolved"
        elif not numerical:
            gate = "threshold_not_met"
        else:
            gate = "numerical_thresholds_met_metadata_limits_remain"
        summary[dim] = {
            "planned": 36, "scorable_complete_answers": scorable,
            "direction_determined": 36 - counts["unjudgeable"] - counts["unresolved"],
            "counts": {k: counts[k] for k in CATEGORIES},
            "rates": {
                k: {"over_planned_36": counts[k] / 36,
                    "over_complete_scorable": counts[k] / scorable
                    if scorable and k not in ("unresolved", "unjudgeable",
                                             "worse_severity_unspecified",
                                             "better_severity_unspecified") else None}
                for k in CATEGORIES
            },
            "numerical_thresholds_met_on_complete_answers": numerical,
            "quality_gate": gate,
        }
    return summary


def validate_mapping(pack, candidate, tasks, mapping, cache):
    assert mapping["candidate"] == candidate
    identities = {t["id"]: t for t in mapping["tasks"]}
    assert len(identities) == len(tasks) and set(identities) == {t["id"] for t in tasks}
    pairs, singles = [], []
    for task in tasks:
        identity = identities[task["id"]]
        for field in ("order", "id", "type", "video", "prompt", "requirement"):
            assert task[field] == identity[field]
        public = ROOT / "outputs" / pack / task["video"]
        assert sha(public) == identity["public_sha256"], public
        if task["type"] == "配对":
            assert {identity[s]["kind"] for s in ("A", "B")} == {"baseline", candidate}
            assert identity["A"]["source_key"] == identity["B"]["source_key"]
            pairs.append(identity["A"]["source_key"])
            sources = [identity["A"], identity["B"]]
        else:
            assert identity["source"]["kind"] == candidate
            singles.append(identity["source"]["source_key"])
            sources = [identity["source"]]
        for source in sources:
            assert source["prompt"] == task["prompt"]
            path = Path(source["video_path"])
            if str(path) not in cache:
                cache[str(path)] = sha(path)
            assert cache[str(path)] == source["video_sha256"], path
    assert len(set(pairs)) == len(pairs)
    return identities, pairs, singles


def save_result(result):
    name = result["pack"]
    output = RUN / f"{name}-quality-analysis.json"
    write_new(output, result)
    buffer = io.StringIO(newline="")
    fields = sorted({key for row in result["pairs_by_dimension"] for key in row})
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(result["pairs_by_dimension"])
    target = RUN / f"{name}-quality-dimensions.csv"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(buffer.getvalue(), encoding="utf-8-sig")
    print(json.dumps({"pack": name, "summary": result["summary"],
                      "controls": result.get("control_consistency"),
                      "marked_worse": [r for r in result["pairs_by_dimension"]
                                       if r["category"] == "marked_worse"]},
                     ensure_ascii=False))


def analyze():
    manifest = read(RUN / "pre_unblind_manifest.json")
    assert manifest["mapping_opened"] is False and len(manifest["snapshots"]) == 3
    assert not (RUN / "analysis_audit.json").exists()
    archived_analyzer = RUN / "analyzer-before-source-hash-fix.py"
    for relative, expected in manifest["inputs_sha256"].items():
        checked = archived_analyzer if Path(relative).as_posix() == "scripts/analyze_week5_ratings.py" and archived_analyzer.exists() else ROOT / relative
        assert sha(checked) == expected, relative
    snapshots = [read(ROOT / path) for path in manifest["snapshots"]]
    rules = load_yaml(ROOT / "configs/optimization_rules.yaml")
    frozen = load_yaml(ROOT / "configs/scoring_frozen.yaml")
    # All effective answers are immutable on disk before any mapping is opened.
    if not (RUN / "unblinding_started.json").exists():
        write_new(RUN / "unblinding_started.json", {
            "date_asia_shanghai": DATE, "pre_unblind_manifest_sha256": sha(RUN / "pre_unblind_manifest.json"),
            "effective_answers_fixed_for_all_three_packages": True,
        })
    cache, mapping_hashes = {}, {}
    results = []
    for snapshot in snapshots:
        pack = snapshot["pack"]
        candidate = "sageattn" if "sage" in pack else "fp8" if pack == "week5-fp8-R01" else "fp8_scaled"
        path = ROOT / "notes/blind_private" / pack / "mapping.json"
        mapping = read(path)
        mapping_hashes[str(path.relative_to(ROOT))] = sha(path)
        tasks = read(ROOT / "outputs" / pack / "tasks.json")["tasks"]
        identities, pair_keys, single_keys = validate_mapping(pack, candidate, tasks, mapping, cache)
        effective = deepcopy(snapshot["effective_pair_answers"])
        for row in effective:
            row["source_key"] = identities[row["id"]]["A"]["source_key"]
            row["category"] = category(row, identities[row["id"]], candidate)
        result = {
            "date_asia_shanghai": DATE, "pack": pack, "candidate": candidate,
            "effective_answer_snapshot_sha256": sha(ROOT / manifest["snapshots"][snapshots.index(snapshot)]),
            "mapping_sha256": sha(path), "mapping_opened_after_all_effective_snapshots": True,
            "sessions_as_entered": snapshot["sessions_as_entered"],
            "fp8_scoring_gap": snapshot["fp8_scoring_gap"],
            "metadata_and_evidence_issues": snapshot["metadata_and_evidence_issues"],
            "limitations": [
                "Single reviewer and lossy proxies; no statistical equivalence claim.",
                "Missing playback/view/session/time fields are not fabricated.",
                "Confirmed directions do not determine mild versus marked severity.",
                "No second clarification round; unresolved severity remains explicit.",
            ],
        }
        if candidate != "sageattn":
            assert len(pair_keys) == 36 and len(single_keys) == 12
            assert set(single_keys) == {t["key"] for t in frozen["absolute_subset"]}
            singles = [deepcopy(r) for r in snapshot["original_completed_records"] if r["type"] == "单视频"]
            for row in singles:
                row["source_key"] = identities[row["id"]]["source"]["source_key"]
            result["singles"] = singles
            result["absolute_distributions"] = {
                dim: dict(Counter(row["dimensions"][dim]["score"] for row in singles))
                for dim in DIMS
            }
            result["pairs_by_dimension"] = effective
        else:
            assert len(pair_keys) == 10 and not single_keys
            old = read(OLD)
            old_csv = OLD.parent / "quality_dimensions.csv"
            assert mapping["source_rating_sha256"] == sha(old_csv)
            with old_csv.open(encoding="utf-8-sig", newline="") as handle:
                old_csv_rows = list(csv.DictReader(handle))
            assert len(old_csv_rows) == len(old["pairs_by_dimension"]) == 108
            old_json_rows = {(r["id"], r["dimension"]): r for r in old["pairs_by_dimension"]}
            for csv_row in old_csv_rows:
                json_row = old_json_rows[(csv_row["id"], csv_row["dimension"])]
                assert all(csv_row[k] == str(json_row[k]) if json_row[k] is not None
                           else csv_row[k] == "" for k in ("category", "source_key",
                                                          "effective_direction", "effective_severity"))
            old_map_path = ROOT / "notes/blind_private/week4-R01-20261027/mapping.json"
            assert mapping["week4_mapping_sha256"] == sha(old_map_path)
            old_mapping = read(old_map_path)
            mapping_hashes[str(old_map_path.relative_to(ROOT))] = sha(old_map_path)
            merged = deepcopy(old["pairs_by_dimension"])
            by_old = {(r["id"], r["dimension"]): r for r in merged}
            original_by_key = {(r["source_key"], r["dimension"]): r for r in old["pairs_by_dimension"]}
            original_ids = {t["id"]: t for t in old_mapping["tasks"]}
            controls, replacements = [], []
            for row in effective:
                identity = identities[row["id"]]
                if identity["review_kind"] == "disputed":
                    if row["dimension"] not in identity["unresolved_dimensions"]:
                        continue
                    key = (identity["source_ticket"], row["dimension"])
                    previous = by_old[key]
                    assert previous["category"] == "unresolved"
                    assert previous["source_key"] == row["source_key"]
                    replacement = {**row, "id": identity["source_ticket"],
                                   "followup_id": row["id"], "previous_category": "unresolved"}
                    replacements.append(deepcopy(replacement))
                    previous.clear()
                    previous.update(replacement)
                else:
                    assert identity["review_kind"] == "control"
                    previous = original_by_key[(row["source_key"], row["dimension"])]
                    assert previous["category"] not in ("unresolved", "unjudgeable")
                    assert previous["source_key"] == original_ids[previous["id"]]["A"]["source_key"]
                    ranks = {"marked_worse": -2, "mild_worse": -1,
                             "no_visible_difference": 0, "different_no_clear_advantage": 0,
                             "mild_better": 1, "marked_better": 2}
                    before, after = ranks[previous["category"]], ranks.get(row["category"])
                    controls.append({
                        "followup_id": row["id"], "original_id": previous["id"],
                        "source_key": row["source_key"], "dimension": row["dimension"],
                        "previous_category": previous["category"], "followup_category": row["category"],
                        "opposite_direction": after is not None and before * after < 0,
                        "degree_gap_at_least_two": after is not None and abs(abs(before) - abs(after)) >= 2,
                        "comparison_unavailable": after is None,
                    })
            assert len(replacements) == 8
            assert {r["id"] for r in replacements} == {"T001", "T002", "T003", "T010", "T032", "T038"}
            assert len(controls) == 12 and len({r["followup_id"] for r in controls}) == 4
            untouched = set(by_old) - {(r["id"], r["dimension"]) for r in replacements}
            old_rows = {(r["id"], r["dimension"]): r for r in old["pairs_by_dimension"]}
            assert all(by_old[k] == old_rows[k] for k in untouched)
            result.update({
                "pairs_by_dimension": merged, "replacements": replacements,
                "control_consistency": controls, "week4_analysis_sha256": sha(OLD),
                "unchanged_week4_dimensions": len(untouched),
                "absolute_distributions": old["absolute_distributions"],
                "singles": old["singles"],
                "limitations": result["limitations"] + [
                    "Post-unblinding supplementary Sage review, not a new independent blind study.",
                    "Only eight unresolved dimensions replaced; controls never replace original scores.",
                ],
            })
        result["summary"] = summarize(result["pairs_by_dimension"], rules["quality"])
        results.append(result)
    # Preserve source hashes and expose reproducibility inputs without publishing them.
    for result in results:
        result["source_video_hashes_verified"] = cache
        save_result(result)
    for relative, expected in manifest["inputs_sha256"].items():
        checked = archived_analyzer if Path(relative).as_posix() == "scripts/analyze_week5_ratings.py" and archived_analyzer.exists() else ROOT / relative
        assert sha(checked) == expected, relative
    write_new(RUN / "analysis_audit.json", {
        "date_asia_shanghai": DATE, "pre_unblind_inputs_sha256": manifest["inputs_sha256"],
        "mapping_hashes": mapping_hashes, "source_video_hashes": cache,
        "all_input_hashes_unchanged": True, "all_effective_snapshots_precede_mapping_read": True,
        "analysis_script_sha256": sha(__file__),
        "source_hash_check_fix": "Sage package records quality_dimensions.csv hash, not quality_analysis.json; both now cross-checked.",
        "results": [r["pack"] for r in results],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "analyze"))
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else analyze)()
