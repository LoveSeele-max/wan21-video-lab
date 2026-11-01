"""Archive returned answers, then audit without revealing the private mapping.

Read XLSX only. Never save, recalculate or correct the reviewer's workbook.
Conflicting direction/severity is retained, not interpreted as a quality pass.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import openpyxl


ROOT = Path(__file__).resolve().parent.parent
DIMENSIONS = (("structure", 3), ("temporal_motion", 5), ("prompt_alignment", 7))
SCORES = {"2-较好", "1-有缺陷", "0-明显失败", "无法判断"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def consistent(direction, severity):
    if direction in ("A更好", "B更好"):
        return severity in ("轻微", "明显")
    if direction == "无明确优劣":
        return severity in ("无明显差异", "有差异但难判优劣")
    return direction == "无法判断" and severity == "无法判断"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.name):
        parser.error("Pack name must be one safe directory name")
    public = ROOT / "outputs" / args.name
    private = ROOT / "notes/blind_private" / args.name
    source = public / "评分表.xlsx"
    source_hash = digest(source)
    returned = private / f"returned-{source_hash[:12]}"
    returned.mkdir(exist_ok=True)
    archived = returned / "original.xlsx"
    if not archived.exists():
        shutil.copy2(source, archived)
    if digest(archived) != source_hash:
        raise ValueError("Archive hash mismatch; preserve originals")
    tasks = json.loads((public / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    if len(tasks) != 48 or Counter(t["type"] for t in tasks) != {"配对": 36, "单视频": 12}:
        raise ValueError("Unexpected public workload")
    # No mapping is opened until the reviewer resolves answer conflicts.
    wb = openpyxl.load_workbook(archived, read_only=True, data_only=False)
    try:
        if wb.sheetnames != ["评分记录", "会话记录"]:
            raise ValueError("Unexpected workbook sheets")
        sheet = wb["评分记录"]
        records, conflicts, issues = [], [], []
        singles = {dim: Counter() for dim, _ in DIMENSIONS}
        pairs = {dim: Counter() for dim, _ in DIMENSIONS}
        for index, task in enumerate(tasks, 8):
            row = [sheet.cell(index, col).value for col in range(1, 17)]
            expected = (task["order"], task["id"], task["type"],
                        task["prompt"], task["requirement"], task["video"])
            actual = (*row[:3], *row[13:16])
            if actual != expected:
                raise ValueError(f"Task identity altered at row {index}")
            item = {
                "excel_row": index, "id": row[1], "type": row[2],
                "dimensions": {}, "playback_status": row[9],
                "problem_time": row[10], "evidence": row[11], "view": row[12],
            }
            if row[9] not in ("可播放", "无法播放"):
                issues.append({"id": row[1], "field": "playback_status", "value": row[9]})
            if row[12] is None:
                issues.append({"id": row[1], "field": "view", "value": None})
            for dimension, col in DIMENSIONS:
                value, severity = row[col:col + 2]
                if row[2] == "单视频":
                    singles[dimension][str(value)] += 1
                    item["dimensions"][dimension] = {"score": value}
                    if value not in SCORES or severity is not None:
                        issues.append({"id": row[1], "dimension": dimension,
                                       "field": "absolute_score_or_unused_severity"})
                    needs_evidence = value in ("1-有缺陷", "0-明显失败", "无法判断")
                else:
                    valid = consistent(value, severity)
                    pairs[dimension]["recorded_pairs"] += 1
                    pairs[dimension]["consistent_answers" if valid else "conflicting_answers"] += 1
                    if valid:
                        pairs[dimension][f"{value}/{severity}"] += 1
                    item["dimensions"][dimension] = {
                        "direction": value, "severity": severity, "consistent": valid}
                    if not valid:
                        conflicts.append({"id": row[1], "excel_row": index,
                                          "dimension": dimension, "direction": value,
                                          "severity": severity, "evidence": row[11]})
                    needs_evidence = (value != "无明确优劣"
                                      or severity != "无明显差异")
                if needs_evidence and not row[11]:
                    issues.append({"id": row[1], "dimension": dimension,
                                   "field": "evidence"})
                if needs_evidence and not row[10]:
                    issues.append({"id": row[1], "dimension": dimension,
                                   "field": "problem_time"})
            if row[9] == "无法播放" and any(row[c] is not None for c in range(3, 9)):
                issues.append({"id": row[1], "field": "unplayable_with_scores"})
            records.append(item)
        extra = list(sheet.iter_rows(min_row=56, max_col=16, values_only=True))
        if any(any(value is not None for value in row) for row in extra):
            raise ValueError("Unexpected additional rating rows")
        sessions = []
        for row in wb["会话记录"].iter_rows(min_row=6, max_row=25, max_col=7,
                                             values_only=True):
            if any(value is not None for value in row):
                sessions.append(dict(zip(
                    ("date", "player", "reviewer", "task_range",
                     "prior_exposure", "discussion", "notes"), row)))
    finally:
        wb.close()
    if digest(source) != source_hash or digest(archived) != source_hash:
        raise ValueError("Source or archive changed during read")
    report = {
        "audit_date_asia_shanghai": datetime.now().astimezone().date().isoformat(),
        "status": "clarification_required_before_unblinding"
                  if conflicts else "answer_consistency_checked_mapping_not_opened",
        "source_sha256": source_hash, "archived_original": str(archived),
        "source_workbook_unchanged": True, "mapping_opened": False,
        "task_count": len(records),
        "type_counts": dict(Counter(r["type"] for r in records)),
        "playback_status_counts": dict(Counter(str(r["playback_status"]) for r in records)),
        "view_counts": dict(Counter(str(r["view"]) for r in records)),
        "absolute_distributions": {k: dict(v) for k, v in singles.items()},
        "pair_answer_consistency_counts_not_quality_counts": {k: dict(v) for k, v in pairs.items()},
        "conflicts": conflicts, "metadata_and_evidence_issues": issues,
        "sessions_as_entered": sessions, "records": records,
        "limits": [
            "Absolute scores describe the candidate alone, not added degradation.",
            "Do not convert conflicting answers to no-difference or mild degradation.",
            "Missing playback, evidence and session entries remain missing, not inferred.",
            "No final scorability, quality threshold or usable verdict before clarification.",
        ],
    }
    output = returned / "pre_unblind_audit.json"
    if output.exists():
        raise FileExistsError(f"Preserve previous audit: {output}")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "source_sha256": source_hash, "archive": str(archived),
        "audit": str(output), "task_count": len(records),
        "conflicting_dimensions": len(conflicts),
        "conflicting_task_ids": sorted({c["id"] for c in conflicts}),
        "absolute_distributions": report["absolute_distributions"],
        "pair_answer_counts": report["pair_answer_consistency_counts_not_quality_counts"],
        "playback_status_counts": report["playback_status_counts"],
        "view_counts": report["view_counts"], "mapping_opened": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
