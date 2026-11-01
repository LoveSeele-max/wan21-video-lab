"""Read returned candidate ratings and report unresolved evidence explicitly."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

import openpyxl

from audit_candidate_ratings import consistent


ROOT = Path(__file__).resolve().parent.parent
DIMENSIONS = ("structure", "temporal_motion", "prompt_alignment")
CATEGORIES = (
    "marked_worse", "mild_worse", "no_visible_difference",
    "different_no_clear_advantage", "mild_better", "marked_better",
    "unjudgeable", "unresolved",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path):
    # The bundled reader lacks PyYAML; use the existing project YAML parser only.
    completed = subprocess.run(
        [str(ROOT / "local/env/python.exe"), "-c",
         "import json,sys,yaml; "
         "print(json.dumps(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))))",
         str(path)],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    return json.loads(completed.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.name):
        parser.error("Pack name must be one safe directory name")
    public = ROOT / "outputs" / args.name
    source = public / "评分表.xlsx"
    source_hash = sha(source)
    private = ROOT / "notes/blind_private" / args.name
    returned = private / f"returned-{source_hash[:12]}"
    original = returned / "original.xlsx"
    decisions_path = returned / "quality_decisions_before_unblinding.json"
    audit_path = returned / "pre_unblind_audit.json"
    output = returned / "quality_analysis.json"
    rows_output = returned / "quality_dimensions.csv"
    if output.exists() or rows_output.exists():
        raise FileExistsError("Preserve prior analysis; use a separately versioned follow-up")
    decisions, audit = load(decisions_path), load(audit_path)
    assert sha(original) == source_hash == decisions["source_sha256"] == audit["source_sha256"]
    assert decisions["mapping_opened_when_this_file_was_created"] is False
    assert audit["mapping_opened"] is False
    tasks = load(public / "tasks.json")["tasks"]
    assert len(tasks) == 48

    # Re-read every rating from the archived original before loading identities.
    workbook = openpyxl.load_workbook(original, read_only=True, data_only=False)
    try:
        assert workbook.sheetnames == ["评分记录", "会话记录"]
        for task, record, values in zip(
            tasks, audit["records"],
            workbook["评分记录"].iter_rows(min_row=8, max_row=55, max_col=16, values_only=True),
            strict=True,
        ):
            assert (values[0], values[1], values[2], *values[13:16]) == (
                task["order"], task["id"], task["type"],
                task["prompt"], task["requirement"], task["video"],
            )
            assert record["id"] == task["id"] and record["type"] == task["type"]
            assert list(values[9:13]) == [
                record["playback_status"], record["problem_time"],
                record["evidence"], record["view"],
            ]
            for i, dimension in enumerate(DIMENSIONS):
                raw = record["dimensions"][dimension]
                if task["type"] == "配对":
                    assert values[3 + 2 * i:5 + 2 * i] == (raw["direction"], raw["severity"])
                else:
                    assert values[3 + 2 * i] == raw["score"]
    finally:
        workbook.close()
    assert len(audit["records"]) == 48
    input_hashes = {
        str(path.relative_to(ROOT)): sha(path) for path in (
            source, original, decisions_path, audit_path,
            returned / "clarification-01.md", returned / "clarification-02.md",
            public / "tasks.json", ROOT / "configs/optimization_rules.yaml",
            ROOT / "configs/scoring_frozen.yaml",
        )
    }
    rules = load_yaml(ROOT / "configs/optimization_rules.yaml")
    frozen = load_yaml(ROOT / "configs/scoring_frozen.yaml")
    quality = rules["quality"]
    overrides = {(r["id"], r["dimension"]): r for r in decisions["clarifications"]}
    absolute_only = decisions["absolute_severity_not_relative_severity"]
    effective = []
    for record in audit["records"]:
        if record["type"] != "配对":
            continue
        for dimension in DIMENSIONS:
            raw = record["dimensions"][dimension]
            override = overrides.get((record["id"], dimension))
            direction = override["effective_direction"] if override else raw["direction"]
            severity = override["effective_severity"] if override else raw["severity"]
            if override:
                assert (raw["direction"], raw["severity"]) == (
                    override["original_direction"], override["original_severity"])
                assert record["evidence"]
            reason = None
            if record["id"] in absolute_only["ids"] and dimension == absolute_only["dimension"]:
                assert direction == absolute_only["original_direction_retained"]
                severity = None
                reason = "absolute_prompt_defect_clarified_relative_subtype_unspecified"
            elif not consistent(direction, severity):
                reason = "original_direction_severity_conflict_not_resolved"
            if record["playback_status"] == "无法播放":
                reason = "unplayable"
            effective.append({
                "id": record["id"], "dimension": dimension,
                "original_direction": raw["direction"], "original_severity": raw["severity"],
                "effective_direction": direction, "effective_severity": severity,
                "clarification_applied": override is not None,
                "unresolved_reason": reason, "evidence": record["evidence"],
                "playback_status": record["playback_status"],
                "problem_time": record["problem_time"], "view": record["view"],
            })
    assert len(effective) == quality["planned_pairs"] * len(DIMENSIONS)
    snapshot_path = returned / "effective_answers_before_unblinding.json"
    if snapshot_path.exists():
        raise FileExistsError("Preserve prior pre-unblinding snapshot")
    snapshot_path.write_text(json.dumps({
        "inputs_sha256": input_hashes, "mapping_opened": False,
        "effective_pair_answers": effective,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    input_hashes[str(snapshot_path.relative_to(ROOT))] = sha(snapshot_path)

    # Interpretations are now saved. Identities cannot affect the rules above.
    mapping_path = private / "mapping.json"
    mapping = load(mapping_path)
    input_hashes[str(mapping_path.relative_to(ROOT))] = sha(mapping_path)
    assert mapping["candidate"] == "sageattn"
    identities = {t["id"]: t for t in mapping["tasks"]}
    assert len(identities) == 48 and set(identities) == {t["id"] for t in tasks}
    pair_keys, single_keys, video_hashes = [], [], {}
    for task in tasks:
        identity = identities[task["id"]]
        for field in ("order", "id", "type", "video", "prompt", "requirement"):
            assert identity[field] == task[field]
        if task["type"] == "配对":
            assert {identity[s]["kind"] for s in ("A", "B")} == {"baseline", "sageattn"}
            assert identity["A"]["source_key"] == identity["B"]["source_key"]
            pair_keys.append(identity["A"]["source_key"])
            sources = [identity["A"], identity["B"]]
        else:
            assert identity["source"]["kind"] == "sageattn"
            single_keys.append(identity["source"]["source_key"])
            sources = [identity["source"]]
        for item in sources:
            assert item["prompt"] == task["prompt"]
            path = Path(item["video_path"])
            if str(path) not in video_hashes:
                video_hashes[str(path)] = sha(path)
            assert video_hashes[str(path)] == item["video_sha256"]
    assert len(set(pair_keys)) == 36
    assert set(single_keys) == {item["key"] for item in frozen["absolute_subset"]}
    assert len(single_keys) == 12

    for item in effective:
        identity = identities[item["id"]]
        item["source_key"] = identity["A"]["source_key"]
        if item["unresolved_reason"]:
            category = "unresolved"
        elif item["effective_direction"] == "无法判断":
            category = "unjudgeable"
        elif item["effective_direction"] == "无明确优劣":
            category = ("no_visible_difference" if item["effective_severity"] == "无明显差异"
                        else "different_no_clear_advantage")
        else:
            winner = item["effective_direction"][0]
            result = "better" if identity[winner]["kind"] == "sageattn" else "worse"
            degree = "mild" if item["effective_severity"] == "轻微" else "marked"
            category = f"{degree}_{result}"
        item["category"] = category
    summary = {}
    for dimension in DIMENSIONS:
        rows = [r for r in effective if r["dimension"] == dimension]
        counts = Counter(r["category"] for r in rows)
        scorable = len(rows) - counts["unresolved"] - counts["unjudgeable"]
        assert sum(counts.values()) == 36
        summary[dimension] = {
            "planned": 36, "scorable_complete_answers": scorable,
            "counts": {category: counts[category] for category in CATEGORIES},
            "rates": {
                category: {
                    "over_planned_36": counts[category] / 36,
                    "over_scorable": counts[category] / scorable
                    if scorable and category not in ("unresolved", "unjudgeable") else None,
                } for category in CATEGORIES
            },
            "scorable_minimum_met": scorable >= quality["min_scorable_pairs_each_dimension"],
            "observed_worse_counts_within_limits": (
                counts["marked_worse"] <= quality["max_marked_worse_each_dimension"]
                and counts["mild_worse"] <= quality["max_mild_worse_each_dimension"]),
            "unresolved_ids": [r["id"] for r in rows if r["category"] == "unresolved"],
            "quality_gate": "insufficient_evidence" if counts["unresolved"] else (
                "numerical_thresholds_met_metadata_limits_remain"
                if scorable >= quality["min_scorable_pairs_each_dimension"]
                and counts["marked_worse"] == 0 and counts["mild_worse"] <= 3
                else "threshold_not_met"),
        }
    singles = []
    for record in audit["records"]:
        if record["type"] == "单视频":
            singles.append({**record, "source_key": identities[record["id"]]["source"]["source_key"]})
    result = {
        "date_asia_shanghai": "2026-10-27", "pack": args.name,
        "inputs_sha256": input_hashes, "source_workbook_unchanged": sha(source) == source_hash,
        "mapping_opened_after_effective_answer_snapshot": True,
        "classification": "结论不足", "quality_passed": False,
        "classification_reason": "Unresolved direction/severity evidence and incomplete metadata; not a confirmed marked quality regression.",
        "summary": summary, "pairs_by_dimension": effective, "singles": singles,
        "absolute_distributions": audit["absolute_distributions"],
        "metadata_and_evidence_issues": audit["metadata_and_evidence_issues"],
        "sessions_as_entered": audit["sessions_as_entered"],
        "source_video_hashes_verified": video_hashes,
        "limits": [
            "Complete-answer counts are conservative analytical eligibility, not proof of complete normal-speed viewing.",
            "Missing playback status is not treated as unplayable or confirmed playable.",
            "Unresolved original answers are not equivalent to reviewer unjudgeable answers.",
            "T019 absolute score and motion evidence may conflict; preserve both.",
            "Single-reviewer lossy proxies and missing session details preclude strict independent-blind or equivalence claims.",
            "General human-motion/steam defects are not automatically candidate-introduced regressions.",
        ],
    }
    assert result["source_workbook_unchanged"] and sha(original) == source_hash
    for relative, expected in input_hashes.items():
        assert sha(ROOT / relative) == expected
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(effective[0]))
    writer.writeheader()
    writer.writerows(effective)
    rows_output.write_text(buffer.getvalue(), encoding="utf-8-sig")
    print(json.dumps({
        "classification": result["classification"], "summary": summary,
        "absolute_distributions": result["absolute_distributions"],
        "verified_unique_source_videos": len(video_hashes),
        "source_workbook_unchanged": result["source_workbook_unchanged"],
        "analysis": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
