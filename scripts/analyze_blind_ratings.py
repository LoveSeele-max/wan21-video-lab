"""Read and archive returned ratings without editing the workbook."""
import collections
import hashlib
import json
from pathlib import Path
import shutil

import openpyxl

ROOT = Path("E:/cstudy/AIGC/V")
PACK = "week3-R01-20261021"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    source = ROOT / "outputs" / PACK / "评分表.xlsx"
    digest = sha(source)
    destination = ROOT / "notes/blind_private" / PACK / f"returned-{digest[:12]}"
    destination.mkdir(exist_ok=True)
    archived = destination / "original.xlsx"
    if not archived.exists():
        shutil.copy2(source, archived)
    assert sha(archived) == digest
    # Only after archiving the received answer, open the private identity mapping.
    mapping = json.loads((destination.parent / "mapping.json").read_text(encoding="utf-8"))
    workbook = openpyxl.load_workbook(archived, data_only=False)
    tasks = json.loads((ROOT / "outputs" / PACK / "tasks.json").read_text(encoding="utf-8"))
    singles = []
    pairs = []
    issues = []
    score_options = {"2-较好", "1-有缺陷", "0-明显失败", "无法判断"}
    for index, task in enumerate(tasks["singles"], 8):
        cells = [workbook.worksheets[0].cell(index, col).value for col in range(1, 13)]
        assert cells[0] == task["id"]
        assert cells[9:] == [task["prompt"], task["requirement"], task["video"]]
        if any(v not in score_options for v in cells[1:4]):
            issues.append(f"{cells[0]} 存在缺失或无效分数")
        if not cells[4]:
            issues.append(f"{cells[0]} 观看状态空白")
        if any(v in {"0-明显失败", "1-有缺陷", "无法判断"} for v in cells[1:4]) and not cells[5]:
            issues.append(f"{cells[0]} 降分/无法判断未填写视频内问题时间")
        identity = mapping["singles"][index - 8]["source"]
        singles.append({
            "id": cells[0], "scores": cells[1:4], "status": cells[4],
            "problem_time": cells[5], "evidence": cells[6], "view": cells[7],
            "seconds": cells[8], "source_id": identity["id"],
            "kind": identity["kind"], "prompt_id": identity["prompt_id"],
        })
    for index, task in enumerate(tasks["pairs"], 8):
        cells = [workbook.worksheets[1].cell(index, col).value for col in range(1, 16)]
        assert cells[0] == task["id"]
        assert cells[12:] == [task["prompt"], task["requirement"], task["video"]]
        identity = mapping["pairs"][index - 8]
        source_side = "A" if identity["A"]["kind"] == "baseline" else "B"
        comparisons = []
        for dimension, (direction, severity) in enumerate(zip(cells[1:7:2], cells[2:7:2])):
            if direction not in {"A更好", "B更好", "无明确优劣", "无法判断"}:
                issues.append(f"{cells[0]} 第 {dimension+1} 项方向缺失/无效")
            conflict = (
                direction in {"A更好", "B更好"} and severity not in {"轻微", "明显"}
            ) or (
                direction == "无明确优劣" and severity not in {"无明显差异", "有差异但难判优劣"}
            ) or (direction == "无法判断" and severity != "无法判断")
            if conflict:
                issues.append(f"{cells[0]} 第 {dimension+1} 项方向与程度冲突：{direction}/{severity}")
            normalized = "来源更好" if direction == source_side + "更好" else (
                "探针更好" if direction in {"A更好", "B更好"} else direction)
            comparisons.append({"direction": direction, "severity": severity,
                                "relative": normalized, "conflict": conflict})
        pairs.append({
            "id": cells[0], "kind": next(identity[s]["kind"] for s in ("A", "B")
                                       if identity[s]["kind"] != "baseline"),
            "source_side": source_side, "comparisons": comparisons,
            "status": cells[7], "problem_time": cells[8], "evidence": cells[9],
            "seconds": cells[10], "view": cells[11],
        })
    session = {f"B{i}": workbook.worksheets[2].cell(i, 2).value
               for i in (6, 7, 8, 9, 10, 11, 12, 17, 18, 19)}
    distributions = {
        dimension: dict(collections.Counter(r["scores"][i] for r in singles))
        for i, dimension in enumerate(("画质与结构", "时序与运动自然度", "提示词符合度"))
    }
    result = {
        "received_date": "2026-10-22", "source_sha256": digest,
        "archive": str(archived), "single_count": len(singles), "pair_count": len(pairs),
        "distributions_all_16_including_probes": distributions,
        "singles": singles, "pairs": pairs, "session": session, "issues": issues,
        "missing_single_duration": sum(r["seconds"] is None for r in singles),
        "missing_pair_duration": sum(r["seconds"] is None for r in pairs),
        "source_workbook_unchanged": sha(source) == digest,
    }
    (destination / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
