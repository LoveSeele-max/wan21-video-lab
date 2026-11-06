"""Archive Week5 answers and apply reviewer-authorized blank defaults.

XLSX is read only here; the companion Artifact Tool script authors copies.
No private mapping is read during intake or verification.
"""

import argparse
from collections import Counter
from copy import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import openpyxl
from lxml import etree

from audit_candidate_ratings import consistent, DIMENSIONS, SCORES


ROOT = Path(__file__).resolve().parent.parent
NAMES = ("week5-sage-followup-R01", "week5-fp8-R01", "week5-fp8s-R01")
RUN = ROOT / "notes/blind_private/week5-rating-intake-20261105"
DELIVERY = ROOT / "outputs/week5-ratings-completed-20261105"
AUTHORIZATION = (
    "用户2026-11-05说明：均已评分完毕；配对视频未填写的评分空格为无差异选项，"
    "单视频未填写的评分空格为2分选项。只填评分选项，不推断观看/会话/证据。"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, data):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def blank(value):
    return value is None or isinstance(value, str) and not value.strip()


def extract(path, tasks):
    wb = openpyxl.load_workbook(path, data_only=False)
    try:
        assert wb.sheetnames == ["评分记录", "会话记录"]
        sheet = wb["评分记录"]
        records = []
        for row, task in enumerate(tasks, 8):
            values = [sheet.cell(row, col).value for col in range(1, 17)]
            assert (*values[:3], *values[14:16]) == (
                task["order"], task["id"], task["type"], task["prompt"],
                task["requirement"]), (row, task["id"])
            records.append({
                "id": task["id"], "type": task["type"], "excel_row": row,
                "dimensions": {
                    dim: {"direction": values[col], "severity": values[col + 1]}
                    if task["type"] == "配对" else {"score": values[col]}
                    for dim, col in DIMENSIONS
                },
                "playback_status": values[9], "problem_time": values[10],
                "absolute_defects": values[11],
                "evidence": values[12], "view": values[13],
            })
        sessions = [
            dict(zip(("date", "player", "reviewer", "task_range",
                      "prior_exposure", "discussion", "notes"), values))
            for values in wb["会话记录"].iter_rows(
                min_row=6, max_row=25, max_col=7, values_only=True)
            if any(not blank(value) for value in values)
        ]
        return records, sessions
    finally:
        wb.close()


def audit(records, sessions):
    conflicts, issues = [], []
    for record in records:
        for dim, answer in record["dimensions"].items():
            if record["type"] == "配对":
                valid = consistent(answer["direction"], answer["severity"])
                if not valid:
                    conflicts.append({"id": record["id"], "dimension": dim, **answer,
                                      "evidence": record["evidence"]})
                needs = (answer["direction"], answer["severity"]) != (
                    "无明确优劣", "无明显差异")
            else:
                valid = answer["score"] in SCORES
                needs = answer["score"] != "2-较好"
                if not valid:
                    issues.append({"id": record["id"], "dimension": dim,
                                   "field": "invalid_score", **answer})
            if needs:
                for field in ("evidence", "problem_time"):
                    present = not blank(record[field])
                    if field == "evidence" and record["type"] == "单视频":
                        present = present or not blank(record["absolute_defects"])
                    if not present:
                        issues.append({"id": record["id"], "dimension": dim,
                                       "field": field})
        for field in ("playback_status", "view"):
            if blank(record[field]):
                issues.append({"id": record["id"], "field": field})
        if record["playback_status"] == "无法播放":
            issues.append({"id": record["id"], "field": "unplayable_with_scores"})
    session_missing = [
        {"session_index": i, "fields": [key for key in (
            "date", "player", "reviewer", "task_range", "prior_exposure", "discussion")
            if blank(session.get(key))]}
        for i, session in enumerate(sessions, 1)
    ]
    return {
        "mapping_opened": False, "records": records, "sessions_as_entered": sessions,
        "conflicts": conflicts, "metadata_and_evidence_issues": issues,
        "session_missing": session_missing,
        "playback_counts": dict(Counter(str(r["playback_status"]) for r in records)),
        "view_counts": dict(Counter(str(r["view"]) for r in records)),
    }


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    DELIVERY.mkdir(parents=True, exist_ok=True)
    packs = []
    for name in NAMES:
        source = ROOT / "outputs" / name / "评分表.xlsx"
        digest = sha(source)
        returned = ROOT / "notes/blind_private" / name / f"returned-{digest[:12]}"
        returned.mkdir(exist_ok=True)
        original = returned / "original.xlsx"
        if not original.exists():
            shutil.copy2(source, original)
        assert sha(original) == digest == sha(source)
        tasks = read(source.parent / "tasks.json")["tasks"]
        expected = Counter({"配对": 10}) if name == NAMES[0] else Counter({"配对": 36, "单视频": 12})
        assert Counter(t["type"] for t in tasks) == expected
        records, sessions = extract(original, tasks)
        fills = []
        for record in records:
            for dim, col in DIMENSIONS:
                answer = record["dimensions"][dim]
                defaults = {"direction": "无明确优劣", "severity": "无明显差异"} \
                    if record["type"] == "配对" else {"score": "2-较好"}
                for offset, (field, default) in enumerate(defaults.items()):
                    if blank(answer[field]):
                        cell = f"{openpyxl.utils.get_column_letter(col + 1 + offset)}{record['excel_row']}"
                        fills.append({"sheet": "评分记录", "cell": cell, "id": record["id"],
                                      "dimension": dim, "field": field,
                                      "before": answer[field], "after": default})
                        answer[field] = default
        report = audit(records, sessions)
        report.update({
            "source_sha256": digest, "source": str(source), "original": str(original),
            "authorization": AUTHORIZATION, "fills": fills,
            "fill_count": len(fills), "snapshot_time": datetime.now().astimezone().isoformat(),
            "status": "clarification_required" if report["conflicts"] else "answers_consistent",
        })
        write_new(returned / "authorized_defaults_pre_unblind.json", report)
        pack = {"name": name, "source": str(source), "source_sha256": digest,
                "original": str(original), "returned": str(returned), "fills": fills,
                "completed": str(DELIVERY / f"{name}-completed.xlsx")}
        packs.append(pack)
        print(json.dumps({"name": name, "fill_count": len(fills),
                          "conflicts": report["conflicts"],
                          "issues": len(report["metadata_and_evidence_issues"]),
                          "sessions": sessions,
                          "non_default_records": [r for r in records if any(
                              a != ({"direction": "无明确优劣", "severity": "无明显差异"}
                                    if r["type"] == "配对" else {"score": "2-较好"})
                              for a in r["dimensions"].values())]}, ensure_ascii=False, default=str))
    write_new(RUN / "intake.json", {"authorization": AUTHORIZATION,
                                  "mapping_opened": False, "packs": packs})


def verify():
    manifest = read(RUN / "intake.json")
    for pack in manifest["packs"]:
        original, completed = Path(pack["original"]), Path(pack["completed"])
        assert sha(original) == pack["source_sha256"] == sha(Path(pack["source"]))
        wb_a = openpyxl.load_workbook(original)
        wb_b = openpyxl.load_workbook(completed)
        fills = {(f["sheet"], f["cell"]): f["after"] for f in pack["fills"]}
        assert wb_a.sheetnames == wb_b.sheetnames
        feature_diffs, unexpected = [], []
        for a, b in zip(wb_a, wb_b, strict=True):
            for row in a:
                for cell in row:
                    target = b[cell.coordinate]
                    expected = fills.get((a.title, cell.coordinate), cell.value)
                    if target.value != expected:
                        unexpected.append([a.title, cell.coordinate, expected, target.value])
                    if (copy(cell.font) != copy(target.font) or copy(cell.fill) != copy(target.fill) or
                            copy(cell.alignment) != copy(target.alignment) or copy(cell.border) != copy(target.border) or
                            cell.number_format != target.number_format or
                            copy(cell.protection) != copy(target.protection)):
                        feature_diffs.append([a.title, cell.coordinate, "cell_format"])
            if a.freeze_panes != b.freeze_panes:
                feature_diffs.append([a.title, "freeze_panes"])
            if list(a.tables) != list(b.tables):
                feature_diffs.append([a.title, "tables"])
            for table_name in a.tables:
                if a.tables[table_name].ref != b.tables[table_name].ref:
                    feature_diffs.append([a.title, table_name, "table_range"])
            validations_a = sorted((str(d.sqref), d.type, d.formula1)
                                   for d in a.data_validations.dataValidation)
            validations_b = sorted((str(d.sqref), d.type, d.formula1)
                                   for d in b.data_validations.dataValidation)
            if validations_a != validations_b:
                feature_diffs.append([a.title, "data_validations"])
            for key, row in a.row_dimensions.items():
                if row.height != b.row_dimensions[key].height:
                    feature_diffs.append([a.title, key, "row_height"])
            for key, col in a.column_dimensions.items():
                if abs((col.width or 0) - (b.column_dimensions[key].width or 0)) > 0.02:
                    feature_diffs.append([a.title, key, "column_width"])
        wb_a.close()
        wb_b.close()
        report = {"original_sha256": pack["source_sha256"],
                  "completed_sha256": sha(completed), "fills": len(fills),
                  "unexpected_values": unexpected, "feature_differences": feature_diffs,
                  "mapping_opened": False}
        with zipfile.ZipFile(original) as za, zipfile.ZipFile(completed) as zb:
            assert za.namelist() == zb.namelist()
            assert zb.testzip() is None
            report["unchanged_zip_members"] = [
                n for n in za.namelist() if za.read(n) == zb.read(n)]
            assert set(za.namelist()) - set(report["unchanged_zip_members"]) == {
                "xl/worksheets/sheet1.xml"}
        tasks = read(Path(pack["source"]).parent / "tasks.json")["tasks"]
        records, sessions = extract(completed, tasks)
        final_audit = audit(records, sessions)
        final_audit.update({
            "source_sha256": pack["source_sha256"], "completed_sha256": sha(completed),
            "authorization": AUTHORIZATION, "fills": pack["fills"],
            "status": "clarification_required" if final_audit["conflicts"] else "answers_consistent",
        })
        write_new(Path(pack["returned"]) / "completed_answers_before_unblinding.json", final_audit)
        write_new(Path(pack["returned"]) / "completed_workbook_verification_final.json", report)
        print(json.dumps({"name": pack["name"], "fills": len(fills),
                          "unexpected_value_count": len(unexpected),
                          "feature_differences": feature_diffs,
                          "completed_sha256": report["completed_sha256"]}, ensure_ascii=False))
        assert not unexpected
        assert not feature_diffs


def preserve():
    """Retain source OOXML features unsupported by the Artifact Tool importer.

    Artifact Tool authors and checks the values. Transplant only those values
    into the original worksheet; keep every other ZIP member byte for byte.
    """
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    manifest = read(RUN / "intake.json")
    for pack in manifest["packs"]:
        completed = Path(pack["completed"])
        exported = Path(pack["returned"]) / "artifact_export.xlsx"
        if exported.exists():
            raise FileExistsError(exported)
        shutil.move(str(completed), str(exported))
        wb = openpyxl.load_workbook(exported, data_only=False, read_only=True)
        try:
            values = {f["cell"]: wb[f["sheet"]][f["cell"]].value for f in pack["fills"]}
            assert all(values[f["cell"]] == f["after"] for f in pack["fills"])
        finally:
            wb.close()
        with zipfile.ZipFile(pack["original"]) as source:
            document = etree.fromstring(source.read("xl/worksheets/sheet1.xml"))
            rows = {int(row.get("r")): row for row in document.findall(f"{{{ns}}}sheetData/{{{ns}}}row")}
            for cell, value in values.items():
                row_index = openpyxl.utils.cell.coordinate_from_string(cell)[1]
                row = rows[row_index]
                target = row.find(f"{{{ns}}}c[@r='{cell}']")
                if target is None:
                    target = etree.Element(f"{{{ns}}}c", r=cell)
                    column = openpyxl.utils.cell.column_index_from_string(
                        openpyxl.utils.cell.coordinate_from_string(cell)[0])
                    position = next((i for i, other in enumerate(row)
                                     if openpyxl.utils.cell.column_index_from_string(
                                         openpyxl.utils.cell.coordinate_from_string(other.get("r"))[0]) > column),
                                    len(row))
                    row.insert(position, target)
                for child in list(target):
                    target.remove(child)
                target.set("t", "inlineStr")
                etree.SubElement(etree.SubElement(target, f"{{{ns}}}is"), f"{{{ns}}}t").text = value
            xml = etree.tostring(document, encoding="UTF-8", xml_declaration=True, standalone=True)
            with zipfile.ZipFile(completed, "w", compression=zipfile.ZIP_DEFLATED) as dest:
                for entry in source.infolist():
                    dest.writestr(entry, xml if entry.filename == "xl/worksheets/sheet1.xml"
                                  else source.read(entry.filename))
        print(json.dumps({"name": pack["name"], "preserved_source_features": True,
                          "artifact_authored_values": len(values)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preserve", "verify"])
    args = parser.parse_args()
    {"prepare": prepare, "preserve": preserve, "verify": verify}[args.mode]()
