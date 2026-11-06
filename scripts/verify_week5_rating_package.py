"""Verify a Week5 reviewer package, then create a private-mapping-free ZIP."""

import argparse
import csv
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

import av
import numpy as np

from blind_pack import read_video
from blind_pack_followup import build_tasks, load_inputs
from metrics import ROOT, sha256, write_json

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
FORBIDDEN_PUBLIC_TERMS = (
    "mapping.json", "baseline.jsonl", "sageattn", "fp8_scaled",
    "P01_s", "P02_s", "P03_s", "P04_s", "P05_s", "P06_s",
    "P07_s", "P08_s", "P09_s", "P10_s", "P11_s", "P12_s",
)


def private_identifiers(mapping):
    values = set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"source_key", "source_ticket"} and isinstance(item, str):
                    values.add(item)
                if key == "id" and isinstance(item, str) and re.fullmatch(r"P\d{2}_s\d+", item):
                    values.add(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(mapping)
    return values


def inspect_workbook(workbook_path, tasks, mapping):
    secrets = private_identifiers(mapping)
    with zipfile.ZipFile(workbook_path) as archive:
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet_names = [node.attrib["name"] for node in book.findall("s:sheets/s:sheet", NS)]
        if sheet_names != ["评分记录", "会话记录"]:
            raise ValueError(f"Unexpected workbook sheets: {sheet_names}")
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(text.text or "" for text in item.findall(".//s:t", NS))
                for item in root.findall("s:si", NS)
            ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        cells = {
            cell.attrib["r"]: cell
            for cell in sheet.findall(".//s:sheetData/s:row/s:c", NS)
        }
        expected_last = 7 + len(tasks)
        for row, task in enumerate(tasks, 8):
            for col, expected in {
                "A": row - 7, "B": task["id"], "C": task["type"],
                "O": task["prompt"], "P": task["requirement"],
            }.items():
                cell = cells.get(f"{col}{row}")
                if cell is None:
                    raise ValueError(f"Missing workbook value {col}{row}")
                raw = cell.find("s:v", NS)
                if raw is None:
                    raise ValueError(f"Missing stored value {col}{row}")
                value = raw.text or ""
                if cell.attrib.get("t") == "s":
                    value = shared_strings[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(node.text or "" for node in cell.findall(".//s:t", NS))
                if value != str(expected):
                    raise ValueError(f"Workbook value mismatch at {col}{row}: {value!r}")
            for col in "DEFGHIJKLMN":
                cell = cells.get(f"{col}{row}")
                if cell is not None and (
                    cell.find("s:v", NS) is not None or cell.find("s:is", NS) is not None
                ):
                    raise ValueError(f"Rating input is not blank at {col}{row}")
        validations = sheet.find("s:dataValidations", NS)
        if validations is None or len(validations) < 10:
            raise ValueError("Expected data validation lists are missing")
        status_cell = cells.get("B5")
        if status_cell is None or status_cell.find("s:f", NS) is None:
            raise ValueError("Status counter formula is missing")
        for member in archive.namelist():
            if not member.endswith(".xml"):
                continue
            text = archive.read(member).decode("utf-8")
            if any(token in text for token in FORBIDDEN_PUBLIC_TERMS):
                raise ValueError(f"Private/config identity leaked into workbook: {member}")
            if any(token in text for token in secrets):
                raise ValueError(f"Source identity leaked into workbook: {member}")
    return {"sheets": sheet_names, "task_rows": len(tasks), "rating_inputs_blank": True}


def verify_followup_video_sources(public, private, mapping):
    from PIL import Image

    source_by_ticket, controls, baseline, sage, requirements, _, _ = load_inputs()
    expected = build_tasks(mapping["seed"], mapping["seed"], source_by_ticket,
                           controls, baseline, sage, requirements)
    private_tasks = mapping["tasks"]
    if len(private_tasks) != 10 or len(expected) != 10:
        raise ValueError("Supplementary mapping does not contain ten tasks")
    if [task["id"] for task in private_tasks] != [task["id"] for task in expected]:
        raise ValueError("Supplementary task order differs from the recorded seed")

    public_tasks = json.loads((public / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    if len(public_tasks) != 10 or any(task["type"] != "配对" for task in public_tasks):
        raise ValueError("Public follow-up workload is not ten pair tasks")
    if any(set(task) != {"order", "id", "type", "video", "prompt", "requirement"}
           for task in public_tasks):
        raise ValueError("Unexpected public task fields")

    checks = []
    for expected_task, private_task, public_task in zip(expected, private_tasks, public_tasks):
        for key in ("id", "source_key", "review_kind", "source_ticket",
                    "unresolved_dimensions", "A", "B"):
            if private_task[key] != expected_task[key]:
                raise ValueError(f"Seed replay mismatch for {private_task['id']} field {key}")
        if (private_task["id"] != public_task["id"]
                or private_task["video"] != public_task["video"]
                or private_task["prompt"] != public_task["prompt"]
                or private_task["requirement"] != public_task["requirement"]):
            raise ValueError(f"Private/public task mismatch: {private_task['id']}")
        if {private_task["A"]["kind"], private_task["B"]["kind"]} != {"baseline", "sageattn"}:
            raise ValueError(f"Pair sides do not contain baseline and candidate: {private_task['id']}")
        if private_task["A"]["source_key"] != private_task["B"]["source_key"]:
            raise ValueError(f"Pair source mismatch: {private_task['id']}")
        path = public / private_task["video"]
        if sha256(path) != private_task["public_sha256"]:
            raise ValueError(f"Proxy hash mismatch: {private_task['id']}")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if (len(container.streams) != 1 or stream.codec_context.name != "h264"
                    or stream.codec_context.format.name != "yuv420p"
                    or stream.width != 1664 or stream.height != 516
                    or float(stream.average_rate) != 16):
                raise ValueError(f"Proxy format mismatch: {private_task['id']}")
            if any(key in container.metadata for key in (
                    "prompt", "seed", "comment", "description", "title")):
                raise ValueError(f"Unexpected MP4 metadata: {private_task['id']}")
            samples = {}
            frame_count = 0
            for index, frame in enumerate(container.decode(stream)):
                frame_count += 1
                if index in (0, 35, 80):
                    samples[index] = frame.to_ndarray(format="rgb24")
        if frame_count != 81 or set(samples) != {0, 35, 80}:
            raise ValueError(f"Incomplete proxy decode: {private_task['id']}")
        errors = []
        for side, source in enumerate((private_task["A"], private_task["B"])):
            if sha256(source["video_path"]) != source["video_sha256"]:
                raise ValueError(f"Source hash mismatch: {source['source_key']}")
            original = read_video(source["video_path"])
            for frame_index, proxy in samples.items():
                crop = proxy[36:, side * 832:(side + 1) * 832]
                error = float(np.mean(
                    (crop.astype(np.float32) - original[frame_index].astype(np.float32)) ** 2
                ))
                errors.append(error)
        max_mse = max(errors)
        if max_mse >= 120:
            raise ValueError(f"Proxy content/orientation mismatch: {private_task['id']} {max_mse}")
        if private_task["id"] == "T101":
            Image.fromarray(samples[35]).save(private / "pair-preview.png")
        checks.append({"id": private_task["id"], "frames": 81,
                       "sampled_indices": [0, 35, 80], "max_proxy_mse": max_mse})

    counts = {
        "disputed_pairs": sum(task["review_kind"] == "disputed" for task in private_tasks),
        "control_pairs": sum(task["review_kind"] == "control" for task in private_tasks),
        "disputed_dimension_answers": sum(
            len(task["unresolved_dimensions"]) for task in private_tasks
            if task["review_kind"] == "disputed"
        ),
        "unique_sources": len({task["source_key"] for task in private_tasks}),
    }
    if counts != {
        "disputed_pairs": 6, "control_pairs": 4,
        "disputed_dimension_answers": 8, "unique_sources": 10,
    }:
        raise ValueError(f"Supplementary task counts do not match preregistration: {counts}")
    return counts, checks


def verify_text_privacy(public, mapping):
    secrets = private_identifiers(mapping)
    files = [path for path in public.rglob("*") if path.is_file()
             and path.suffix.lower() not in {".mp4", ".xlsx"}]
    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        if any(token in text for token in FORBIDDEN_PUBLIC_TERMS):
            raise ValueError(f"Private/config identity leaked into public text: {path.name}")
        if any(token in text for token in secrets):
            raise ValueError(f"Source identity leaked into public text: {path.name}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    public = ROOT / "outputs" / args.name
    private = ROOT / "notes/blind_private" / args.name
    if not public.is_dir() or not private.is_dir():
        raise FileNotFoundError("Both public and private package folders must exist")
    archive = public.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(f"Preserve existing reviewer ZIP: {archive}")

    mapping = json.loads((private / "mapping.json").read_text(encoding="utf-8"))
    public_tasks_data = json.loads((public / "tasks.json").read_text(encoding="utf-8"))
    tasks = public_tasks_data["tasks"]
    if len(tasks) not in (10, 48):
        raise ValueError(f"Unexpected task count: {len(tasks)}")
    if mapping.get("candidate") == "sageattn":
        if len(tasks) != 10:
            raise ValueError("Sage supplementary package must have ten pairs")
        followup_counts, video_checks = verify_followup_video_sources(public, private, mapping)
    else:
        if len(tasks) != 48 or mapping.get("candidate") not in {"fp8", "fp8_scaled"}:
            raise ValueError("This verifier expects the Sage follow-up or an FP8 candidate package")
        if sum(task["type"] == "配对" for task in tasks) != 36 \
                or sum(task["type"] == "单视频" for task in tasks) != 12:
            raise ValueError("FP8 package workload is not 36 pairs and 12 singles")
        followup_counts, video_checks = None, None

    if [task["order"] for task in tasks] != list(range(1, len(tasks) + 1)):
        raise ValueError("Public task order is not continuous")
    workbook = public / "评分表.xlsx"
    workbook_check = inspect_workbook(workbook, tasks, mapping)
    public_text_files = verify_text_privacy(public, mapping)
    expected_files = {task["video"] for task in tasks}
    expected_files.update({
        "tasks.json", "评分表.xlsx", "备用评分.csv", "评分标准.txt", "先读我.txt",
    })
    actual_files = {
        path.relative_to(public).as_posix() for path in public.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(f"Unexpected reviewer files: {sorted(actual_files ^ expected_files)}")

    csv_path = public / "备用评分.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(tasks):
        raise ValueError("CSV task coverage mismatch")
    for index, (row, task) in enumerate(zip(rows, tasks), 1):
        if row["order"] != str(index) or row["id"] != task["id"]:
            raise ValueError(f"CSV task identity mismatch at row {index}")
        rating_fields = (
            ("结构方向", "结构程度", "时序方向", "时序程度",
             "符合度方向", "符合度程度", "观看状态",
             "相对差异证据或无法判断原因")
            if len(tasks) == 10 else
            ("结构评分或方向", "结构程度", "时序评分或方向", "时序程度",
             "符合度评分或方向", "符合度程度", "观看状态",
             "观察证据/无法判断原因")
        )
        for field in rating_fields:
            if row.get(field):
                raise ValueError(f"CSV rating field is not blank: {task['id']} {field}")

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for relative in sorted(expected_files):
            handle.write(public / relative, arcname=f"{args.name}/{relative}")
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise ValueError("Reviewer ZIP CRC check failed")
        archived = set(handle.namelist())
        if len(archived) != len(expected_files) or any(
                "blind_private" in item or "mapping" in item for item in archived):
            raise ValueError("Reviewer ZIP has missing, duplicate, or private members")

    report = {
        "status": "passed_package_checks",
        "reviewer_files": len(expected_files),
        "task_count": len(tasks),
        "workbook": workbook_check,
        "zip_members": len(expected_files),
        "zip_sha256": sha256(archive),
        "zip_bytes": archive.stat().st_size,
        "public_file_sha256": {
            path.relative_to(public).as_posix(): sha256(path)
            for path in sorted(public.rglob("*")) if path.is_file()
        },
        "private_mapping_excluded": True,
    }
    if followup_counts is not None:
        report["followup_counts"] = followup_counts
        report["video_checks"] = video_checks
    write_json(private / "verification.json", report)
    print(f"PASS: blank workbook, task coverage, privacy, ZIP CRC. ZIP SHA256: {report['zip_sha256']}")
    print(f"Reviewer ZIP: {archive} ({archive.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
