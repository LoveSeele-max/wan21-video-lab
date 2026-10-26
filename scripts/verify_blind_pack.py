"""Validate and zip only the reviewer-facing package, retaining audit privately."""
import collections
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import av
import numpy as np
from PIL import Image

from blind_pack import DEFAULT_PACK, KINDS, select_jobs
from metrics import ROOT, baseline_jobs, read_jsonl, read_video, sha256, write_json


def mse(a, b):
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def main():
    public = ROOT / "outputs" / DEFAULT_PACK
    private = ROOT / "notes/blind_private" / DEFAULT_PACK
    mapping = json.loads((private / "mapping.json").read_text(encoding="utf-8"))
    tasks = json.loads((public / "tasks.json").read_text(encoding="utf-8"))
    baseline = baseline_jobs()
    probes = read_jsonl(ROOT / "notes/eval/probes/manifest.jsonl")
    singles, pairs = select_jobs(baseline, probes, mapping["seed"])
    assert [r["id"] for r in singles] == [r["source"]["id"] for r in mapping["singles"]]
    assert [[r["left"]["id"], r["right"]["id"]] for r in pairs] == [
        [r["A"]["id"], r["B"]["id"]] for r in mapping["pairs"]]
    ordinary = [r["source"] for r in mapping["singles"] if r["source"]["kind"] == "baseline"]
    artificial = [r["source"] for r in mapping["singles"] if r["source"]["kind"] != "baseline"]
    assert len(ordinary) == 12 and len({r["prompt_id"] for r in ordinary}) == 12
    assert collections.Counter(r["kind"] for r in artificial) == collections.Counter(KINDS)
    assert len(mapping["singles"]) == 16 and len(mapping["pairs"]) == 4
    assert all(set(r) == {"id", "video", "prompt", "requirement"}
               for r in tasks["singles"] + tasks["pairs"])
    video_audit = []
    for phase in ("singles", "pairs"):
        for row in mapping[phase]:
            path = public / row["video"]
            assert sha256(path) == row["public_sha256"]
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                assert len(container.streams) == 1 and float(stream.average_rate) == 16
                assert stream.codec_context.name == "h264"
                assert stream.codec_context.format.name == "yuv420p"
                assert not any(k in container.metadata for k in (
                    "prompt", "seed", "comment", "description", "title"))
                selected = {i: f.to_ndarray(format="rgb24")
                            for i, f in enumerate(container.decode(stream)) if i in (0, 35, 80)}
                assert stream.frames == 81 and len(selected) == 3
                expected_shape = (480, 832, 3) if phase == "singles" else (516, 1664, 3)
                assert all(f.shape == expected_shape for f in selected.values())
            if phase == "singles":
                originals = read_video(row["source"]["video_path"])
                error = max(mse(selected[i], originals[i]) for i in selected)
                assert error < 120, f"Excessive proxy error: {row['id']}"
            else:
                a = read_video(row["A"]["video_path"])
                b = read_video(row["B"]["video_path"])
                errors = [max(mse(selected[i][36:, :832], a[i]),
                              mse(selected[i][36:, 832:], b[i])) for i in selected]
                error = max(errors)
                assert error < 120, f"Pair orientation/content error: {row['id']}"
                if row["id"] == "C001":
                    Image.fromarray(selected[35]).save(private / "pair-video-preview.jpg")
            video_audit.append({"id": row["id"], "decoded_frames": 81,
                                "checked_indices": [0, 35, 80], "max_proxy_mse": error})
    # The workbook has no private sheets; score/status/duration fields remain blank.
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    workbook_path = public / "评分表.xlsx"
    with zipfile.ZipFile(workbook_path) as workbook:
        book = ET.fromstring(workbook.read("xl/workbook.xml"))
        assert [s.attrib["name"] for s in book.findall("s:sheets/s:sheet", ns)] == [
            "单视频评分", "配对评分", "会话记录"]
        for sheet_name, rows, cols in (
            ("sheet1.xml", range(8, 24), "BCDEFGHI"),
            ("sheet2.xml", range(8, 12), "BCDEFGHIJKL"),
        ):
            sheet = ET.fromstring(workbook.read("xl/worksheets/" + sheet_name))
            cells = {c.attrib["r"]: c for c in sheet.findall(".//s:sheetData/s:row/s:c", ns)}
            for row in rows:
                for col in cols:
                    cell = cells.get(f"{col}{row}")
                    if cell is not None:
                        assert cell.find("s:v", ns) is None and cell.find("s:is", ns) is None
            assert sheet.find("s:dataValidations", ns) is not None
            assert sheet.find("s:sheetViews/s:sheetView/s:pane", ns) is not None
        for member in workbook.namelist():
            if member.endswith(".xml"):
                text = workbook.read(member).decode("utf-8")
                assert not any(token in text for token in (
                    "P01_s", "P04_s", "P07_s", "local_warp", "mapping.json",
                    "baseline.jsonl", "E:/cstudy", "E:\\cstudy"))
    expected_files = {t["video"] for t in tasks["singles"] + tasks["pairs"]}
    expected_files.update({"tasks.json", "单视频备用评分.csv", "配对备用评分.csv",
                           "评分标准.txt", "先读我.txt", "评分表.xlsx"})
    actual_files = {p.relative_to(public).as_posix() for p in public.rglob("*") if p.is_file()}
    assert actual_files == expected_files, "Unexpected files in public folder"
    archive = public.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError("Do not overwrite an existing delivered ZIP")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for relative in sorted(expected_files):
            handle.write(public / relative, arcname=f"{DEFAULT_PACK}/{relative}")
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        assert len(handle.namelist()) == 26
        assert all("private" not in p and "mapping" not in p for p in handle.namelist())
    write_json(private / "verification.json", {
        "status": "ok", "date": "2026-10-21", "single_videos": 16, "pair_videos": 4,
        "unique_baseline_prompts": 12, "probe_types": 4, "rating_inputs_blank": True,
        "private_mapping_excluded": True, "archive_members": 26,
        "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
        "video_checks": video_audit,
        "public_file_sha256": {p: sha256(public / p) for p in sorted(expected_files)},
    })
    print("PASS: frozen randomization, source coverage, video format/content, blank ratings, privacy, ZIP CRC.")
    print(f"Reviewer ZIP: {archive}; {archive.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
