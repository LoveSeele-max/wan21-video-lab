"""Prepare neutral review videos and separate the private randomization mapping."""
import argparse
import csv
import json
import random
import secrets
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from metrics import ROOT, baseline_jobs, read_jsonl, read_video, sha256, write_json

DEFAULT_PACK = "week3-R01-20261021"
KINDS = ("static", "flicker", "blur", "local_warp")


def select_jobs(baseline, probes, seed):
    rng = random.Random(seed)
    groups = {}
    for row in baseline:
        groups.setdefault(row["prompt_id"], []).append(row)
    if len(groups) != 12 or any(len(rows) != 3 for rows in groups.values()):
        raise ValueError("Expected 12 prompts with three seeds each")
    singles = [dict(rng.choice(sorted(groups[key], key=lambda r: r["id"])))
               for key in sorted(groups)]
    selected_probes = []
    for kind in KINDS:
        candidates = sorted((p for p in probes if p["kind"] == kind), key=lambda p: p["id"])
        if len(candidates) != 3:
            raise ValueError(f"Expected three candidates for {kind}")
        selected_probes.append(dict(rng.choice(candidates)))
    singles.extend(selected_probes)
    rng.shuffle(singles)
    pairs = []
    by_id = {r["id"]: r for r in baseline}
    for probe in selected_probes:
        sides = [dict(by_id[probe["source_key"]]), dict(probe)]
        rng.shuffle(sides)
        pairs.append({"left": sides[0], "right": sides[1]})
    rng.shuffle(pairs)
    return singles, pairs


def encode(path, arrays, width, height):
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=16)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium", "threads": "4"}
        for index, array in enumerate(arrays):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        if (len(frames), stream.width, stream.height, float(stream.average_rate)) != (
                81, width, height, 16):
            raise RuntimeError(f"Encoded video failed format check: {path.name}")
        if len(container.streams) != 1 or any(
                k not in {"major_brand", "minor_version", "compatible_brands", "encoder"}
                for k in container.metadata):
            raise RuntimeError(f"Unexpected public metadata: {path.name}")


def pair_arrays(left, right):
    header = Image.new("RGB", (1664, 36), "#181818")
    draw = ImageDraw.Draw(header)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 23)
    draw.text((20, 3), "A", font=font, fill="white")
    draw.text((852, 3), "B", font=font, fill="white")
    header = np.asarray(header)
    for a, b in zip(left, right):
        yield np.concatenate((header, np.concatenate((a, b), axis=1)), axis=0)


def action_requirements():
    text = (ROOT / "notes/week3_rubric.md").read_text(encoding="utf-8")
    result = {}
    for line in text.splitlines():
        if line.startswith("| P") and line.count("|") == 4:
            _, key, _, requirement, _ = line.split("|")
            result[key.strip()] = requirement.strip()
    if len(result) != 12:
        raise ValueError("Cannot read twelve frozen action requirements")
    return result, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_PACK)
    args = parser.parse_args()
    if Path(args.name).name != args.name or "/" in args.name or "\\" in args.name:
        raise ValueError("Pack name must be a single directory name")
    public = ROOT / "outputs" / args.name
    private = ROOT / "notes/blind_private" / args.name
    if public.exists() or private.exists():
        raise FileExistsError("Preserve existing review package and ratings")
    baseline = baseline_jobs()
    probes = read_jsonl(ROOT / "notes/eval/probes/manifest.jsonl")
    requirements, rubric = action_requirements()
    seed = secrets.randbits(128)
    singles, pairs = select_jobs(baseline, probes, seed)
    for source in singles + [s for p in pairs for s in (p["left"], p["right"])]:
        if sha256(source["video_path"]) != source["video_sha256"]:
            raise ValueError(f"Source hash mismatch: {source['id']}")
    (public / "01_single").mkdir(parents=True)
    (public / "02_pairs").mkdir()
    private.mkdir(parents=True)
    mapping = {
        "seed": seed, "randomization": "Python random.Random; sorted candidates then choice/shuffle",
        "author_script_sha256": sha256(__file__),
        "rubric_sha256": sha256(ROOT / "notes/week3_rubric.md"),
        "baseline_manifest_sha256": sha256(ROOT / "notes/eval/baseline.jsonl"),
        "probe_manifest_sha256": sha256(ROOT / "notes/eval/probes/manifest.jsonl"),
        "proxy_encoding": "libx264 CRF18 medium yuv420p 16fps; no source metadata/audio",
        "reviewer": "R01", "date": "2026-10-21", "singles": [], "pairs": [],
        "note": "Rating proxies are lossy; metrics use originals. Same encoding for all sides.",
    }
    write_json(private / "mapping.json", mapping)
    public_tasks = {"pack": args.name, "reviewer": "R01", "singles": [], "pairs": []}
    for index, source in enumerate(singles, 1):
        identifier = f"V{index:03d}"
        relative = f"01_single/{identifier}.mp4"
        encode(public / relative, read_video(source["video_path"]), 832, 480)
        task = {"id": identifier, "video": relative, "prompt": source["prompt"],
                "requirement": requirements[source["prompt_id"]]}
        public_tasks["singles"].append(task)
        mapping["singles"].append(dict(
            task, source=source, public_sha256=sha256(public / relative)))
        write_json(private / "mapping.json", mapping)
        print(f"{identifier}: encoded and decoded", flush=True)
    for index, pair in enumerate(pairs, 1):
        identifier = f"C{index:03d}"
        relative = f"02_pairs/{identifier}.mp4"
        left, right = pair["left"], pair["right"]
        if left["prompt"] != right["prompt"] or left["source_key"] != right["source_key"]:
            raise ValueError("Invalid pair identity")
        encode(public / relative, pair_arrays(
            read_video(left["video_path"]), read_video(right["video_path"])), 1664, 516)
        task = {"id": identifier, "video": relative, "prompt": left["prompt"],
                "requirement": requirements[left["prompt_id"]]}
        public_tasks["pairs"].append(task)
        mapping["pairs"].append(dict(task, A=left, B=right,
                                     public_sha256=sha256(public / relative)))
        write_json(private / "mapping.json", mapping)
        print(f"{identifier}: encoded and decoded", flush=True)
    write_json(public / "tasks.json", public_tasks)
    for key, filename, fields in (
        ("singles", "单视频备用评分.csv", ["id", "prompt", "requirement", "video",
          "画质与结构", "时序与运动自然度", "提示词符合度", "观看状态", "问题时间",
          "观察证据", "查看方式", "评分耗时秒", "评分者", "日期"]),
        ("pairs", "配对备用评分.csv", ["id", "prompt", "requirement", "video",
          "结构方向", "结构程度", "时序方向", "时序程度", "符合度方向", "符合度程度",
          "观看状态", "问题时间", "观察证据", "查看方式", "评分耗时秒", "评分者", "日期"]),
    ):
        with (public / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(public_tasks[key])
    # Omit calibration history, model identity, project paths and source scores.
    rating_rules = rubric[rubric.index("## 2."):rubric.index("## 6.")]
    (public / "评分标准.txt").write_text(
        "人工评分标准 v1.0（2026-10-18 冻结）\n\n" + rating_rules, encoding="utf-8-sig")
    (public / "先读我.txt").write_text(
        "视频独立试评 / R01\n\n"
        "1. 先读评分标准.txt，准备同一播放器，正常速度播放。\n"
        "2. 打开评分表.xlsx，在会话记录页填写评分者编号 R01、日期和播放器。\n"
        "3. 先完成单视频评分页的 V001 到 V016，对照每行提示词及动作要求，"
        "播放 01_single 中的对应视频。三项独立评分，不求总分。\n"
        "4. 第一遍完整观看，第二遍确认问题。可暂停、慢放，但要记录查看方式。"
        "证据写可见现象和视频内秒数，不猜原因。\n"
        "5. 看不到关键部位时可按维度选无法判断；明确没有要求的动作时符合度为 0。"
        "无法播放时记录状态、原因，分数留空，不删除行。\n"
        "6. 每条从开始播放到填完记录计时，单位为秒。"
        "会话记录页分别填单视频阶段、配对阶段总用时及休息秒数；"
        "总用时包含播放、重看和填表，排除读标准、准备及休息。\n"
        "7. 完成并保存单视频评分后，再打开 02_pairs，按 C001 到 C004 评分。"
        "画面左侧 A、右侧 B，三项分别选 A更好/无明确优劣/B更好/无法判断，"
        "分别记录程度；没有总体胜负。\n"
        "8. 独立作答，评分过程中先不讨论好坏，也不查原始文件或实验记录。"
        "完成后把评分表.xlsx 返回组织者。日期留空，按实际评分日期填写。\n\n"
        "CSV 是没有 Excel 时的备用表，只填一种即可，避免出现两份冲突答案。"
        "所有评分项均为空白；视频目录按编号排序即可。\n",
        encoding="utf-8-sig")
    print(f"Public package prepared: {public}", flush=True)
    print("Private mapping retained separately; never include it in the reviewer ZIP.", flush=True)


if __name__ == "__main__":
    main()
