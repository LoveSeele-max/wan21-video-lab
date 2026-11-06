"""Build a private-mapped, 10-pair supplementary SageAttention review package."""

import argparse
import csv
import json
import random
import secrets
from pathlib import Path

import numpy as np
import yaml

from blind_pack import action_requirements, encode, pair_arrays
from metrics import ROOT, baseline_jobs, read_jsonl, read_video, sha256, write_json

WEEK4_PACK = "week4-R01-20261027"
WEEK4_RETURN = "notes/blind_private/week4-R01-20261027/returned-bf3a3a2dd254"
DISPUTED_TICKETS = ("T001", "T002", "T003", "T010", "T032", "T038")
PRIVATE_REVIEWER = "R01"


def load_inputs():
    returned = ROOT / WEEK4_RETURN
    old_mapping = json.loads(
        (ROOT / "notes/blind_private" / WEEK4_PACK / "mapping.json").read_text(encoding="utf-8"))
    old_tasks = {task["id"]: task for task in old_mapping["tasks"]}
    ratings_path = returned / "quality_dimensions.csv"
    with ratings_path.open(encoding="utf-8-sig", newline="") as handle:
        ratings = list(csv.DictReader(handle))

    source_by_ticket = {}
    for ticket in DISPUTED_TICKETS:
        if ticket not in old_tasks or old_tasks[ticket]["type"] != "配对":
            raise ValueError(f"Missing original paired task: {ticket}")
        sides = old_tasks[ticket]
        a, b = sides["A"], sides["B"]
        if a["source_key"] != b["source_key"]:
            raise ValueError(f"Original pair has mismatched sources: {ticket}")
        rows = [row for row in ratings if row["id"] == ticket]
        unresolved = {row["dimension"] for row in rows if row["category"] == "unresolved"}
        if not rows or not unresolved:
            raise ValueError(f"Expected unresolved dimension evidence: {ticket}")
        source_by_ticket[ticket] = {
            "ticket": ticket, "source_key": a["source_key"],
            "unresolved_dimensions": sorted(unresolved),
        }

    by_source = {}
    for row in ratings:
        by_source.setdefault(row["source_key"], []).append(row)
    controls = []
    disputed_sources = {item["source_key"] for item in source_by_ticket.values()}
    for source_key, rows in by_source.items():
        if source_key in disputed_sources or len(rows) != 3:
            continue
        if all(row["category"] != "unresolved"
               and row["effective_direction"] and row["effective_severity"] for row in rows):
            controls.append(source_key)
    if len(controls) < 4:
        raise ValueError("Fewer than four fully resolved control pairs")

    baseline = {row["source_key"]: row for row in baseline_jobs()}
    manifest = read_jsonl(ROOT / "notes/eval/sageattn.jsonl")
    sage = {}
    for row in manifest:
        if row["status"] != "ok":
            raise ValueError(f"Sage source is not successful: {row['key']}")
        key = row["key"]
        sage[key] = {
            "id": key, "kind": "sageattn", "source_key": key,
            "prompt_id": row["prompt_id"], "prompt": row["prompt"],
            "video_path": str(ROOT / "outputs/eval/sageattn" / f"{key}.mp4"),
            "video_sha256": row["video_sha256"],
        }
    if len(baseline) != 36 or len(sage) != 36 or baseline.keys() != sage.keys():
        raise ValueError("Expected the same 36 baseline and Sage samples")
    requirements, rubric = action_requirements()
    return source_by_ticket, controls, baseline, sage, requirements, rubric, ratings_path


def build_tasks(name, seed, source_by_ticket, controls, baseline, sage, requirements):
    rng = random.Random(seed)
    selections = [
        {"review_kind": "disputed", **item}
        for item in source_by_ticket.values()
    ]
    selected_controls = rng.sample(sorted(controls), 4)
    selections.extend({
        "review_kind": "control", "source_key": key,
        "ticket": None, "unresolved_dimensions": [],
    } for key in selected_controls)
    rng.shuffle(selections)

    tasks = []
    for index, selection in enumerate(selections, 101):
        key = selection["source_key"]
        sides = [dict(baseline[key]), dict(sage[key])]
        rng.shuffle(sides)
        if sides[0]["prompt"] != sides[1]["prompt"]:
            raise ValueError(f"Prompt mismatch: {key}")
        tasks.append({
            "order": index - 100,
            "id": f"T{index:03d}",
            "type": "配对",
            "A": sides[0],
            "B": sides[1],
            "prompt": sides[0]["prompt"],
            "requirement": requirements[sides[0]["prompt_id"]],
            "review_kind": selection["review_kind"],
            "source_ticket": selection["ticket"],
            "source_key": key,
            "unresolved_dimensions": selection["unresolved_dimensions"],
        })
    if len({task["source_key"] for task in tasks}) != 10:
        raise ValueError("Supplementary package must contain ten unique video pairs")
    return tasks


def write_public_material(public, name, tasks, rubric):
    public_tasks = {
        "pack": name, "reviewer": PRIVATE_REVIEWER,
        "tasks": [{
            "order": task["order"], "id": task["id"], "type": task["type"],
            "video": task["video"], "prompt": task["prompt"],
            "requirement": task["requirement"],
        } for task in tasks],
    }
    write_json(public / "tasks.json", public_tasks)
    fields = [
        "order", "id", "结构方向", "结构程度", "时序方向", "时序程度",
        "符合度方向", "符合度程度", "观看状态", "问题时间（视频秒）",
        "绝对缺陷记录（非配对结论）", "相对差异证据或无法判断原因",
        "查看方式", "prompt", "requirement", "video",
    ]
    with (public / "备用评分.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in public_tasks["tasks"]:
            writer.writerow({field: task.get(field) for field in fields})
    rating_rules = rubric[rubric.index("## 2."):rubric.index("## 6.")]
    (public / "评分标准.txt").write_text(
        "配对补充评分标准 v1.0\n\n"
        "配对栏只记录 A/B 的相对方向和程度。对任一视频本身的绝对缺陷，"
        "如有，请单独写在“绝对缺陷记录”栏，不要据此推定哪侧相对更差。\n"
        "静止本身不扣时序分；是否满足动作要求由提示词符合度判断。"
        "不要求记录人工耗时。\n\n" + rating_rules,
        encoding="utf-8-sig")
    (public / "先读我.txt").write_text(
        "匿名视频配对补充评分 / R01\n\n"
        "本包是对既有视频的补充评分。评分者此前已经看过部分待评视频；"
        "请在会话记录中如实填写既往观看范围、播放器、实际日期及讨论情况。"
        "本次不能视为新一轮严格独立盲评。\n\n"
        "1. 阅读评分标准.txt，按 T101 到 T110 的顺序观看对应视频。\n"
        "2. 每段视频左侧标为 A，右侧标为 B。结构、时序与运动、提示词符合度分别判断。\n"
        "3. 每个维度分别填写方向和程度。方向选 A更好、无明确优劣、B更好或无法判断；"
        "程度选无明显差异、有差异但难判优劣、轻微、明显或无法判断。\n"
        "4. 对单侧视频本身的绝对缺陷，另填“绝对缺陷记录”；不要将绝对缺陷直接当成配对差异。\n"
        "5. 写可观察到的证据和视频内秒数。静止本身不扣时序分；明确动作要求由符合度判断。\n"
        "6. 无法播放时记录状态和原因，方向及程度留空。无法判断须按维度填写，不当作无差异。\n"
        "7. 先独立作答，不查实验记录、身份映射或自动指标。评分完成前不讨论身份。\n"
        "8. 不要求记录人工耗时。完成后只返回评分表.xlsx。\n\n"
        "CSV 是无法使用 Excel 时的备用表，评分者只使用一种格式，避免冲突版本。\n",
        encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="week5-sage-followup-R01")
    parser.add_argument("--resume", action="store_true",
                        help="Complete a package whose ten proxies and private mapping already exist")
    args = parser.parse_args()
    if not args.name or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in args.name):
        raise ValueError("Use a neutral single directory name")
    public = ROOT / "outputs" / args.name
    private = ROOT / "notes/blind_private" / args.name
    if public.with_suffix(".zip").exists():
        raise FileExistsError("Preserve an existing reviewer ZIP")
    if public.exists() or private.exists():
        if not args.resume or not public.is_dir() or not private.is_dir():
            raise FileExistsError("Existing paths require --resume and both package directories")
        mapping_path = private / "mapping.json"
        if not mapping_path.is_file():
            raise FileNotFoundError("Cannot resume without the private mapping")
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        tasks = mapping.get("tasks", [])
        expected = {task["video"] for task in tasks}
        actual = {path.relative_to(public).as_posix()
                  for path in (public / "videos").glob("*.mp4")}
        other_files = {path.relative_to(public).as_posix()
                       for path in public.rglob("*") if path.is_file()
                       and path.suffix.lower() != ".mp4"}
        if (len(tasks) != 10 or actual != expected
                or not other_files <= {"tasks.json", "备用评分.csv"}
                or len({task["source_key"] for task in tasks}) != 10):
            raise ValueError("Existing partial package does not match a resumable ten-video set")
        public_tasks = json.loads((public / "tasks.json").read_text(encoding="utf-8"))
        expected_public = [{
            "order": task["order"], "id": task["id"], "type": task["type"],
            "video": task["video"], "prompt": task["prompt"],
            "requirement": task["requirement"],
        } for task in tasks]
        if public_tasks.get("tasks") != expected_public:
            raise ValueError("Existing public task list differs from the private mapping")
        csv_path = public / "备用评分.csv"
        if csv_path.exists():
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                if len(list(csv.reader(handle))) != 1:
                    raise ValueError("Refusing to replace a CSV that may contain ratings")
        for task in tasks:
            if sha256(public / task["video"]) != task["public_sha256"]:
                raise ValueError(f"Existing proxy hash mismatch: {task['id']}")
        _, rubric = action_requirements()
        write_public_material(public, args.name, tasks, rubric)
    else:
        if args.resume:
            raise FileNotFoundError("--resume requested but no partial package exists")
        source_by_ticket, controls, baseline, sage, requirements, rubric, ratings_path = load_inputs()
        seed = secrets.randbits(128)
        tasks = build_tasks(args.name, seed, source_by_ticket, controls, baseline, sage, requirements)
        for task in tasks:
            for side in (task["A"], task["B"]):
                if sha256(side["video_path"]) != side["video_sha256"]:
                    raise ValueError(f"Source hash mismatch: {side['source_key']}")

        (public / "videos").mkdir(parents=True)
        private.mkdir(parents=True)
        mapping = {
            "seed": seed,
            "candidate": "sageattn",
            "reviewer": PRIVATE_REVIEWER,
            "created_date": __import__("datetime").date.today().isoformat(),
            "randomization": "Python random.Random; sample four controls, shuffle ten tasks and sides",
            "source_rating_sha256": sha256(ratings_path),
            "week4_mapping_sha256": sha256(
                ROOT / "notes/blind_private" / WEEK4_PACK / "mapping.json"),
            "author_script_sha256": sha256(__file__),
            "rubric_sha256": sha256(ROOT / "notes/week3_rubric.md"),
            "proxy_encoding": "libx264 CRF18 medium yuv420p 16fps; paired layout A left, B right",
            "prior_exposure": "R01 reviewed these source videos in Week4; supplementary post-unblinding review",
            "tasks": [],
        }
        write_json(private / "mapping.json", mapping)
        for task in tasks:
            relative = f"videos/{task['id']}.mp4"
            task["video"] = relative
            encode(public / relative, pair_arrays(
                read_video(task["A"]["video_path"]), read_video(task["B"]["video_path"])),
                1664, 516)
            private_task = {
                key: task[key] for key in (
                    "order", "id", "type", "review_kind", "source_ticket",
                    "source_key", "unresolved_dimensions")
            }
            private_task.update({
                "video": relative, "prompt": task["prompt"],
                "requirement": task["requirement"], "A": task["A"], "B": task["B"],
                "public_sha256": sha256(public / relative),
            })
            mapping["tasks"].append(private_task)
            write_json(private / "mapping.json", mapping)
            print(f"{task['id']}: paired proxy encoded and decoded", flush=True)
        write_public_material(public, args.name, tasks, rubric)
    print(f"PUBLIC: {public}", flush=True)
    print(f"PRIVATE: {private}; mapping must not be shared with reviewer.", flush=True)


if __name__ == "__main__":
    main()
