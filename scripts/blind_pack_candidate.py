"""Prepare the frozen candidate review workload with a private random mapping."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import secrets

import yaml

from blind_pack import action_requirements, encode, pair_arrays
from metrics import ROOT, baseline_jobs, read_jsonl, read_video, sha256, write_json


def select_tasks(baseline, candidate, subset, seed):
    rng = random.Random(seed)
    left = {r["source_key"]: r for r in baseline}
    right = {r["source_key"]: r for r in candidate}
    assert len(left) == len(right) == 36 and left.keys() == right.keys()
    tasks = []
    for key in sorted(left):
        sides = [left[key], right[key]]
        rng.shuffle(sides)
        tasks.append({"type": "配对", "A": sides[0], "B": sides[1]})
    assert len(subset) == len(set(subset)) == 12 and set(subset) <= right.keys()
    tasks.extend({"type": "单视频", "source": right[key]} for key in subset)
    rng.shuffle(tasks)
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=["sageattn", "fp8", "fp8_scaled"], required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.name):
        raise ValueError("Use a neutral single directory name")
    public = ROOT / "outputs" / args.name
    private = ROOT / "notes/blind_private" / args.name
    if public.exists() or private.exists():
        raise FileExistsError("Preserve existing package, mapping and answers")
    frozen = yaml.safe_load((ROOT / "configs/scoring_frozen.yaml").read_text(encoding="utf-8"))
    assert frozen["status"] == "frozen" and not frozen["human_timing_required"]
    assert (frozen["paired_review_per_config"], frozen["absolute_review_per_config"]) == (36, 12)
    baseline = baseline_jobs()
    rows = read_jsonl(ROOT / f"notes/eval/{args.candidate}.jsonl")
    candidate = [{
        "id": r["key"], "source_key": r["key"], "kind": args.candidate,
        "prompt_id": r["prompt_id"], "prompt": r["prompt"],
        "video_path": str(ROOT / "outputs/eval" / args.candidate / f"{r['key']}.mp4"),
        "video_sha256": r["video_sha256"],
    } for r in rows if r["status"] == "ok"]
    requirements, rubric = action_requirements()
    seed = secrets.randbits(128)
    tasks = select_tasks(baseline, candidate,
                         [r["key"] for r in frozen["absolute_subset"]], seed)
    for source in baseline + candidate:
        assert sha256(source["video_path"]) == source["video_sha256"]
    (public / "videos").mkdir(parents=True)
    private.mkdir(parents=True)
    mapping = {
        "seed": seed, "candidate": args.candidate,
        "randomization": "Python random.Random; sorted keys, pair side shuffle then all 48 task shuffle",
        "reviewer": "R01", "created_date": datetime.now().astimezone().date().isoformat(),
        "author_script_sha256": sha256(__file__),
        "scoring_frozen_sha256": sha256(ROOT / "configs/scoring_frozen.yaml"),
        "rubric_sha256": sha256(ROOT / "notes/week3_rubric.md"),
        "baseline_manifest_sha256": sha256(ROOT / "notes/eval/baseline.jsonl"),
        "candidate_manifest_sha256": sha256(ROOT / f"notes/eval/{args.candidate}.jsonl"),
        "proxy_encoding": "libx264 CRF18 medium yuv420p 16fps; same settings on all sides",
        "limits": "Lossy review proxies; metrics use originals. One returning reviewer, prior exposure must be recorded.",
        "tasks": [],
    }
    write_json(private / "mapping.json", mapping)
    public_tasks = {"pack": args.name, "reviewer": "R01", "tasks": []}
    for i, task in enumerate(tasks, 1):
        identifier = f"T{i:03d}"
        relative = f"videos/{identifier}.mp4"
        source = task["source"] if task["type"] == "单视频" else task["A"]
        if task["type"] == "单视频":
            encode(public / relative, read_video(source["video_path"]), 832, 480)
        else:
            assert task["A"]["source_key"] == task["B"]["source_key"]
            assert task["A"]["prompt"] == task["B"]["prompt"]
            encode(public / relative, pair_arrays(read_video(task["A"]["video_path"]),
                                                read_video(task["B"]["video_path"])),
                   1664, 516)
        exposed = {"order": i, "id": identifier, "type": task["type"],
                   "video": relative, "prompt": source["prompt"],
                   "requirement": requirements[source["prompt_id"]]}
        public_tasks["tasks"].append(exposed)
        mapping["tasks"].append({**exposed, **task,
                                 "public_sha256": sha256(public / relative)})
        write_json(private / "mapping.json", mapping)
        print(f"{i}/48 {identifier}: encoded and format checked", flush=True)
    write_json(public / "tasks.json", public_tasks)
    fields = ["order", "id", "type", "结构评分或方向", "结构程度", "时序评分或方向",
              "时序程度", "符合度评分或方向", "符合度程度", "观看状态",
              "问题时间（视频秒）", "观察证据或无法判断原因", "查看方式",
              "prompt", "requirement", "video"]
    with (public / "备用评分.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(public_tasks["tasks"])
    rules = rubric[rubric.index("## 2."):rubric.index("## 6.")]
    (public / "评分标准.txt").write_text(
        "人工评分标准v1.0；三项独立判断，不合成总分。\n"
        "不要求记录人工耗时；问题时间指视频内秒数。\n\n" + rules,
        encoding="utf-8-sig")
    (public / "先读我.txt").write_text(
        "匿名视频评分 / R01\n\n"
        "1. 阅读评分标准.txt；打开评分表.xlsx，先填写会话记录中的日期、播放器、既往观看和讨论情况。\n"
        "2. 按评分记录页T001到T048的混排顺序观看videos中的同编号视频，不按文件类型重新分组。\n"
        "3. 单视频：三个“评分/方向”栏分别选2-较好、1-有缺陷、0-明显失败或无法判断；程度栏留空。\n"
        "4. 配对：左侧A、右侧B，三项分别选A更好、无明确优劣、B更好或无法判断，再各选程度。\n"
        "   A/B更好对应轻微或明显；无明确优劣对应无明显差异或有差异但难判优劣；无法判断对应无法判断。\n"
        "5. 第一遍正常速度完整看，第二遍确认。暂停/慢放/逐帧时记录查看方式；"
        "仅逐帧可见的小问题记录现象，不自动判明显失败。\n"
        "6. 静止本身不扣时序分；要求动作未发生由符合度判断。明确可见失败不能写无法判断。\n"
        "7. 非最高档、存在差异或无法判断时，写观察证据和视频内位置，如1.2-1.8秒或全程。"
        "不猜原因，不猜身份。\n"
        "8. 无法播放时标记状态，评分和程度留空，写错误，保留该行。空白是未评，不是0分。\n"
        "9. 保持同一播放器、正常速度、相近显示条件。宽屏配对可全屏，确保左右同等可见；"
        "看不清时按维度记录，勿强行判断。\n"
        "10. 可以分多次完成，每次在会话记录中新增一行，填写实际日期、编号范围和必要备注。"
        "不要求记录逐条或阶段耗时。\n"
        "11. 评分期间不查实验记录、映射或自动指标，先独立作答，再讨论。"
        "评分结束前不提供身份解释。完成后返回评分表.xlsx。\n\n"
        "备用评分.csv仅供无法使用Excel时选择，二选一填写，避免冲突版本。"
        "CSV使用者同时另附评分者编号、实际日期、播放器及既往接触/讨论情况。\n",
        encoding="utf-8-sig")
    print("PUBLIC:", public, flush=True)
    print("PRIVATE: mapping retained separately; not for the reviewer.", flush=True)


if __name__ == "__main__":
    main()
