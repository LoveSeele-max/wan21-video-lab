"""Verify candidate review randomization and video sources; do not score quality."""

import argparse
from collections import Counter
import json

import av
import numpy as np
from PIL import Image
import yaml

from blind_pack import action_requirements
from blind_pack_candidate import select_tasks
from metrics import ROOT, baseline_jobs, read_jsonl, sha256, write_json


def sampled_video(path):
    samples = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert len(container.streams) == 1
        assert float(stream.average_rate) == 16
        assert stream.codec_context.name == "h264"
        assert stream.codec_context.format.name == "yuv420p"
        assert not any(k in container.metadata for k in
                       ("prompt", "seed", "comment", "description", "title"))
        count = 0
        for index, frame in enumerate(container.decode(stream)):
            if index in (0, 35, 80):
                samples[index] = frame.to_ndarray(format="rgb24")
            count += 1
        assert count == 81 and len(samples) == 3
    return samples


def mse(a, b):
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    public = ROOT / "outputs" / args.name
    private = ROOT / "notes/blind_private" / args.name
    mapping = json.loads((private / "mapping.json").read_text(encoding="utf-8"))
    tasks = json.loads((public / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    candidate_name = mapping["candidate"]
    rows = read_jsonl(ROOT / f"notes/eval/{candidate_name}.jsonl")
    candidate = [{
        "id": r["key"], "source_key": r["key"], "kind": candidate_name,
        "prompt_id": r["prompt_id"], "prompt": r["prompt"],
        "video_path": str(ROOT / "outputs/eval" / candidate_name / f"{r['key']}.mp4"),
        "video_sha256": r["video_sha256"],
    } for r in rows]
    frozen = yaml.safe_load((ROOT / "configs/scoring_frozen.yaml").read_text(encoding="utf-8"))
    expected = select_tasks(baseline_jobs(), candidate,
                            [r["key"] for r in frozen["absolute_subset"]], mapping["seed"])
    assert len(tasks) == len(mapping["tasks"]) == len(expected) == 48
    assert Counter(t["type"] for t in tasks) == {"配对": 36, "单视频": 12}
    requirements, _ = action_requirements()
    checks = []
    preview_saved = False
    for i, (task, secret, source_task) in enumerate(zip(tasks, mapping["tasks"], expected), 1):
        assert set(task) == {"order", "id", "type", "video", "prompt", "requirement"}
        assert task["order"] == i and task["id"] == f"T{i:03d}"
        assert all(secret[k] == v for k, v in task.items())
        assert all(secret[k] == v for k, v in source_task.items())
        path = public / task["video"]
        assert sha256(path) == secret["public_sha256"]
        samples = sampled_video(path)
        sources = [secret["source"]] if task["type"] == "单视频" else [secret["A"], secret["B"]]
        assert task["prompt"] == sources[0]["prompt"]
        assert task["requirement"] == requirements[sources[0]["prompt_id"]]
        errors = []
        for side, source in enumerate(sources):
            assert sha256(source["video_path"]) == source["video_sha256"]
            original = sampled_video(source["video_path"])
            for frame_id, actual in samples.items():
                if task["type"] == "单视频":
                    assert actual.shape == (480, 832, 3)
                    crop = actual
                else:
                    assert actual.shape == (516, 1664, 3)
                    crop = actual[36:, side * 832:(side + 1) * 832]
                errors.append(mse(crop, original[frame_id]))
        # Same engineering proxy check used in the Week3 package verifier.
        assert max(errors) < 120, task["id"]
        if task["type"] == "配对" and not preview_saved:
            Image.fromarray(samples[35]).save(private / "pair-preview.png")
            preview_saved = True
        checks.append({"id": task["id"], "type": task["type"], "decoded_frames": 81,
                       "checked_source_indices": [0, 35, 80], "max_proxy_mse": max(errors)})
        print(f"{i}/48 {task['id']}: source and proxy checks passed", flush=True)
    report = {
        "status": "passed_video_and_randomization_checks_only",
        "date": "2026-10-27", "pair_count": 36, "single_count": 12,
        "seed_replay_and_fixed_subset_verified": True,
        "all_pair_sides_and_prompts_verified": True,
        "public_proxy_count": 48, "video_checks": checks,
        "mapping_sha256": sha256(private / "mapping.json"),
        "tasks_sha256": sha256(public / "tasks.json"),
        "verifier_sha256": sha256(__file__),
        "limits": [
            "Lossy proxies; three frames per source checked for content/orientation.",
            "MSE guard is a packaging engineering check, not a generation-quality threshold.",
            "No claim of strict independent blind assessment or reviewer agreement.",
        ],
    }
    write_json(private / "video_verification.json", report)
    print("PASS: all review videos and private randomization replay", flush=True)


if __name__ == "__main__":
    main()
