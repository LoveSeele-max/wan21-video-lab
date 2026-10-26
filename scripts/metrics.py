"""Frozen Day 4 metrics; originals are read-only and errors remain in the output."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
SCORES = ("ssim_mean", "ssim_min", "clip_mean", "clip_min")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def read_video(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    frames = np.asarray(frames)
    if frames.shape != (81, 480, 832, 3) or abs(fps - 16) > 1e-6:
        raise ValueError(f"Unexpected video format: {frames.shape}, fps={fps}")
    return frames


def ssim_scores(frames):
    from skimage.metrics import structural_similarity

    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    return [float(structural_similarity(
        a, b, data_range=255, win_size=11, gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False, K1=0.01, K2=0.03,
    )) for a, b in zip(gray, gray[1:])]


class ClipScorer:
    def __init__(self, device):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        torch.set_num_threads(4)
        directory = ROOT / "local/models/clip-vit-base-patch32"
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        if provenance["revision"] != REVISION:
            raise ValueError("CLIP revision mismatch")
        for filename, expected in provenance["files"].items():
            if sha256(directory / filename) != expected:
                raise ValueError(f"CLIP hash mismatch: {filename}")
        self.model = CLIPModel.from_pretrained(
            directory, local_files_only=True, attn_implementation="eager"
        ).to(device=device, dtype=torch.float32).eval()
        self.processor = CLIPProcessor.from_pretrained(directory, local_files_only=True)

    def score(self, frames, prompt):
        values = []
        torch = self.torch
        for start in range(0, 21, 7):
            images = [frames[index] for index in list(range(0, 81, 4))[start:start + 7]]
            inputs = self.processor(
                text=[prompt], images=images, return_tensors="pt",
                padding=True, truncation=False,
            ).to(self.device)
            with torch.inference_mode():
                output = self.model(**inputs)
                # CLIP forward returns normalized projected embeddings.
                cosine = output.image_embeds @ output.text_embeds.T
            values.extend(cosine[:, 0].cpu().tolist())
        return values


def baseline_jobs():
    import yaml

    rows = read_jsonl(ROOT / "notes/eval/baseline.jsonl")
    frozen = yaml.safe_load((ROOT / "configs/prompts_frozen.yaml").read_text(encoding="utf-8"))
    expected = {f"{p['id']}_s{s}": p for p in frozen["prompts"] for s in frozen["seeds"]}
    if len(rows) != 36 or {r["key"] for r in rows} != set(expected):
        raise ValueError("Baseline coverage differs from frozen set")
    for row in rows:
        if row["status"] != "ok" or row["prompt"] != expected[row["key"]]["prompt"]:
            raise ValueError(f"Baseline not valid: {row['key']}")
    return [{
        "id": r["key"], "kind": "baseline", "source_key": r["key"],
        "prompt_id": r["prompt_id"], "prompt": r["prompt"],
        "video_path": str(ROOT / "outputs/eval/baseline" / f"{r['key']}.mp4"),
        "video_sha256": r["video_sha256"],
    } for r in rows]


def classify(value, low, high):
    if value < low or value > high:
        return "能区分"
    if high > low and min(value - low, high - value) <= 0.05 * (high - low):
        return "部分区分"
    return "不能区分"


def report(rows, directory):
    baseline = [r for r in rows if r["kind"] == "baseline" and r["status"] == "ok"]
    if len(baseline) != 36 or any(r["status"] != "ok" for r in rows):
        return
    by_id = {r["id"]: r for r in baseline}
    ranges = {score: [min(r[score] for r in baseline), max(r[score] for r in baseline)]
              for score in SCORES}
    table = []
    for row in rows:
        if row["kind"] == "baseline":
            continue
        for score in SCORES:
            low, high = ranges[score]
            table.append({
                "id": row["id"], "kind": row["kind"], "metric": score,
                "value": row[score], "baseline_low": low, "baseline_high": high,
                "source_delta": row[score] - by_id[row["source_key"]][score],
                "range_class": classify(row[score], low, high),
                "outside_direction": "low" if row[score] < low else
                                     "high" if row[score] > high else "inside",
            })
    write_csv(directory / "probe_comparison.csv", table)
    write_json(directory / "baseline_ranges.json", ranges)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row
                               if k not in ("ssim_pairs", "clip_frames")))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "notes/week3_metrics_run")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve existing results; choose another --output: {args.output}")
    jobs = baseline_jobs() + read_jsonl(ROOT / "notes/eval/probes/manifest.jsonl")
    args.output.mkdir(parents=True)
    started = time.time()
    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "transformers", "numpy", "av", "opencv-python", "scikit-image", "scipy")}
    write_json(args.output / "run.json", {
        "status": "running", "device": args.device, "versions": versions,
        "clip_revision": REVISION, "started_unix": started,
        "protocol_sha256": sha256(ROOT / "notes/week3_metrics.md"),
        "script_sha256": sha256(__file__), "clip_frame_indices": list(range(0, 81, 4)),
        "input_manifest_sha256": {
            "baseline": sha256(ROOT / "notes/eval/baseline.jsonl"),
            "probes": sha256(ROOT / "notes/eval/probes/manifest.jsonl"),
        },
    })
    scorer = ClipScorer(args.device)
    rows = []
    with (args.output / "per_video.jsonl").open("w", encoding="utf-8") as handle:
        for job in jobs:
            row = dict(job)
            try:
                if sha256(job["video_path"]) != job["video_sha256"]:
                    raise ValueError("Input video hash mismatch")
                frames = read_video(job["video_path"])
                ssim = ssim_scores(frames)
                clip = scorer.score(frames, job["prompt"])
                row.update(status="ok", ssim_mean=float(np.mean(ssim)), ssim_min=min(ssim),
                           clip_mean=float(np.mean(clip)), clip_min=min(clip),
                           ssim_pairs=ssim, clip_frames=clip)
            except Exception as error:
                row.update(status="error", error=f"{type(error).__name__}: {error}")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
            print(f"{len(rows)}/{len(jobs)} {row['id']}: {row['status']}", flush=True)
    write_csv(args.output / "metrics.csv", rows)
    report(rows, args.output)
    metadata = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
    errors = sum(r["status"] != "ok" for r in rows)
    metadata.update(status="ok" if not errors else "error", errors=errors,
                    completed=len(rows), elapsed_seconds=time.time() - started)
    write_json(args.output / "run.json", metadata)
    print(f"Metrics saved: {args.output}; errors={errors}", flush=True)
    return int(errors > 0)


if __name__ == "__main__":
    sys.exit(main())
