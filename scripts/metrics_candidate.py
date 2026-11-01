"""Compute candidate metrics with the unchanged Week3 functions, on CPU FP32.

Preserve baseline results and write to a fresh output directory. Paired metric
deltas are anomaly hints only; no automatic quality pass/fail or new thresholds.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

import metrics


ROOT = metrics.ROOT


def prepare(candidate):
    rows = metrics.read_jsonl(ROOT / f"notes/eval/{candidate}.jsonl")
    reference_path = ROOT / "notes/week3_metrics_run/per_video.jsonl"
    reference = {r["id"]: r for r in metrics.read_jsonl(reference_path)
                 if r["kind"] == "baseline"}
    jobs = metrics.baseline_jobs()
    assert len(rows) == len({r["key"] for r in rows}) == 36
    assert set(reference) == {j["id"] for j in jobs} == {r["key"] for r in rows}
    reference_run = json.loads(
        (ROOT / "notes/week3_metrics_run/run.json").read_text(encoding="utf-8"))
    assert reference_run["status"] == "ok" and reference_run["device"] == "cpu"
    assert reference_run["clip_revision"] == metrics.REVISION
    assert reference_run["script_sha256"] == metrics.sha256(Path(metrics.__file__))
    assert reference_run["protocol_sha256"] == metrics.sha256(ROOT / "notes/week3_metrics.md")
    assert reference_run["input_manifest_sha256"]["baseline"] == metrics.sha256(
        ROOT / "notes/eval/baseline.jsonl")
    versions = {name: importlib.metadata.version(name)
                for name in reference_run["versions"]}
    assert versions == reference_run["versions"], "Metric dependency versions changed"
    by_key = {r["key"]: r for r in rows}
    selected = []
    for job in jobs:
        r, b = by_key[job["id"]], reference[job["id"]]
        assert r["status"] == b["status"] == "ok"
        assert r["prompt"] == b["prompt"] == job["prompt"]
        assert b["video_sha256"] == job["video_sha256"]
        assert metrics.sha256(job["video_path"]) == b["video_sha256"]
        path = ROOT / "outputs/eval" / candidate / f"{r['key']}.mp4"
        assert metrics.sha256(path) == r["video_sha256"]
        selected.append({
            "id": r["key"], "kind": candidate, "source_key": r["key"],
            "prompt_id": r["prompt_id"], "prompt": r["prompt"],
            "video_path": str(path), "video_sha256": r["video_sha256"],
        })
    return selected, reference, versions, reference_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=["sageattn", "fp8", "fp8_scaled"],
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(f"Preserve existing output: {options.output}")
    jobs, baseline, versions, reference_path = prepare(options.candidate)
    if options.dry_run:
        print(f"Validated {len(jobs)} candidate/baseline pairs and video hashes; "
              "frozen protocol, functions and dependency versions match.")
        return 0
    options.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    metadata = {
        "status": "running", "candidate": options.candidate,
        "device": "cpu", "dtype": "float32", "versions": versions,
        "clip_revision": metrics.REVISION,
        "clip_frame_indices": list(range(0, 81, 4)),
        "started_unix": started,
        "protocol_sha256": metrics.sha256(ROOT / "notes/week3_metrics.md"),
        "metric_functions_script_sha256": metrics.sha256(Path(metrics.__file__)),
        "runner_sha256": metrics.sha256(__file__),
        "input_manifest_sha256": metrics.sha256(
            ROOT / f"notes/eval/{options.candidate}.jsonl"),
        "baseline_per_video_sha256": metrics.sha256(reference_path),
        "baseline_run_sha256": metrics.sha256(ROOT / "notes/week3_metrics_run/run.json"),
        "quality_verdict": "not evaluated; metric deltas are anomaly hints only",
    }
    run_path = options.output / "run.json"
    metrics.write_json(run_path, metadata)
    rows = []
    try:
        scorer = metrics.ClipScorer("cpu")
        with (options.output / "per_video.jsonl").open("w", encoding="utf-8") as handle:
            for job in jobs:
                row = dict(job)
                try:
                    assert metrics.sha256(job["video_path"]) == job["video_sha256"]
                    frames = metrics.read_video(job["video_path"])
                    ssim = metrics.ssim_scores(frames)
                    clip = scorer.score(frames, job["prompt"])
                    assert len(ssim) == 80 and len(clip) == 21
                    assert np.isfinite(ssim).all() and np.isfinite(clip).all()
                    row.update(
                        status="ok", ssim_mean=float(np.mean(ssim)), ssim_min=min(ssim),
                        clip_mean=float(np.mean(clip)), clip_min=min(clip),
                        ssim_pairs=ssim, clip_frames=clip)
                except Exception as error:
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows.append(row)
                print(f"{len(rows)}/36 {row['id']}: {row['status']}", flush=True)
        metrics.write_csv(options.output / "metrics.csv", rows)
        paired = []
        for row in rows:
            if row["status"] != "ok":
                continue
            b = baseline[row["id"]]
            pair = {"id": row["id"], "prompt_id": row["prompt_id"]}
            for score in metrics.SCORES:
                pair[f"baseline_{score}"] = b[score]
                pair[f"candidate_{score}"] = row[score]
                pair[f"delta_{score}"] = row[score] - b[score]
            paired.append(pair)
        metrics.write_csv(options.output / "paired_comparison.csv", paired)
        # Rank all pairs; no post-hoc numerical threshold or exclusion.
        rankings = {
            score: [
                {"id": p["id"], "delta": p[f"delta_{score}"]}
                for p in sorted(paired, key=lambda p: abs(p[f"delta_{score}"]),
                                reverse=True)]
            for score in metrics.SCORES
        }
        metrics.write_json(options.output / "delta_rankings.json", rankings)
        metadata.update(
            status="ok" if all(r["status"] == "ok" for r in rows) else "error",
            completed=len(rows), errors=sum(r["status"] != "ok" for r in rows),
            ssim_pair_values=sum(len(r.get("ssim_pairs", [])) for r in rows),
            clip_frame_values=sum(len(r.get("clip_frames", [])) for r in rows))
    except Exception:
        metadata.update(status="error", traceback=traceback.format_exc())
        raise
    finally:
        metadata["elapsed_seconds"] = time.time() - started
        metrics.write_json(run_path, metadata)
    print(f"Metrics saved: {options.output}; status={metadata['status']}", flush=True)
    return 0 if metadata["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
