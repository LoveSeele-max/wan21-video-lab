"""Inspect saved Week4 profile artifacts on CPU; never run model inference.

Retain the Week2 External-id filter, then recover Triton launches without
External ids through CUDA launch correlation and matching CPU/GPU ProfilerStep
intervals. Apply the same rule to the historical baseline. Derived artifacts
go in notes/week4; raw notes/week2 and outputs/week2 artifacts stay untouched.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_name(name):
    name = name.lower()
    if name == "_attn_fwd":
        return "sage_attention"
    if name == "quant_per_block_int8_kernel":
        return "sage_quantization"
    if "flash" in name and "fwd" in name:
        return "flash_attention"
    if "gemm" in name or "gemv" in name:
        return "gemm_gemv"
    return "other_kernels"


def analyze_trace(path):
    events = read(path)["traceEvents"]
    cpu_ids = {e.get("args", {}).get("External id") for e in events
               if e.get("cat") == "cpu_op"} - {None}
    steps = [e for e in events
             if e.get("name", "").startswith("ProfilerStep#")]
    cpu_steps = [e for e in steps if e.get("cat") == "user_annotation"]
    gpu_steps = [e for e in steps if e.get("cat") == "gpu_user_annotation"]
    expected = {f"ProfilerStep#{i}" for i in range(18, 22)}
    assert len(cpu_steps) == len(gpu_steps) == 4
    assert {e["name"] for e in cpu_steps} == expected
    assert {e["name"] for e in gpu_steps} == expected
    launches = defaultdict(list)
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            correlation = e.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation].append(e)
    groups = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
    per_step = defaultdict(Counter)
    direct_count, recovered_count = 0, 0
    recovered_names, excluded = Counter(), []
    all_kernel_names = Counter()
    for kernel in events:
        if kernel.get("cat") != "kernel" or kernel.get("ph") != "X":
            continue
        all_kernel_names[kernel["name"]] += 1
        # 1 us tolerance accounts for timestamp rounding at GPU boundaries.
        gpu_matches = [
            s for s in gpu_steps
            if s["ts"] - 1 <= kernel["ts"]
            and kernel["ts"] + kernel["dur"] <= s["ts"] + s["dur"] + 1
        ]
        step_names = set()
        correlation = kernel.get("args", {}).get("correlation")
        for launch in launches.get(correlation, []):
            for step in cpu_steps:
                if (launch.get("pid") == step.get("pid")
                        and launch.get("tid") == step.get("tid")
                        and step["ts"] <= launch["ts"]
                        <= step["ts"] + step["dur"]
                        and any(g["name"] == step["name"] for g in gpu_matches)):
                    step_names.add(step["name"])
        direct = kernel.get("args", {}).get("External id") in cpu_ids
        if not direct and not step_names:
            excluded.append(kernel)
            continue
        assert len(step_names) == 1, kernel["name"]
        if direct:
            direct_count += 1
        else:
            recovered_count += 1
            recovered_names[kernel["name"]] += 1
        group = group_name(kernel["name"])
        groups[group]["count"] += 1
        groups[group]["duration_us"] += kernel["dur"]
        per_step[next(iter(step_names))][group] += 1
    first_gpu_step = min(e["ts"] for e in gpu_steps)
    assert all(e["ts"] + e["dur"] <= first_gpu_step for e in excluded)
    total = sum(g["duration_us"] for g in groups.values())
    for group in groups.values():
        group["percent_of_selected_kernel_sum"] = group["duration_us"] / total * 100
    diagnostics = Counter(
        e["name"] for e in events
        if any(term in e.get("name", "").lower() for term in (
            "command buffer full", "buffer flush", "activity buffer request",
            "cupti_error", "dropped")))
    return {
        "trace_sha256": digest(path),
        "cpu_steps": [{k: e[k] for k in ("name", "ts", "dur")} for e in cpu_steps],
        "gpu_steps": [{k: e[k] for k in ("name", "ts", "dur")} for e in gpu_steps],
        "direct_external_id_kernel_count": direct_count,
        "launch_correlation_recovered_kernel_count": recovered_count,
        "recovered_names": dict(recovered_names),
        "excluded_kernel_count": len(excluded),
        "excluded_all_before_first_gpu_step": True,
        "selected_kernel_duration_sum_us": total,
        "groups": dict(groups),
        "per_step_counts": {k: dict(v) for k, v in per_step.items()},
        "all_trace_flash_fwd_kernel_count": sum(
            count for name, count in all_kernel_names.items()
            if group_name(name) == "flash_attention"),
        "diagnostic_counts": dict(diagnostics),
    }


def inspect_video_and_latent(folder, output_dir):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from safetensors.torch import load_file
    import torch

    latent_paths, video_paths = list(folder.glob("*.safetensors")), list(folder.glob("*.mp4"))
    assert len(latent_paths) == len(video_paths) == 1
    tensors = load_file(str(latent_paths[0]), device="cpu")
    tensor_stats = {}
    for name, tensor in tensors.items():
        finite = bool(torch.isfinite(tensor).all().item())
        assert finite
        tensor_stats[name] = {
            "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "all_finite": finite, "min": tensor.float().min().item(),
            "max": tensor.float().max().item(),
        }
    assert any(list(t.shape) in ([16, 21, 60, 104], [1, 16, 21, 60, 104])
               for t in tensors.values())
    cap = cv2.VideoCapture(str(video_paths[0]))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, samples, black_frames, sizes = 0, {}, [], set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        sizes.add(tuple(frame.shape[:2]))
        if not np.any(frame):
            black_frames.append(frames)
        if frames in (0, 40, 80):
            samples[frames] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames += 1
    cap.release()
    assert frames == 81 and abs(fps - 16) < 1e-8
    assert sizes == {(480, 832)} and set(samples) == {0, 40, 80}
    assert not black_frames
    stats = {str(i): {"mean": float(f.mean()),
                     "variance": float(f.astype(np.float64).var()),
                     "min": int(f.min()), "max": int(f.max())}
             for i, f in samples.items()}
    sheet = Image.new("RGB", (1248, 272), "white")
    draw = ImageDraw.Draw(sheet)
    for column, index in enumerate((0, 40, 80)):
        image = Image.fromarray(samples[index]).resize(
            (416, 240), Image.Resampling.LANCZOS)
        sheet.paste(image, (416 * column, 32))
        draw.text((416 * column + 8, 9), f"Frame {index} / 80 (20-step profile)",
                  fill="black")
    sheet.save(output_dir / "frames_0_40_80.png")
    return {
        "latent": tensor_stats,
        "video": {"frames": frames, "fps": fps, "height": 480, "width": 832,
                  "all_zero_frame_indices": black_frames,
                  "sampled_frame_statistics": stats},
        "artifact_hashes": {p.name: digest(p) for p in (latent_paths[0], video_paths[0])},
        "visual_inspection": "pending",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--baseline-session", default="R5-profile-20261015-221034-3813a433")
    options = parser.parse_args()
    for name in (options.session, options.baseline_session):
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name):
            parser.error("Session names accept letters, digits, hyphen and underscore")
    records = ROOT / "notes/week2" / options.session
    session, result = read(records / "session.json"), read(records / "run-01/result.json")
    assert session["status"] == result["status"] == "ok"
    assert session["options"]["mode"] == result["mode"] == "profile"
    assert session["bench_sha256"] == digest(ROOT / "scripts/bench.py")
    assert session["config_sha256"] == digest(ROOT / "configs/eval_sageattn.yaml")
    assert session["commit"] == "f8a1b03794a49239a3539015075f5123d6c07d66"
    assert session["effective_config"]["infer_steps"] == result["effective_args"]["infer_steps"] == 20
    assert result["effective_args"]["attn_mode"] == "sageattn"
    assert result["forward_calls"] == 40 and result["profiler"]["trace_ready"]
    assert result["profiler"]["schedule_forwards"] == {"wait": 16, "warmup": 2, "active": 4}
    assert not any(result["effective_args"][key] for key in (
        "fp8", "fp8_scaled", "fp8_fast", "fp8_t5", "compile", "blocks_to_swap"))
    bench = runpy.run_path(str(ROOT / "scripts/bench.py"))
    assert bench["make_summary"]([result]) == read(records / "summary.json")
    candidate = analyze_trace(records / "run-01/trace.json")
    baseline = analyze_trace(ROOT / "notes/week2" / options.baseline_session / "run-01/trace.json")
    assert candidate["groups"]["sage_attention"]["count"] == 240
    assert candidate["groups"]["sage_quantization"]["count"] == 480
    assert candidate["all_trace_flash_fwd_kernel_count"] == 0
    assert baseline["groups"]["flash_attention"]["count"] == 240
    for counts in candidate["per_step_counts"].values():
        assert counts["sage_attention"] == 60 and counts["sage_quantization"] == 120
    output_dir = ROOT / "notes/week4" / ("profile-audit-" + options.session)
    output_dir.mkdir(exist_ok=False)
    artifacts = inspect_video_and_latent(
        ROOT / "outputs/week2" / options.session / "run-01", output_dir)
    audit = {
        "session": options.session,
        "status": "local_profile_content_checks_passed_server_hashes_pending",
        "analysis_script_sha256": digest(Path(__file__)),
        "raw_note_hashes": {
            str(p.relative_to(records)): digest(p)
            for p in sorted(records.rglob("*")) if p.is_file()},
        "profile_elapsed_seconds_diagnostic_only": result["elapsed_seconds"],
        "whole_card_peak": "not sampled",
        "forward_calls": 40,
        "candidate": candidate, "historical_baseline": baseline,
        "artifact_checks": artifacts,
        "server_hash_comparison": "pending user-provided trace and artifact hashes",
        "limits": [
            "Historical R5 is a mechanism reference, not a contemporary speed baseline.",
            "Kernel duration sums and classification shares are not end-to-end latency.",
            "Sage attention plus quantization excludes smoothing, casts, allocations and host work.",
            "The original External-id-only rule misses the 720 recovered Triton kernels.",
            "Capture is limited to four middle forwards; event loss is not independently ruled out.",
            "No quality verdict from this profile or its three sampled frames.",
            "No model inference was rerun locally; CPU artifact inspection only.",
        ],
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("AUDIT:", output_dir / "audit.json")
    print("CANDIDATE GROUPS:", json.dumps(candidate["groups"]))
    print("ARTIFACT CHECKS:", json.dumps(artifacts))


if __name__ == "__main__":
    main()
