"""Measure the pinned upstream latent-only decode without memory-history hooks.

One fresh process, no warmup, one diagnostic observation. The decode interval
includes VAE load, GPU transfer, decode and output conversion to float32/CPU;
MP4 encoding is excluded. Peaks are absolute allocator bytes, not deltas.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
COMMIT = "f8a1b03794a49239a3539015075f5123d6c07d66"
LATENT_SHA256 = "45b483abe74e4cc29943b15c40931aa11ff265d652abf964132636303f0def16"
LATENT_RELATIVE = "outputs/eval/baseline/P01_s42_latent.safetensors"
VAE_RELATIVE = "models/wan21/split_files/vae/wan_2.1_vae.safetensors"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def upstream_args(output, cache_cpu):
    args = [
        "--task", "t2v-1.3B",
        "--vae", str(ROOT / VAE_RELATIVE),
        "--vae_dtype", "bfloat16",
        "--latent_path", str(ROOT / LATENT_RELATIVE),
        "--video_size", "480", "832",
        "--video_length", "81",
        "--fps", "16",
        "--output_type", "video",
        "--save_path", str(output),
        "--device", "cuda:0",
    ]
    if cache_cpu:
        args.append("--vae_cache_cpu")
    return args


def measure_decode(torch, original, records):
    def wrapped(latent, args, cfg):
        if records:
            raise RuntimeError("Expected exactly one decode invocation")
        device = args.device
        torch.cuda.synchronize(device)
        before = {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
        }
        torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        video = original(latent, args, cfg)
        torch.cuda.synchronize(device)
        seconds = perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        row = {
            "seconds": seconds,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_mib": peak_allocated / 1024**2,
            "peak_reserved_mib": peak_reserved / 1024**2,
            "before_bytes": before,
            "after_bytes": {
                "allocated": torch.cuda.memory_allocated(device),
                "reserved": torch.cuda.memory_reserved(device),
            },
            "input_dtype": str(latent.dtype),
            "input_shape": list(latent.shape),
            "output_dtype": str(video.dtype),
            "output_device": str(video.device),
            "output_shape": list(video.shape),
        }
        records.append(row)
        if row["input_shape"] != [1, 16, 21, 60, 104]:
            raise RuntimeError("Unexpected upstream decode input shape")
        if row["output_shape"] != [3, 81, 480, 832] or str(video.device) != "cpu":
            raise RuntimeError("Unexpected decoded output")
        return video
    return wrapped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vae-cache-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="No torch imports or writes")
    options = parser.parse_args(argv)
    output = options.output.resolve()
    cli = upstream_args(output, options.vae_cache_cpu)
    if options.dry_run:
        print(json.dumps({
            "root": str(ROOT), "upstream_args": cli,
            "expected_latent_sha256": LATENT_SHA256,
            "physical_gpu": 0, "memory_history": False,
            "observation_count": 1, "warmup": 0,
            "interval": "decode_latent including VAE load and CPU conversion; excluding MP4 save",
        }, ensure_ascii=False, indent=2))
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "0"):
        raise RuntimeError("Require physical GPU0; conflicting CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(ROOT / "models/.hf-home")
    os.environ["HF_HUB_CACHE"] = str(ROOT / "models/.hf-home/hub")
    repo = ROOT / "musubi-tuner"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True)
    if commit != COMMIT or dirty.strip():
        raise RuntimeError("Pinned clean upstream required; inspect commit and git status")
    latent_path, vae_path = ROOT / LATENT_RELATIVE, ROOT / VAE_RELATIVE
    if sha(latent_path) != LATENT_SHA256:
        raise RuntimeError("Fixed latent hash mismatch")
    vae_sha256 = sha(vae_path)
    import torch
    from safetensors import safe_open

    if torch.__version__ != "2.9.1+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError(f"Inspect environment drift: torch={torch.__version__}, CUDA={torch.version.cuda}")
    with safe_open(str(latent_path), framework="pt", device="cpu") as handle:
        latent = handle.get_tensor("latent")
        metadata = handle.metadata() or {}
    if (str(latent.dtype) != "torch.float32" or list(latent.shape) != [16, 21, 60, 104]
            or not bool(torch.isfinite(latent).all().item())):
        raise RuntimeError("Fixed latent dtype/shape/finite check failed")
    expected_metadata = {
        "seeds": "42", "prompt": "A person waving one hand, plain background, static camera.",
        "guidance_scale": "5.0", "height": "480", "width": "832",
        "video_length": "81", "infer_steps": "20",
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise RuntimeError("Fixed latent metadata mismatch")
    del latent
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    sys.path.insert(0, str(repo / "src"))
    from musubi_tuner import wan_generate_video as wan

    smi = subprocess.check_output(["nvidia-smi"], text=True)
    sources = {
        relative: sha(repo / relative) for relative in (
            "src/musubi_tuner/wan_generate_video.py",
            "src/musubi_tuner/wan/modules/vae.py",
        )
    }
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "commit": commit, "upstream_sources_sha256": sources,
        "script_sha256": sha(__file__), "python": sys.version,
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "physical_gpu": 0, "gpu": torch.cuda.get_device_name(0),
        "smi_before": smi, "upstream_args": cli, "memory_history": False,
        "vae_sha256": vae_sha256, "vae_cache_cpu": options.vae_cache_cpu,
        "latent_path": str(latent_path), "latent_sha256": LATENT_SHA256,
        "latent_dtype": "torch.float32", "latent_shape": [16, 21, 60, 104],
        "latent_finite": True, "latent_metadata": metadata,
        "decode_records": [], "warmup": 0, "observation_count": 1,
        "interval": "decode_latent including VAE load, GPU transfers, and float32 CPU output; excluding MP4 encoding",
        "peak_semantics": "absolute process allocator peaks, not whole-card nvidia-smi or deltas",
        "limitations": "Single fresh-process diagnostic, not full-generation speed or quality acceptance",
    }
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "input.json", report)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["wan_generate_video.py", *cli]))
            stack.enter_context(patch.object(wan, "decode_latent",
                                            measure_decode(torch, wan.decode_latent, report["decode_records"])))
            wan.main()
        if len(report["decode_records"]) != 1:
            raise RuntimeError("Expected one measured decode")
        videos = list(output.glob("*.mp4"))
        if len(videos) != 1 or videos[0].stat().st_size == 0:
            raise RuntimeError("Expected one nonempty video")
        report["video_path"] = str(videos[0])
        report["video_sha256"] = sha(videos[0])
        report["video_validation"] = "nonempty file and hash only; full MP4 decode verification pending"
        if sha(latent_path) != LATENT_SHA256:
            raise RuntimeError("Latent changed during measurement")
        report["status"] = "ok"
    except BaseException:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_new(output / "result.json", report)
        print("RESULT_JSON=" + str(output / "result.json"))
        print(json.dumps({
            "status": report["status"], "decode_records": report["decode_records"],
            "video_path": report.get("video_path"),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
