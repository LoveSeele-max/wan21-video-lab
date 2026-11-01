"""Day 1 operator diagnostic only; no model loading or quality verdict.

Compare BF16 random inputs at the two planned Wan shapes, batch size 1.
Use synchronized wall time, 3 warmups and 10 measured calls per operator.
The SDPA reference explicitly selects PyTorch flash; this is not the full
upstream wrapper or its automatic backend selection. Timing includes Sage's
quantization/smoothing and SDPA's output layout conversion.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter
import traceback
import uuid


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES = (("self", 32760, 32760), ("cross", 32760, 512))


def parse_options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, default=0,
                        help="Physical index under PCI_BUS_ID device order")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    return args


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def measure(torch, fn):
    for _ in range(3):
        out = fn()
        torch.cuda.synchronize()
        del out
    values = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = perf_counter()
        out = fn()
        torch.cuda.synchronize()
        values.append((perf_counter() - start) * 1000)
        del out
    return {"unit": "ms", "values": values, "mean": statistics.mean(values),
            "min": min(values), "max": max(values)}


def compare(torch, reference, candidate):
    result = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "reference_finite": bool(torch.isfinite(reference).all().item()),
        "candidate_finite": bool(torch.isfinite(candidate).all().item()),
    }
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise RuntimeError(f"Output contract mismatch: {result}")
    if not result["reference_finite"] or not result["candidate_finite"]:
        return result
    ref = reference.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    error = (ref - cand).abs()
    result.update({
        "max_absolute_error": error.max().item(),
        "mean_absolute_error": error.mean().item(),
        "cosine_similarity_flattened": torch.nn.functional.cosine_similarity(
            ref, cand, dim=0).item(),
    })
    return result


def run_case(torch, sageattention, sdpa_kernel, backend, row):
    generator = torch.Generator(device="cuda").manual_seed(row["seed"])
    lq, lk = row["q_length"], row["kv_length"]
    q = torch.randn((lq, 12, 128), generator=generator,
                    device="cuda", dtype=torch.bfloat16)
    k = torch.randn((lk, 12, 128), generator=generator,
                    device="cuda", dtype=torch.bfloat16)
    v = torch.randn((lk, 12, 128), generator=generator,
                    device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, lq], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, lk], dtype=torch.int32, device="cuda")
    row["input_sha256"] = {
        name: hashlib.sha256(tensor.cpu().contiguous().view(torch.uint8)
                             .numpy().tobytes()).hexdigest()
        for name, tensor in (("q", q), ("k", k), ("v", v))
    }
    row["input_strides"] = {
        name: list(tensor.stride())
        for name, tensor in (("q", q), ("k", k), ("v", v))
    }
    # Views match the upstream SDPA input layout for batch size 1.
    q_sdpa = q.unsqueeze(0).transpose(1, 2)
    k_sdpa = k.unsqueeze(0).transpose(1, 2)
    v_sdpa = v.unsqueeze(0).transpose(1, 2)

    def reference():
        out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2).contiguous().squeeze(0)

    def candidate():
        return sageattention.sageattn_varlen(
            q, k, v, cu_q, cu_k, lq, lk,
            is_causal=False, smooth_k=True)

    with torch.inference_mode(), sdpa_kernel(backend.FLASH_ATTENTION):
        # The first calls may compile kernels; they are never timed.
        ref = reference()
        cand = candidate()
        torch.cuda.synchronize()
        row["accuracy"] = compare(torch, ref, cand)
        del ref, cand
        print(json.dumps({"case": row["name"], "accuracy": row["accuracy"]}),
              flush=True)
        if not (row["accuracy"]["reference_finite"]
                and row["accuracy"]["candidate_finite"]):
            raise RuntimeError("Nonfinite output; stop before timing")
        row["sdpa_flash"] = measure(torch, reference)
        row["sageattn_varlen"] = measure(torch, candidate)
    row["operator_speedup_sdpa_over_sage"] = (
        row["sdpa_flash"]["mean"] / row["sageattn_varlen"]["mean"])
    row["status"] = "completed"
    print(json.dumps(row), flush=True)


def main():
    args = parse_options()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    # Keep compilation artifacts in this project, not the user's global cache.
    os.environ["TRITON_CACHE_DIR"] = str(PROJECT_ROOT / ".cache" / "triton")
    os.environ["CUDA_CACHE_PATH"] = str(PROJECT_ROOT / ".cache" / "cuda")
    for key in ("TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    session = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
               + "-" + uuid.uuid4().hex[:8])
    output_dir = PROJECT_ROOT / "notes" / "week4" / ("varlen-" + session)
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "result.json"
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "python": sys.executable,
        "script_sha256": sha256(__file__),
        "requested_physical_gpu_index": args.gpu_index,
        "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
        "triton_cache_dir": os.environ["TRITON_CACHE_DIR"],
        "cuda_cache_path": os.environ["CUDA_CACHE_PATH"],
        "seed": args.seed,
        "batch": 1, "heads": 12, "head_dim": 128, "dtype": "bfloat16",
        "input_distribution": "independent standard normal, random activations",
        "warmup": 3, "measured": 10,
        "timing": "synchronized wall time; compilation excluded",
        "reference": "PyTorch SDPA, forced FLASH_ATTENTION, contiguous output",
        "candidate": "sageattn_varlen, smooth_k=True",
        "accuracy_scope": "one untimed output per operator per case",
        "quality_verdict": "not evaluated; no numerical acceptance threshold",
        "cases": [],
    }
    save_report(result_path, report)
    print("RESULT:", result_path, flush=True)
    try:
        import torch
        import triton
        import sageattention
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; GPU diagnostic not run")
        torch.cuda.set_device(0)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected GPU does not support BF16")
        props = torch.cuda.get_device_properties(0)
        report["environment"] = {
            "torch": torch.__version__, "torch_path": torch.__file__,
            "torch_cuda_build": torch.version.cuda,
            "triton": triton.__version__, "triton_path": triton.__file__,
            "sageattention": version("sageattention"),
            "sageattention_path": sageattention.__file__,
            "gpu_name": props.name,
            "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_total_memory_bytes": props.total_memory,
        }
        smi = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False)
        (output_dir / "nvidia-smi.txt").write_text(
            smi.stdout + smi.stderr, encoding="utf-8")
        # Archive the actual installed modules used for the operator diagnosis.
        entry = inspect.unwrap(sageattention.sageattn_varlen)
        report["source_sha256"] = {}
        for name, fn in (
            ("core", entry),
            ("quant_per_block_varlen",
             entry.__globals__["per_block_int8_varlen_triton"]),
            ("attn_qk_int8_block_varlen",
             entry.__globals__["attn_false_varlen"]),
        ):
            source = Path(inspect.getsourcefile(fn))
            (output_dir / (name + ".py")).write_bytes(source.read_bytes())
            report["source_sha256"][name] = {
                "installed_path": str(source), "sha256": sha256(source)}
        save_report(result_path, report)
        for name, lq, lk in CASES:
            row = {"name": name, "q_length": lq, "kv_length": lk,
                   "seed": args.seed, "status": "running"}
            report["cases"].append(row)
            save_report(result_path, report)
            print("START:", name, "q_length:", lq, "kv_length:", lk, flush=True)
            run_case(torch, sageattention, sdpa_kernel, SDPBackend, row)
            save_report(result_path, report)
        report["status"] = "completed_operator_check_only"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save_report(result_path, report)
        print("STATUS:", report["status"], flush=True)
        print("RESULT:", result_path, flush=True)
    return 0 if report["status"] == "completed_operator_check_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
