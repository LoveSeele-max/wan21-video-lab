"""Instrument one fixed latent-only VAE decode; preserve its numerical code."""

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys
import traceback
from unittest.mock import patch

import vae_decode_measure as base

MAX_ENTRIES = 200000
BASE_SHA256 = "656c0224eea84b460fe3c04243e2a8f72e6f86c4f0dc9e126bd24133402e0a24"
BASELINE = base.ROOT / "outputs/week5-vae/decode-baseline-01/result.json"


def tensor_info(tensor):
    storage = tensor.untyped_storage()
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "device": str(tensor.device), "data_ptr": hex(tensor.data_ptr()),
        "storage_data_ptr": hex(storage.data_ptr()),
        "storage_bytes": storage.nbytes(),
        "logical_bytes": tensor.numel() * tensor.element_size(),
    }


def cache_info(torch, owner):
    entries, unique = [], {}
    for index, value in enumerate(owner._feat_map):
        if isinstance(value, torch.Tensor):
            info = tensor_info(value)
            entries.append({"index": index, **info})
            unique[(info["device"], info["storage_data_ptr"])] = info["storage_bytes"]
    return {
        "tensor_entries": entries,
        "unique_storage_bytes_by_device": {
            device: sum(size for (d, _), size in unique.items() if d == device)
            for device in sorted({d for d, _ in unique})
        },
        "note": "Full underlying storage bytes, deduplicated by device/base pointer; not allocator block bytes.",
    }


def observed_decode(torch, original, checkpoints, counts):
    def wrapped(owner, latent, scale):
        handles = []

        def record(label, values):
            tensors = [tensor_info(t) for t in values if isinstance(t, torch.Tensor)]
            checkpoints.append({
                "sequence": len(checkpoints), "label": label,
                "decoder_calls_started": counts["decoder"],
                "allocated_bytes": torch.cuda.memory_allocated(0),
                "reserved_bytes": torch.cuda.memory_reserved(0),
                "max_allocated_so_far_bytes": torch.cuda.max_memory_allocated(0),
                "tensors": tensors, "feat_cache": cache_info(torch, owner),
            })

        def pre(label):
            def hook(module, args):
                if label == "decoder":
                    counts["decoder"] += 1
                record(label + "_before", args)
            return hook

        def post(label):
            def hook(module, args, output):
                record(label + "_after", (output,))
            return hook

        try:
            for label, module in (("conv2", owner.conv2), ("decoder", owner.decoder)):
                handles.append(module.register_forward_pre_hook(pre(label)))
                handles.append(module.register_forward_hook(post(label)))
            return original(owner, latent, scale)
        finally:
            for handle in handles:
                handle.remove()
    return wrapped


def history_decode(torch, measured, state, output):
    def wrapped(*args, **kwargs):
        memory = torch.cuda.memory
        memory._record_memory_history(
            enabled="all", context="all", stacks="all",
            max_entries=MAX_ENTRIES, clear_history=True,
        )
        state["history_started"] = True
        active_error = False
        try:
            return measured(*args, **kwargs)
        except BaseException:
            active_error = True
            state["decode_error"] = traceback.format_exc()
            raise
        finally:
            try:
                snapshot = memory._snapshot()
                snapshot_path = output / "vae_snapshot.pickle"
                with snapshot_path.open("xb") as handle:
                    pickle.dump(snapshot, handle)
                traces = snapshot["device_traces"]
                state["snapshot"] = {
                    "path": str(snapshot_path),
                    "bytes": snapshot_path.stat().st_size,
                    "sha256": base.sha(snapshot_path),
                    "device_trace_lengths": [len(trace) for trace in traces],
                    "device_trace_actions": [dict(Counter(e["action"] for e in trace)) for trace in traces],
                    "history_may_be_truncated": any(len(trace) >= MAX_ENTRIES for trace in traces),
                    "segments_at_snapshot": len(snapshot["segments"]),
                }
                if not traces or not traces[0]:
                    raise RuntimeError("No GPU0 allocator history captured")
            except BaseException:
                state["snapshot_error"] = traceback.format_exc()
                if not active_error:
                    raise
            finally:
                memory._record_memory_history(enabled=None)
                state["history_stopped"] = True
    return wrapped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    output = options.output.resolve()
    if options.dry_run:
        print(json.dumps({
            "physical_gpu": 0, "base_script_sha256_required": BASE_SHA256,
            "memory_history": True, "max_entries": MAX_ENTRIES,
            "baseline_result": str(BASELINE), "output": str(output),
            "latent_only_args": base.upstream_args(output, False),
            "hooks": ["conv2_before/after", "decoder_before/after"],
            "additional_sync_inside_hooks": False,
            "numerical_operations_changed": False,
        }, ensure_ascii=False, indent=2))
        return
    if base.sha(Path(base.__file__)) != BASE_SHA256:
        raise RuntimeError("Verified base measurement script changed")
    reference = json.loads(BASELINE.read_text(encoding="utf-8"))
    if reference["status"] != "ok" or reference["memory_history"] or reference["vae_cache_cpu"]:
        raise RuntimeError("Require the successful uninstrumented baseline")
    if output.exists():
        raise FileExistsError(output)
    state = {
        "probe_script_sha256": base.sha(__file__),
        "base_script_sha256": BASE_SHA256,
        "baseline_result_sha256": base.sha(BASELINE),
        "max_entries": MAX_ENTRIES,
        "history_started": False, "history_stopped": False,
        "checkpoints": [], "counts": {"decoder": 0},
        "limitations": [
            "Timing and allocation behavior may be affected by instrumentation; not a speed benchmark.",
            "Boundary tensor/cache observations do not capture every transient or prove cache size at the exact global peak.",
            "Snapshots cover the PyTorch CUDA allocator, not all driver or external allocations.",
        ],
    }
    original_factory, original_save = base.measure_decode, base.save_new

    def factory(torch, original, records):
        measured = original_factory(torch, original, records)
        from musubi_tuner.wan.modules.vae import WanVAE_
        original_decode = WanVAE_.decode

        def run(*args, **kwargs):
            observer = observed_decode(torch, original_decode, state["checkpoints"], state["counts"])
            with patch.object(WanVAE_, "decode", observer):
                video = history_decode(torch, measured, state, output)(*args, **kwargs)
            if state["counts"]["decoder"] != 21 or len(state["checkpoints"]) != 44:
                raise RuntimeError("Expected 21 decoder calls and 44 boundary observations")
            return video
        return run

    def save(path, value):
        report = deepcopy(value)
        for key in ("commit", "latent_sha256", "vae_sha256", "torch", "cuda_build"):
            if report[key] != reference[key]:
                raise RuntimeError("Baseline/probe identity mismatch: " + key)
        report["memory_history"] = True
        report["probe_script_sha256"] = state["probe_script_sha256"]
        report["limitations"] += "; instrumented memory-history/hook diagnostic"
        report["probe"] = deepcopy(state)
        if path.name == "result.json":
            report["same_video_bytes_as_uninstrumented_baseline"] = (
                report.get("video_sha256") == reference["video_sha256"])
            original_save(output / "probe.json", state)
        original_save(path, report)

    with ExitStack() as stack:
        stack.enter_context(patch.object(base, "measure_decode", factory))
        stack.enter_context(patch.object(base, "save_new", save))
        base.main(["--output", str(output)])
    print("PROBE_SUMMARY=" + json.dumps({
        "snapshot": state.get("snapshot"),
        "decoder_calls": state["counts"]["decoder"],
        "checkpoint_count": len(state["checkpoints"]),
        "result_json": str(output / "result.json"),
        "probe_json": str(output / "probe.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
