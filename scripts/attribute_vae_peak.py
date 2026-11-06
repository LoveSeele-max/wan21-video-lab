"""Check source-backed attribution for this fixed VAE history capture."""

import argparse
import csv
import json
from pathlib import Path

from vae_decode_measure import ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    options = parser.parse_args()
    directory = options.analysis.resolve()
    output = directory / "attribution.json"
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    assert audit["analyzer_sha256"] == sha(ROOT / "scripts/analyze_vae_history.py")
    peak = audit["peak"]
    assert peak["event"] == 28595 and peak["before_checkpoint"]["decoder_call"] == 21
    evidence = audit["cache_epoch_evidence"]
    selected, pending = [], []
    indices = sorted({entry["cache_index"] for entry in evidence})
    assert indices == list(range(32))
    for index in indices:
        alive = [entry for entry in evidence if entry["cache_index"] == index and entry["alive_at_peak"]]
        assert alive
        old = [entry for entry in alive if entry["observation"] == "decoder_before"]
        new = [entry for entry in alive if entry["observation"] == "decoder_after"]
        assert len(old) <= 1 and len(new) <= 1
        selected.append((old or new)[0])
        if old and new and old[0]["allocation_event"] != new[0]["allocation_event"]:
            pending.append(new[0])
    assert len(pending) == 1 and pending[0]["cache_index"] == 25
    assert pending[0]["allocation_event"] == 28588
    assert pending[0]["shape"] == [1, 96, 2, 480, 832]
    assert selected[25]["free_requested_event"] == 28604 > peak["event"]
    cache_bytes = sum(e["storage_bytes"] for e in selected)
    pending_bytes = sum(e["storage_bytes"] for e in pending)
    assert cache_bytes == 3728824320 and pending_bytes == 306708480
    assert cache_bytes + pending_bytes == audit["cache_observed_before_or_after_storages_alive_at_peak_bytes"]
    vae = ROOT / "local/musubi-tuner/src/musubi_tuner/wan/modules/vae.py"
    source = vae.read_text(encoding="utf-8").splitlines()
    assert "cache_x =" in source[204] and ".clone()" in source[204]
    assert "x = layer(" in source[208] and "feat_cache[idx]" in source[208]
    assert "feat_cache[idx] = cache_x" in source[209]
    native = directory / "pytorch-v2.9.1-NaiveDilatedConvolution.cu"
    allocator = directory / "pytorch-v2.9.1-CUDACachingAllocator.cpp"
    assert sha(native) == "4017e8584f75cb2c40e983f355a425edd10eb33818b00ae4f45a9ffbf6ce16eb"
    assert sha(allocator) == "3d43b5834f6143f0465832cb15a906c3098ffbbf6048e75361d38901269d7599"
    native_lines = native.read_text(encoding="utf-8").splitlines()
    assert "columns.resize_({nInputPlane * m, output_vsize})" in native_lines[208]
    assert any("slow_conv_dilated_all_cuda_template" in f["name"] for f in peak["trigger_event_full_stack"])
    # Independently check requested-byte conservation from the peak CSV.
    with (directory / "peak_live_allocations.csv").open(newline="", encoding="utf-8") as handle:
        live = list(csv.DictReader(handle))
    assert sum(int(e["requested_bytes"]) for e in live) == peak["requested_bytes"]
    assert sum(int(e["block_bytes"]) for e in live) == peak["allocated_bytes"]
    largest = peak["largest_live_allocations"][0]
    assert largest["requested_bytes"] == 8281128960 and largest["block_bytes"] == 8281653248
    assert 2592 * 1597440 * 2 == largest["requested_bytes"]
    result = {
        "verification_status": "ANALYZED",
        "script_sha256": sha(__file__),
        "input_audit_sha256": sha(directory / "audit.json"),
        "peak_event": peak["event"],
        "decoder_call": 21,
        "assigned_cache_storage_bytes_reconstructed": cache_bytes,
        "assigned_cache_storage_mib_reconstructed": cache_bytes / 1024**2,
        "assigned_cache_epochs": selected,
        "pending_cache_clone_bytes": pending_bytes,
        "pending_cache_clone_mib": pending_bytes / 1024**2,
        "pending_cache_clone": pending,
        "assignment_evidence": (
            "vae.py:205 clones the new cache; line 209 executes convolution using old feat_cache[idx]; "
            "line 210 replaces the entry only after return. At event 28595 the old index 25 "
            "and the new clone are both alive; indices 0..24 have already been replaced, "
            "indices 25..31 retain their old allocations. Membership is reconstructed, not an interior hook."
        ),
        "largest_allocation": {
            "kind": "native slow_conv_dilated3d columns temporary buffer",
            "requested_bytes": 8281128960,
            "allocator_block_bytes": 8281653248,
            "percent_of_total_allocated_peak_requested_basis": 8281128960 / peak["allocated_bytes"] * 100,
            "native_source": "pytorch v2.9.1 aten/src/ATen/native/cuda/NaiveDilatedConvolution.cu:195-209",
            "shape_reconstruction": {
                "shape": [2592, 1597440], "dtype": "bfloat16",
                "formula": "[96*3*3*3, 4*480*832], 2 bytes per element",
                "evidence_level": "source/shape/byte-consistent reconstruction, not directly shape-hooked",
            },
        },
        "source_files": {
            native.name: {
                "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cuda/NaiveDilatedConvolution.cu",
                "sha256": sha(native),
            },
            allocator.name: {
                "url": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/c10/cuda/CUDACachingAllocator.cpp",
                "sha256": sha(allocator),
            },
        },
        "limits": [
            "Backend dispatch reason for selecting slow_conv_dilated3d is not measured",
            "Transient shapes are reconstructed, not directly observed at an interior hook",
            "Cache offload may change temporary overlaps, transfers and backend behavior; do not subtract cache bytes to predict a measured peak",
            "No optimization performed and no speed benefit inferred",
        ],
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in (
        "peak_event", "decoder_call", "assigned_cache_storage_mib_reconstructed",
        "pending_cache_clone_mib",
    )}, indent=2))


if __name__ == "__main__":
    main()
