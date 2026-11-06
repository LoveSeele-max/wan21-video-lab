"""Audit the returned fixed VAE probe and replay its native allocator history."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import pickle
import subprocess

import av

from vae_decode_measure import COMMIT, ROOT, sha


MIB = 1024**2
EXPECTED_SNAPSHOT = "c8682923be50a74c6884acc3e5bc1bdf5443ae33c364763092a9ceecbdd62b63"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def table(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vae_frames(event):
    return [f for f in event["frames"] if f["filename"].endswith("/wan/modules/vae.py")]


def replay(snapshot):
    """Replay only default native pools; fail rather than approximate block bytes."""
    settings = snapshot["allocator_settings"]
    assert settings["max_split_size"] == -1 and not settings["expandable_segments"]
    assert not any(settings["roundup_power2_divisions"].values())
    trace = snapshot["device_traces"][0]
    assert all(e["stream"] == 0 for e in trace)
    segment_types = {s["address"]: s["segment_type"] for s in snapshot["segments"]}
    free, live, pending, segments = {}, {}, {}, {}
    allocations, points = [], []
    allocated = requested = active = reserved = peak = 0
    peak_live, peak_index = {}, None
    first_time = trace[0]["time_us"]
    for index, event in enumerate(trace):
        action, addr, size = event["action"], event["addr"], event["size"]
        if action == "segment_alloc":
            assert addr not in segments
            segments[addr] = size
            free[addr] = (size, addr)
            reserved += size
        elif action == "alloc":
            assert addr not in live and addr not in pending
            available, segment = free.pop(addr)
            rounded = max(512, (size + 511) // 512 * 512)
            assert available >= rounded
            remaining = available - rounded
            split = remaining >= 512 if segment_types[segment] == "small" else remaining > MIB
            block = rounded if split else available
            if split:
                free[addr + block] = (remaining, segment)
            allocation = {
                "allocation_event": index, "address": hex(addr),
                "requested_bytes": size, "block_bytes": block,
                "free_requested_event": None, "free_completed_event": None,
                "segment": segment, "event": event,
            }
            allocations.append(allocation)
            live[addr] = allocation
            allocated += block
            requested += size
            active += block
            if allocated > peak:
                peak, peak_index, peak_live = allocated, index, dict(live)
        elif action == "free_requested":
            allocation = live.pop(addr)
            assert allocation["requested_bytes"] == size
            allocation["free_requested_event"] = index
            pending[addr] = allocation
            allocated -= allocation["block_bytes"]
            requested -= size
        elif action == "free_completed":
            allocation = pending.pop(addr)
            assert allocation["requested_bytes"] == size
            allocation["free_completed_event"] = index
            block, segment = allocation["block_bytes"], allocation["segment"]
            active -= block
            # Coalesce adjacent inactive blocks in the same segment.
            following = free.get(addr + block)
            if following and following[1] == segment:
                free.pop(addr + block)
                block += following[0]
            previous = next(
                (a for a, (b, s) in free.items() if s == segment and a + b == addr), None
            )
            if previous is not None:
                block += free.pop(previous)[0]
                addr = previous
            free[addr] = (block, segment)
        else:
            raise AssertionError("Unsupported history action: " + action)
        assert 0 <= requested <= allocated <= active <= reserved
        points.append({
            "event": index, "seconds_since_first_event": (event["time_us"] - first_time) / 1e6,
            "action": action, "allocated_bytes": allocated,
            "requested_bytes": requested, "active_bytes": active,
            "reserved_bytes": reserved, "max_allocated_so_far_bytes": peak,
        })
    assert not pending
    assert allocated == sum(s["allocated_size"] for s in snapshot["segments"])
    assert active == sum(s["active_size"] for s in snapshot["segments"])
    assert reserved == sum(s["total_size"] for s in snapshot["segments"])
    # Check every final block, including allocator slack and coalesced free space.
    final_live = {int(a["address"], 16): a for a in live.values()}
    for segment in snapshot["segments"]:
        for block in segment["blocks"]:
            addr = block["address"]
            if block["state"] == "inactive":
                assert free[addr] == (block["size"], segment["address"])
            else:
                a = final_live[addr]
                assert block["state"] == "active_allocated"
                assert (a["block_bytes"], a["requested_bytes"]) == (block["size"], block["requested_size"])
    return allocations, points, peak_index, peak_live


def checkpoint_candidates(checkpoints, allocations, points):
    by_address = {}
    for allocation in allocations:
        by_address.setdefault(allocation["address"], []).append(allocation)
    matches = []
    for checkpoint in checkpoints:
        observations = checkpoint["tensors"] + checkpoint["feat_cache"]["tensor_entries"]
        expected = ("allocated_bytes", "reserved_bytes", "max_allocated_so_far_bytes")
        candidates = []
        for point in points:
            index = point["event"]
            if any(point[k] != checkpoint[k] for k in expected):
                continue
            epochs = []
            for tensor in observations:
                if tensor["device"] != "cuda:0":
                    continue
                found = [
                    a for a in by_address.get(tensor["storage_data_ptr"], [])
                    if a["allocation_event"] <= index
                    and (a["free_requested_event"] is None or index < a["free_requested_event"])
                    and a["requested_bytes"] == tensor["storage_bytes"]
                ]
                if len(found) != 1:
                    break
                epochs.append(found[0]["allocation_event"])
            else:
                candidates.append((index, epochs))
        assert candidates, checkpoint["sequence"]
        matches.append(candidates)
    # Enforce observed checkpoint order in both directions.
    lower = 0
    for i, candidates in enumerate(matches):
        matches[i] = [c for c in candidates if c[0] >= lower]
        assert matches[i]
        lower = matches[i][0][0]
    upper = len(points)
    for i in reversed(range(len(matches))):
        matches[i] = [c for c in matches[i] if c[0] <= upper]
        assert matches[i]
        upper = matches[i][-1][0]
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    directory, output = options.directory.resolve(), options.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    initial, result, probe = (read(directory / name) for name in ("input.json", "result.json", "probe.json"))
    baseline_dir = ROOT / "outputs/week5-vae/decode-baseline-01"
    baseline = read(baseline_dir / "result.json")
    fixed = read(ROOT / "notes/week5_vae_fixed_input.json")
    summary = read(ROOT / "notes/week5/vae-history-01-user-summary.json")["reported_summary"]
    assert result["status"] == "ok" and initial["status"] == "running"
    assert initial["decode_records"] == [] and result["probe"] == probe
    for key, value in initial.items():
        if key not in ("status", "decode_records", "probe"):
            assert result[key] == value, key
    assert result["commit"] == COMMIT == fixed["upstream_commit"]
    for key in (
        "commit", "upstream_sources_sha256", "script_sha256", "python", "torch", "cuda_build",
        "physical_gpu", "gpu", "vae_sha256", "vae_cache_cpu", "latent_path", "latent_sha256",
        "latent_dtype", "latent_shape", "latent_finite", "latent_metadata", "warmup",
        "observation_count", "interval", "peak_semantics",
    ):
        assert result[key] == baseline[key], key
    assert result["script_sha256"] == sha(ROOT / "scripts/vae_decode_measure.py")
    assert result["probe_script_sha256"] == probe["probe_script_sha256"] == sha(ROOT / "scripts/vae_memory_probe.py")
    assert probe["base_script_sha256"] == result["script_sha256"]
    assert probe["baseline_result_sha256"] == sha(baseline_dir / "result.json")
    assert result["memory_history"] and probe["history_started"] and probe["history_stopped"]
    assert not any("error" in k for k in probe)
    assert probe["counts"] == {"decoder": 21} and len(probe["checkpoints"]) == 44
    assert result["latent_sha256"] == fixed["latent_sha256"] and result["latent_metadata"] == fixed["metadata"]
    cli, old_cli = result["upstream_args"], baseline["upstream_args"]
    assert cli[:-3] == old_cli[:-3] and cli[-2:] == old_cli[-2:]
    assert cli[cli.index("--save_path") + 1] == str(PurePosixPath(result["video_path"]).parent)
    sources = {}
    for relative, expected in result["upstream_sources_sha256"].items():
        blob = subprocess.check_output(["git", "-C", str(ROOT / "local/musubi-tuner"), "show", COMMIT + ":" + relative])
        assert hashlib.sha256(blob).hexdigest() == expected
        assert (ROOT / "local/musubi-tuner" / relative).read_bytes().replace(b"\r\n", b"\n") == blob
        sources[relative] = expected
    snapshot_path = directory / "vae_snapshot.pickle"
    assert sha(snapshot_path) == EXPECTED_SNAPSHOT == probe["snapshot"]["sha256"]
    assert snapshot_path.stat().st_size == 7781984 == probe["snapshot"]["bytes"]
    assert probe["snapshot"] == summary["snapshot"]
    # The trusted experiment snapshot is hash-checked before pickle deserialization.
    with snapshot_path.open("rb") as handle:
        snapshot = pickle.load(handle)
    assert [len(t) for t in snapshot["device_traces"]] == [29327]
    actions = [dict(Counter(e["action"] for e in t)) for t in snapshot["device_traces"]]
    assert actions == probe["snapshot"]["device_trace_actions"]
    assert not probe["snapshot"]["history_may_be_truncated"] and probe["max_entries"] == 200000
    assert len(snapshot["segments"]) == 37
    row = result["decode_records"][0]
    assert len(result["decode_records"]) == 1
    assert row["input_shape"] == [1, 16, 21, 60, 104] and row["input_dtype"] == "torch.float32"
    assert row["output_shape"] == [3, 81, 480, 832] and row["output_device"] == "cpu"
    assert row["output_dtype"] == "torch.float32" and row["before_bytes"] == {"allocated": 0, "reserved": 0}
    assert row["seconds"] > 0
    for field in ("allocated", "reserved"):
        assert row[f"peak_{field}_bytes"] / MIB == row[f"peak_{field}_mib"]
    video = directory / PurePosixPath(result["video_path"]).name
    assert list(directory.glob("*.mp4")) == [video]
    assert sha(video) == result["video_sha256"] == baseline["video_sha256"]
    assert sha(ROOT / "outputs/eval/baseline/P01_s42.mp4") == sha(video)
    assert result["same_video_bytes_as_uninstrumented_baseline"] is True
    rgb_hash, frame_count, black = hashlib.sha256(), 0, 0
    with av.open(str(video)) as container:
        assert len(container.streams.video) == 1 and not container.streams.audio
        stream = container.streams.video[0]
        assert float(stream.average_rate) == 16 and stream.codec_context.name == "h264"
        assert stream.codec_context.format.name == "yuv420p"
        for frame in container.decode(stream):
            assert (frame.width, frame.height) == (832, 480)
            pixels = frame.to_ndarray(format="rgb24")
            rgb_hash.update(pixels.tobytes())
            black += int(not pixels.any())
            frame_count += 1
    assert frame_count == 81 and black == 0
    allocations, points, peak_index, peak_live = replay(snapshot)
    assert points[peak_index]["allocated_bytes"] == row["peak_allocated_bytes"]
    assert max(p["reserved_bytes"] for p in points) == row["peak_reserved_bytes"]
    assert points[-1]["allocated_bytes"] == row["after_bytes"]["allocated"]
    matches = checkpoint_candidates(probe["checkpoints"], allocations, points)
    checkpoints = []
    for checkpoint, candidates in zip(probe["checkpoints"], matches):
        checkpoints.append({
            "sequence": checkpoint["sequence"], "label": checkpoint["label"],
            "decoder_call": checkpoint["decoder_calls_started"],
            "first_matching_event": candidates[0][0], "last_matching_event": candidates[-1][0],
            "allocated_mib": checkpoint["allocated_bytes"] / MIB,
            "max_allocated_mib": checkpoint["max_allocated_so_far_bytes"] / MIB,
            "cache_storage_mib": checkpoint["feat_cache"]["unique_storage_bytes_by_device"].get("cuda:0", 0) / MIB,
        })
    peak_rows = []
    for address, allocation in sorted(peak_live.items(), key=lambda item: item[1]["block_bytes"], reverse=True):
        event = allocation["event"]
        peak_rows.append({
            "allocation_event": allocation["allocation_event"], "address": hex(address),
            "requested_bytes": allocation["requested_bytes"], "block_bytes": allocation["block_bytes"],
            "requested_mib": allocation["requested_bytes"] / MIB,
            "block_mib": allocation["block_bytes"] / MIB,
            "free_requested_event": allocation["free_requested_event"],
            "vae_stack": " > ".join(f'{f["name"]}:{f["line"]}' for f in vae_frames(event)),
            "native_slow_conv": any("slow_conv_dilated" in f["name"] for f in event["frames"]),
        })
    # Match cache observations to allocation epochs, never to addresses alone.
    before = max(
        (c for c in checkpoints if c["label"] == "decoder_before" and c["last_matching_event"] < peak_index),
        key=lambda c: c["sequence"],
    )
    after = next(c for c in checkpoints if c["sequence"] > before["sequence"] and c["label"] == "decoder_after")
    assert peak_index < after["first_matching_event"]
    cache_evidence = []
    for sequence in (before["sequence"], after["sequence"]):
        checkpoint = probe["checkpoints"][sequence]
        candidates = matches[sequence]
        for tensor in checkpoint["feat_cache"]["tensor_entries"]:
            epochs = {
                a["allocation_event"] for a in allocations
                if a["address"] == tensor["storage_data_ptr"]
                and a["requested_bytes"] == tensor["storage_bytes"]
                and any(
                    a["allocation_event"] <= i
                    and (a["free_requested_event"] is None or i < a["free_requested_event"])
                    for i, _ in candidates
                )
            }
            assert len(epochs) == 1
            epoch = next(iter(epochs))
            a = next(a for a in allocations if a["allocation_event"] == epoch)
            cache_evidence.append({
                "observation": checkpoint["label"], "cache_index": tensor["index"],
                "allocation_event": epoch, "address": a["address"],
                "shape": tensor["shape"], "dtype": tensor["dtype"],
                "storage_bytes": tensor["storage_bytes"],
                "alive_at_peak": epoch <= peak_index and (
                    a["free_requested_event"] is None or peak_index < a["free_requested_event"]
                ),
                "free_requested_event": a["free_requested_event"],
            })
    cache_unique = {
        c["allocation_event"]: c for c in cache_evidence if c["alive_at_peak"]
    }
    audit = {
        "verification_status": "ANALYZED",
        "local_record_and_video_audit": "passed",
        "analyzer_sha256": sha(__file__),
        "files_sha256": {p.name: sha(p) for p in directory.iterdir() if p.is_file()},
        "fixed_source_git_blob_hashes": sources,
        "decode_record": row, "baseline_decode_record": baseline["decode_records"][0],
        "snapshot_actions": actions, "snapshot_size_bytes": snapshot_path.stat().st_size,
        "decoded_video": {
            "frames": frame_count, "fps": 16, "width": 832, "height": 480,
            "all_black_frames": black, "rgb_frames_sha256": rgb_hash.hexdigest(),
            "identical_bytes_to_original_and_uninstrumented_baselines": True,
        },
        "peak": {
            **points[peak_index], "live_allocations": len(peak_live),
            "allocator_slack_bytes": points[peak_index]["allocated_bytes"] - points[peak_index]["requested_bytes"],
            "largest_live_allocations": peak_rows[:25],
            "trigger_event_full_stack": snapshot["device_traces"][0][peak_index]["frames"],
            "before_checkpoint": before, "after_checkpoint": after,
        },
        "cache_epoch_evidence": cache_evidence,
        "cache_observed_before_or_after_storages_alive_at_peak_bytes": sum(c["storage_bytes"] for c in cache_unique.values()),
        "cache_interpretation": "Live observed storages include the pending new cache clone; exact _feat_map membership requires source assignment order.",
        "checks": [
            "Trusted snapshot hash/size checked before unpickling",
            "Input/result/probe/script/fixed-source/baseline identity checked",
            "All allocation/free epochs matched without truncation",
            "Default native block split/coalescing replay matches every final snapshot block",
            "Replayed absolute allocated and reserved peaks match recorded peaks exactly",
            "Checkpoint allocated/reserved/max and all observed storage epochs matched in order",
            "Video fully decoded and byte-identical to both baselines",
        ],
        "limits": [
            "No local GPU rerun and no independent local read of latent or VAE weights",
            "Native allocator replay supports only this default single-stream capture",
            "History includes allocator requests, not complete tensor shapes",
            "Cache boundary shapes do not directly measure transient convolution shapes",
            "One fixed input; no speed or overall quality conclusion",
            "Memory-viz HTML uses an external PyTorch JavaScript module; not offline standalone",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", audit)
    save(output / "cache_epochs.json", cache_evidence)
    table(output / "peak_live_allocations.csv", peak_rows)
    table(output / "allocator_timeline.csv", points)
    table(output / "checkpoints.csv", checkpoints)
    from torch.cuda import _memory_viz
    with (output / "memory_viz.html").open("x", encoding="utf-8") as handle:
        handle.write(_memory_viz.trace_plot(snapshot))
    print(json.dumps({
        "audit": str(output / "audit.json"), "status": audit["local_record_and_video_audit"],
        "peak": {k: v for k, v in audit["peak"].items() if k not in ("trigger_event_full_stack", "largest_live_allocations")},
        "cache_observed_storages_alive_at_peak_mib": audit["cache_observed_before_or_after_storages_alive_at_peak_bytes"] / MIB,
    }, indent=2))


if __name__ == "__main__":
    main()
