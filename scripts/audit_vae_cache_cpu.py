"""Audit the fixed CPU-cache decode against the verified uninstrumented baseline."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

import av

from vae_decode_measure import COMMIT, ROOT, sha


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_cli(args):
    args = list(args)
    index = args.index("--save_path")
    del args[index:index + 2]
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    directory, output = options.directory.resolve(), options.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    initial, result = read(directory / "input.json"), read(directory / "result.json")
    baseline_path = ROOT / "outputs/week5-vae/decode-baseline-01/result.json"
    baseline = read(baseline_path)
    baseline_audit_path = ROOT / "notes/week5/vae-baseline-01-local-audit.json"
    baseline_audit = read(baseline_audit_path)
    fixed = read(ROOT / "notes/week5_vae_fixed_input.json")
    summary_path = ROOT / "notes/week5/vae-cache-cpu-01-user-summary.json"
    summary = read(summary_path)
    assert initial["status"] == "running" and initial["decode_records"] == []
    assert result["status"] == "ok" and len(result["decode_records"]) == 1
    assert set(initial).issubset(result)
    for key, value in initial.items():
        if key not in ("status", "decode_records"):
            assert result[key] == value, key
    assert baseline["status"] == "ok" and not baseline["vae_cache_cpu"]
    assert result["vae_cache_cpu"] is True and result["memory_history"] is False
    assert result["physical_gpu"] == 0 and result["warmup"] == 0 and result["observation_count"] == 1
    assert result["commit"] == COMMIT == fixed["upstream_commit"]
    assert result["script_sha256"] == sha(ROOT / "scripts/vae_decode_measure.py")
    assert result["script_sha256"] == "656c0224eea84b460fe3c04243e2a8f72e6f86c4f0dc9e126bd24133402e0a24"
    for key in (
        "commit", "upstream_sources_sha256", "script_sha256", "python", "torch", "cuda_build",
        "physical_gpu", "gpu", "memory_history", "vae_sha256", "latent_path", "latent_sha256",
        "latent_dtype", "latent_shape", "latent_finite", "latent_metadata", "warmup",
        "observation_count", "interval", "peak_semantics", "limitations",
    ):
        assert result[key] == baseline[key], key
    assert result["latent_sha256"] == fixed["latent_sha256"]
    assert result["latent_dtype"] == fixed["dtype"] and result["latent_shape"] == fixed["shape"]
    assert result["latent_metadata"] == fixed["metadata"] and result["latent_finite"] is True
    cli = normalized_cli(result["upstream_args"])
    assert cli.count("--vae_cache_cpu") == 1
    cli.remove("--vae_cache_cpu")
    assert cli == normalized_cli(baseline["upstream_args"])
    assert result["upstream_args"][result["upstream_args"].index("--save_path") + 1] == str(
        PurePosixPath(result["video_path"]).parent
    )
    source_checks = {}
    for relative, expected in result["upstream_sources_sha256"].items():
        blob = subprocess.check_output([
            "git", "-C", str(ROOT / "local/musubi-tuner"), "show", COMMIT + ":" + relative,
        ])
        local = (ROOT / "local/musubi-tuner" / relative).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == expected
        assert local.replace(b"\r\n", b"\n") == blob
        source_checks[relative] = {
            "server_and_fixed_git_blob_sha256": expected,
            "local_worktree_sha256": hashlib.sha256(local).hexdigest(),
            "local_equals_fixed_blob_after_crlf_to_lf": True,
        }
    # Verify the comparison's previously audited original files still have their recorded hashes.
    for relative, expected in baseline_audit["files_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    row, base_row = result["decode_records"][0], baseline["decode_records"][0]
    assert result["decode_records"] == summary["reported_summary"]["decode_records"]
    assert result["video_path"] == summary["reported_summary"]["video_path"]
    for key in ("input_dtype", "input_shape", "output_dtype", "output_device", "output_shape", "before_bytes"):
        assert row[key] == base_row[key], key
    assert row["before_bytes"] == {"allocated": 0, "reserved": 0}
    assert row["seconds"] > 0 and row["peak_allocated_bytes"] <= row["peak_reserved_bytes"]
    for field in ("allocated", "reserved"):
        assert row[f"peak_{field}_bytes"] / 1024**2 == row[f"peak_{field}_mib"]
    video = directory / PurePosixPath(result["video_path"]).name
    assert list(directory.glob("*.mp4")) == [video]
    video_hash = sha(video)
    assert video_hash == result["video_sha256"]
    assert video_hash == baseline["video_sha256"] == sha(ROOT / "outputs/eval/baseline/P01_s42.mp4")
    frame_hash, count, black = hashlib.sha256(), 0, 0
    with av.open(str(video)) as container:
        assert len(container.streams.video) == 1 and len(container.streams.audio) == 0
        stream = container.streams.video[0]
        assert float(stream.average_rate) == 16
        assert stream.codec_context.name == "h264" and stream.codec_context.format.name == "yuv420p"
        for frame in container.decode(stream):
            assert (frame.width, frame.height) == (832, 480)
            pixels = frame.to_ndarray(format="rgb24")
            frame_hash.update(pixels.tobytes())
            black += int(not pixels.any())
            count += 1
    assert count == 81 and black == 0
    assert frame_hash.hexdigest() == baseline_audit["decoded_video"]["rgb_frames_sha256"]
    comparison = {
        "allocated_reduction_bytes": base_row["peak_allocated_bytes"] - row["peak_allocated_bytes"],
        "allocated_reduction_mib": base_row["peak_allocated_mib"] - row["peak_allocated_mib"],
        "allocated_reduction_percent": (base_row["peak_allocated_bytes"] - row["peak_allocated_bytes"])
        / base_row["peak_allocated_bytes"] * 100,
        "reserved_reduction_mib": base_row["peak_reserved_mib"] - row["peak_reserved_mib"],
        "seconds_increase": row["seconds"] - base_row["seconds"],
        "elapsed_time_ratio": row["seconds"] / base_row["seconds"],
    }
    for key, value in comparison.items():
        assert value == summary["locally_recomputed_comparison"][key], key
    attribution_path = ROOT / "notes/week5/vae-history-01-analysis-v2/attribution.json"
    attribution = read(attribution_path)
    history_audit_path = attribution_path.parent / "audit.json"
    assert attribution["input_audit_sha256"] == sha(history_audit_path)
    comparison["history_reconstructed_assigned_cache_mib"] = attribution["assigned_cache_storage_mib_reconstructed"]
    comparison["allocated_reduction_minus_reconstructed_cache_mib"] = (
        comparison["allocated_reduction_mib"] - comparison["history_reconstructed_assigned_cache_mib"]
    )
    assert comparison["allocated_reduction_minus_reconstructed_cache_mib"] == 0.8125
    audit = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_status": "ANALYZED",
        "status": "passed_local_record_identity_hashes_and_full_mp4_decode",
        "audit_script_sha256": sha(__file__),
        "files_sha256": {
            str(p.relative_to(ROOT)): sha(p) for p in (
                directory / "input.json", directory / "result.json", video,
                baseline_path, baseline_audit_path, summary_path, attribution_path,
                ROOT / "scripts/vae_decode_measure.py",
            )
        },
        "source_checks": source_checks,
        "only_intended_upstream_argument_change_except_output_directory": "--vae_cache_cpu",
        "cache_cpu_enabled_in_original_record": True,
        "baseline_record": base_row, "cache_cpu_record": row,
        "comparison": comparison,
        "decoded_video": {
            "frames": count, "fps": 16, "width": 832, "height": 480,
            "all_black_frames": black, "rgb_frames_sha256": frame_hash.hexdigest(),
            "mp4_sha256": video_hash,
            "bytes_identical_to_uninstrumented_and_frozen_baselines": True,
        },
        "limits": [
            "Server GPU run performed by user; no independent local GPU reproduction",
            "Latent and VAE weight identities are server records, not locally read bodies",
            "No separately supplied server hashes for input/result JSON",
            "Exact MP4/RGB identity for this sample does not prove raw decoded-tensor equality",
            "Single fresh-process diagnostic, not a formal speed or full-generation benchmark",
            "No CPU-cache allocation history; remaining peak location/backend not measured",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "status": audit["status"], "decoded_video": audit["decoded_video"],
        "comparison": comparison, "audit": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
