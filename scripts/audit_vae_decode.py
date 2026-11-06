"""Verify returned latent-only decode records and fully decode their MP4."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    output = options.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    directory = options.directory.resolve()
    initial, result = read(directory / "input.json"), read(directory / "result.json")
    fixed = read(ROOT / "notes/week5_vae_fixed_input.json")
    reported = read(ROOT / "notes/week5/vae-baseline-01-user-return.json")["reported_summary"]
    assert initial["status"] == "running" and initial["decode_records"] == []
    assert result["status"] == "ok" and len(result["decode_records"]) == 1
    assert set(initial).issubset(result)
    for key, value in initial.items():
        if key not in ("status", "decode_records"):
            assert result[key] == value, key
    assert result["commit"] == COMMIT == fixed["upstream_commit"]
    assert result["script_sha256"] == sha(ROOT / "scripts/vae_decode_measure.py")
    source_checks = {}
    repo = ROOT / "local/musubi-tuner"
    for relative, expected in result["upstream_sources_sha256"].items():
        blob = subprocess.check_output(["git", "-C", str(repo), "show", COMMIT + ":" + relative])
        local = (repo / relative).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == expected, relative
        assert local.replace(b"\r\n", b"\n") == blob, relative
        source_checks[relative] = {
            "server_sha256": expected,
            "fixed_git_blob_sha256": hashlib.sha256(blob).hexdigest(),
            "local_worktree_raw_sha256": hashlib.sha256(local).hexdigest(),
            "local_matches_fixed_blob_after_crlf_to_lf": True,
            "raw_local_bytes_equal_server": hashlib.sha256(local).hexdigest() == expected,
        }
    assert result["physical_gpu"] == 0 and result["torch"] == "2.9.1+cu128"
    assert result["cuda_build"] == "12.8" and not result["vae_cache_cpu"]
    assert not result["memory_history"] and result["warmup"] == 0 and result["observation_count"] == 1
    for key in ("latent_path", "latent_sha256"):
        assert result[key] == fixed[key], key
    assert result["latent_dtype"] == fixed["dtype"] and result["latent_shape"] == fixed["shape"]
    assert result["latent_finite"] is True and result["latent_metadata"] == fixed["metadata"]
    cli = result["upstream_args"]
    assert cli[cli.index("--task") + 1] == "t2v-1.3B"
    assert cli[cli.index("--vae_dtype") + 1] == "bfloat16"
    assert cli[cli.index("--device") + 1] == "cuda:0"
    assert cli[cli.index("--output_type") + 1] == "video"
    assert "--vae_cache_cpu" not in cli and "--dit" not in cli and "--t5" not in cli
    row = result["decode_records"][0]
    assert result["decode_records"] == reported["decode_records"]
    assert result["video_path"] == reported["video_path"]
    assert row["peak_allocated_bytes"] / 1024**2 == row["peak_allocated_mib"]
    assert row["peak_reserved_bytes"] / 1024**2 == row["peak_reserved_mib"]
    assert row["seconds"] > 0 and row["peak_allocated_bytes"] <= row["peak_reserved_bytes"]
    assert row["input_dtype"] == "torch.float32" and row["input_shape"] == [1, 16, 21, 60, 104]
    assert row["output_dtype"] == "torch.float32" and row["output_device"] == "cpu"
    assert row["output_shape"] == [3, 81, 480, 832]
    video = directory / PurePosixPath(result["video_path"]).name
    assert list(directory.glob("*.mp4")) == [video]
    assert sha(video) == result["video_sha256"]
    baseline = next(
        json.loads(line) for line in (ROOT / "notes/eval/baseline.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["key"] == "P01_s42"
    )
    baseline_video = ROOT / "outputs/eval/baseline/P01_s42.mp4"
    assert sha(baseline_video) == baseline["video_sha256"] == sha(video)
    frame_hash = hashlib.sha256()
    black = 0
    count = 0
    with av.open(str(video)) as container:
        assert len(container.streams.video) == 1 and len(container.streams.audio) == 0
        stream = container.streams.video[0]
        assert float(stream.average_rate) == 16
        assert stream.codec_context.name == "h264"
        assert stream.codec_context.format.name == "yuv420p"
        for frame in container.decode(stream):
            assert (frame.width, frame.height) == (832, 480)
            pixels = frame.to_ndarray(format="rgb24")
            black += int(not pixels.any())
            frame_hash.update(pixels.tobytes())
            count += 1
    assert count == 81 and black == 0
    stage = read(ROOT / "notes/week4/fp8_stages_audit.json")["cases"]["baseline"]["stage_table"]["decode_latent"]
    audit = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_local_record_identity_hashes_and_full_mp4_decode",
        "audit_script_sha256": sha(__file__),
        "files_sha256": {
            str(path.relative_to(ROOT)): sha(path) for path in (
                directory / "input.json", directory / "result.json", video,
                ROOT / "scripts/vae_decode_measure.py",
            )
        },
        "recorded_source_hashes_match_fixed_git_blobs": True,
        "source_checks": source_checks,
        "server_latent_identity_matches_registration": True,
        "video_bytes_identical_to_frozen_P01_s42_baseline": True,
        "decoded_video": {
            "frames": count, "fps": 16, "width": 832, "height": 480,
            "all_black_frames": black, "rgb_frames_sha256": frame_hash.hexdigest(),
        },
        "decode_record": row,
        "allocated_difference_from_week4_mib": row["peak_allocated_mib"] - stage["allocated_mib"],
        "reserved_difference_from_week4_mib": row["peak_reserved_mib"] - stage["reserved_mib"],
        "vae_weight_hash_as_recorded_by_server": result["vae_sha256"],
        "limits": [
            "No independent local read of the latent or server VAE weight.",
            "One fixed sample diagnostic; no GPU rerun or quality acceptance.",
            "Matching output bytes do not locate the peak allocation source.",
            "No independent server-side digest was supplied for the two JSON files.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "status": audit["status"], "decoded_video": audit["decoded_video"],
        "same_bytes_as_baseline": True, "audit": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
