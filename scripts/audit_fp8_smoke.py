"""Audit received FP8 smoke artifacts on CPU without invoking generation."""

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import av
import numpy as np
from PIL import Image, ImageDraw
from safetensors import safe_open
import torch
import yaml

import bench


ROOT = Path(__file__).resolve().parent.parent
PACK = "W4-fp8-smoke-artifacts-20261027"
DATA = ROOT / "notes/week4" / PACK
CASES = (
    ("fp8", "W4-fp8-smoke-stages-20261027-233635-3ec2be40"),
    ("fp8_scaled", "W4-fp8-scaled-smoke-stages-20261027-234012-aee81310"),
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    output = ROOT / "notes/week4/fp8_smoke_audit.json"
    preview_path = ROOT / "notes/week4/fp8_smoke_preview.png"
    if output.exists() or preview_path.exists():
        raise FileExistsError("Preserve existing audit and preview")
    torch.set_num_threads(4)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in DATA.rglob("*") if p.is_file()}
    assert len(hashes) == 10
    archive = ROOT / "notes/week4" / f"{PACK}.tar.gz"
    member_count = 0
    with tarfile.open(archive, "r:gz") as package:
        members = [member for member in package.getmembers() if not member.isdir()]
        assert len(members) == 10
        expected = {p.relative_to(DATA).as_posix() for p in DATA.rglob("*") if p.is_file()}
        assert {member.name for member in members} == expected
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk()
            with package.extractfile(member) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == sha(DATA / member.name)
            member_count += 1
    hashes[str(archive.relative_to(ROOT))] = sha(archive)
    hashes["scripts/bench.py"] = sha(ROOT / "scripts/bench.py")
    report = {
        "verified_date_asia_shanghai": "2026-10-28",
        "status": "passed_local_smoke_checks_only",
        "source_root": str(DATA),
        "archive_sha256": hashes[str(archive.relative_to(ROOT))],
        "archive_members_match_extracted_files": member_count,
        "server_archive_sha256_supplied": False,
        "cases": {},
        "limits": [
            "Server execution is user-operated; this audit reads received artifacts on CPU.",
            "Local archive/extraction hashes agree; no server checksum was supplied for independent transfer comparison.",
            "Successful decode, finite latents and non-black frames do not establish video quality or performance benefit.",
            "No external whole-card GPU memory sampling was requested in these smoke runs.",
            "Interpreter invocation follows the supplied command; session metadata does not independently record executable path.",
            "No continuous second-GPU occupancy evidence was supplied.",
        ],
    }
    preview = Image.new("RGB", (1248, 528), "white")
    draw = ImageDraw.Draw(preview)
    sessions = []
    args_by_case = {}
    for row, (name, identifier) in enumerate(CASES):
        directory = DATA / "notes/week2" / identifier
        session = load(directory / "session.json")
        result = load(directory / "run-01/result.json")
        summary = load(directory / "summary.json")
        config_path = ROOT / "configs" / f"eval_{name}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        hashes[str(config_path.relative_to(ROOT))] = sha(config_path)
        assert session["status"] == result["status"] == "ok"
        assert session["bench_sha256"] == hashes["scripts/bench.py"]
        assert session["config_sha256"] == hashes[str(config_path.relative_to(ROOT))]
        assert session["commit"] == bench.EXPECTED_COMMIT
        assert session["options"]["mode"] == result["mode"] == "stages"
        assert session["options"]["steps"] == 2 and session["options"]["warmup"] == 0
        assert session["options"]["repeat"] == 1 and session["options"]["gpu_index"] == 0
        assert session["options"]["sample_smi"] is False
        assert session["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert session["environment"]["HF_HUB_OFFLINE"] == "1"
        expected_config = {**config, "infer_steps": 2}
        assert session["effective_config"] == expected_config
        args = result["effective_args"]
        args_by_case[name] = args
        for key, value in expected_config.items():
            if key not in ("config_name", "save_path"):
                assert args[key] == value, (name, key)
        assert args["device"] == "cuda:0" and result["warmup"] is False
        assert args["attn_mode"] == "torch" and args["fp8_fast"] is False
        assert result["forward_calls"] == len(result["forward_seconds"]) == 4
        assert result["checks"]["expected_forward_calls"] == 4
        assert all(result["checks"][key] for key in (
            "each_stage_called_once", "forward_count_matches", "timing_gap_below_2_percent"))
        assert set(result["calls"]) == set(bench.STAGE_NAMES)
        assert all(value == 1 for value in result["calls"].values())
        assert all(value > 0 for value in result["forward_seconds"])
        assert bench.aggregate_stages(result["stages"]) == result["stage_summary"]
        assert sum(item["seconds"] for item in result["stages"]) == result["stage_sum_seconds"]
        gap = abs(result["elapsed_seconds"] - result["stage_sum_seconds"]) / result["elapsed_seconds"] * 100
        assert gap == result["timing_gap_percent"] and gap < 2
        for metric in ("peak_allocated_bytes", "peak_reserved_bytes"):
            assert max(item[metric] for item in result["stages"]) == result[metric]
        assert bench.make_summary([result]) == summary
        artifacts = DATA / "outputs/week2" / identifier / "run-01"
        assert result["output_directory"] == args["save_path"] == (
            f"/home/user/projects/wan21-video/outputs/week2/{identifier}/run-01")
        latent_paths = list(artifacts.glob("*.safetensors"))
        video_paths = list(artifacts.glob("*.mp4"))
        assert len(latent_paths) == len(video_paths) == 1
        tensors = {}
        with safe_open(latent_paths[0], framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                assert tuple(tensor.shape[-4:]) == (16, 21, 60, 104)
                value = tensor.float()
                finite = bool(torch.isfinite(value).all())
                tensors[key] = {
                    "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                    "finite": finite, "min": float(value.min()),
                    "max": float(value.max()), "mean": float(value.mean()),
                    "std": float(value.std()),
                }
                assert finite
            metadata = handle.metadata()
        assert tensors
        count, nonblack, frame_means, frame_stds, samples = 0, 0, [], [], {}
        with av.open(str(video_paths[0])) as container:
            assert len(container.streams.video) == 1
            stream = container.streams.video[0]
            assert float(stream.average_rate) == 16
            assert stream.codec_context.width == 832 and stream.codec_context.height == 480
            codec = stream.codec_context.name
            for index, frame in enumerate(container.decode(stream)):
                image = frame.to_ndarray(format="rgb24")
                assert image.shape == (480, 832, 3)
                nonblack += int(image.max() > 0)
                frame_means.append(float(image.mean()))
                frame_stds.append(float(image.std()))
                if index in (0, 40, 80):
                    samples[index] = Image.fromarray(image).resize((416, 240))
                count += 1
        assert count == nonblack == 81 and min(frame_stds) > 0
        for column, index in enumerate((0, 40, 80)):
            draw.text((column * 416 + 5, row * 264 + 5), f"{name} / frame {index}", fill="black")
            preview.paste(samples[index], (column * 416, row * 264 + 24))
        report["cases"][name] = {
            "session": identifier, "session_status": session["status"], "result_status": result["status"],
            "flags": {key: args[key] for key in ("attn_mode", "fp8", "fp8_scaled", "fp8_fast", "fp8_t5")},
            "config_and_bench_hashes_verified": True, "summary_recomputed_exactly": True,
            "stage_checks_recomputed": True, "forward_calls": 4,
            "elapsed_seconds_smoke_only": result["elapsed_seconds"],
            "allocated_peak_mib": result["peak_allocated_bytes"] / 1024**2,
            "reserved_peak_mib": result["peak_reserved_bytes"] / 1024**2,
            "allocated_peak_stage": [
                key for key, value in result["stage_summary"].items()
                if value["peak_allocated_bytes"] == result["peak_allocated_bytes"]],
            "stage_summary": result["stage_summary"],
            "latents": tensors, "latent_metadata": metadata,
            "latent_sha256": sha(latent_paths[0]), "video_sha256": sha(video_paths[0]),
            "video": {
                "decoded_frames": count, "nonblack_frames": nonblack,
                "width": 832, "height": 480, "fps": 16, "codec": codec,
                "frame_mean_range": [min(frame_means), max(frame_means)],
                "frame_pixel_std_range": [min(frame_stds), max(frame_stds)],
                "preview_frame_indices": [0, 40, 80],
            },
        }
        sessions.append(session)
    behavioral_diffs = {
        key for key in args_by_case["fp8"]
        if args_by_case["fp8"][key] != args_by_case["fp8_scaled"][key]
    }
    assert behavioral_diffs == {"fp8", "fp8_scaled", "save_path"}
    report["effective_argument_differences"] = sorted(behavioral_diffs)
    for key in ("python", "torch", "cuda_build", "gpu", "commit", "environment",
                "generation_module", "allocator_backend", "tokenizer_directory", "tokenizer_revision"):
        assert sessions[0][key] == sessions[1][key]
    report["matching_recorded_runtime"] = {
        key: sessions[0][key] for key in ("python", "torch", "cuda_build", "gpu", "commit")}
    checkout = ROOT / "local/musubi-tuner"
    source_paths = {
        "wan_generate_video.py": "src/musubi_tuner/wan_generate_video.py",
        "wan_model.py": "src/musubi_tuner/wan/modules/model.py",
        "fp8_optimization_utils.py": "src/musubi_tuner/modules/fp8_optimization_utils.py",
    }
    source_checks = {}
    for saved, relative in source_paths.items():
        blob = subprocess.run(
            ["git", "-C", str(checkout), "show", f"{bench.EXPECTED_COMMIT}:{relative}"],
            check=True, capture_output=True,
        ).stdout
        saved_path = ROOT / "notes/source" / saved
        assert saved_path.read_bytes().replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n")
        hashes[str(saved_path.relative_to(ROOT))] = sha(saved_path)
        source_checks[saved] = {
            "git_revision": bench.EXPECTED_COMMIT, "git_path": relative,
            "matches_saved_source_ignoring_crlf": True,
            "git_blob_sha256": hashlib.sha256(blob).hexdigest(),
        }
    report["fixed_source_comparisons"] = source_checks
    report["audit_script_sha256"] = sha(__file__)
    for relative, expected in hashes.items():
        assert sha(ROOT / relative) == expected
    report["inputs_sha256"] = hashes
    report["input_files_unchanged_after_audit"] = True
    preview.save(preview_path)
    report["preview"] = {"path": str(preview_path), "sha256": sha(preview_path)}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "audit": str(output), "preview": str(preview_path),
        "archive_sha256": report["archive_sha256"],
        "cases": {key: {field: value[field] for field in (
            "elapsed_seconds_smoke_only", "allocated_peak_mib", "reserved_peak_mib",
            "allocated_peak_stage", "flags", "latents", "video",
        )} for key, value in report["cases"].items()},
        "source_comparisons": source_checks,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
