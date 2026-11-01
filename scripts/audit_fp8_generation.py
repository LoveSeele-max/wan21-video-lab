"""Validate the returned 72-video archive and retained generation evidence."""

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

import av

import gen_eval


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SHA = "4412ce239e719935db16148916d33d21b6914ae671ad930a4231d61aa7722869"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]


def main():
    output = ROOT / "notes/week4/fp8_generation_audit.json"
    if output.exists():
        raise FileExistsError("Preserve existing audit")
    archive = ROOT / "notes/week4/W4-fp8-eval-72-20261029.tar.gz"
    assert gen_eval.file_hash(archive) == EXPECTED_SHA
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        files = [m for m in members if m.isfile()]
        assert len({m.name for m in members}) == len(members)
        for member in members:
            p = PurePosixPath(member.name)
            assert not p.is_absolute() and ".." not in p.parts and ":" not in member.name
            assert member.isdir() or member.isfile()
            allowed = any(
                member.name == f"notes/eval/{name}.jsonl"
                or member.name.startswith(f"notes/eval/{name}/")
                or member.name == f"notes/eval/{name}"
                or (member.isfile() and member.name.startswith(f"outputs/eval/{name}/")
                    and p.suffix == ".mp4" and len(p.parts) == 4)
                for name in ("fp8", "fp8_scaled"))
            assert allowed, member.name
            target = (ROOT / member.name).resolve()
            assert target.is_relative_to(ROOT.resolve())
            assert member.size <= 50_000_000
        content = {}
        for member in files:
            with package.extractfile(member) as handle:
                data = handle.read()
            target = ROOT / member.name
            if target.exists():
                assert target.read_bytes() == data, f"Preserve differing file: {target}"
            content[member.name] = data
        for name, data in content.items():
            target = ROOT / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(data)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}
    report = {
        "verified_date_asia_shanghai": "2026-10-29",
        "status": "passed_local_generation_receipts_video_hashes_and_decode",
        "archive_sha256": EXPECTED_SHA, "server_local_archive_hash_match": True,
        "archive_file_count": len(files), "datasets": {},
        "limits": [
            "Latents and output .attempts were not transferred; no independent local latent hash/finite audit.",
            "Server summarize checks latent/video hashes; local audit independently checks final MP4s.",
            "Coverage and decode success are not human quality acceptance.",
            "Concurrent generation timing is not a speed benchmark.",
        ],
    }
    total_audit_path = ROOT / "notes/week4/fp8_total_audit.json"
    reference_runtime = load(total_audit_path)["matching_recorded_runtime"]
    hashes[str(total_audit_path.relative_to(ROOT))] = gen_eval.file_hash(total_audit_path)
    common_args = None
    for name in ("fp8", "fp8_scaled"):
        options = gen_eval.parse_options(["--config", str(ROOT / f"configs/eval_{name}.yaml"), "--dry-run"])
        plan = gen_eval.load_plan(options)
        folder = ROOT / f"notes/eval/{name}"
        marker, coverage = load(folder / "dataset.json"), load(folder / "coverage.json")
        assert marker == {"fingerprint": plan["fingerprint"], "identity": plan["identity"]}
        assert coverage["planned"] == 36 and coverage["counts"] == {"ok": 36}
        assert coverage["fingerprint"] == plan["fingerprint"]
        manifest = lines(ROOT / f"notes/eval/{name}.jsonl")
        jobs = {job["key"]: job for job in plan["jobs"]}
        assert len(manifest) == len(jobs) == 36 and {r["key"] for r in manifest} == set(jobs)
        assert set(p.stem for p in (ROOT / f"outputs/eval/{name}").glob("*.mp4")) == set(jobs)
        receipts = sorted(folder.glob("attempts/*/attempt-*/result.json"))
        by_key = {}
        for path in receipts:
            record = load(path)
            assert record["key"] in jobs
            assert path.parent.name == f"attempt-{record['attempt']:02d}"
            by_key.setdefault(record["key"], []).append(record)
        assert set(by_key) == set(jobs)
        gpu_counts, attempts, decoded = Counter(), Counter(), {}
        for row in manifest:
            key = row["key"]
            records = by_key[key]
            assert 1 <= len(records) <= 2
            assert sorted(r["attempt"] for r in records) == list(range(1, len(records) + 1))
            final = max(records, key=lambda r: r["attempt"])
            assert row["attempts"] == len(records)
            assert {k: v for k, v in row.items() if k != "attempts"} == final
            assert row["status"] == "ok" and row["actual_steps"] == 20
            assert row["identity"] == plan["identity"] and row["fingerprint"] == plan["fingerprint"]
            assert all(row[k] == v for k, v in jobs[key].items())
            assert row["config_name"] == f"eval_{name}"
            for runtime_key in ("torch", "cuda_build", "gpu", "python", "commit", "tokenizer_revision", "tokenizer_directory"):
                assert row["runtime"][runtime_key] == reference_runtime[runtime_key]
            gpu = row["runtime"]["physical_gpu"]
            assert gpu in (0, 1)
            assert list(jobs).index(key) % 2 == gpu
            gpu_counts[gpu] += 1
            args = row["effective_args"]
            for k, v in plan["config"].items():
                if k not in ("config_name", "save_path", "prompt", "seed"):
                    assert args[k] == v
            assert args["seed"] == row["seed"] and args["prompt"] == row["prompt"]
            assert args["device"] == "cuda:0"
            assert args["save_path"] == row["attempt_output_directory"]
            assert args["save_path"] == (
                f"/home/user/projects/wan21-video/outputs/eval/{name}/.attempts/{key}/attempt-{row['attempt']:02d}")
            normalized = {k: v for k, v in args.items() if k not in (
                "fp8", "fp8_scaled", "save_path", "seed", "prompt")}
            if common_args is None:
                common_args = normalized
            assert normalized == common_args
            video = ROOT / f"outputs/eval/{name}/{key}.mp4"
            assert row["video_path"] == f"/home/user/projects/wan21-video/outputs/eval/{name}/{key}.mp4"
            assert gen_eval.file_hash(video) == row["video_sha256"]
            assert row["latent_path"] == f"/home/user/projects/wan21-video/outputs/eval/{name}/{key}_latent.safetensors"
            assert len(row["latent_sha256"]) == 64
            count = nonblack = 0
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                assert float(stream.average_rate) == 16
                assert (stream.codec_context.width, stream.codec_context.height) == (832, 480)
                for frame in container.decode(stream):
                    pixels = frame.to_ndarray(format="rgb24")
                    assert pixels.shape == (480, 832, 3)
                    nonblack += int(pixels.max() > 0)
                    count += 1
            assert count == 81 and nonblack == 81
            decoded[key] = {"frames": count, "nonblack_frames": nonblack, "sha256": row["video_sha256"]}
            attempts.update(r["status"] for r in records)
        events_files = sorted(folder.glob("events-*.jsonl"))
        events = [event for path in events_files for event in lines(path)]
        assert all(e["key"] in jobs for e in events)
        for records in by_key.values():
            for record in records:
                transitions = [e for e in events if e.get("key") == record["key"]
                               and e.get("attempt") == record["attempt"]]
                assert sum(e["status"] == "running" for e in transitions) == 1
                assert sum(e["status"] == record["status"] for e in transitions) == 1
                terminal = next(e for e in transitions if e["status"] == record["status"])
                assert terminal == record
        skipped = [e for e in events if e["status"] == "skipped"]
        assert all(e["reason"] == "verified successful receipt" for e in skipped)
        failures = [r for records in by_key.values() for r in records if r["status"] != "ok"]
        failure_events = [e for e in events if e["status"] not in ("running", "ok", "skipped")]
        locks = list((folder / "locks").glob("*")) if (folder / "locks").exists() else []
        assert not locks
        report["datasets"][name] = {
            "fingerprint": plan["fingerprint"], "unique_successes": 36,
            "local_mp4_hash_matches": 36, "decoded_frames_each": 81,
            "gpu_counts": dict(gpu_counts), "attempt_records": len(receipts),
            "attempt_statuses": dict(attempts),
            "retried_samples": [key for key, records in by_key.items() if len(records) > 1],
            "failure_attempts": failures, "event_counts": dict(Counter(e["status"] for e in events)),
            "event_file_count": len(events_files), "failure_events": failure_events,
            "verified_skips": skipped, "no_residual_locks": True, "videos": decoded,
        }
        for input_path in (options.config, options.prompts, ROOT / "scripts/gen_eval.py", ROOT / "scripts/bench.py"):
            hashes[str(input_path.relative_to(ROOT))] = gen_eval.file_hash(input_path)
        print(f"{name}: 36 video hashes/decode and receipt/event checks passed", flush=True)
    assert gen_eval.file_hash(archive) == EXPECTED_SHA
    assert all(gen_eval.file_hash(ROOT / name) == digest for name, digest in hashes.items())
    report["inputs_sha256"] = hashes
    report["inputs_unchanged"] = True
    report["audit_script_sha256"] = gen_eval.file_hash(__file__)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "audit": str(output), "status": report["status"],
        "datasets": {k: {field: value for field, value in v.items() if field != "videos"}
                     for k, v in report["datasets"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
