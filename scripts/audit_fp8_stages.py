"""Safely unpack and reconcile the received F6-F8 stage records."""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tarfile

import yaml

import bench


ROOT = Path(__file__).resolve().parent.parent
PACK = "W4-fp8-stages-records-20261029"
DATA = ROOT / "notes/week4" / PACK
SESSIONS = (
    ("baseline", "W4-F6-torch-stages-stages-20261029-151302-78d70e8b"),
    ("fp8", "W4-F7-fp8-stages-stages-20261029-152149-75a61563"),
    ("fp8_scaled", "W4-F8-fp8-scaled-stages-stages-20261029-152931-00dfa303"),
)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    output = ROOT / "notes/week4/fp8_stages_audit.json"
    if output.exists():
        raise FileExistsError("Preserve previous audit")
    archive = ROOT / "notes/week4" / f"{PACK}.tar.gz"
    expected = {
        f"notes/week2/{session}/{suffix}"
        for _, session in SESSIONS
        for suffix in ("session.json", "summary.json", "warmup-01/result.json", "run-01/result.json")
    }
    # Validate the entire archive before writing any files; never overwrite differences.
    with tarfile.open(archive, "r:gz") as package:
        files = [member for member in package.getmembers() if not member.isdir()]
        assert len(files) == len(expected) == 12
        assert {member.name for member in files} == expected
        content = {}
        for member in package.getmembers():
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and ".." not in path.parts and ":" not in member.name
            assert member.isdir() or member.isfile()
            assert (DATA / member.name).resolve().is_relative_to(DATA.resolve())
            if member.isfile():
                assert member.size < 2_000_000
                with package.extractfile(member) as handle:
                    content[member.name] = handle.read()
        for relative, data in content.items():
            path = DATA / relative
            if path.exists():
                assert path.read_bytes() == data, f"Preserve existing file: {path}"
        for relative, data in content.items():
            path = DATA / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as handle:
                    handle.write(data)
    hashes = {str((DATA / name).relative_to(ROOT)): sha(DATA / name) for name in expected}
    hashes[str(archive.relative_to(ROOT))] = sha(archive)
    hashes["scripts/bench.py"] = sha(ROOT / "scripts/bench.py")
    total_path = ROOT / "notes/week4/fp8_total_audit.json"
    total = load(total_path)
    hashes[str(total_path.relative_to(ROOT))] = sha(total_path)
    report = {
        "verified_date_asia_shanghai": "2026-10-29",
        "status": "passed_local_stage_records_audit_diagnostic_only",
        "archive_sha256": sha(archive), "archive_members_match_extraction": 12,
        "server_archive_sha256_supplied": False, "cases": {},
        "limits": [
            "User-operated server run, locally audited saved records only.",
            "Local archive and extraction match; no server checksum supplied.",
            "One measured stages run per configuration; warmups excluded from comparisons.",
            "Stage instrumentation adds synchronization overhead; use total audit for speed thresholds.",
            "No whole-card sampling or output videos/latents supplied for these stage sessions.",
            "Runtime fields match; full environment, executable path and absence of concurrent work not independently monitored.",
        ],
    }
    canonical_args = None
    formal = {}
    for name, identifier in SESSIONS:
        folder = DATA / "notes/week2" / identifier
        session, summary = load(folder / "session.json"), load(folder / "summary.json")
        config_path = ROOT / "configs" / f"eval_{name}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        hashes[str(config_path.relative_to(ROOT))] = sha(config_path)
        assert session["status"] == "ok"
        assert session["commit"] == bench.EXPECTED_COMMIT
        assert session["bench_sha256"] == hashes["scripts/bench.py"]
        assert session["config_sha256"] == sha(config_path)
        assert session["effective_config"] == config
        options = session["options"]
        assert (options["mode"], options["steps"], options["warmup"], options["repeat"],
                options["gpu_index"], options["sample_smi"], options["online"]) == (
                    "stages", 20, 1, 1, 0, False, False)
        assert options["config"] == f"configs/eval_{name}.yaml"
        assert identifier.startswith(options["tag"] + "-stages-")
        for key, value in total["matching_recorded_runtime"].items():
            assert session[key] == value, (name, key)
        assert {p.parent.name for p in folder.glob("*/result.json")} == {"warmup-01", "run-01"}
        raw, runs = [], []
        for run_name in ("warmup-01", "run-01"):
            result = load(folder / run_name / "result.json")
            args = result["effective_args"]
            assert result["status"] == "ok" and result["mode"] == "stages"
            assert result["warmup"] is (run_name == "warmup-01")
            for key, value in config.items():
                if key not in ("config_name", "save_path"):
                    assert args[key] == value, (name, key)
            assert args["device"] == "cuda:0"
            assert args["save_path"] == result["output_directory"] == (
                f"/home/user/projects/wan21-video/outputs/week2/{identifier}/{run_name}")
            normalized = {key: value for key, value in args.items()
                          if key not in ("fp8", "fp8_scaled", "save_path")}
            if canonical_args is None:
                canonical_args = normalized
            assert normalized == canonical_args
            assert result["forward_calls"] == len(result["forward_seconds"]) == 40
            assert all(math.isfinite(v) and v > 0 for v in result["forward_seconds"])
            assert set(result["calls"]) == set(bench.STAGE_NAMES)
            assert all(v == 1 for v in result["calls"].values())
            assert result["checks"]["expected_forward_calls"] == 40
            assert all(result["checks"][key] for key in (
                "each_stage_called_once", "forward_count_matches", "timing_gap_below_2_percent"))
            stages = bench.aggregate_stages(result["stages"])
            assert stages == result["stage_summary"]
            assert sum(r["seconds"] for r in result["stages"]) == result["stage_sum_seconds"]
            gap = abs(result["elapsed_seconds"] - result["stage_sum_seconds"]) / result["elapsed_seconds"] * 100
            assert gap == result["timing_gap_percent"] and gap < 2
            for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
                assert max(r[field] for r in result["stages"]) == result[field]
            assert "external_gpu_sampling" not in result and "profiler" not in result
            assert math.isfinite(result["elapsed_seconds"]) and result["elapsed_seconds"] > 0
            raw.append(result)
            runs.append({
                "run": run_name, "warmup": result["warmup"],
                "seconds": result["elapsed_seconds"], "forward_calls": 40,
                "timing_gap_percent": gap,
            })
            if run_name == "run-01":
                formal[name] = result
        assert bench.make_summary(raw) == summary
        result = formal[name]
        stage_table = {
            key: {"seconds": value["seconds"],
                  "allocated_mib": value["peak_allocated_bytes"] / 1024**2,
                  "reserved_mib": value["peak_reserved_bytes"] / 1024**2}
            for key, value in result["stage_summary"].items()
        }
        report["cases"][name] = {
            "session": identifier, "runs": runs, "summary_recomputed_exactly": True,
            "recorded_runtime_matches_total": True, "stage_table": stage_table,
            "formal_seconds_diagnostic": result["elapsed_seconds"],
            "allocated_peak_mib": result["peak_allocated_bytes"] / 1024**2,
            "reserved_peak_mib": result["peak_reserved_bytes"] / 1024**2,
            "allocated_peak_stages": [
                key for key, value in stage_table.items()
                if value["allocated_mib"] == result["peak_allocated_bytes"] / 1024**2],
            "forward_seconds": bench.summary(result["forward_seconds"]),
        }
    comparisons = {}
    base = report["cases"]["baseline"]
    for name in ("fp8", "fp8_scaled"):
        candidate = report["cases"][name]
        comparisons[name] = {}
        for stage in base["stage_table"]:
            b, c = base["stage_table"][stage], candidate["stage_table"][stage]
            comparisons[name][stage] = {
                "seconds_saved": b["seconds"] - c["seconds"],
                "allocated_saved_mib": b["allocated_mib"] - c["allocated_mib"],
                "allocated_reduction_fraction": (b["allocated_mib"] - c["allocated_mib"]) / b["allocated_mib"]
                if b["allocated_mib"] else None,
                "reserved_saved_mib": b["reserved_mib"] - c["reserved_mib"],
            }
    report["stage_comparisons_diagnostic_only"] = comparisons
    for relative, digest in hashes.items():
        assert sha(ROOT / relative) == digest
    report["inputs_sha256"] = hashes
    report["inputs_unchanged"] = True
    report["audit_script_sha256"] = sha(__file__)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "audit": str(output),
        "archive_sha256": report["archive_sha256"],
        "cases": report["cases"], "comparisons": comparisons,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
