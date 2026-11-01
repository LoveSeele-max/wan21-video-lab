"""Reconcile received F0-F5 timing records and whole-card samples read-only."""

import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import tarfile

import yaml

import bench


ROOT = Path(__file__).resolve().parent.parent
PACK = "W4-fp8-total-records-20261028"
DATA = ROOT / "notes/week4" / PACK
SESSIONS = (
    ("baseline", "W4-F0-torch-total-20261028-003533-43e7c074"),
    ("fp8", "W4-F1-fp8-total-20261028-133106-bf2ccdec"),
    ("fp8_scaled", "W4-F2-fp8-scaled-total-20261028-135518-9da4d16a"),
    ("baseline", "W4-F3-torch-confirm-total-20261028-142252-ac3ea804"),
    ("fp8", "W4-F4-fp8-confirm-total-20261028-144843-947405db"),
    ("fp8_scaled", "W4-F5-fp8-scaled-confirm-total-20261028-153815-14aeaba4"),
)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    output = ROOT / "notes/week4/fp8_total_audit.json"
    if output.exists():
        raise FileExistsError("Preserve prior audit")
    files = [path for path in DATA.rglob("*") if path.is_file()]
    assert len(files) == 60
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
    archive = ROOT / "notes/week4" / f"{PACK}.tar.gz"
    with tarfile.open(archive, "r:gz") as package:
        members = [m for m in package.getmembers() if not m.isdir()]
        assert len(members) == len(files)
        assert {m.name for m in members} == {p.relative_to(DATA).as_posix() for p in files}
        for member in members:
            assert member.isfile()
            with package.extractfile(member) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == sha(DATA / member.name)
    hashes[str(archive.relative_to(ROOT))] = sha(archive)
    hashes["scripts/bench.py"] = sha(ROOT / "scripts/bench.py")
    rules_path = ROOT / "configs/optimization_rules.yaml"
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))["benchmark"]
    hashes[str(rules_path.relative_to(ROOT))] = sha(rules_path)
    report = {
        "verified_date_asia_shanghai": "2026-10-28",
        "status": "passed_local_total_records_and_sampling_audit",
        "archive_sha256": sha(archive), "archive_members_match_extraction": len(files),
        "server_archive_sha256_supplied": False, "sessions": {}, "pooled": {},
        "limits": [
            "User-operated server execution; local saved-record audit only.",
            "Archive and extraction agree locally; no server archive hash supplied.",
            "Recorded runtime matches, but full system/environment or sys.executable path not independently established.",
            "No continuous second-card/process occupancy monitoring supplied.",
            "Whole-card samples at 100 ms may miss transient peaks and include setup/reporting.",
            "No output videos/latents transferred or quality assessed in this timing audit.",
            "No additional total rounds after the registered confirmation round.",
        ],
    }
    runtime = None
    canonical_args = None
    pooled = {name: [] for name in ("baseline", "fp8", "fp8_scaled")}
    for config_name, identifier in SESSIONS:
        folder = DATA / "notes/week2" / identifier
        session = load(folder / "session.json")
        saved_summary = load(folder / "summary.json")
        config_path = ROOT / "configs" / f"eval_{config_name}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        hashes[str(config_path.relative_to(ROOT))] = sha(config_path)
        assert session["status"] == "ok"
        assert session["effective_config"] == config
        assert session["bench_sha256"] == hashes["scripts/bench.py"]
        assert session["config_sha256"] == sha(config_path)
        assert session["commit"] == bench.EXPECTED_COMMIT
        options = session["options"]
        assert (options["mode"], options["steps"], options["warmup"], options["repeat"],
                options["gpu_index"], options["sample_smi"], options["online"]) == (
                    "total", None, 1, 3, 0, True, False)
        assert options["config"] == f"configs/eval_{config_name}.yaml"
        assert identifier.startswith(options["tag"] + "-total-")
        assert session["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert session["environment"]["HF_HUB_OFFLINE"] == "1"
        current_runtime = {key: session[key] for key in (
            "python", "torch", "cuda_build", "gpu", "commit", "environment",
            "generation_module", "allocator_backend", "tokenizer_directory", "tokenizer_revision")}
        if runtime is None:
            runtime = current_runtime
        assert runtime == current_runtime
        run_names = ("warmup-01", "run-01", "run-02", "run-03")
        assert {p.parent.name for p in folder.glob("*/result.json")} == set(run_names)
        raw_results, rows = [], []
        for run_name in run_names:
            run_folder = folder / run_name
            result = load(run_folder / "result.json")
            assert result["status"] == "ok" and result["mode"] == "total"
            assert result["warmup"] is (run_name == "warmup-01")
            args = result["effective_args"]
            for key, expected in config.items():
                if key not in ("config_name", "save_path"):
                    assert args[key] == expected, (identifier, run_name, key)
            expected_save = f"/home/user/projects/wan21-video/outputs/week2/{identifier}/{run_name}"
            assert args["save_path"] == result["output_directory"] == expected_save
            assert args["device"] == "cuda:0"
            normalized = {key: value for key, value in args.items()
                          if key not in ("save_path", "fp8", "fp8_scaled")}
            if canonical_args is None:
                canonical_args = normalized
            assert normalized == canonical_args
            assert result["stages"] == [] and result["calls"] == {} and result["forward_seconds"] == []
            assert "stage_summary" not in result and "profiler" not in result
            assert math.isfinite(result["elapsed_seconds"]) and result["elapsed_seconds"] > 0
            sampling = result["external_gpu_sampling"]
            assert sampling["warning"] is None and sampling["interval_ms"] == 100
            assert sampling["path"] == (
                f"/home/user/projects/wan21-video/notes/week2/{identifier}/{run_name}/gpu.csv")
            with (run_folder / "gpu.csv").open(encoding="utf-8", newline="") as stream:
                samples = list(csv.reader(stream))
            timestamps, used = [], []
            for sample in samples:
                assert len(sample) == 4 and int(sample[1]) == 0
                timestamps.append(datetime.strptime(sample[0].strip(), "%Y/%m/%d %H:%M:%S.%f"))
                memory, utilization = float(sample[2]), float(sample[3])
                assert math.isfinite(memory) and 0 <= memory <= 49140
                assert math.isfinite(utilization) and 0 <= utilization <= 100
                used.append(memory)
            assert samples and len(samples) == sampling["samples"]
            assert all(a <= b for a, b in zip(timestamps, timestamps[1:]))
            assert max(used) == sampling["sampled_peak_used_mib"]
            row = {
                "run": run_name, "warmup": result["warmup"],
                "seconds": result["elapsed_seconds"],
                "allocated_mib": result["peak_allocated_bytes"] / 1024**2,
                "reserved_mib": result["peak_reserved_bytes"] / 1024**2,
                "whole_card_sampled_peak_mib": max(used),
                "csv_rows": len(samples),
            }
            rows.append(row)
            raw_results.append(result)
            if not result["warmup"]:
                pooled[config_name].append(row)
        assert bench.make_summary(raw_results) == saved_summary
        report["sessions"][identifier] = {
            "config": config_name, "summary_recomputed_exactly": True,
            "configuration_and_identity_verified": True, "runs": rows,
        }
    for name, rows in pooled.items():
        assert len(rows) == 6
        report["pooled"][name] = {
            "measured_runs": 6, "elapsed_seconds": bench.summary([r["seconds"] for r in rows]),
            "allocated_peak_mib": bench.summary([r["allocated_mib"] for r in rows]),
            "reserved_peak_mib": bench.summary([r["reserved_mib"] for r in rows]),
            "whole_card_sampled_peak_mib": bench.summary([r["whole_card_sampled_peak_mib"] for r in rows]),
            "all_six_runs": rows,
        }
    base = report["pooled"]["baseline"]["elapsed_seconds"]
    for name in ("fp8", "fp8_scaled"):
        candidate = report["pooled"][name]["elapsed_seconds"]
        reduction = (base["mean"] - candidate["mean"]) / base["mean"]
        overlap = max(base["min"], candidate["min"]) <= min(base["max"], candidate["max"])
        report["pooled"][name].update({
            "latency_reduction_fraction": reduction, "closed_ranges_overlap": overlap,
            "speed_threshold_passed": reduction >= rules["minimum_latency_reduction_fraction"] and not overlap,
            "confirmation_rounds_completed": 1, "further_total_rounds_allowed": False,
        })
    for relative, digest in hashes.items():
        assert sha(ROOT / relative) == digest
    report["inputs_sha256"] = hashes
    report["inputs_unchanged"] = True
    report["matching_recorded_runtime"] = runtime
    report["audit_script_sha256"] = sha(__file__)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "audit": str(output), "files": len(files),
        "archive_sha256": report["archive_sha256"],
        "total_csv_rows": sum(r["csv_rows"] for s in report["sessions"].values() for r in s["runs"]),
        "pooled": {name: {key: value for key, value in values.items() if key != "all_six_runs"}
                   for name, values in report["pooled"].items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
