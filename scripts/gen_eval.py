"""Generate the frozen evaluation set using the Week 2 single-sample path.

No torch/Musubi imports for --dry-run, --summarize or unit tests.
Timing is diagnostic (possibly concurrent), never a latency benchmark.
Successful outputs require a matching receipt and verified MP4/latent hashes.
Failures and interrupted attempts remain on disk. At most two attempts/sample.
"""

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime
from time import perf_counter
import traceback
import uuid

import bench

ROOT = Path(__file__).resolve().parent.parent
MAX_ATTEMPTS = 2
TUNABLE = {
    "attn_mode", "fp8", "fp8_scaled", "fp8_fast", "fp8_t5",
    "blocks_to_swap", "vae_cache_cpu", "compile",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str),
                             encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs/prompts_frozen.yaml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval_baseline.yaml")
    parser.add_argument("--shard", default="0/1", help="Zero-based shard, e.g. 0/2")
    parser.add_argument("--gpu-index", type=int, default=0, help="Physical GPU index")
    parser.add_argument("--steps", type=int, choices=[2], help="Isolated 2-step smoke only")
    parser.add_argument("--limit", type=int, help="First N selected samples in this shard")
    parser.add_argument("--prompt-id", help="Select one frozen prompt, e.g. P01")
    parser.add_argument("--seed", type=int, help="Select one frozen seed")
    parser.add_argument("--tag", default="", help="Separate diagnostic output namespace")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="List jobs; no model imports or writes")
    mode.add_argument("--summarize", action="store_true", help="Merge receipts; no model imports")
    options = parser.parse_args(argv)
    match = re.fullmatch(r"(\d+)/(\d+)", options.shard)
    if not match:
        parser.error("--shard must be INDEX/COUNT")
    options.shard_index, options.shard_count = map(int, match.groups())
    if not 0 <= options.shard_index < options.shard_count:
        parser.error("Require 0 <= shard index < shard count")
    if options.gpu_index < 0 or (options.limit is not None and options.limit < 1):
        parser.error("GPU index must be >= 0 and limit must be positive")
    if options.tag and not re.fullmatch(r"[A-Za-z0-9_-]+", options.tag):
        parser.error("--tag accepts letters, digits, underscore and hyphen")
    return options


def load_plan(options):
    import yaml

    manifest = yaml.safe_load(options.prompts.read_text(encoding="utf-8"))
    config = yaml.safe_load(options.config.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(config, dict):
        raise ValueError("Both YAML files must be mappings")
    if manifest.get("status") != "frozen" or manifest.get("generation_settings_status") != "frozen":
        raise ValueError("Evaluation settings are not frozen")
    settings = manifest["generation_settings"]
    if not isinstance(settings.get("cpu_noise"), bool):
        raise ValueError("cpu_noise must be explicitly true or false")
    prompts, seeds = manifest["prompts"], manifest["seeds"]
    if not prompts or not seeds:
        raise ValueError("Empty evaluation set")
    ids = [item["id"] for item in prompts]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"P\d{2}", key) for key in ids):
        raise ValueError("Prompt IDs must be unique Pxx identifiers")
    if any(not isinstance(item.get("prompt"), str) or not item["prompt"].strip() for item in prompts):
        raise ValueError("Every prompt must contain text")
    if (any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError("Seeds must be distinct uint32 integers")
    if len(prompts) * len(seeds) != manifest["expected_video_count_per_config"]:
        raise ValueError("Frozen coverage count does not match prompts and seeds")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config.get("config_name", "")):
        raise ValueError("Invalid config_name")
    for key, value in settings.items():
        if key != "precision" and key not in TUNABLE and config.get(key) != value:
            raise ValueError(f"Config differs from frozen generation setting: {key}")
    if settings["precision"] != "bf16" or config.get("output_type") != "both":
        raise ValueError("Require BF16 base weights and output_type=both")
    for name in ("dit", "vae", "t5"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError(f"Missing model path: {name}")
    jobs = [{"prompt_id": item["id"], "prompt": item["prompt"], "seed": seed,
             "key": f"{item['id']}_s{seed}"} for item in prompts for seed in seeds]
    if options.prompt_id is not None and options.prompt_id not in ids:
        raise ValueError("Unknown --prompt-id")
    if options.seed is not None and options.seed not in seeds:
        raise ValueError("Unknown --seed")
    # Shard the full ordered Cartesian product before applying diagnostic filters.
    selected = [job for index, job in enumerate(jobs)
                if index % options.shard_count == options.shard_index
                and (options.prompt_id is None or job["prompt_id"] == options.prompt_id)
                and (options.seed is None or job["seed"] == options.seed)]
    if options.limit is not None:
        selected = selected[:options.limit]
    if options.steps is not None:
        config["infer_steps"] = options.steps
    name = config["config_name"].removeprefix("eval_")
    if options.steps is not None:
        name += f"-smoke{options.steps}"
    if options.tag:
        name += f"-{options.tag}"
    identity = {
        "prompts_sha256": file_hash(options.prompts),
        "config_sha256": file_hash(options.config),
        "bench_sha256": file_hash(Path(bench.__file__)),
        "script_sha256": file_hash(Path(__file__)),
        "commit": bench.EXPECTED_COMMIT, "actual_steps": config["infer_steps"],
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"manifest": manifest, "config": config, "jobs": jobs, "selected": selected,
            "name": name, "identity": identity, "fingerprint": fingerprint,
            "notes": ROOT / "notes/eval" / name,
            "outputs": ROOT / "outputs/eval" / name}


def bind_dataset(plan):
    """Publish one complete identity atomically, including concurrent shards."""
    directory = plan["notes"]
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "dataset.json"
    temporary = directory / f".identity-{uuid.uuid4().hex}.tmp"
    value = {"fingerprint": plan["fingerprint"], "identity": plan["identity"]}
    temporary.write_text(json.dumps(value), encoding="utf-8")
    try:
        try:
            os.link(temporary, marker)
        except FileExistsError:
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if existing != value:
                raise ValueError("Dataset identity changed; use a new config/tag, do not mix results")
    finally:
        temporary.unlink(missing_ok=True)


def records_for(plan, job):
    directory = plan["notes"] / "attempts" / job["key"]
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("attempt-*/result.json"))]


def verified_success(record, plan):
    if record.get("status") != "ok" or record.get("fingerprint") != plan["fingerprint"]:
        return False
    for kind in ("video", "latent"):
        path = Path(record[f"{kind}_path"])
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if file_hash(path) != record[f"{kind}_sha256"]:
            return False
    return True


def check_runtime(record, runtime):
    keys = ("torch", "cuda_build", "gpu", "python", "commit", "tokenizer_revision")
    previous = record.get("runtime", {})
    if any(previous.get(key) != runtime.get(key) for key in keys):
        raise ValueError("Runtime changed since prior attempt; use a new diagnostic tag")


def append_event(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def generate_sample(wan, torch, base_args, job, directory):
    """Match bench total's call sequence without its Week 2 output directory."""
    args = copy.deepcopy(base_args)
    args.prompt, args.seed, args.save_path = job["prompt"], job["seed"], str(directory)
    args.device = torch.device("cuda:0")
    height, width, _ = wan.check_inputs(args)
    settings = wan.get_generation_settings(args)
    effective = copy.deepcopy(vars(args))
    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    started = perf_counter()
    latent = wan.generate(args, settings)
    gc.collect()
    wan.clean_memory_on_device(args.device)
    wan.save_output(latent, args, settings.cfg, height, width)
    torch.cuda.synchronize(args.device)
    elapsed = perf_counter() - started
    result = {"elapsed_seconds": elapsed,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
              "effective_args": effective,
              "effective_negative_prompt": args.negative_prompt or settings.cfg.sample_neg_prompt}
    del latent
    gc.collect()
    wan.clean_memory_on_device(args.device)
    return result


def attempt_sample(plan, job, generator, oom_type, log_path, runtime):
    lock = plan["notes"] / "locks" / job["key"]
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f"Sample locked: {lock}; verify no live worker before removing stale lock")
    try:
        records = records_for(plan, job)
        if records:
            check_runtime(records[-1], runtime)
        if records and verified_success(records[-1], plan):
            append_event(log_path, {"status": "skipped", "key": job["key"],
                                    "reason": "verified successful receipt"})
            return "skipped"
        attempts = plan["notes"] / "attempts" / job["key"]
        count = len(list(attempts.glob("attempt-*")))
        if count >= MAX_ATTEMPTS:
            append_event(log_path, {"status": "exhausted", "key": job["key"],
                                    "reason": "two attempts already used; inspect retained records"})
            return "exhausted"
        number = count + 1
        directory = plan["outputs"] / ".attempts" / job["key"] / f"attempt-{number:02d}"
        result_path = attempts / f"attempt-{number:02d}" / "result.json"
        record = {
            **job, "config_name": plan["config"]["config_name"],
            "fingerprint": plan["fingerprint"], "identity": plan["identity"],
            "attempt": number, "status": "running", "runtime": runtime,
            "started_at": datetime.now().astimezone().isoformat(),
            "actual_steps": plan["config"]["infer_steps"],
            "timing_note": "single-sample generate/cleanup/save; may run concurrently; not benchmark",
            "attempt_output_directory": str(directory),
        }
        atomic_json(result_path, record)
        append_event(log_path, record)
        started = perf_counter()
        try:
            directory.mkdir(parents=True, exist_ok=False)
            record.update(generator(job, directory))
            videos = list(directory.glob("*.mp4"))
            latents = list(directory.glob("*_latent.safetensors"))
            if len(videos) != 1 or len(latents) != 1:
                raise RuntimeError("Expected exactly one MP4 and one latent")
            for kind, source, suffix in (("video", videos[0], ".mp4"),
                                         ("latent", latents[0], "_latent.safetensors")):
                if source.stat().st_size == 0:
                    raise RuntimeError(f"Empty {kind} output")
                # Keep attempt outputs as evidence; publish a separate stable filename.
                destination = plan["outputs"] / f"{job['key']}{suffix}"
                temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
                try:
                    shutil.copyfile(source, temporary)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                record[f"{kind}_path"] = str(destination)
                record[f"{kind}_sha256"] = file_hash(destination)
            record["status"] = "ok"
        except BaseException as error:
            record.update(status="oom" if isinstance(error, oom_type) else
                          "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "error",
                          error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc(),
                          time_until_failure_seconds=perf_counter() - started)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            record["finished_at"] = datetime.now().astimezone().isoformat()
            atomic_json(result_path, record)
            append_event(log_path, record)
        return record["status"]
    finally:
        lock.rmdir()


def summarize(plan):
    """One final row per planned sample, including missing/corrupt/failed ones."""
    rows = []
    for job in plan["jobs"]:
        records = records_for(plan, job)
        row = {**job, "status": "missing", "attempts": len(records)}
        if records:
            row.update(records[-1])
            if row["status"] == "ok" and not verified_success(row, plan):
                row.update(status="invalid_output", error="Output or hash does not match receipt")
        rows.append(row)
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    output = plan["notes"].with_suffix(".jsonl")
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n"
                                     for row in rows), encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(plan["notes"] / "coverage.json", {
        "planned": len(rows), "counts": counts, "fingerprint": plan["fingerprint"],
        "note": "Generation coverage only; human scorability is recorded separately",
    })
    print(f"Coverage: {counts}; final manifest: {output}", flush=True)
    return rows


def load_runtime(plan, options, stack):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible != str(options.gpu_index):
        raise ValueError("CUDA_VISIBLE_DEVICES conflicts with --gpu-index")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu_index)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(ROOT / "models/.hf-home")
    os.environ["HF_HUB_CACHE"] = str(ROOT / "models/.hf-home/hub")
    repo = ROOT / "musubi-tuner"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != bench.EXPECTED_COMMIT:
        raise RuntimeError(f"Unexpected upstream commit: {commit}")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True).strip():
        raise RuntimeError("Upstream worktree is dirty")
    for name in ("dit", "vae", "t5"):
        if not Path(plan["config"][name]).is_file():
            raise FileNotFoundError(f"Missing {name}: {plan['config'][name]}")
    import torch
    sys.path.insert(0, str(repo / "src"))
    from musubi_tuner import wan_generate_video as wan

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with the server project environment")
    args = bench.build_args(wan, plan["config"])
    if args.cfg_skip_mode != "none" or args.slg_mode is not None or args.from_file:
        raise ValueError("Require full CFG, no SLG and the single-prompt generation path")
    tokenizer = bench.install_tokenizer(stack, wan)
    runtime = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0), "physical_gpu": options.gpu_index,
               "python": sys.version, "commit": commit,
               "tokenizer_directory": tokenizer, "tokenizer_revision": bench.TOKENIZER_REVISION}
    return wan, torch, args, runtime


def main(argv=None):
    options = parse_options(argv)
    plan = load_plan(options)
    print(f"Dataset: {plan['name']}; steps: {plan['config']['infer_steps']}; "
          f"shard: {options.shard}; selected: {len(plan['selected'])}", flush=True)
    if options.dry_run:
        for job in plan["selected"]:
            print(json.dumps(job, ensure_ascii=False))
        return 0
    if options.summarize:
        # Summaries must match the exact frozen dataset already used for generation.
        marker = plan["notes"] / "dataset.json"
        if not marker.is_file():
            raise FileNotFoundError("Dataset not started; no receipts to summarize")
        if json.loads(marker.read_text(encoding="utf-8"))["fingerprint"] != plan["fingerprint"]:
            raise ValueError("Dataset identity changed")
        rows = summarize(plan)
        return 0 if all(row["status"] == "ok" for row in rows) else 1
    if not plan["selected"]:
        raise ValueError("No selected samples")
    with ExitStack() as stack:
        wan, torch, args, runtime = load_runtime(plan, options, stack)
        bind_dataset(plan)
        log = plan["notes"] / f"events-{options.shard_index}-of-{options.shard_count}-{uuid.uuid4().hex}.jsonl"
        generator = lambda job, directory: generate_sample(wan, torch, args, job, directory)
        failed = False
        for job in plan["selected"]:
            status = attempt_sample(plan, job, generator, torch.OutOfMemoryError, log, runtime)
            print(f"{job['key']}: {status}", flush=True)
            failed |= status not in ("ok", "skipped")
            # Error frames have unwound; clean before attempting another sample.
            gc.collect()
            wan.clean_memory_on_device(torch.device("cuda:0"))
            torch.cuda.synchronize()
        print(f"Worker log: {log}", flush=True)
    print("After both shards finish, run --summarize to build the complete manifest.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
