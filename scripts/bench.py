"""Week 2 measurements for the pinned Wan2.1 T2V 1.3B checkout.

total: generate through video save, without stage/forward instrumentation.
stages: adjacent intervals, absolute allocator peaks, forward timing/call checks.
profile: 20-step generation; capture 4 forwards (2 CFG steps), diagnostic only.
cold: launcher-before-spawn through child save completion, including imports and
      setup, excluding interpreter shutdown. OS disk caches are not cleared.

Warmup does not keep models resident: every iteration reloads T5, DiT and VAE.
Each iteration gets fresh args and resets peaks before timing. Stage boundaries
synchronize CUDA and reset peaks; total/cold do neither inside generation.
For stages, max(interval peaks) covers all recorded intervals; it need not equal
an independent total-mode run. Both allocated and reserved are absolute bytes.
Optional nvidia-smi sampling covers the whole card every 100 ms, may include
setup/reporting outside timing, and can miss short peaks. All modes share the
same local tokenizer compatibility wrapper. Formal runs default to offline.
"""

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter
import traceback
import uuid
from contextlib import ExitStack
from datetime import datetime
from functools import wraps
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_COMMIT = "f8a1b03794a49239a3539015075f5123d6c07d66"
TOKENIZER_REVISION = "a064a6c71f5be440641209c07bf2a5ce7a2ff5e4"
STAGE_NAMES = (
    "prepare_t2v_inputs", "load_text_encoder", "load_dit_models",
    "run_sampling", "save_latent", "decode_latent", "load_vae", "save_video",
)
LOAD_NAMES = {"load_text_encoder", "load_dit_models", "load_vae"}


def parse_options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["total", "stages", "profile", "cold"],
                        default="total")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs" / "baseline_bf16.yaml")
    parser.add_argument("--steps", type=int, help="Override; 2 for smoke only")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--gpu-index", type=int, default=0,
                        help="Physical GPU index; expose only this GPU")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--sample-smi", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--launch-start", type=float, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.warmup < 0 or options.repeat < 1 or options.gpu_index < 0:
        parser.error("Require warmup >= 0, repeat >= 1, gpu-index >= 0")
    if options.steps is not None and options.steps < 1:
        parser.error("--steps must be positive")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not options.tag or any(c not in allowed for c in options.tag):
        parser.error("--tag accepts letters, digits, underscore and hyphen")
    return options


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


def summary(values):
    if not values:
        return None
    return {"n": len(values), "mean": statistics.mean(values),
            "min": min(values), "max": max(values)}


def build_args(wan, config):
    cli_args = []
    for name, value in config.items():
        if name == "config_name" or value is None:
            continue
        if isinstance(value, bool):
            if value:
                cli_args.append(f"--{name}")
            continue
        cli_args.append(f"--{name}")
        values = value if isinstance(value, list) else [value]
        cli_args.extend(str(item) for item in values)
    with patch.object(sys, "argv", ["wan_generate_video.py", *cli_args]):
        return wan.setup_args(wan.parse_args())


class StageRecorder:
    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.rows, self.forward_seconds = [], []
        self.calls = {name: 0 for name in STAGE_NAMES}
        self.label, self.started = None, None

    def boundary(self, next_label):
        cuda = self.torch.cuda
        cuda.synchronize(self.device)
        now = perf_counter()
        if self.label is not None:
            self.rows.append({
                "stage": self.label, "seconds": now - self.started,
                "peak_allocated_bytes": cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": cuda.max_memory_reserved(self.device),
            })
        if next_label is not None:
            cuda.reset_peak_memory_stats(self.device)
        self.label, self.started = next_label, now

    def install(self, stack, wan):
        def make_wrapper(name, original):
            @wraps(original)
            def wrapped(*args, **kwargs):
                previous = self.label
                self.calls[name] += 1
                self.boundary(name)
                try:
                    return original(*args, **kwargs)
                finally:
                    # Preserve failures if the CUDA context cannot synchronize.
                    if sys.exc_info()[0] is None:
                        self.boundary(previous)
            return wrapped
        for name in STAGE_NAMES:
            original = getattr(wan, name)
            stack.enter_context(patch.object(wan, name, make_wrapper(name, original)))


def install_tokenizer(stack, wan):
    from huggingface_hub import try_to_load_from_cache
    from musubi_tuner.wan.modules import t5 as t5_module

    cache_dir = PROJECT_ROOT / "models" / ".hf-home" / "hub"
    cached = try_to_load_from_cache(
        "Wan-AI/Wan2.1-T2V-14B", "google/umt5-xxl/tokenizer_config.json",
        revision=TOKENIZER_REVISION, cache_dir=str(cache_dir),
    )
    if not isinstance(cached, str) or not Path(cached).is_file():
        raise RuntimeError(f"Pinned tokenizer not found in project cache: {cache_dir}")
    directory = str(Path(cached).parent)
    original_encoder = wan.T5EncoderModel
    original_tokenizer = t5_module.HuggingfaceTokenizer

    def create_tokenizer(*args, **kwargs):
        if kwargs.get("subfolder") is None:
            kwargs.pop("subfolder", None)
        return original_tokenizer(*args, **kwargs)

    def create_encoder(**kwargs):
        kwargs["tokenizer_path"] = directory
        return original_encoder(**kwargs)

    stack.enter_context(patch.object(t5_module, "HuggingfaceTokenizer", create_tokenizer))
    stack.enter_context(patch.object(wan, "T5EncoderModel", create_encoder))
    create_tokenizer(name=directory, seq_len=512, clean="whitespace", subfolder=None)
    print("Tokenizer precheck passed.", flush=True)
    return directory


class SmiSampler:
    def __init__(self, path, gpu_index):
        self.path, self.gpu_index = path, gpu_index
        self.process, self.file = None, None

    def start(self):
        self.file = self.path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen([
                "nvidia-smi", f"--id={self.gpu_index}",
                "--query-gpu=timestamp,index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits", "--loop-ms=100",
            ], stdout=self.file, stderr=subprocess.STDOUT)
        except BaseException:
            self.file.close()
            raise

    def stop(self):
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            self.file.close()
        samples = []
        with self.path.open(encoding="utf-8") as file:
            for row in csv.reader(file):
                try:
                    samples.append(float(row[2].strip()))
                except (ValueError, IndexError):
                    continue
        return {
            "path": str(self.path), "interval_ms": 100,
            "scope": "whole card; includes brief setup/reporting outside timed interval",
            "samples": len(samples),
            "sampled_peak_used_mib": max(samples) if samples else None,
            "warning": None if samples else "No valid samples; inspect gpu.csv",
        }


def collect_profile(torch, run_dir, stack):
    state = {"trace_ready": False, "schedule_forwards": {"wait": 16, "warmup": 2, "active": 4}}

    def ready(prof):
        prof.export_chrome_trace(str(run_dir / "trace.json"))
        table = prof.key_averages().table(sort_by="self_device_time_total", row_limit=40)
        (run_dir / "operators.txt").write_text(table, encoding="utf-8")
        events = [{
            "name": item.key, "calls": item.count,
            "self_device_time_us": item.self_device_time_total,
            "device_time_us": item.device_time_total,
            "self_cpu_time_us": item.self_cpu_time_total,
        } for item in prof.key_averages()]
        events.sort(key=lambda item: item["self_device_time_us"], reverse=True)
        write_json(run_dir / "operators.json", events)
        state["trace_ready"] = True

    prof = stack.enter_context(torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=16, warmup=2, active=4, repeat=1),
        on_trace_ready=ready, record_shapes=False, profile_memory=False, with_stack=False,
    ))
    return prof, state


def install_forward(stack, wan, torch, device, durations=None, profiler=None):
    original = wan.WanModel.forward
    state = {"count": 0}

    @wraps(original)
    def wrapped(model, *args, **kwargs):
        if durations is not None:
            torch.cuda.synchronize(device)
            started = perf_counter()
        output = original(model, *args, **kwargs)
        if durations is not None:
            torch.cuda.synchronize(device)
            durations.append(perf_counter() - started)
        state["count"] += 1
        if profiler is not None:
            profiler.step()
        return output

    stack.enter_context(patch.object(wan.WanModel, "forward", wrapped))
    return state


def aggregate_stages(rows):
    grouped = {}
    for row in rows:
        target = grouped.setdefault(row["stage"], {
            "seconds": 0.0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0})
        target["seconds"] += row["seconds"]
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            target[key] = max(target[key], row[key])
    return grouped


def run_once(wan, torch, args, options, run_dir, warmup):
    run_dir.mkdir(parents=True)
    args.save_path = str(PROJECT_ROOT / "outputs" / "week2" / run_dir.parent.name / run_dir.name)
    args.device = torch.device("cuda:0")
    recorder = StageRecorder(torch, args.device) if options.mode == "stages" else None
    sampler = SmiSampler(run_dir / "gpu.csv", options.gpu_index) if options.sample_smi else None
    result = {
        "status": "running", "mode": options.mode, "warmup": warmup,
        "effective_args": copy.deepcopy(vars(args)), "output_directory": args.save_path,
        "timing_includes": "generate + cleanup + latent/video save",
        "timing_excludes": "imports, tokenizer precheck, argument/settings preparation, reporting",
        "stages": [], "calls": {}, "forward_seconds": [],
    }
    started = None
    forward_state, profile_state = None, None
    try:
        height, width, _ = wan.check_inputs(args)
        settings = wan.get_generation_settings(args)
        if sampler is not None:
            sampler.start()
        with ExitStack() as stack:
            if recorder is not None:
                recorder.install(stack, wan)
                forward_state = install_forward(stack, wan, torch, args.device,
                                                durations=recorder.forward_seconds)
            elif options.mode == "profile":
                profiler, profile_state = collect_profile(torch, run_dir, stack)
                forward_state = install_forward(stack, wan, torch, args.device, profiler=profiler)
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            started = perf_counter()
            if recorder is not None:
                recorder.boundary("other")
            latent = wan.generate(args, settings)
            gc.collect()
            wan.clean_memory_on_device(args.device)
            wan.save_output(latent, args, settings.cfg, height, width)
            torch.cuda.synchronize(args.device)
            if recorder is not None:
                recorder.boundary(None)
            finished = perf_counter()
            result["elapsed_seconds"] = finished - started
            if options.launch_start is not None:
                result["launch_to_video_saved_seconds"] = finished - options.launch_start
                result["cold_boundary"] = "parent before subprocess spawn -> child save completion"
            if recorder is None:
                result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
                result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
            del latent

        if profile_state is not None:
            result["profiler"] = profile_state
            if not profile_state["trace_ready"]:
                raise RuntimeError("Profiler did not complete its scheduled capture.")
        if recorder is not None:
            grouped = aggregate_stages(recorder.rows)
            stage_sum = sum(row["seconds"] for row in recorder.rows)
            gap = abs(result["elapsed_seconds"] - stage_sum) / result["elapsed_seconds"] * 100
            result.update({
                "stage_summary": grouped, "stage_sum_seconds": stage_sum,
                "timing_gap_percent": gap,
                "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in recorder.rows),
                "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in recorder.rows),
                "excluding_load_intervals_seconds": result["elapsed_seconds"] -
                    sum(value["seconds"] for name, value in grouped.items() if name in LOAD_NAMES),
                "load_exclusion_note": "Subtract loader function intervals only; transfer/cleanup may remain",
            })
            result["checks"] = {
                "each_stage_called_once": all(n == 1 for n in recorder.calls.values()),
                "forward_count_matches": forward_state["count"] == args.infer_steps * 2,
                "expected_forward_calls": args.infer_steps * 2,
                "timing_gap_below_2_percent": gap < 2,
            }
            keys = ("each_stage_called_once", "forward_count_matches", "timing_gap_below_2_percent")
            if not all(result["checks"][key] for key in keys):
                raise RuntimeError("Stage validation failed; inspect result.json checks.")
        result["status"] = "ok"
    except BaseException as error:
        result.update({
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_stage": recorder.label if recorder is not None else None,
        })
        if started is not None:
            result["time_until_failure_seconds"] = perf_counter() - started
        raise
    finally:
        if recorder is not None:
            result.update(stages=recorder.rows, calls=recorder.calls,
                          forward_seconds=recorder.forward_seconds)
        if forward_state is not None:
            result["forward_calls"] = forward_state["count"]
        if sampler is not None and sampler.process is not None:
            try:
                result["external_gpu_sampling"] = sampler.stop()
            except Exception as error:
                result["sampler_error"] = str(error)
        write_json(run_dir / "result.json", result)
        print(f"Result saved: {run_dir / 'result.json'}", flush=True)
    gc.collect()
    wan.clean_memory_on_device(args.device)
    return result


def make_summary(results):
    measured = [item for item in results if not item["warmup"] and item["status"] == "ok"]
    report = {
        "measured_runs": len(measured),
        "elapsed_seconds": summary([item["elapsed_seconds"] for item in measured]),
        "peak_allocated_bytes": summary([item["peak_allocated_bytes"] for item in measured]),
        "peak_reserved_bytes": summary([item["peak_reserved_bytes"] for item in measured]),
        "stage_seconds": {}, "stage_peak_allocated_bytes": {}, "stage_peak_reserved_bytes": {},
    }
    names = sorted({name for item in measured for name in item.get("stage_summary", {})})
    for name in names:
        rows = [item["stage_summary"][name] for item in measured]
        report["stage_seconds"][name] = summary([row["seconds"] for row in rows])
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            report[f"stage_{key}"][name] = summary([row[key] for row in rows])
    report["excluding_load_intervals_seconds"] = summary([
        item["excluding_load_intervals_seconds"] for item in measured
        if "excluding_load_intervals_seconds" in item])
    report["launch_to_video_saved_seconds"] = summary([
        item["launch_to_video_saved_seconds"] for item in measured
        if "launch_to_video_saved_seconds" in item])
    forwards = [value for item in measured for value in item["forward_seconds"]]
    report["forward_seconds"] = summary(forwards)
    steps = [sum(values[index:index + 2])
             for item in measured for values in [item["forward_seconds"]]
             for index in range(0, len(values) - 1, 2)]
    report["paired_forward_seconds"] = summary(steps)
    report["paired_forward_note"] = "Two forwards only; excludes scheduler/loop overhead"
    return report


def main(argv=None):
    options = parse_options(argv)
    if options.mode == "cold" and options.launch_start is None:
        # No torch/YAML/Musubi imports in the launcher.
        started = perf_counter()
        command = [sys.executable, str(Path(__file__).resolve()),
                   *(sys.argv[1:] if argv is None else argv),
                   "--launch-start", str(started)]
        return subprocess.call(command)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible != str(options.gpu_index):
        raise ValueError("CUDA_VISIBLE_DEVICES conflicts with --gpu-index.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu_index)
    os.environ["HF_HUB_OFFLINE"] = "0" if options.online else "1"
    # Use the cache verified during smoke tests, regardless of shell defaults.
    os.environ["HF_HOME"] = str(PROJECT_ROOT / "models" / ".hf-home")
    os.environ["HF_HUB_CACHE"] = str(PROJECT_ROOT / "models" / ".hf-home" / "hub")

    import yaml
    import torch
    sys.path.insert(0, str(PROJECT_ROOT / "musubi-tuner" / "src"))
    from musubi_tuner import wan_generate_video as wan

    config = yaml.safe_load(options.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping.")
    if options.steps is not None:
        config["infer_steps"] = options.steps
    base_args = build_args(wan, config)
    if base_args.task != "t2v-1.3B" or base_args.output_type != "both":
        raise ValueError("Week 2 requires t2v-1.3B and output_type=both.")
    if base_args.cfg_skip_mode != "none" or base_args.slg_mode is not None:
        raise ValueError("Require CFG on every step and SLG disabled.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the server project environment.")
    for name in ("dit", "vae", "t5"):
        if not Path(getattr(base_args, name)).is_file():
            raise FileNotFoundError(f"Missing {name} weight: {getattr(base_args, name)}")
    repo = PROJECT_ROOT / "musubi-tuner"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Unexpected upstream commit: {commit}")
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True)
    if dirty.strip():
        raise RuntimeError("Upstream worktree has changes; inspect git status before baseline.")
    if options.mode == "profile" and base_args.infer_steps != 20:
        raise ValueError("Profile mode requires the frozen 20-step configuration.")
    if options.mode in ("cold", "profile"):
        options.warmup, options.repeat = 0, 1

    session = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    session_dir = PROJECT_ROOT / "notes" / "week2" / f"{options.tag}-{options.mode}-{session}"
    session_dir.mkdir(parents=True)
    results = []
    metadata = {
        "status": "setup", "options": vars(options), "effective_config": config,
        "config_sha256": hashlib.sha256(options.config.read_bytes()).hexdigest(),
        "bench_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "commit": commit, "python": sys.version, "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_CACHE", "HF_ENDPOINT",
            "HF_HUB_OFFLINE", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")},
        "generation_module": wan.__file__, "allocator_backend": torch.cuda.get_allocator_backend(),
    }
    print(f"Mode: {options.mode}; actual steps: {base_args.infer_steps}; "
          f"warmup: {options.warmup}; measured: {options.repeat}", flush=True)
    print(f"Session directory: {session_dir}", flush=True)
    try:
        with ExitStack() as stack:
            metadata["tokenizer_directory"] = install_tokenizer(stack, wan)
            metadata["tokenizer_revision"] = TOKENIZER_REVISION
            write_json(session_dir / "session.json", metadata)
            for index in range(options.warmup + options.repeat):
                warmup = index < options.warmup
                kind = "warmup" if warmup else "run"
                number = index + 1 if warmup else index - options.warmup + 1
                run_dir = session_dir / f"{kind}-{number:02d}"
                result = run_once(wan, torch, copy.deepcopy(base_args), options, run_dir, warmup)
                results.append(result)
                print(f"{kind}-{number:02d}: {result['elapsed_seconds']:.3f} s; "
                      f"allocated peak {result['peak_allocated_bytes'] / 1024**2:.1f} MiB",
                      flush=True)
            write_json(session_dir / "summary.json", make_summary(results))
            metadata["status"] = "ok"
            print(f"Summary saved: {session_dir / 'summary.json'}", flush=True)
    except BaseException as error:
        metadata.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(session_dir / "session.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
