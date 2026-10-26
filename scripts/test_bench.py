"""CPU-only control-flow checks; no claim of CUDA accuracy or real model testing."""

import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("week2_bench", Path(__file__).with_name("bench.py"))
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class FakeRuntime:
    def __init__(self):
        self.clock = 0.0
        self.resets = 0
        self.fail = False
        self.profiler = None
        self.seeds = []
        runtime = self

        class Model:
            def forward(self):
                runtime.clock += 1.0
                if runtime.fail:
                    raise RuntimeError("simulated forward failure")
                return "prediction"

        class Profile:
            def __init__(self, **kwargs):
                self.callback = kwargs["on_trace_ready"]
                self.steps = 0

            def __enter__(self):
                runtime.profiler = self
                return self

            def __exit__(self, *args):
                return False

            def step(self):
                self.steps += 1
                if self.steps == 22:
                    self.callback(self)

            def export_chrome_trace(self, path):
                Path(path).write_text("{}", encoding="utf-8")

            def key_averages(self):
                class Events(list):
                    def table(self, **kwargs):
                        return "fake device table"
                return Events([SimpleNamespace(
                    key="fake_op", count=4, self_device_time_total=10,
                    device_time_total=12, self_cpu_time_total=2)])

        def advance(*args, **kwargs):
            self.clock += 0.1

        def reset(device):
            self.resets += 1

        self.torch = SimpleNamespace(
            device=lambda name: name,
            cuda=SimpleNamespace(
                synchronize=lambda device: None,
                reset_peak_memory_stats=reset,
                max_memory_allocated=lambda device: 1000,
                max_memory_reserved=lambda device: 2000),
            profiler=SimpleNamespace(
                profile=Profile, schedule=lambda **kwargs: kwargs,
                ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda")),
        )
        self.wan = SimpleNamespace(
            WanModel=Model,
            check_inputs=lambda args: (*args.video_size, args.video_length),
            get_generation_settings=lambda args: SimpleNamespace(cfg="config"),
            load_text_encoder=advance, load_dit_models=advance, load_vae=advance,
            save_latent=advance, save_video=advance, clean_memory_on_device=advance,
        )

        def prepare(*args):
            self.wan.load_text_encoder()
            advance()

        def sampling(args):
            model = Model()
            for _ in range(args.infer_steps * 2):
                model.forward()

        def generate(args, settings):
            self.seeds.append(args.seed)
            args.seed = 999
            self.wan.prepare_t2v_inputs()
            self.wan.load_dit_models()
            self.wan.run_sampling(args)
            return "latent"

        def decode():
            self.wan.load_vae()
            advance()

        def save_output(*args):
            self.wan.save_latent()
            self.wan.decode_latent()
            self.wan.save_video()

        self.wan.prepare_t2v_inputs = prepare
        self.wan.run_sampling = sampling
        self.wan.generate = generate
        self.wan.decode_latent = decode
        self.wan.save_output = save_output

    def args(self, steps=2):
        return SimpleNamespace(
            infer_steps=steps, video_size=[480, 832], video_length=81,
            seed=42, task="t2v-1.3B",
        )

    def options(self, mode):
        return SimpleNamespace(mode=mode, sample_smi=False, gpu_index=0, launch_start=None)


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = FakeRuntime()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(bench, "PROJECT_ROOT", self.root))
        self.stack.enter_context(patch.object(bench, "perf_counter", lambda: self.runtime.clock))
        self.stack.enter_context(patch.object(bench.gc, "collect", lambda: None))

    def run_mode(self, mode, name="run-01", warmup=False, steps=2):
        return bench.run_once(
            self.runtime.wan, self.runtime.torch, self.runtime.args(steps),
            self.runtime.options(mode), self.root / "session" / name, warmup)

    def test_stage_coverage_counts_and_restoration(self):
        original = self.runtime.wan.WanModel.forward
        result = self.run_mode("stages")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["forward_calls"], 4)
        self.assertTrue(all(n == 1 for n in result["calls"].values()))
        self.assertAlmostEqual(result["stage_sum_seconds"], result["elapsed_seconds"])
        self.assertEqual(len(result["forward_seconds"]), 4)
        self.assertAlmostEqual(result["stage_summary"]["decode_latent"]["seconds"], 0.1)
        self.assertIs(self.runtime.wan.WanModel.forward, original)
        self.assertEqual(json.loads((self.root / "session/run-01/result.json").read_text())["status"], "ok")

    def test_total_no_forward_wrapper_or_midrun_resets(self):
        original = self.runtime.wan.WanModel.forward
        result = self.run_mode("total")
        self.assertEqual(self.runtime.resets, 1)
        self.assertEqual(result["forward_seconds"], [])
        self.assertNotIn("forward_calls", result)
        self.assertIs(self.runtime.wan.WanModel.forward, original)
        self.assertEqual(result["peak_reserved_bytes"], 2000)

    def test_failure_saved_and_wrappers_restored(self):
        original = self.runtime.wan.WanModel.forward
        stage_original = self.runtime.wan.run_sampling
        self.runtime.fail = True
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            self.run_mode("stages")
        result = json.loads((self.root / "session/run-01/result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed_stage"], "run_sampling")
        self.assertIs(self.runtime.wan.WanModel.forward, original)
        self.assertIs(self.runtime.wan.run_sampling, stage_original)

    def test_profile_schedule_exports_and_restoration(self):
        original = self.runtime.wan.WanModel.forward
        result = self.run_mode("profile", steps=20)
        self.assertEqual(result["forward_calls"], 40)
        self.assertEqual(self.runtime.profiler.steps, 40)
        self.assertTrue(result["profiler"]["trace_ready"])
        self.assertTrue((self.root / "session/run-01/operators.json").is_file())
        self.assertIs(self.runtime.wan.WanModel.forward, original)

    def test_summary_excludes_warmup(self):
        warm = self.run_mode("stages", "warmup-01", True)
        measured = self.run_mode("stages")
        warm["elapsed_seconds"] = 99999
        report = bench.make_summary([warm, measured])
        self.assertEqual(report["elapsed_seconds"]["n"], 1)
        self.assertEqual(report["elapsed_seconds"]["mean"], measured["elapsed_seconds"])
        self.assertEqual(self.runtime.seeds, [42, 42])
        self.assertEqual(report["paired_forward_seconds"]["n"], 2)

    def test_build_args_restores_argv_and_boolean_flags(self):
        original = bench.sys.argv
        wan = SimpleNamespace(
            parse_args=lambda: bench.sys.argv[1:],
            setup_args=lambda args: args)
        args = bench.build_args(wan, {
            "config_name": "test", "video_size": [480, 832], "fp8": False,
            "compile": True, "negative_prompt": None})
        self.assertEqual(args, ["--video_size", "480", "832", "--compile"])
        self.assertIs(bench.sys.argv, original)

    def test_cold_launcher_passes_clock_to_child(self):
        with patch.object(bench.subprocess, "call", return_value=0) as call:
            self.assertEqual(bench.main(["--mode", "cold"]), 0)
        command = call.call_args.args[0]
        self.assertIn("--launch-start", command)
        self.assertEqual(command[-1], "0.0")

    def test_sampler_parses_whole_card_samples(self):
        sampler = bench.SmiSampler(self.root / "gpu.csv", 0)
        sampler.path.write_text("time, 0, 816, 0\ntime, 0, 999, 25\n", encoding="utf-8")
        sampler.file = sampler.path.open("a", encoding="utf-8")
        sampler.process = SimpleNamespace(poll=lambda: 0)
        result = sampler.stop()
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["sampled_peak_used_mib"], 999)

    def test_main_repeats_fresh_args_and_saves_session(self):
        runtime = self.runtime
        runtime.torch.__version__ = "fake"
        runtime.torch.version = SimpleNamespace(cuda="fake")
        runtime.torch.cuda.is_available = lambda: True
        runtime.torch.cuda.get_device_name = lambda index: "fake GPU"
        runtime.torch.cuda.get_allocator_backend = lambda: "fake"

        def parse():
            parser = argparse.ArgumentParser()
            parser.add_argument("--task")
            parser.add_argument("--output_type")
            parser.add_argument("--infer_steps", type=int)
            parser.add_argument("--seed", type=int)
            parser.add_argument("--video_size", nargs=2, type=int)
            parser.add_argument("--video_length", type=int)
            for name in ("dit", "vae", "t5"):
                parser.add_argument("--" + name)
            parser.set_defaults(cfg_skip_mode="none", slg_mode=None)
            return parser.parse_args()

        runtime.wan.parse_args = parse
        runtime.wan.setup_args = lambda args: args
        runtime.wan.__file__ = "fake checkout"
        config = {
            "task": "t2v-1.3B", "output_type": "both", "infer_steps": 2,
            "seed": 42, "video_size": [480, 832], "video_length": 81,
        }
        for name in ("dit", "vae", "t5"):
            weight = self.root / (name + ".fake")
            weight.touch()
            config[name] = str(weight)
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        modules = {
            "torch": runtime.torch,
            "yaml": SimpleNamespace(safe_load=json.loads),
            "musubi_tuner": SimpleNamespace(wan_generate_video=runtime.wan),
        }
        def git(command, **kwargs):
            return bench.EXPECTED_COMMIT if "rev-parse" in command else ""
        with patch.dict(bench.sys.modules, modules), \
                patch.dict(bench.os.environ, {
                    "CUDA_VISIBLE_DEVICES": "0", "HF_HOME": "wrong-home",
                    "HF_HUB_CACHE": "wrong-hub"}), \
                patch.object(bench, "install_tokenizer", return_value="cached"), \
                patch.object(bench.subprocess, "check_output", side_effect=git):
            code = bench.main([
                "--mode", "stages", "--config", str(config_path),
                "--warmup", "1", "--repeat", "3"])
            self.assertEqual(bench.os.environ["HF_HOME"],
                             str(self.root / "models" / ".hf-home"))
            self.assertEqual(bench.os.environ["HF_HUB_CACHE"],
                             str(self.root / "models" / ".hf-home" / "hub"))
        self.assertEqual(code, 0)
        self.assertEqual(runtime.seeds, [42, 42, 42, 42])
        summaries = list(self.root.glob("notes/week2/*/summary.json"))
        self.assertEqual(len(summaries), 1)
        report = json.loads(summaries[0].read_text())
        self.assertEqual(report["measured_runs"], 3)
        self.assertEqual(report["forward_seconds"]["n"], 12)

    def test_tokenizer_compatibility_and_restoration(self):
        directory = self.root / "tokenizer"
        directory.mkdir()
        cached = directory / "tokenizer_config.json"
        cached.write_text("{}", encoding="utf-8")
        received = []
        def tokenizer(**kwargs):
            received.append(kwargs)
        t5 = SimpleNamespace(HuggingfaceTokenizer=tokenizer)
        cache_queries = []
        def query_cache(*args, **kwargs):
            cache_queries.append(kwargs)
            return str(cached)
        def encoder(**kwargs):
            t5.HuggingfaceTokenizer(
                name=kwargs["tokenizer_path"], subfolder=None)
            return "encoder"
        wan = SimpleNamespace(T5EncoderModel=encoder)
        modules = {
            "huggingface_hub": SimpleNamespace(
                try_to_load_from_cache=query_cache),
            "musubi_tuner.wan.modules": SimpleNamespace(t5=t5),
        }
        with patch.dict(bench.sys.modules, modules):
            with ExitStack() as stack:
                self.assertEqual(bench.install_tokenizer(stack, wan), str(directory))
                self.assertEqual(wan.T5EncoderModel(tokenizer_path=None), "encoder")
            self.assertIs(wan.T5EncoderModel, encoder)
            self.assertIs(t5.HuggingfaceTokenizer, tokenizer)
        self.assertTrue(all("subfolder" not in kwargs for kwargs in received))
        self.assertEqual(cache_queries[0]["cache_dir"],
                         str(self.root / "models" / ".hf-home" / "hub"))


if __name__ == "__main__":
    unittest.main()
