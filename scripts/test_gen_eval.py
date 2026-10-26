"""CPU-only tests for evaluation orchestration; not model/CUDA validation."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gen_eval as gen


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.options = gen.parse_options([])
        self.plan = gen.load_plan(self.options)
        self.plan["notes"] = self.root / "notes/baseline"
        self.plan["outputs"] = self.root / "outputs/baseline"
        gen.bind_dataset(self.plan)
        self.job = self.plan["jobs"][0]
        self.log = self.plan["notes"] / "events-test.jsonl"
        self.calls = 0

    def generate(self, job, directory):
        self.calls += 1
        (directory / "video.mp4").write_bytes(b"fake-video")
        (directory / "fake_latent.safetensors").write_bytes(b"fake-latent")
        return {"elapsed_seconds": 1.0, "peak_allocated_bytes": 100}

    def attempt(self, generator=None):
        return gen.attempt_sample(self.plan, self.job, generator or self.generate,
                                  MemoryError, self.log, {"test": True})

    def test_shards_disjoint_and_complete(self):
        a = gen.load_plan(gen.parse_options(["--shard", "0/2"]))["selected"]
        b = gen.load_plan(gen.parse_options(["--shard", "1/2"]))["selected"]
        self.assertEqual(len(a), 18)
        self.assertEqual(len(b), 18)
        self.assertFalse({x["key"] for x in a} & {x["key"] for x in b})
        self.assertEqual({x["key"] for x in a + b}, {x["key"] for x in self.plan["jobs"]})

    def test_verified_skip_and_corrupt_output_recovery(self):
        self.assertEqual(self.attempt(), "ok")
        self.assertEqual(self.attempt(), "skipped")
        self.assertEqual(self.calls, 1)
        (self.plan["outputs"] / f"{self.job['key']}.mp4").write_bytes(b"corrupt")
        self.assertEqual(self.attempt(), "ok")
        self.assertEqual(self.calls, 2)
        self.assertEqual(len(gen.records_for(self.plan, self.job)), 2)

    def test_orphan_video_is_not_a_success(self):
        self.plan["outputs"].mkdir(parents=True)
        (self.plan["outputs"] / f"{self.job['key']}.mp4").write_bytes(b"partial")
        self.assertEqual(self.attempt(), "ok")
        self.assertEqual(self.calls, 1)

    def test_failure_history_retry_limit_and_coverage(self):
        def oom(*args):
            raise MemoryError("fake OOM")
        self.assertEqual(self.attempt(oom), "oom")
        def error(*args):
            raise RuntimeError("fake save failure")
        self.assertEqual(self.attempt(error), "error")
        self.assertEqual(self.attempt(), "exhausted")
        records = gen.records_for(self.plan, self.job)
        self.assertEqual([r["status"] for r in records], ["oom", "error"])
        self.assertTrue(all("traceback" in r for r in records))
        rows = gen.summarize(self.plan)
        self.assertEqual(len(rows), 36)
        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(sum(r["status"] == "missing" for r in rows), 35)

    def test_interruption_recorded_and_lock_released(self):
        def interrupt(*args):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.attempt(interrupt)
        self.assertEqual(gen.records_for(self.plan, self.job)[0]["status"], "interrupted")
        self.assertFalse((self.plan["notes"] / "locks" / self.job["key"]).exists())
        self.assertEqual(self.attempt(), "ok")

    def test_missing_output_is_error(self):
        self.assertEqual(self.attempt(lambda *args: {}), "error")
        self.assertEqual(gen.records_for(self.plan, self.job)[0]["error_type"], "RuntimeError")

    def test_dataset_identity_change_rejected(self):
        changed = dict(self.plan, fingerprint="different")
        with self.assertRaises(ValueError):
            gen.bind_dataset(changed)

    def test_changed_runtime_does_not_skip_or_overwrite(self):
        self.attempt()
        with self.assertRaises(ValueError):
            gen.attempt_sample(self.plan, self.job, self.generate, MemoryError,
                               self.log, {"torch": "different"})
        self.assertEqual(self.calls, 1)

    def test_abandoned_attempt_does_not_reuse_directory(self):
        directory = self.plan["notes"] / "attempts" / self.job["key"] / "attempt-01"
        directory.mkdir(parents=True)
        outputs = self.plan["outputs"] / ".attempts" / self.job["key"] / "attempt-01"
        outputs.mkdir(parents=True)
        (outputs / "partial.mp4").write_bytes(b"partial")
        self.assertEqual(self.attempt(), "ok")
        self.assertEqual(gen.records_for(self.plan, self.job)[0]["attempt"], 2)
        self.assertTrue((outputs / "partial.mp4").is_file())

    def test_smoke_namespace_separate_and_formal_config_unchanged(self):
        smoke = gen.load_plan(gen.parse_options(["--steps", "2", "--limit", "2"]))
        self.assertEqual(smoke["config"]["infer_steps"], 2)
        self.assertEqual(len(smoke["selected"]), 2)
        self.assertNotEqual(smoke["name"], self.plan["name"])
        self.assertNotEqual(smoke["fingerprint"], self.plan["fingerprint"])
        self.assertEqual(self.plan["config"]["infer_steps"], 20)

    def test_pending_and_changed_frozen_settings_rejected(self):
        import yaml
        manifest = copy.deepcopy(self.plan["manifest"])
        manifest["generation_settings"]["cpu_noise"] = None
        manifest_path = self.root / "pending.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        options = gen.parse_options(["--prompts", str(manifest_path)])
        with self.assertRaises(ValueError):
            gen.load_plan(options)
        config = copy.deepcopy(self.plan["config"])
        config["video_length"] = 17
        config_path = self.root / "changed.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with self.assertRaises(ValueError):
            gen.load_plan(gen.parse_options(["--config", str(config_path)]))

    def test_lock_conflict_does_not_start_generation(self):
        lock = self.plan["notes"] / "locks" / self.job["key"]
        lock.mkdir(parents=True)
        with self.assertRaises(RuntimeError):
            self.attempt()
        self.assertEqual(self.calls, 0)

    def test_single_sample_call_sequence_and_fresh_args(self):
        order = []
        cuda = SimpleNamespace(
            synchronize=lambda *args: order.append("sync"),
            reset_peak_memory_stats=lambda *args: order.append("reset"),
            max_memory_allocated=lambda *args: 100,
            max_memory_reserved=lambda *args: 200,
        )
        torch = SimpleNamespace(cuda=cuda, device=lambda x: x)
        def generate(args, settings):
            order.append(("generate", args.prompt, args.seed))
            args.seed = 999
            return "latent"
        wan = SimpleNamespace(
            check_inputs=lambda args: (480, 832, 81),
            get_generation_settings=lambda args: SimpleNamespace(
                cfg=SimpleNamespace(sample_neg_prompt="default negative")),
            generate=generate,
            clean_memory_on_device=lambda *args: order.append("clean"),
            save_output=lambda *args: order.append("save"),
        )
        args = SimpleNamespace(prompt="placeholder", seed=0, negative_prompt=None)
        with patch.object(gen.gc, "collect", lambda: order.append("gc")):
            result = gen.generate_sample(wan, torch, args, self.job, self.root)
        self.assertEqual(order[:7], [
            "sync", "reset", ("generate", self.job["prompt"], 42), "gc", "clean", "save", "sync"])
        self.assertEqual(args.seed, 0)
        self.assertEqual(result["effective_args"]["seed"], 42)
        self.assertEqual(result["effective_negative_prompt"], "default negative")

    def test_summary_flags_corruption(self):
        self.attempt()
        (self.plan["outputs"] / f"{self.job['key']}.mp4").unlink()
        rows = gen.summarize(self.plan)
        self.assertEqual(rows[0]["status"], "invalid_output")


if __name__ == "__main__":
    unittest.main()
