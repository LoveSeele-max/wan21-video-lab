"""Small behavioral tests for temporal scope and metric limitations."""
import unittest
import numpy as np

from make_probes import transform
from metrics import classify, ssim_scores


class MetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.frames = rng.integers(0, 256, (81, 48, 64, 3), dtype=np.uint8)

    def test_static_is_identical_and_ssim_one(self):
        static = transform(self.frames, "static", (8, 8, 40, 40))
        self.assertTrue(np.all(static == self.frames[0]))
        self.assertEqual(ssim_scores(static), [1.0] * 80)

    def test_local_changes_preserve_everything_outside_scope(self):
        for kind in ("flicker", "blur", "local_warp"):
            result = transform(self.frames, kind, (8, 8, 40, 40))
            self.assertTrue(np.array_equal(result[:30], self.frames[:30]))
            self.assertTrue(np.array_equal(result[41:], self.frames[41:]))
            self.assertTrue(all(not np.array_equal(result[i], self.frames[i])
                                for i in range(30, 41)))
            if kind == "local_warp":
                self.assertTrue(np.array_equal(result[:, :8], self.frames[:, :8]))
                self.assertTrue(np.array_equal(result[:, :, 40:], self.frames[:, :, 40:]))

    def test_range_boundaries_and_zero_width(self):
        self.assertEqual(classify(1.01, 0, 1), "能区分")
        self.assertEqual(classify(-0.01, 0, 1), "能区分")
        self.assertEqual(classify(0.02, 0, 1), "部分区分")
        self.assertEqual(classify(0.5, 0, 1), "不能区分")
        self.assertEqual(classify(1, 1, 1), "不能区分")


if __name__ == "__main__":
    unittest.main()
