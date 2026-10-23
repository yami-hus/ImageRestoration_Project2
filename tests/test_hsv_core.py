import unittest
import cv2
import numpy as np

import hsv_core_mock as core   # TV1 xong thì đổi thành: import hsv_core as core
from tools.synthetic import make_synthetic


class TestSegmentHSV(unittest.TestCase):
    def setUp(self):
        self.img, self.masks = make_synthetic()

    def test_alg01_green_only(self):  # ALG-01
        mask, _ = core.segment_hsv(self.img, (50, 100, 100), (70, 255, 255))
        self.assertTrue(np.array_equal(mask, self.masks["green"]))

    def test_alg02_red_wraparound(self):  # ALG-02
        mask, _ = core.segment_hsv(self.img, (170, 100, 100), (10, 255, 255))
        expected = cv2.bitwise_or(self.masks["red_low"], self.masks["red_high"])
        self.assertTrue(np.array_equal(mask, expected))

    def test_alg03_full_range(self):  # ALG-03
        mask, _ = core.segment_hsv(self.img, (0, 0, 0), (179, 255, 255))
        self.assertTrue(np.all(mask == 255))

    def test_alg04_empty_range(self):  # ALG-04
        mask, _ = core.segment_hsv(self.img, (0, 0, 0), (0, 0, 0))
        self.assertEqual(int(mask.sum()), 0)

    def test_alg05_invalid_params(self):  # ALG-05, mock chưa ném lỗi
        with self.assertRaises(ValueError):
            core.segment_hsv(self.img, (0, 200, 0), (179, 100, 255))   # S min > S max
        with self.assertRaises(ValueError):
            core.segment_hsv(self.img, (0, 0, 0), (200, 255, 255))     # H ngoài thang

    def test_alg06_output_format(self):  # ALG-06
        mask, result = core.segment_hsv(self.img, (50, 100, 100), (70, 255, 255))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, self.img.shape[:2])
        self.assertTrue(set(np.unique(mask)) <= {0, 255})
        self.assertEqual(result.shape, self.img.shape)

    def test_alg07_bgr_vs_rgb(self):  # ALG-07
        red = ((170, 100, 100), (10, 255, 255))
        red_truth = cv2.bitwise_or(self.masks["red_low"], self.masks["red_high"])

        mask_bgr, _ = core.segment_hsv(self.img, *red)
        self.assertTrue(np.array_equal(mask_bgr, red_truth))  # đúng kênh: bắt đúng 2 ô đỏ

        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # nhầm kênh có chủ đích
        mask_rgb, _ = core.segment_hsv(rgb, *red)
        overlap = cv2.bitwise_and(mask_rgb, red_truth)
        self.assertEqual(int(overlap.sum()), 0)  # sai kênh: hai ô đỏ thật không còn trong mask


@unittest.skipUnless(hasattr(core, "iou") and hasattr(core, "dice"),
                     "chờ TV1 cài iou/dice")
class TestMetrics(unittest.TestCase):  # MET-01
    def setUp(self):
        self.a = np.zeros((20, 20), np.uint8)
        self.a[0:10, 0:10] = 255                  # 100 px
        self.b = np.zeros((20, 20), np.uint8)
        self.b[0:10, 5:15] = 255                  # 100 px, giao 50 px

    def test_identical(self):
        self.assertAlmostEqual(core.iou(self.a, self.a), 1.0)
        self.assertAlmostEqual(core.dice(self.a, self.a), 1.0)

    def test_disjoint(self):
        c = np.zeros((20, 20), np.uint8)
        c[15:20, 15:20] = 255
        self.assertAlmostEqual(core.iou(self.a, c), 0.0)
        self.assertAlmostEqual(core.dice(self.a, c), 0.0)

    def test_partial_overlap(self):
        self.assertAlmostEqual(core.iou(self.a, self.b), 50 / 150)
        self.assertAlmostEqual(core.dice(self.a, self.b), 100 / 200)


if __name__ == "__main__":
    unittest.main()