"""evaluate.py 纯函数单元测试（不加载模型）。"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate import levenshtein, match_boxes, parse_det_gt, quad_iou


def _q(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


class TestLevenshtein(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(levenshtein("abc", "abc"), 0)

    def test_substitute(self):
        self.assertEqual(levenshtein("abc", "axc"), 1)

    def test_insert(self):
        self.assertEqual(levenshtein("abc", "abxc"), 1)

    def test_delete(self):
        self.assertEqual(levenshtein("abc", "ab"), 1)

    def test_empty(self):
        self.assertEqual(levenshtein("", "ab"), 2)


class TestParseDetGt(unittest.TestCase):
    def test_parse_one_quad(self):
        content = "933,255,954,255,956,277,936,277,###\n"
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        try:
            quads = parse_det_gt(Path(path))
            self.assertEqual(len(quads), 1)
            np.testing.assert_allclose(
                quads[0], [[933, 255], [954, 255], [956, 277], [936, 277]]
            )
        finally:
            os.unlink(path)

    def test_skip_malformed(self):
        content = "not,numbers,here\n\n"
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        try:
            self.assertEqual(parse_det_gt(Path(path)), [])
        finally:
            os.unlink(path)


class TestQuadIoU(unittest.TestCase):
    def test_identical(self):
        q = _q(0, 0, 10, 10)
        self.assertAlmostEqual(quad_iou(q, q), 1.0)

    def test_disjoint(self):
        a = _q(0, 0, 10, 10)
        b = _q(50, 50, 60, 60)
        self.assertEqual(quad_iou(a, b), 0.0)

    def test_half_overlap(self):
        # 交集 5x10=50，并集 10x10 + 10x10 - 50 = 150 -> 1/3
        a = _q(0, 0, 10, 10)
        b = _q(5, 0, 15, 10)
        self.assertAlmostEqual(quad_iou(a, b), 50 / 150, places=4)


class TestMatchBoxes(unittest.TestCase):
    def test_match_one(self):
        gt = [_q(0, 0, 10, 10)]
        pred = [_q(1, 1, 9, 9)]
        self.assertEqual(match_boxes(gt, pred, 0.5), 1)

    def test_no_match_low_iou(self):
        gt = [_q(0, 0, 10, 10)]
        pred = [_q(50, 50, 60, 60)]
        self.assertEqual(match_boxes(gt, pred, 0.5), 0)

    def test_greedy_one_to_one(self):
        gt = [_q(0, 0, 10, 10), _q(100, 0, 110, 10)]
        pred = [_q(0, 0, 10, 10), _q(0, 0, 10, 10)]
        self.assertEqual(match_boxes(gt, pred, 0.5), 1)


if __name__ == "__main__":
    unittest.main()
