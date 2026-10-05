"""evaluate_linelevel.py 行合并函数单元测试。"""

import unittest

import numpy as np

from evaluate_linelevel import merge_to_lines


def _q(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


class TestMergeToLines(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(merge_to_lines([]), [])

    def test_two_words_same_line(self):
        words = [_q(0, 0, 50, 20), _q(60, 0, 110, 20)]
        self.assertEqual(len(merge_to_lines(words)), 1)

    def test_two_separate_lines(self):
        words = [_q(0, 0, 50, 20), _q(0, 60, 50, 80)]
        self.assertEqual(len(merge_to_lines(words)), 2)

    def test_merged_box_tight(self):
        words = [_q(0, 0, 50, 20), _q(60, 0, 110, 20)]
        line = merge_to_lines(words)[0]
        self.assertAlmostEqual(float(line[0][0]), 0.0)   # xmin
        self.assertAlmostEqual(float(line[1][0]), 110.0)  # xmax
        self.assertAlmostEqual(float(line[0][1]), 0.0)   # ymin
        self.assertAlmostEqual(float(line[2][1]), 20.0)  # ymax

    def test_tolerates_small_stagger(self):
        # 同行轻微错落（y 中心差 < 0.5 字高）仍合并
        words = [_q(0, 0, 50, 20), _q(60, 3, 110, 23)]
        self.assertEqual(len(merge_to_lines(words)), 1)


if __name__ == "__main__":
    unittest.main()
