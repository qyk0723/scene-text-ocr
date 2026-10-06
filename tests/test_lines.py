"""行聚类与标注解析的单元测试（端到端评测依赖这两块），不加载模型。"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluator import (
    cluster_line_groups,
    merge_to_lines,
    parse_det_gt_flagged,
    parse_det_gt_full,
)


def _q(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


class TestClusterLineGroups(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(cluster_line_groups([]), [])

    def test_two_words_same_line_share_a_group(self):
        groups = cluster_line_groups([_q(0, 0, 50, 20), _q(60, 0, 110, 20)])
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0]), [0, 1])

    def test_two_lines_are_separate(self):
        groups = cluster_line_groups([_q(0, 0, 50, 20), _q(0, 60, 50, 80)])
        self.assertEqual(len(groups), 2)

    def test_groups_are_consistent_with_merge_to_lines(self):
        """分组数必须与 merge_to_lines 的行数一致——端到端评测靠这个对应关系。"""
        rng = np.random.default_rng(0)
        for _ in range(30):
            n = int(rng.integers(1, 12))
            quads = []
            for _ in range(n):
                x = float(rng.integers(0, 400))
                y = float(rng.integers(0, 200))
                w = float(rng.integers(20, 80))
                h = float(rng.integers(10, 30))
                quads.append(_q(x, y, x + w, y + h))
            self.assertEqual(len(cluster_line_groups(quads)), len(merge_to_lines(quads)))

    def test_group_indices_cover_every_input_exactly_once(self):
        quads = [_q(0, 0, 50, 20), _q(60, 3, 110, 23), _q(0, 60, 50, 80)]
        flat = [i for g in cluster_line_groups(quads) for i in g]
        self.assertEqual(sorted(flat), list(range(len(quads))))


class TestParseDetGtFull(unittest.TestCase):
    def _write(self, content):
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        f.write(content)
        f.close()
        return Path(f.name)

    def test_returns_texts_and_flags(self):
        p = self._write(
            "0,0,10,0,10,10,0,10,Hello\n"
            "20,0,30,0,30,10,20,10,###\n"
            "40,0,50,0,50,10,40,10,World\n"
        )
        try:
            quads, flags, texts = parse_det_gt_full(p)
            self.assertEqual(len(quads), 3)
            self.assertEqual(list(flags), [False, True, False])
            self.assertEqual(texts, ["Hello", "###", "World"])
            # 与只返回标记的接口必须一致
            q2, f2 = parse_det_gt_flagged(p)
            self.assertEqual(len(q2), 3)
            self.assertEqual(list(f2), list(flags))
        finally:
            os.unlink(p)

    def test_missing_transcription(self):
        p = self._write("0,0,10,0,10,10,0,10\n")
        try:
            quads, flags, texts = parse_det_gt_full(p)
            self.assertEqual(list(flags), [False])   # 无转写字段不算 do-not-care
            self.assertEqual(texts, [""])
        finally:
            os.unlink(p)


if __name__ == "__main__":
    unittest.main()
