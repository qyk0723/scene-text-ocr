"""do-not-care 协议（ICDAR2015 deteval 口径）的单元测试，不加载模型。

背景：检测 GT 中约 60% 的框转写为 ``###``（do-not-care）。官方协议不把它们计入
召回分母，命中它们的预测也既不算 TP 也不算 FP。旧实现丢弃了转写字段，因此无法
表达这一口径——这些测试守护新增的实现。
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluator import (
    match_boxes,
    match_boxes_dnc,
    parse_det_gt,
    parse_det_gt_flagged,
    prf_from_counts,
)


def _q(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


class TestParseDetGtFlagged(unittest.TestCase):
    def _write(self, content):
        f = tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return Path(f.name)

    def test_marks_dont_care_rows(self):
        p = self._write(
            "0,0,10,0,10,10,0,10,real text\n"
            "20,0,30,0,30,10,20,10,###\n"
            "40,0,50,0,50,10,40,10,###\n"
        )
        try:
            quads, flags = parse_det_gt_flagged(p)
            self.assertEqual(len(quads), 3)
            self.assertEqual(list(flags), [False, True, True])
            # 与旧接口的几何解析必须完全一致
            self.assertEqual(len(parse_det_gt(p)), 3)
            np.testing.assert_allclose(quads[0], parse_det_gt(p)[0])
        finally:
            os.unlink(p)

    def test_missing_transcription_is_not_dont_care(self):
        p = self._write("0,0,10,0,10,10,0,10\n")
        try:
            quads, flags = parse_det_gt_flagged(p)
            self.assertEqual(len(quads), 1)
            self.assertEqual(list(flags), [False])
        finally:
            os.unlink(p)


class TestMatchBoxesDnc(unittest.TestCase):
    def test_ignores_dont_care_and_overlapping_preds(self):
        gt = [_q(0, 0, 10, 10), _q(20, 0, 30, 10)]      # A=care, B=dont-care
        dnc = np.array([False, True])
        preds = [
            _q(0, 0, 10, 10),      # 命中 A -> TP
            _q(20, 0, 30, 10),     # 命中 B -> ignored
            _q(21, 0, 29, 10),     # 未匹配但压在 B 上 -> ignored
            _q(100, 100, 110, 110)  # 无关 -> FP
        ]
        r = match_boxes_dnc(gt, dnc, preds, 0.5)
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["matched_dnc"], 1)
        self.assertEqual(r["ignored"], 2)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(r["fn"], 0)

    def test_fn_counts_only_care_boxes(self):
        gt = [_q(0, 0, 10, 10), _q(20, 0, 30, 10), _q(200, 0, 210, 10)]
        dnc = np.array([False, True, False])            # 第三个是 care 但无预测
        preds = [_q(0, 0, 10, 10)]
        r = match_boxes_dnc(gt, dnc, preds, 0.5)
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fn"], 1)                    # 只数 care 的漏检，不含 dont-care
        self.assertEqual(r["ignored"], 0)

    def test_legacy_differs_on_dont_care(self):
        """同一组输入，legacy 把命中 dont-care 也算匹配，deteval 不算。"""
        gt = [_q(0, 0, 10, 10), _q(20, 0, 30, 10)]
        dnc = np.array([False, True])
        preds = [_q(0, 0, 10, 10), _q(20, 0, 30, 10)]
        self.assertEqual(match_boxes(gt, preds, 0.5), 2)          # legacy：2
        self.assertEqual(match_boxes_dnc(gt, dnc, preds, 0.5)["tp"], 1)  # deteval：1

    def test_all_care_reduces_to_plain_matching(self):
        gt = [_q(0, 0, 10, 10), _q(50, 0, 60, 10)]
        dnc = np.array([False, False])
        preds = [_q(0, 0, 10, 10), _q(50, 0, 60, 10), _q(200, 0, 210, 10)]
        r = match_boxes_dnc(gt, dnc, preds, 0.5)
        self.assertEqual((r["tp"], r["fp"], r["ignored"], r["fn"]), (2, 1, 0, 0))


class TestPrfFromCounts(unittest.TestCase):
    def test_basic(self):
        m = prf_from_counts(1, 1, 1)
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["f1"], 0.5)

    def test_all_zero(self):
        m = prf_from_counts(0, 0, 0)
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (0.0, 0.0, 0.0))

    def test_matches_legacy_formula(self):
        """legacy 口径必须能由 (matched, pred-matched, gt-matched) 复现原公式。"""
        matched, pred, gt = 1355, 2208, 5230
        m = prf_from_counts(matched, pred - matched, gt - matched)
        self.assertAlmostEqual(m["precision"], matched / pred, places=12)
        self.assertAlmostEqual(m["recall"], matched / gt, places=12)
        p, r = matched / pred, matched / gt
        self.assertAlmostEqual(m["f1"], 2 * p * r / (p + r), places=12)


if __name__ == "__main__":
    unittest.main()
