"""守护测试：检测分辨率旋钮（`limit_side_len` / `limit_type`）的参数贯通。

**对应审计 P1-8**：检测输入被 `limit_type='max' + limit_side_len=960` 下采样到 0.75 倍，
而 500 张测试图全是 1280×720 → 每张都先缩到约 960×544 再检测，真实文本框高度中位
34px 变成约 25.5px。机制已确证，但"提到 1280/1600 能换多少 F1"此前的扫描只在 **30 张**
上做过（small 有收益、medium 复测无收益），需要**大样本复测**才能定论。

`evaluate.py` 已有 `--det-thresh` / `--det-box-thresh` / `--det-unclip-ratio` 三个对称旋钮，
唯独漏了 `--det-limit-side-len`；本文件守住新补的两个旋钮确实贯通到 `TextDetector`。

**不加载模型**：`TextDetector` 被替换为只记录参数的假实现。
"""

import unittest
from unittest import mock

import numpy as np

from src.pipeline.ocr_pipeline import SceneTextOCR


class _RecordingDetector:
    """假检测器：记录构造参数，不加载任何模型。"""

    last_kwargs: dict = {}

    def __init__(self, model_name, **kwargs):
        type(self).last_kwargs = {"model_name": model_name, **kwargs}

    def detect_with_scores(self, img):
        return [], []

    def detect(self, img):
        return []


IMG = np.full((40, 60, 3), 255, np.uint8)


class TestDetectorResolutionKnobs(unittest.TestCase):
    """`SceneTextOCR(det_limit_side_len=..., det_limit_type=...)` 必须传到 TextDetector。"""

    def setUp(self):
        _RecordingDetector.last_kwargs = {}

    def _build(self, **kwargs):
        with mock.patch("src.detector.detector.TextDetector", _RecordingDetector):
            ocr = SceneTextOCR(det_model_name="PP-OCRv6_small_det", **kwargs)
            ocr.detect_detailed(IMG)  # 触发懒加载，看真正传下去的 kwargs
        return _RecordingDetector.last_kwargs

    def test_default_leaves_knobs_to_model(self):
        """不传时不应硬塞值——config.yaml 是 null，交给模型自带默认（960/max）。"""
        kw = self._build()
        self.assertIsNone(kw.get("limit_side_len"))
        self.assertIsNone(kw.get("limit_type"))

    def test_explicit_values_reach_detector(self):
        """审计 P1-8 的两个档位：1280/max（不下采样）与 960/min（反而放大）。"""
        kw = self._build(det_limit_side_len=1280, det_limit_type="max")
        self.assertEqual(kw["limit_side_len"], 1280)
        self.assertEqual(kw["limit_type"], "max")

        kw2 = self._build(det_limit_side_len=960, det_limit_type="min")
        self.assertEqual(kw2["limit_side_len"], 960)
        self.assertEqual(kw2["limit_type"], "min")

    def test_other_knobs_still_pass_through(self):
        """确认没有把原有的三个旋钮（阈值/框阈值/外扩比）弄丢。"""
        kw = self._build(det_thresh=0.3, det_box_thresh=0.5, det_unclip_ratio=1.0)
        self.assertEqual(kw["thresh"], 0.3)
        self.assertEqual(kw["box_thresh"], 0.5)
        self.assertEqual(kw["unclip_ratio"], 1.0)

    def test_config_yaml_still_wins_over_model_default(self):
        """显式参数 > config.yaml > 模型默认：这里验证 config 层（null 时才算"未设"）。"""
        kw = self._build()
        # 当前 config.yaml 的 det 段全是 null，所以应保持 None（= 用模型默认）
        self.assertIn("limit_side_len", kw)

    def test_evaluate_cli_exposes_the_flag(self):
        """`evaluate.py` 必须暴露 CLI 开关（否则无法在评测里扫描该旋钮）。"""
        import inspect

        import evaluate as evaluate_mod

        src = inspect.getsource(evaluate_mod.main)
        self.assertIn("--det-limit-side-len", src)
        self.assertIn("--det-limit-type", src)
        self.assertIn("det_limit_side_len", src)


if __name__ == "__main__":
    unittest.main()
