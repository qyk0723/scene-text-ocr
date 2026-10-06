"""健壮性守护测试：覆盖审计中发现的三个真实缺陷，不加载模型。

每个测试都对应一个已发生过的实际缺陷：
1. crop_box 对退化框静默返回整张原图（识别器随后输出垃圾文本）；
2. render_report 把 IoU 阈值写死为 0.5，--iou 0.3 时报告正文说谎；
3. make_figures 用 ImageEnhancer() 默认（全关）生成"预处理后"面板，与原图逐像素相同；
4. cv2.imread 在含非 ASCII 的路径上返回 None（项目位于 毕业设计 目录下）。
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.evaluator.runner import render_report
from src.pipeline.geometry import crop_box
from src.pipeline.ocr_pipeline import imread_unicode
from src.preprocess.enhancer import ImageEnhancer


def _q(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


class TestCropBoxDegenerate(unittest.TestCase):
    """退化框必须返回空数组，而不是静默返回整张原图。"""

    def setUp(self):
        self.img = np.full((60, 120, 3), 255, np.uint8)

    def _assert_empty(self, box):
        out = crop_box(self.img, box)
        self.assertEqual(out.size, 0, "退化框应返回空数组，而不是原图尺寸的图")

    def test_identical_points(self):
        self._assert_empty(np.array([[10, 10]] * 4, np.float32))

    def test_zero_width(self):
        self._assert_empty(np.array([[10, 10], [10, 10], [10, 40], [10, 40]], np.float32))

    def test_zero_height(self):
        self._assert_empty(np.array([[10, 10], [40, 10], [40, 10], [10, 10]], np.float32))

    def test_normal_box_unaffected(self):
        out = crop_box(self.img, _q(20, 10, 80, 30))
        self.assertEqual(out.shape[0], 20)  # 高 = 字高
        self.assertEqual(out.shape[1], 60)  # 宽 = 字宽


class TestRenderReportIou(unittest.TestCase):
    """报告正文必须写明**实际使用**的 IoU 阈值。"""

    @staticmethod
    def _det_result():
        return [{
            "task": "检测", "images": 1, "gt_boxes": 10, "pred_boxes": 5,
            "matched": 3, "precision": 0.6, "recall": 0.3, "f1": 0.4, "avg_time": 1.0,
        }]

    def test_reports_actual_threshold(self):
        text = render_report(self._det_result(), "m", 0.3)
        self.assertIn("IoU 阈值 0.3", text)
        self.assertNotIn("IoU 阈值 0.5", text)

    def test_default_threshold_is_half(self):
        self.assertIn("IoU 阈值 0.5", render_report(self._det_result(), "m"))


class TestEnhancerDefaults(unittest.TestCase):
    """ImageEnhancer() 默认无操作——这正是 preprocess_compare 出错的根因。"""

    def setUp(self):
        self.img = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype(np.uint8)

    def test_default_is_noop(self):
        self.assertTrue(np.array_equal(self.img, ImageEnhancer().process(self.img)))

    def test_explicit_operators_do_change_image(self):
        out = ImageEnhancer(denoise=True, contrast=True, sharpen=True).process(self.img)
        self.assertFalse(
            np.array_equal(self.img, out),
            "显式开启算子后图像应发生变化，否则 preprocess_compare 又会生成两张相同的面板",
        )


class TestImreadUnicode(unittest.TestCase):
    """含非 ASCII 字符的路径必须能读图。

    注：不断言 cv2.imread 一定失败（那是环境细节）；只断言我们的封装能读。
    """

    def test_reads_image_under_non_ascii_path(self):
        tmp = tempfile.mkdtemp(prefix="stocr_test_")
        try:
            d = Path(tmp) / "毕业设计"
            d.mkdir()
            p = d / "样本.png"
            ok, buf = cv2.imencode(".png", np.full((8, 8, 3), 200, np.uint8))
            self.assertTrue(ok)
            with open(p, "wb") as f:  # 用 Python IO，避免 tofile 的编码问题
                f.write(buf.tobytes())

            img = imread_unicode(p)
            self.assertIsNotNone(img, "imread_unicode 应能读含中文的路径")
            self.assertEqual(img.shape, (8, 8, 3))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_file_returns_none(self):
        self.assertIsNone(imread_unicode(Path(tempfile.gettempdir()) / "不存在的图_xyz.jpg"))


if __name__ == "__main__":
    unittest.main()
