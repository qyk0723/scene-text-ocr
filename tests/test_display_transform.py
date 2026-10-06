"""守护测试：预处理改变尺寸时的**坐标系映射**（审计第 13 轮的界面缺陷）。

**对应缺陷**：`app.py` 勾选「小字放大」且上传图长边 <800 时，`ImageEnhancer` 把图放大到 800，
检测跑在放大图上（框属于放大图的坐标系），却直接画在原图上 → 框整体错位、大半跑到画布外
（实测 299×400 → proc 598×800，124/156 个坐标点越界）。

修复：`ImageEnhancer.process_with_info()` 返回 `ProcessInfo`（按实际形状测出比值），
调用方据此把框映射回原图。

**不加载模型**：全部用 `np.zeros` 合成图，秒级完成。
"""

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.preprocess.enhancer import ImageEnhancer, ProcessInfo
from src.visualize.draw import (
    _load_font,
    annotate_results,
    draw_results,
    map_boxes_to_original,
)


def _q(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _max_dev(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """两个框对应顶点的最大像素偏差。"""
    return float(np.abs(np.asarray(box_a, np.float64) - np.asarray(box_b, np.float64)).max())


def _in_bounds(box: np.ndarray, h: int, w: int) -> bool:
    return bool((box[:, 0] >= 0).all() and (box[:, 0] <= w).all()
                and (box[:, 1] >= 0).all() and (box[:, 1] <= h).all())


class TestProcessInfoReportsRealGeometry(unittest.TestCase):
    """ProcessInfo 报的形状/系数必须与**实际**输出图一致。"""

    def test_upscale_reports_target_shape_and_scale(self):
        """审计实测场景：299×400 的图，upscale 目标长边 800 → proc 598×800。"""
        img = np.zeros((400, 299, 3), np.uint8)  # (h, w)
        enhancer = ImageEnhancer(upscale=True, upscale_min_long_side=800)
        proc, info = enhancer.process_with_info(img)

        self.assertEqual(proc.shape[:2], (800, 598))
        self.assertTrue(info.resized)
        self.assertEqual(info.image_shape, (800, 598))
        self.assertEqual(info.orig_shape, (400, 299))
        self.assertAlmostEqual(info.scale_x, 299 / 598, places=12)
        self.assertAlmostEqual(info.scale_y, 400 / 800, places=12)
        # 系数必须与"实际形状之比"吻合，而不是从算子参数推算出来的巧合
        self.assertAlmostEqual(info.scale_x, img.shape[1] / proc.shape[1], places=9)
        self.assertAlmostEqual(info.scale_y, img.shape[0] / proc.shape[0], places=9)

    def test_no_resize_when_long_side_already_large(self):
        """长边 ≥800 时算子不动图，info 应报"未改变尺寸"且系数为 1。"""
        img = np.zeros((720, 1280, 3), np.uint8)
        proc, info = ImageEnhancer(upscale=True, upscale_min_long_side=800).process_with_info(img)

        self.assertEqual(proc.shape[:2], img.shape[:2])
        self.assertFalse(info.resized)
        self.assertEqual((info.scale_x, info.scale_y), (1.0, 1.0))

    def test_default_enhancer_is_identity(self):
        """默认全关：图不变、未改变尺寸（与 test_robustness 的 no-op 断言一致）。"""
        img = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype(np.uint8)
        proc, info = ImageEnhancer().process_with_info(img)

        self.assertTrue(np.array_equal(img, proc))
        self.assertFalse(info.resized)

    def test_scale_matches_each_axis_independently(self):
        """两个改尺寸算子叠加（先缩到 300，再放大到 600）也要逐轴正确。

        原图宽 1001 是奇数：缩放取整后 ``scale_x = 1001/600``、``scale_y = 300/180``
        不再相等，混用两轴系数的实现会被抓出来。
        """
        img = np.zeros((300, 1001, 3), np.uint8)
        enhancer = ImageEnhancer(resize=True, resize_long_side=300,
                                 upscale=True, upscale_min_long_side=600)
        proc, info = enhancer.process_with_info(img)

        self.assertEqual(proc.shape[:2], (180, 600))
        self.assertTrue(info.resized)
        self.assertAlmostEqual(info.scale_x, 1001 / 600, places=9)
        self.assertAlmostEqual(info.scale_y, 300 / 180, places=9)
        self.assertNotAlmostEqual(info.scale_x, info.scale_y, places=3)


class TestRoundTripMapping(unittest.TestCase):
    """映射必须与**真实缩放**互逆——方向或系数写错，这里立刻红。"""

    def test_round_trip_within_one_pixel(self):
        img = np.zeros((400, 299, 3), np.uint8)
        proc, info = ImageEnhancer(upscale=True, upscale_min_long_side=800).process_with_info(img)

        # 原图上的一块区域 -> 放大图上的对应区域（正变换）
        orig_box = _q(40, 60, 150, 90)
        proc_box = orig_box * np.array([proc.shape[1] / img.shape[1],
                                        proc.shape[0] / img.shape[0]], np.float32)

        back = info.map_box(proc_box)
        self.assertLess(_max_dev(back, orig_box), 1.0,
                        "映射回原图后应与原始位置一致（误差 < 1px）")

    def test_map_point_matches_map_box(self):
        img = np.zeros((299, 400, 3), np.uint8)  # 高 299、宽 400，长边 400
        proc, info = ImageEnhancer(upscale=True, upscale_min_long_side=800).process_with_info(img)

        self.assertAlmostEqual(info.map_point(100.0, 50.0)[0],
                               float(info.map_box(np.array([[100.0, 50.0]], np.float32))[0, 0]))
        self.assertAlmostEqual(info.map_point(100.0, 50.0)[1],
                               float(info.map_box(np.array([[100.0, 50.0]], np.float32))[0, 1]))

    def test_identity_mapping_does_not_alter_boxes(self):
        img = np.zeros((720, 1280, 3), np.uint8)
        _, info = ImageEnhancer(upscale=True, upscale_min_long_side=800).process_with_info(img)
        box = _q(10, 20, 30, 40)

        self.assertTrue(np.array_equal(info.map_box(box), box))

    def test_mapping_does_not_mutate_input_box(self):
        """map_box 是只读的：调用方可能还要用原始（预处理图坐标系）框。"""
        info = ProcessInfo.from_shapes((400, 299), (800, 598))
        box = _q(100, 100, 200, 200)
        before = box.copy()

        info.map_box(box)
        self.assertTrue(np.array_equal(box, before))


class TestOutOfCanvasRegression(unittest.TestCase):
    """回归：修复前画到画布外的框，映射后必须全部落在原图内。"""

    def setUp(self):
        self.img = np.zeros((400, 299, 3), np.uint8)
        self.proc, self.info = ImageEnhancer(
            upscale=True, upscale_min_long_side=800
        ).process_with_info(self.img)

    def test_audit_measured_boxes_are_now_in_bounds(self):
        """审计实测的坐标范围 x:[14,528] y:[54,689]（proc 598×800 坐标系）。"""
        boxes = [
            _q(14, 54, 528, 120),
            _q(100, 600, 400, 689),
            _q(0, 0, 598, 800),  # 贴满整张 proc 的极端框
        ]
        mapped = map_boxes_to_original(boxes, self.info)

        for box in mapped:
            self.assertTrue(
                _in_bounds(box, *self.img.shape[:2]),
                f"映射后仍有坐标越界（原图 400×299）: {box.tolist()}",
            )

    def test_unmapped_boxes_really_were_out_of_bounds(self):
        """反向确认：不映射时确实越界——否则上面的回归测试是空的。"""
        box = _q(14, 54, 528, 689)
        self.assertFalse(_in_bounds(box, *self.img.shape[:2]))

    def test_clipping_handles_oversized_boxes(self):
        """防御：预处理图大于原图时，超出边界的框被裁剪回原图内。"""
        info = ProcessInfo.from_shapes((100, 100), (200, 200))
        mapped = info.map_box(_q(0, 0, 400, 400))  # proc 系的 400 -> 原图系 200

        self.assertTrue(_in_bounds(mapped, 100, 100))


class TestCompatibilityAndContract(unittest.TestCase):
    """向后兼容 + 绘制契约（防止有人再把 proc 坐标系的框直接画到原图上）。"""

    IMG = np.full((400, 299, 3), 255, np.uint8)

    def test_process_still_returns_single_ndarray(self):
        """process() 的签名与返回值不能变——evaluate_ablation / main / make_figures 都在用。"""
        out = ImageEnhancer(upscale=True, upscale_min_long_side=800).process(self.IMG)

        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape[:2], (800, 598))

    def test_process_matches_process_with_info_pixels(self):
        enhancer = ImageEnhancer(denoise=True, upscale=True, upscale_min_long_side=800)
        proc_a = enhancer.process(self.IMG)
        proc_b, _ = enhancer.process_with_info(self.IMG)

        self.assertTrue(np.array_equal(proc_a, proc_b),
                        "process() 与 process_with_info() 必须是同一条算子链")

    def test_draw_results_keeps_canvas_size(self):
        out = draw_results(self.IMG, [_q(20, 30, 120, 60)], ["测试 text"])

        self.assertEqual(out.shape, self.IMG.shape)

    def test_annotated_output_stays_on_original_canvas(self):
        """把 proc 系的框交给 draw_results 只会画到画布外；映射后必须完全落在原图内。"""
        proc, info = ImageEnhancer(
            upscale=True, upscale_min_long_side=800
        ).process_with_info(self.IMG)
        boxes = [_q(120, 200, 400, 300)]  # proc 598×800 坐标系，远超原图 299 宽
        mapped = map_boxes_to_original(boxes, info)

        self.assertEqual(proc.shape[:2], (800, 598))
        for box in mapped:
            self.assertTrue(_in_bounds(box, *self.IMG.shape[:2]))
        self.assertEqual(draw_results(self.IMG, mapped, ["x"]).shape, self.IMG.shape)

    def test_font_is_cached(self):
        """_load_font 必须缓存：原先每张图每行标注都从磁盘重载字体。"""
        first = _load_font(16)
        hits_before = _load_font.cache_info().hits
        second = _load_font(16)

        self.assertIs(first, second, "同一字号应复用同一个字体对象")
        self.assertGreater(_load_font.cache_info().hits, hits_before,
                           "第二次调用必须命中 lru_cache")


class TestDependencyBoundaries(unittest.TestCase):
    """依赖边界（审计 P2-21）：绘制模块**不碰 GUI**，`app` **不在导入期加载模型**。

    原先 `draw_results` 只住在 `app.py`，于是 `make_figures.py` 仅仅为了画标注就要
    `import app` → 拉起整个 Gradio GUI 依赖（实测 import gradio ≈ 7.7s）；
    且 `app` 在模块级构造 `SceneTextOCR()`，导入即加载模型。

    两条都用**独立子进程**验证：本进程里 `app` 可能已被别的测试导入过，看不出来。
    合并成一次子进程（每次 spawn 都要付一遍解释器+导入成本）。
    """

    def test_import_boundaries_in_fresh_interpreter(self):
        """独立子进程验证：绘制模块不引入 gradio/app，且 `import app` 不加载模型。

        只在子进程里 import `app`（**不 import gradio**）：gradio 的导入实测约
        7.7s，把它拉进测试环境会让这个守护测试本身变得昂贵。
        """
        import json
        import subprocess
        import sys

        code = (
            "import json, sys;"
            "import src.visualize.draw;"
            "d = {'gradio_after_draw': 'gradio' in sys.modules,"
            "     'app_after_draw': 'app' in sys.modules};"
            "import app;"
            "d['app_ocr_is_none'] = app._OCR is None;"
            "print(json.dumps(d))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=600,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        result = json.loads(out.stdout.strip().splitlines()[-1])

        self.assertFalse(result["gradio_after_draw"], "绘制模块不应引入 gradio")
        self.assertFalse(result["app_after_draw"], "绘制模块不应引入 app")
        self.assertTrue(result["app_ocr_is_none"],
                        "import app 时不应构造 SceneTextOCR（否则等于加载模型）")

    def test_make_figures_no_longer_imports_app(self):
        """`make_figures.py` 不得再依赖 `app`（否则画图仍会拉起 GUI）。"""
        src = (Path(__file__).resolve().parent.parent / "make_figures.py").read_text(
            encoding="utf-8")
        self.assertNotIn("from app import", src)
        self.assertNotIn("import app\n", src)
        self.assertIn("from src.visualize.draw import", src)

    def test_app_reexports_drawing_helpers(self):
        """`app` 仍再导出绘制/映射函数，避免 `from app import ...` 的既有用法失效。"""
        import app as app_mod

        self.assertIs(app_mod.draw_results, draw_results)
        self.assertIs(app_mod._load_font, _load_font)
        self.assertIs(app_mod.annotate_results, annotate_results)
        self.assertIs(app_mod.map_boxes_to_original, map_boxes_to_original)


class TestModelIsLoadedLazily(unittest.TestCase):
    """`app._get_ocr()` 必须**只在首次使用时**构造管线（导入期不构造）。"""

    def test_get_ocr_returns_singleton(self):
        """`_get_ocr()` 必须复用同一个实例（只加载一次模型）。"""
        import app as app_mod

        sentinel = object()
        with mock.patch.object(app_mod, "_OCR", sentinel):
            self.assertIs(app_mod._get_ocr(), sentinel)

    def test_get_ocr_constructs_once_when_unset(self):
        """未初始化时才构造，且构造后缓存住。"""
        import app as app_mod

        made = []

        class _Fake:
            det_model_name = "PP-OCRv6_small_det"

            def __init__(self):
                made.append(1)

        with mock.patch.object(app_mod, "_OCR", None), \
             mock.patch.object(app_mod, "SceneTextOCR", _Fake):
            first = app_mod._get_ocr()
            second = app_mod._get_ocr()

        self.assertIs(first, second)
        self.assertEqual(len(made), 1, "只应构造一次")
        self.assertEqual(app_mod._model_label(), "PP-OCRv6 small")


if __name__ == "__main__":
    unittest.main()
