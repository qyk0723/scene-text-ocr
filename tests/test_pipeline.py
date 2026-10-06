"""管线集成测试：`SceneTextOCR` 的编排、缓存与对位（审计 §待办第 3 项）。

**覆盖缺口**：审计时 55 项测试**全是纯函数与 `ImageEnhancer`**，`SceneTextOCR`
零覆盖——而它正是「检测 → 排序 → 裁剪 → 识别 → 缓存」的中枢，历次真实缺陷
（退化框静默返回整图、boxes/texts 失去对位、缓存键身份不一致）都在这一层。

**不加载模型**：把 `TextDetector` / `TextRecognizer` 替换为假实现，
因此本文件是**秒级**的（审计建议的"用很小的图跑几秒"会真的加载模型、每次 8–17s，
不适合放进单测；真机冒烟由 `scripts/validate_app_display.py` 负责）。
"""

import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from src.pipeline.ocr_pipeline import SceneTextOCR


def _q(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


class _FakeDetector:
    """假检测器：按脚本返回固定框与置信度，并记录被调用了多少次。"""

    calls = 0
    next_boxes: list = []
    next_scores: list = []

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs

    def detect_with_scores(self, img):
        type(self).calls += 1
        return ([np.asarray(b, np.float32) for b in type(self).next_boxes],
                list(type(self).next_scores))

    def detect(self, img):
        return self.detect_with_scores(img)[0]


class _FakeRecognizer:
    """假识别器：文本 = 裁剪图的高度（便于断言"这个框喂进了哪块裁剪"）。"""

    calls = 0

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs

    def recognize(self, crop):
        type(self).calls += 1
        return f"h{crop.shape[0]}", 0.9


def _patched(boxes, scores):
    """把管线里的检测/识别换成假实现。"""
    _FakeDetector.calls = 0
    _FakeRecognizer.calls = 0
    _FakeDetector.next_boxes = boxes
    _FakeDetector.next_scores = scores
    return (
        mock.patch("src.detector.detector.TextDetector", _FakeDetector),
        mock.patch("src.recognizer.recognizer.TextRecognizer", _FakeRecognizer),
    )


IMG = np.full((80, 200, 3), 255, np.uint8)
IMG[10:20, 10:60] = 0     # 框 1 目标
IMG[30:40, 10:60] = 0     # 框 2 目标（y 更大 → 排序在后）


class TestPipelineOrdering(unittest.TestCase):
    """检测 → 排序：输出顺序必须是"从上到下、从左到右"。"""

    def test_boxes_are_sorted_top_to_bottom(self):
        ocr = SceneTextOCR()
        # 故意乱序给出：中间的框先给，底下的框最后给
        boxes = [_q(10, 30, 60, 40), _q(100, 60, 150, 70), _q(10, 10, 60, 20)]
        p1, p2 = _patched(boxes, [0.9, 0.9, 0.9])
        with p1, p2:
            out_boxes, out_texts, out_scores, elapsed = ocr.run_detailed(IMG)

        ys = [float(b[:, 1].min()) for b in out_boxes]
        self.assertEqual(ys, sorted(ys), "输出框必须按 y 从小到大")
        self.assertEqual(len(out_boxes), len(out_texts), "boxes 与 texts 必须等长")
        self.assertEqual(len(out_boxes), len(out_scores), "boxes 与 scores 必须等长")
        self.assertGreaterEqual(elapsed, 0.0)

    def test_boxes_and_texts_stay_aligned(self):
        """文本必须与它所属的框一一对位（历史缺陷：退化框会打乱对位）。"""
        ocr = SceneTextOCR()
        # 高度不同 → 假识别器返回的 h 值不同，可据此反推每个框喂了哪块裁剪
        boxes = [_q(10, 10, 60, 30), _q(10, 50, 60, 60)]
        p1, p2 = _patched(boxes, [0.9, 0.9])
        with p1, p2:
            out_boxes, out_texts, out_scores, _ = ocr.run_detailed(IMG)

        # 第一个框高 20（y 10→30）、第二个框高 10（y 50→60）
        self.assertEqual(out_texts[0], "h20")
        self.assertEqual(out_texts[1], "h10")


class TestPipelineCaching(unittest.TestCase):
    """结果缓存：同图二次识别不再跑检测；不同图不能互相污染。"""

    def test_second_call_hits_cache(self):
        ocr = SceneTextOCR()
        boxes = [_q(10, 10, 60, 20)]
        p1, p2 = _patched(boxes, [0.9])
        with p1, p2:
            first = ocr.run_detailed(IMG)
            self.assertEqual(_FakeDetector.calls, 1)
            second = ocr.run_detailed(IMG)
            self.assertEqual(_FakeDetector.calls, 1, "第二次应命中缓存，不再跑检测")
            self.assertEqual(_FakeRecognizer.calls, 1, "第二次应命中缓存，不再跑识别")

        self.assertEqual(first[1], second[1])          # texts 相同
        self.assertLessEqual(second[3], first[3])      # 缓存命中更快

    def test_different_images_do_not_collide(self):
        ocr = SceneTextOCR()
        other = np.full((80, 200, 3), 128, np.uint8)
        boxes = [_q(10, 10, 60, 20)]
        p1, p2 = _patched(boxes, [0.9])
        with p1, p2:
            ocr.run_detailed(IMG)
            ocr.run_detailed(other)
        self.assertEqual(_FakeDetector.calls, 2, "内容不同的两张图必须各跑一次")

    def test_cache_is_lru_bounded(self):
        """缓存必须是**有界 LRU**（上限 32），不能无限增长。"""
        ocr = SceneTextOCR()
        boxes = [_q(10, 10, 60, 20)]
        p1, p2 = _patched(boxes, [0.9])
        with p1, p2:
            for i in range(ocr._cache_max + 3):
                img = np.full((40, 40, 3), i % 256, np.uint8)
                img[0, 0, 0] = i % 256  # 保证每张图内容唯一
                ocr.run_detailed(img)

        self.assertLessEqual(len(ocr._cache), ocr._cache_max,
                             "缓存条数不得超过上限")

    def test_cache_returns_fresh_elapsed_time(self):
        """缓存命中返回的耗时必须是**本次**的（约 0），而不是上次的旧值。"""
        ocr = SceneTextOCR()
        boxes = [_q(10, 10, 60, 20)]
        p1, p2 = _patched(boxes, [0.9])
        with p1, p2:
            ocr.run_detailed(IMG)
            _boxes, _texts, _scores, cached_elapsed = ocr.run_detailed(IMG)

        self.assertLess(cached_elapsed, 0.05,
                        "缓存命中应返回约 0 的耗时，而不是复述上次的耗时")


class TestPipelinePathAndArrayIdentity(unittest.TestCase):
    """缓存键的**身份**统一（审计 P2-21 的修复守护）。

    原实现按输入类型分两种算法：路径取 ``sha256(文件字节)``、数组取
    ``sha256(数组字节 + str(shape))``——同一张图经不同入口进来就是**两个身份**，
    会重复推理、缓存里同时存两份同样结果（实测已确证）。
    现已统一为「解码后数组的内容哈希」，本组测试锁定该行为。
    """

    def setUp(self):
        self.tmp = Path(__file__).resolve().parent.parent / ".tmp_tests" / "pipeline"
        self.tmp.mkdir(parents=True, exist_ok=True)
        # 三通道等值图：编码成 png 再读回，字节序与 arr.tobytes() 一致
        gray = np.full((80, 200, 3), 255, np.uint8)
        gray[10:20, 10:60] = 0
        self.path = self.tmp / "same.png"
        ok, buf = cv2.imencode(".png", gray)
        assert ok
        self.path.write_bytes(buf.tobytes())
        self.arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        self.assertEqual(self.arr.tobytes(), gray.tobytes())

    def test_same_content_via_path_and_array_shares_one_key(self):
        """同一张图：路径入口与数组入口必须**合并为同一个缓存身份**。"""
        ocr = SceneTextOCR()
        boxes = [_q(10, 10, 60, 20)]
        p1, p2 = _patched(boxes, [0.9])
        with p1, p2:
            ocr.run_detailed(self.path)
            ocr.run_detailed(self.arr)

        self.assertEqual(_FakeDetector.calls, 1,
                         "内容相同就该命中同一个缓存键（P2-21 的修复点）")
        self.assertEqual(len(ocr._cache), 1, "缓存里不应出现两份同样的结果")

    def test_cache_key_is_content_hash_of_array(self):
        """键的构成：``sha256(数组字节 + str(shape))``，与入口类型无关。"""
        import hashlib

        ocr = SceneTextOCR()
        expect = hashlib.sha256(
            self.arr.tobytes() + str(self.arr.shape).encode()).hexdigest()
        self.assertEqual(ocr._cache_key(self.arr), expect)
        # 路径输入解码后是同一份数组 → 同一个键
        self.assertEqual(ocr._cache_key(ocr._load_any(self.path)), expect)

    def test_path_input_reads_file_once_per_cold_call(self):
        """**冷调用只读一次文件**（原实现算键读一次、加载再读一次，共两次）。"""
        ocr = SceneTextOCR()
        boxes = [_q(10, 10, 60, 20)]
        p1, p2 = _patched(boxes, [0.9])
        reads = 0
        real_fromfile = np.fromfile

        def counting_fromfile(*a, **kw):
            nonlocal reads
            reads += 1
            return real_fromfile(*a, **kw)

        with p1, p2, mock.patch.object(np, "fromfile", counting_fromfile):
            ocr.run_detailed(self.path)

        self.assertEqual(reads, 1, "解码只应发生一次（imread_unicode 内部 np.fromfile）")

    def test_missing_path_raises_filenotfound(self):
        ocr = SceneTextOCR()
        with self.assertRaises(FileNotFoundError):
            ocr.run_detailed(self.tmp / "不存在.png")

class TestPipelineDegenerateBox(unittest.TestCase):
    """退化框：裁剪为空时必须跳过识别，且**保持 boxes/texts 对位**。"""

    def test_degenerate_box_keeps_alignment(self):
        ocr = SceneTextOCR()
        good = _q(10, 10, 60, 20)
        degenerate = _q(100, 10, 100, 10)  # 四点重合
        p1, p2 = _patched([good, degenerate], [0.9, 0.8])
        with p1, p2:
            out_boxes, out_texts, out_scores, _ = ocr.run_detailed(IMG)

        self.assertEqual(len(out_boxes), 2)
        self.assertEqual(len(out_texts), 2, "退化框也要占位，否则文本与框错位")
        self.assertEqual(len(out_scores), 2)
        self.assertEqual(out_scores[0], 0.9)
        self.assertEqual(out_scores[1], 0.0, "退化框的置信度记 0")
        self.assertEqual(_FakeRecognizer.calls, 1, "退化框不应送进识别器")


class TestPipelineRunWrapper(unittest.TestCase):
    """`run()` 是 `run_detailed()` 的三元组包装，行为必须一致。"""

    def test_run_returns_three_tuple_consistent_with_run_detailed(self):
        ocr = SceneTextOCR()
        boxes = [_q(10, 10, 60, 20)]
        p1, p2 = _patched(boxes, [0.9])
        with p1, p2:
            b1, t1, e1 = ocr.run(IMG)
            b2, t2, s2, e2 = ocr.run_detailed(IMG)

        self.assertEqual(len(b1), len(t1))
        self.assertEqual([t for t in t1], [t for t in t2])
        self.assertEqual(len(b2), len(s2))
        self.assertAlmostEqual(e1, e2, places=3)


if __name__ == "__main__":
    unittest.main()
