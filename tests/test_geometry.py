"""src/pipeline/geometry.py 纯函数单元测试。"""

import unittest

import numpy as np

from src.pipeline.geometry import crop_box, sort_boxes


def _q(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


class TestSortBoxes(unittest.TestCase):
    def test_top_to_bottom(self):
        boxes = [_q(0, 50, 10, 60), _q(0, 0, 10, 10)]
        out = sort_boxes(boxes)
        self.assertAlmostEqual(float(out[0][0][1]), 0.0)

    def test_left_to_right_same_line(self):
        boxes = [_q(100, 0, 110, 10), _q(0, 0, 10, 10)]
        out = sort_boxes(boxes)
        self.assertAlmostEqual(float(out[0][0][0]), 0.0)

    def test_empty_and_single(self):
        self.assertEqual(sort_boxes([]), [])
        self.assertEqual(len(sort_boxes([_q(0, 0, 10, 10)])), 1)


class TestCropBox(unittest.TestCase):
    def test_axis_aligned_crop(self):
        img = np.full((50, 100, 3), 255, np.uint8)
        crop = crop_box(img, _q(20, 10, 80, 30))
        self.assertEqual(crop.shape[0], 20)  # 高 = 字高
        self.assertEqual(crop.shape[1], 60)  # 宽 = 字宽

    def test_rotated_crop_is_horizontal_strip(self):
        img = np.full((100, 200, 3), 255, np.uint8)
        box = np.array([[20, 50], [180, 10], [200, 30], [40, 70]], dtype=np.float32)
        crop = crop_box(img, box)
        self.assertLessEqual(crop.shape[0], crop.shape[1])  # 水平条（宽 >= 高）


if __name__ == "__main__":
    unittest.main()
