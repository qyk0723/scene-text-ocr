"""倾斜角估计的单元测试（纯函数，不加载模型）。

这些测试针对一个**真实发生过的缺陷**（审计 P2-19）：原实现用
``cv2.minAreaRect(全部前景像素)`` 估角，实测带符号误差平均 **−59.5°**
（典型输出是 ``-90`` / ``0`` 这类量化值），导致消融把「仅倾斜校正」测成
−39.5 点——那不是"转得不准"，而是**按无意义角度乱转**。
根因是 OpenCV ``minAreaRect`` 的角度按宽高谁更长而模 90° 歧义。

换成投影轮廓方差法后，带符号误差均值 **−0.05°**、绝对误差 **3.2°**。
下面的测试把这一行为固定住，防止回退。
"""
import unittest

import cv2
import numpy as np

from src.preprocess.enhancer import estimate_skew_angle


def _text_like_image(h: int = 240, w: int = 640) -> np.ndarray:
    """造一张类文本文档图：白底 + 若干条黑色横条（模拟文本行）。"""
    img = np.full((h, w), 255, np.uint8)
    for y in range(30, h - 20, 34):
        cv2.rectangle(img, (40, y), (w - 40, y + 14), 0, -1)
    return img


def _binarize(gray: np.ndarray) -> np.ndarray:
    """复刻 enhancer 的二值化（含深色背景反相）。"""
    _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if t.mean() > 127.5:
        t = 255 - t
    return t


def _rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


class TestEstimateSkewAngle(unittest.TestCase):
    def test_blank_image_returns_none(self):
        """全白（无内容）不应编出一个角度。"""
        self.assertIsNone(estimate_skew_angle(_binarize(np.full((120, 200), 255, np.uint8))))

    def test_requires_grayscale(self):
        with self.assertRaises(ValueError):
            estimate_skew_angle(np.zeros((10, 10, 3), np.uint8))

    def test_upright_image_gives_near_zero(self):
        t = _binarize(_text_like_image())
        a = estimate_skew_angle(t)
        self.assertIsNotNone(a)
        self.assertLess(abs(a), 0.5, f"正向图应估出 ≈0°，实得 {a}")

    def test_estimates_known_rotations(self):
        """注入已知旋转，估计值应约为 -θ（deskew 会再转 +估计值来抵消）。"""
        base = _text_like_image()
        for theta in (-12, -8, -4, 4, 8, 12):
            rot = _rotate(base, theta)
            a = estimate_skew_angle(_binarize(rot))
            self.assertIsNotNone(a, f"θ={theta} 时不应返回 None")
            self.assertAlmostEqual(
                a, -theta, delta=2.0,
                msg=f"注入 θ={theta}°，期望估计 ≈{-theta}°，实得 {a:.2f}°",
            )

    def test_sign_convention_lets_deskew_cancel_the_rotation(self):
        """关键约定：按估计角再转一次后，残余角应接近 0（否则符号反了，会转成两倍倾斜）。"""
        base = _text_like_image()
        rot = _rotate(base, 10)
        a = estimate_skew_angle(_binarize(rot))
        residual = estimate_skew_angle(_binarize(_rotate(rot, a)))
        self.assertLess(abs(residual), 1.0,
                        f"抵消后残余应 ≈0°，实得 {residual:.2f}°（符号可能反了）")

    def test_robust_to_speckle_noise(self):
        """**原缺陷的回归测试**：零散噪点不应改变估计结果。

        原实现对全部前景像素取一个 minAreaRect，极值由离文字最远的杂点决定，
        于是少量噪点就能把角度带偏几十度。
        """
        base = _text_like_image()
        clean = estimate_skew_angle(_binarize(base))

        rng = np.random.default_rng(0)
        noisy = base.copy()
        # 在四角与边缘撒孤立噪点（正是原实现最怕的情形）
        for _ in range(60):
            y = int(rng.integers(0, noisy.shape[0]))
            x = int(rng.integers(0, noisy.shape[1]))
            noisy[y, x] = 0
        a = estimate_skew_angle(_binarize(noisy))

        self.assertIsNotNone(a)
        self.assertLess(abs(a - clean), 2.0,
                        f"加入零散噪点后估计从 {clean:.2f}° 漂到 {a:.2f}°")


if __name__ == "__main__":
    unittest.main()
