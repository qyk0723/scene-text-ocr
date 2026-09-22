"""图像预处理模块：灰度化、去噪、对比度增强、锐化、二值化、倾斜校正、尺寸归一化。

每个功能独立开关，通过构造参数控制。默认开启：去噪 + 对比度增强 + 锐化。
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class ImageEnhancer:
    """可开关的图像预处理管线。

    用法::

        enhancer = ImageEnhancer()
        processed = enhancer.process(image_bgr)

    ``image_bgr`` 为 OpenCV 读取的 BGR 三通道 uint8 数组。
    """

    def __init__(
        self,
        grayscale: bool = False,
        denoise: bool = True,
        contrast: bool = True,
        sharpen: bool = True,
        binarize: bool = False,
        deskew: bool = False,
        resize: bool = False,
        denoise_method: str = "median",
        binarize_method: str = "otsu",
        resize_long_side: int = 1280,
    ) -> None:
        """初始化并配置开关。

        参数:
            grayscale: 灰度化。
            denoise: 去噪（默认开）。
            contrast: 对比度增强 CLAHE（默认开）。
            sharpen: 锐化（默认开）。
            binarize: 二值化。
            deskew: 倾斜校正。
            resize: 尺寸归一化（长边缩放）。
            denoise_method: 去噪方法 "median" 或 "gaussian"。
            binarize_method: 二值化方法 "otsu" 或 "adaptive"。
            resize_long_side: resize 时目标长边长度（像素）。
        """
        self.grayscale = grayscale
        self.denoise = denoise
        self.contrast = contrast
        self.sharpen = sharpen
        self.binarize = binarize
        self.deskew = deskew
        self.resize = resize
        self.denoise_method = denoise_method
        self.binarize_method = binarize_method
        self.resize_long_side = resize_long_side

    # ---- 各预处理步骤 ----

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """BGR -> 单通道灰度。"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, img: np.ndarray) -> np.ndarray:
        """中值或高斯去噪。"""
        if self.denoise_method == "gaussian":
            return cv2.GaussianBlur(img, (3, 3), 0)
        return cv2.medianBlur(img, 3)

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """CLAHE 对比度增强。彩色图作用于 LAB 亮度通道。"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if img.ndim == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        return clahe.apply(img)

    def sharpen_image(self, img: np.ndarray) -> np.ndarray:
        """拉普拉斯锐化。"""
        kernel = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
        )
        return cv2.filter2D(img, -1, kernel)

    def binarize_image(self, img: np.ndarray) -> np.ndarray:
        """Otsu 或自适应二值化。输入彩色先转灰度。"""
        gray = img if img.ndim == 2 else self.to_grayscale(img)
        if self.binarize_method == "adaptive":
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def deskew_image(self, img: np.ndarray) -> np.ndarray:
        """简单倾斜校正：按文本区域最小外接矩形角度旋转。"""
        gray = img if img.ndim == 2 else self.to_grayscale(img)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 10:
            return img  # 几乎无内容，不旋转
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) < 0.5:
            return img

        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            img, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def resize_long_edge(self, img: np.ndarray) -> np.ndarray:
        """长边缩放到固定值，保持宽高比。"""
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side <= self.resize_long_side:
            return img
        scale = self.resize_long_side / long_side
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ---- 管线 ----

    def process(self, image: np.ndarray) -> np.ndarray:
        """按开启的开关顺序执行预处理，返回 BGR 三通道 uint8 图。"""
        img = image

        if self.grayscale:
            img = self.to_grayscale(img)

        if self.denoise:
            img = self.remove_noise(img)

        if self.contrast:
            img = self.enhance_contrast(img)

        if self.sharpen:
            img = self.sharpen_image(img)

        if self.binarize:
            img = self.binarize_image(img)

        if self.deskew:
            img = self.deskew_image(img)

        if self.resize:
            img = self.resize_long_edge(img)

        # 统一成 BGR 三通道，供 PaddleOCR 使用
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img
