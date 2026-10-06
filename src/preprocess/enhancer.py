"""图像预处理模块：灰度化、去噪、对比度增强、锐化、二值化、倾斜校正、尺寸归一化、小字放大。

每个功能独立开关，通过构造参数控制。默认全关，按需显式开启
（消融结论：清晰图预处理有害，各算子适用域见 docs/evaluation_report.md 第七节）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import get_config


def estimate_skew_angle(
    thresh: np.ndarray,
    max_angle: float = 15.0,
    coarse: float = 1.0,
    fine: float = 0.1,
    max_side: int = 500,
) -> Optional[float]:
    """估计文本倾斜角（度）；无法可靠估计时返回 ``None``。

    使用**投影轮廓方差法**：文本行与图像行对齐时，逐行前景像素数的分布最尖锐
    （方差最大）。在候选角度上旋转并计算该方差，取最大者。

    **为什么不再用 ``cv2.minAreaRect``**（审计 P2-19 的深入核查结论）：
    原实现对全部前景像素取一个最小外接矩形，实测在**单行裁剪图**与**整页图**上
    都给出不可用的结果——典型输出是 ``-90`` / ``0`` 这类量化值，带符号误差平均达
    **−59.5°**。根因是 OpenCV ``minAreaRect`` 的角度定义**按宽高谁更长而模 90° 歧义**
    （同一个矩形可能报成 ``a`` 或 ``a-90``），原代码的
    ``-(90 + angle) if angle < -45 else -angle`` 只能处理其中一种情形。
    这解释了消融里「仅倾斜校正」把字符准确率从 0.5956 打到 0.2003——
    它不是"转得不够准"，而是**按一个无意义的角度乱转**。

    本方法在注入已知旋转的验证里误差 ≈ 0.1–0.5°（见 `tests/test_skew.py`）。
    """
    if thresh.ndim != 2:
        raise ValueError("estimate_skew_angle 需要单通道二值图")

    binary = thresh > 0
    fg = float(binary.mean())
    if fg < 1e-4 or fg > 0.98:
        return None  # 近乎全空或近乎全满，无法判断

    h, w = thresh.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(thresh, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    else:
        small = thresh
    sh, sw = small.shape[:2]
    if sh < 8 or sw < 8:
        return None

    center = (sw / 2.0, sh / 2.0)
    cache: dict = {}

    def score(a: float) -> float:
        """角度 a 下的投影轮廓方差；越大说明行越对齐。"""
        key = round(a, 4)
        if key in cache:
            return cache[key]
        m = cv2.getRotationMatrix2D(center, a, 1.0)
        rot = cv2.warpAffine(small, m, (sw, sh), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        prof = (rot > 0).sum(axis=1).astype(np.float64)
        cache[key] = float(prof.var())
        return cache[key]

    best_a, best_s = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + 1e-9, coarse):
        s = score(float(a))
        if s > best_s:
            best_s, best_a = s, float(a)
    for a in np.arange(best_a - coarse, best_a + coarse + 1e-9, fine):
        s = score(float(a))
        if s > best_s:
            best_s, best_a = s, float(a)
    return float(best_a)


@dataclass(frozen=True)
class ProcessInfo:
    """``process_with_info()`` 的伴生信息：**预处理图坐标系如何映射回原图**。

    **为什么需要它**（审计第 13 轮的真实缺陷）：``upscale``（小字放大）会把长边不足
    目标值的图放大，于是 ``process()`` 的输出比输入大。检测框是在**预处理图**上算出来的，
    属于预处理图的坐标系；若直接把它们画到原图上就会整体错位
    （实测 299×400 的图勾选后 124/156 个坐标点越界）。

    ``process()`` 只返回图像，调用方拿不到任何变换信息，是这个缺陷的结构性原因。
    本类补上这一半：调用方拿 ``image_shape`` 与 ``scale_x/scale_y`` 就能把检测框映射回原图。

    ``scale`` 是**按实际形状测出来的比值**（原图 / 预处理图），不是从算子参数推算的，
    因此以后再加任何改变尺寸的算子（``resize``、``deskew``……）也不必改这里。
    """

    resized: bool
    """预处理是否改变了尺寸。**为 False 时坐标可直接通用**，无需缩放。"""

    image_shape: Tuple[int, int]
    """预处理图的 ``(高, 宽)``。"""

    scale_x: float
    """``原图宽 / 预处理图宽``。"""

    scale_y: float
    """``原图高 / 预处理图高``。"""

    orig_shape: Tuple[int, int] = (0, 0)
    """原图 ``(高, 宽)``，用于把映射结果裁剪回合法范围。"""

    def map_point(self, x: float, y: float) -> Tuple[float, float]:
        """把预处理图坐标系的**一个点**映射回原图坐标系。"""
        return x * self.scale_x, y * self.scale_y

    @classmethod
    def from_shapes(
        cls, orig_shape: Tuple[int, int], proc_shape: Tuple[int, int]
    ) -> "ProcessInfo":
        """按原图/预处理图的 ``(高, 宽)`` 构造映射信息（等价于 :meth:`ImageEnhancer.process_with_info` 的推导）。

        供只拿到两张图、没有 enhancer 实例的场合使用（例如 ``app.annotate_results``）。
        """
        h, w = int(orig_shape[0]), int(orig_shape[1])
        ph, pw = int(proc_shape[0]), int(proc_shape[1])
        resized = (ph, pw) != (h, w)
        return cls(
            resized=resized,
            image_shape=(ph, pw),
            scale_x=(w / pw) if resized else 1.0,
            scale_y=(h / ph) if resized else 1.0,
            orig_shape=(h, w),
        )

    def map_box(self, box: np.ndarray) -> np.ndarray:
        """把预处理图坐标系的 ``(N, 2)`` 框映射回原图坐标系。

        映射后按原图尺寸**裁剪**：检测框本就被上游裁剪到预处理图边界内，
        缩放回去必然落在原图内；这里多做一步是为了防御上游行为变化
        （例如框超出边界时不会画到画布外）。
        """
        out = np.asarray(box, dtype=np.float32).copy()
        if not self.resized or out.size == 0:
            return out
        out[:, 0] = np.round(out[:, 0] * self.scale_x)
        out[:, 1] = np.round(out[:, 1] * self.scale_y)
        if self.orig_shape[0] > 0 and self.orig_shape[1] > 0:
            h, w = self.orig_shape
            out[:, 0] = np.clip(out[:, 0], 0, w)
            out[:, 1] = np.clip(out[:, 1], 0, h)
        return out


class ImageEnhancer:
    """可开关的图像预处理管线。

    用法::

        enhancer = ImageEnhancer()
        processed = enhancer.process(image_bgr)

    ``image_bgr`` 为 OpenCV 读取的 BGR 三通道 uint8 数组。

    若要在预处理图上跑模型、再把结果（检测框等）画回**原图**，改用
    ``process_with_info()`` 并按其返回的 :class:`ProcessInfo` 映射坐标。
    """

    def __init__(
        self,
        grayscale: bool = False,
        denoise: bool = False,
        contrast: bool = False,
        sharpen: bool = False,
        binarize: bool = False,
        deskew: bool = False,
        resize: bool = False,
        upscale: bool = False,
        denoise_method: str = "median",
        binarize_method: str = "otsu",
        resize_long_side: Optional[int] = None,
        upscale_min_long_side: Optional[int] = None,
    ) -> None:
        """初始化并配置开关。

        参数:
            grayscale: 灰度化。
            denoise: 去噪（默认关）。
            contrast: 对比度增强 CLAHE（默认关）。
            sharpen: 锐化（默认关）。
            binarize: 二值化。
            deskew: 倾斜校正。
            resize: 尺寸归一化（长边缩放）。
            upscale: 小字放大（长边不足时放大）。
            denoise_method: 去噪方法 "median" 或 "gaussian"。
            binarize_method: 二值化方法 "otsu" 或 "adaptive"。
            resize_long_side: resize 时目标长边长度（像素）。
            upscale_min_long_side: 放大时目标长边长度（像素）。
        """
        self.grayscale = grayscale
        self.denoise = denoise
        self.contrast = contrast
        self.sharpen = sharpen
        self.binarize = binarize
        self.deskew = deskew
        self.resize = resize
        self.upscale = upscale
        self.denoise_method = denoise_method
        self.binarize_method = binarize_method
        pre = get_config().get("preprocess", {})
        self.resize_long_side = (
            resize_long_side if resize_long_side is not None
            else pre.get("resize_long_side", 1280)
        )
        self.upscale_min_long_side = (
            upscale_min_long_side if upscale_min_long_side is not None
            else pre.get("upscale_min_long_side", 800)
        )

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
        # 前景占比过半说明选到的是背景（深色底图），反相
        if thresh.mean() > 127.5:
            thresh = 255 - thresh

        angle = estimate_skew_angle(thresh)
        if angle is None or abs(angle) < 0.5:
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

    def upscale_small_text(self, img: np.ndarray) -> np.ndarray:
        """长边不足目标值时放大（小字放大）。"""
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side >= self.upscale_min_long_side:
            return img
        scale = self.upscale_min_long_side / long_side
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # ---- 管线 ----

    def process(self, image: np.ndarray) -> np.ndarray:
        """按开启的开关顺序执行预处理，返回 BGR 三通道 uint8 图。

        只返回图像。**若预处理改变了尺寸，调用方无法把结果坐标映射回原图**——
        需要映射时请用 :meth:`process_with_info`。
        """
        return self._pipeline(image)[0]

    def process_with_info(self, image: np.ndarray) -> Tuple[np.ndarray, ProcessInfo]:
        """等价于 :meth:`process`，但**一并返回坐标映射信息**。

        返回 ``(预处理图, info)``；用 ``info.map_box(box)`` / ``info.map_point(x, y)``
        可把在预处理图上算出的坐标（检测框等）映射回原图坐标系。

        典型用法（Gradio 界面，见 ``app.py``）::

            enhancer = ImageEnhancer(upscale=True)
            proc, info = enhancer.process_with_info(img_bgr)
            boxes, texts, scores, elapsed = ocr.run_detailed(proc)
            # boxes 属于 proc 的坐标系，必须映射回 img_bgr 才能在原图上画
            draw_results(img_bgr, [info.map_box(b) for b in boxes], texts)
        """
        proc, img = self._pipeline(image)
        return proc, ProcessInfo.from_shapes(img.shape[:2], proc.shape[:2])

    def _pipeline(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """执行算子链，返回 ``(BGR 三通道结果图, 原始输入图)``。

        原图一并返回，是为了让 :meth:`process_with_info` **按实际形状**算出映射系数，
        而不是从算子参数反推（反推在叠加多个改尺寸算子时会错）。
        """
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

        if self.upscale:
            img = self.upscale_small_text(img)

        # 统一成 BGR 三通道，供 PaddleOCR 使用
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img, image
