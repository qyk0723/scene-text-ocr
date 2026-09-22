"""场景文字检测与识别统一封装。

对外只暴露一个类 SceneTextOCR：
    - 懒加载 PaddleOCR 模型（PP-OCR 预训练，不训练）
    - 输入图片路径，输出文本框坐标、识别文本、耗时
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np


class SceneTextOCR:
    """基于 PaddleOCR 的文字检测 + 识别管线。

    用法::

        ocr = SceneTextOCR(lang="ch")
        boxes, texts, elapsed = ocr.run("path/to/image.jpg")
    """

    def __init__(self, lang: str = "ch", device: str = "cpu") -> None:
        """初始化。

        参数:
            lang: 识别语言，默认 "ch"（中英文）。
            device: 推理设备，默认 "cpu"。
        """
        self.lang = lang
        self.device = device
        # 懒加载：首次调用 run 时才真正实例化模型
        self._engine = None

    def _get_engine(self):
        """延迟创建 PaddleOCR 实例，避免 import 与模型加载开销集中在构造时。"""
        if self._engine is None:
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                lang=self.lang,
                use_angle_cls=True,
                device=self.device,
                enable_mkldnn=False,
            )
        return self._engine

    def run(self, image_path: str) -> Tuple[List[np.ndarray], List[str], float]:
        """对单张图片执行检测 + 识别。

        参数:
            image_path: 输入图片路径。

        返回:
            (boxes, texts, elapsed)
            boxes: 文本框坐标列表，每个元素为 shape (4, 2) 的 numpy 数组，
                   四点为左上、右上、右下、左下。
            texts: 与 boxes 一一对应的识别文本列表。
            elapsed: 检测 + 识别总耗时（秒）。
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        engine = self._get_engine()

        start = time.perf_counter()
        raw = engine.ocr(str(path))
        elapsed = time.perf_counter() - start

        boxes: List[np.ndarray] = []
        texts: List[str] = []

        # raw 为 None 表示未检出任何文字
        if raw:
            result = raw[0]
            dt_polys = result.get("dt_polys") or []
            rec_texts = result.get("rec_texts") or []
            for poly, text in zip(dt_polys, rec_texts):
                box = np.asarray(poly, dtype=np.float32)
                boxes.append(box)
                texts.append(text)

        return boxes, texts, elapsed
