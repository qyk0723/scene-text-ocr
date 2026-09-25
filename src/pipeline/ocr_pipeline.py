"""场景文字检测与识别统一封装。

对外只暴露一个类 SceneTextOCR：
    - 懒加载 PaddleOCR 模型（PP-OCR 预训练，不训练）
    - 输入图片路径，输出文本框坐标、识别文本、耗时
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


class SceneTextOCR:
    """基于 PaddleOCR 的文字检测 + 识别管线。

    用法::

        ocr = SceneTextOCR(lang="ch")
        boxes, texts, elapsed = ocr.run("path/to/image.jpg")
    """

    def __init__(
        self,
        lang: str = "ch",
        device: str = "cpu",
        det_thresh: Optional[float] = None,
        det_box_thresh: Optional[float] = None,
    ) -> None:
        """初始化。

        参数:
            lang: 识别语言，默认 "ch"（中英文）。
            device: 推理设备，默认 "cpu"。
            det_thresh: 检测过滤阈值，None 用 PaddleOCR 默认。
            det_box_thresh: 检测框阈值，None 用 PaddleOCR 默认。
        """
        self.lang = lang
        self.device = device
        self.det_thresh = det_thresh
        self.det_box_thresh = det_box_thresh
        # 懒加载：首次调用 run 时才真正实例化模型
        self._engine = None

    def _get_engine(self):
        """延迟创建 PaddleOCR 实例，避免 import 与模型加载开销集中在构造时。"""
        if self._engine is None:
            from paddleocr import PaddleOCR

            kwargs = dict(
                lang=self.lang,
                device=self.device,
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            if self.det_thresh is not None:
                kwargs["text_det_thresh"] = self.det_thresh
            if self.det_box_thresh is not None:
                kwargs["text_det_box_thresh"] = self.det_box_thresh
            self._engine = PaddleOCR(**kwargs)
        return self._engine

    def _resolve_input(self, image: Union[str, np.ndarray]) -> Union[str, np.ndarray]:
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                raise FileNotFoundError(f"图片不存在: {image}")
            return str(path)
        return image

    def _parse(
        self, raw
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """解析 ocr() 返回，提取检测框、文本、置信度。"""
        boxes: List[np.ndarray] = []
        texts: List[str] = []
        scores: List[float] = []

        if raw:
            result = raw[0]
            dt_polys = result.get("dt_polys") or []
            rec_texts = result.get("rec_texts") or []
            for poly, text in zip(dt_polys, rec_texts):
                boxes.append(np.asarray(poly, dtype=np.float32))
                texts.append(text)
            scores = [float(s) for s in (result.get("rec_scores") or [])]
            scores += [0.0] * (len(texts) - len(scores))

        return boxes, texts, scores

    def run_detailed(
        self, image: Union[str, np.ndarray]
    ) -> Tuple[List[np.ndarray], List[str], List[float], float]:
        """同 run()，额外返回每个文本框的识别置信度。"""
        ocr_input = self._resolve_input(image)
        engine = self._get_engine()

        start = time.perf_counter()
        raw = engine.ocr(ocr_input)
        elapsed = time.perf_counter() - start

        boxes, texts, scores = self._parse(raw)
        return boxes, texts, scores, elapsed

    def run(
        self, image: Union[str, np.ndarray]
    ) -> Tuple[List[np.ndarray], List[str], float]:
        """对单张图片执行检测 + 识别。

        参数:
            image: 输入图片路径，或 BGR 三通道 numpy 数组。

        返回:
            (boxes, texts, elapsed)
            boxes: 文本框坐标列表，每个元素为 shape (4, 2) 的 numpy 数组，
                   四点为左上、右上、右下、左下。
            texts: 与 boxes 一一对应的识别文本列表。
            elapsed: 检测 + 识别总耗时（秒）。
        """
        boxes, texts, _scores, elapsed = self.run_detailed(image)
        return boxes, texts, elapsed
