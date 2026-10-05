"""场景文字检测与识别统一封装。

SceneTextOCR 编排「检测 → 排序 → 裁剪 → 识别」：
    - 检测、识别模型可独立选择（默认读 config.yaml，部署用 small）
    - run() 返回 (文本框, 文本, 耗时)；run_detailed() 额外返回置信度
    - 按图片内容 hash 缓存结果（LRU 32 条），同图二次识别约 0s
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from src.config import get_config
from src.pipeline.geometry import crop_box, sort_boxes


class SceneTextOCR:
    """检测 + 识别管线。

    用法::

        ocr = SceneTextOCR()  # 读 config.yaml 默认（small）
        boxes, texts, elapsed = ocr.run("path/to/image.jpg")
    """

    def __init__(
        self,
        lang: Optional[str] = None,
        device: Optional[str] = None,
        det_thresh: Optional[float] = None,
        det_box_thresh: Optional[float] = None,
        det_model_name: Optional[str] = None,
        rec_model_name: Optional[str] = None,
    ) -> None:
        cfg = get_config()
        model = cfg.get("model", {})
        det = cfg.get("det", {})

        self.lang = lang or model.get("lang", "ch")
        self.device = device or model.get("device", "cpu")
        self.det_model_name = det_model_name or model.get("det")
        self.rec_model_name = rec_model_name or model.get("rec")
        self.det_thresh = det_thresh if det_thresh is not None else det.get("thresh")
        self.det_box_thresh = det_box_thresh if det_box_thresh is not None else det.get("box_thresh")
        self.det_unclip_ratio = det.get("unclip_ratio")
        self.det_limit_side_len = det.get("limit_side_len")

        # 懒加载
        self._detector = None
        self._recognizer = None
        # 结果缓存：图片内容 hash -> 结果，LRU 上限 32 条
        self._cache: OrderedDict = OrderedDict()
        self._cache_max = 32

    def _get_detector(self):
        if self._detector is None:
            from src.detector.detector import TextDetector

            self._detector = TextDetector(
                self.det_model_name,
                device=self.device,
                thresh=self.det_thresh,
                box_thresh=self.det_box_thresh,
                unclip_ratio=self.det_unclip_ratio,
                limit_side_len=self.det_limit_side_len,
            )
        return self._detector

    def _get_recognizer(self):
        if self._recognizer is None:
            from src.recognizer.recognizer import TextRecognizer

            self._recognizer = TextRecognizer(self.rec_model_name, device=self.device)
        return self._recognizer

    def _load_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """输入路径或 BGR 数组，返回 BGR uint8 数组。"""
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                raise FileNotFoundError(f"图片不存在: {image}")
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"图片读取失败: {image}")
            return img
        return image

    def _cache_key(self, image: Union[str, np.ndarray]) -> str:
        if isinstance(image, (str, Path)):
            data = Path(image).read_bytes()
        else:
            arr = np.asarray(image)
            data = arr.tobytes() + str(arr.shape).encode()
        return hashlib.sha256(data).hexdigest()

    def run_detailed(
        self, image: Union[str, np.ndarray]
    ) -> Tuple[List[np.ndarray], List[str], List[float], float]:
        """检测 + 识别，返回 (文本框, 文本, 置信度, 耗时)。命中缓存耗时约 0。"""
        start = time.perf_counter()
        key = self._cache_key(image)
        if key in self._cache:
            result = self._cache.pop(key)
            self._cache[key] = result  # 移到队尾（LRU）
            boxes, texts, scores, _elapsed = result
            return boxes, texts, scores, time.perf_counter() - start

        img = self._load_image(image)
        detector = self._get_detector()
        recognizer = self._get_recognizer()

        start = time.perf_counter()
        boxes = sort_boxes(detector.detect(img))
        texts: List[str] = []
        scores: List[float] = []
        for box in boxes:
            crop = crop_box(img, box)
            text, score = recognizer.recognize(crop)
            texts.append(text)
            scores.append(score)
        elapsed = time.perf_counter() - start

        self._cache[key] = (boxes, texts, scores, elapsed)
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return boxes, texts, scores, elapsed

    def run(
        self, image: Union[str, np.ndarray]
    ) -> Tuple[List[np.ndarray], List[str], float]:
        """检测 + 识别，返回 (文本框, 文本, 耗时)。"""
        boxes, texts, _scores, elapsed = self.run_detailed(image)
        return boxes, texts, elapsed
