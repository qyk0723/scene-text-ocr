"""文字识别模块：调用 PaddleX 识别模型，输出文本与置信度。"""

from __future__ import annotations

from typing import Tuple

import numpy as np

import paddlex


class TextRecognizer:
    """封装 PaddleX 识别模型，输入单行图（BGR），输出 (文本, 置信度)。"""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        kwargs: dict = {"engine_config": {"run_mode": "paddle"}}
        if device:
            kwargs["device"] = device
        self._model = paddlex.create_model(model_name, **kwargs)

    def recognize(self, image_bgr: np.ndarray) -> Tuple[str, float]:
        """识别单行裁剪图，返回 (文本, 置信度)。"""
        results = list(self._model.predict(image_bgr))
        if not results:
            return "", 0.0
        r = results[0]
        text = r.get("rec_text")
        score = r.get("rec_score")
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        if isinstance(score, (list, tuple, np.ndarray)):
            score = score[0] if len(score) else 0.0
        return str(text or ""), float(score or 0.0)
