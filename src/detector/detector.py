"""文字检测模块：调用 PaddleX 检测模型，输出文本框。"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

import paddlex


class TextDetector:
    """封装 PaddleX 检测模型，输入 BGR 图，输出文本框列表。"""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        thresh: Optional[float] = None,
        box_thresh: Optional[float] = None,
        unclip_ratio: Optional[float] = None,
        limit_side_len: Optional[int] = None,
    ) -> None:
        kwargs: dict = {"engine_config": {"run_mode": "paddle"}}
        if device:
            kwargs["device"] = device
        if thresh is not None:
            kwargs["thresh"] = thresh
        if box_thresh is not None:
            kwargs["box_thresh"] = box_thresh
        if unclip_ratio is not None:
            kwargs["unclip_ratio"] = unclip_ratio
        if limit_side_len is not None:
            kwargs["limit_side_len"] = limit_side_len
        self._model = paddlex.create_model(model_name, **kwargs)

    def detect(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """检测文字框，返回 [(4,2) float32 四边形, ...]。"""
        results = list(self._model.predict(image_bgr))
        if not results:
            return []
        polys = results[0].get("dt_polys")
        if polys is None:
            return []
        return [np.asarray(p, dtype=np.float32) for p in polys]
