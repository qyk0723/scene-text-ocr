"""文字检测模块：调用 PaddleX 检测模型，输出文本框。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.config import get_config


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
        limit_type: Optional[str] = None,
    ) -> None:
        # paddlex 在**此处**才导入：它的导入很重（数秒），而本模块要能被
        # 不加载模型的测试与工具引用（审计 P2-21 的同类问题）。
        import paddlex

        run_mode = get_config().get("inference", {}).get("run_mode", "paddle")
        kwargs: dict = {"engine_config": {"run_mode": run_mode}}
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
        if limit_type is not None:
            kwargs["limit_type"] = limit_type
        self._model = paddlex.create_model(model_name, **kwargs)

    def detect_with_scores(self, image_bgr: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """检测文字框，返回 (四边形列表, 置信度列表)。

        保留 ``dt_scores`` 是为了支持**离线**阈值敏感性分析：一次推理拿到分数后，
        换阈值只需重算匹配，无需重新跑模型（见 evaluate.py --dump-pred）。
        """
        results = list(self._model.predict(image_bgr))
        if not results:
            return [], []
        r = results[0]
        polys = r.get("dt_polys")
        if polys is None:
            return [], []
        boxes = [np.asarray(p, dtype=np.float32) for p in polys]
        scores = r.get("dt_scores")
        if scores is None:
            scores = [float("nan")] * len(boxes)
        scores = [float(s) for s in scores]
        if len(scores) != len(boxes):  # 防御：长度不一致时以框数为准
            scores = (scores + [float("nan")] * len(boxes))[: len(boxes)]
        return boxes, scores

    def detect(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """检测文字框，返回 [(4,2) float32 四边形, ...]。"""
        return self.detect_with_scores(image_bgr)[0]
