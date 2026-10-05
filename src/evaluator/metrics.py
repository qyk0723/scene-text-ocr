"""评估指标与几何工具（纯函数，不加载模型）。"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """两个凸四边形的 IoU。"""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
    except cv2.error:
        return 0.0
    if inter <= 0:
        return 0.0
    area_a = cv2.contourArea(a)
    area_b = cv2.contourArea(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt_boxes: List[np.ndarray], pred_boxes: List[np.ndarray], iou_thresh: float) -> int:
    """贪心一对一匹配，返回匹配上的对数。"""
    matched_gt = set()
    matched = 0
    for p in pred_boxes:
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = quad_iou(p, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thresh:
            matched_gt.add(best_j)
            matched += 1
    return matched


def levenshtein(a: str, b: str) -> int:
    """编辑距离。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = cur
    return prev[-1]


def merge_to_lines(quads: List[np.ndarray], y_center_ratio: float = 0.5) -> List[np.ndarray]:
    """把词级四边形按 y 中心聚类成行，返回每行的贴合合并框。

    用 y 中心距离聚类（阈值 = 0.5 × 中位字高），容忍同行错落、
    不并相邻行；合并框 y 取 y 中心 ± 半字高，避免跨度膨胀。
    """
    words = []
    for q in quads:
        ys = q[:, 1]
        xs = q[:, 0]
        yc = float((ys.min() + ys.max()) / 2.0)
        h = float(ys.max() - ys.min())
        words.append({
            "q": q, "yc": yc, "h": h,
            "xmin": float(xs.min()), "xmax": float(xs.max()),
        })
    if not words:
        return []

    med_h = float(np.median([w["h"] for w in words]))
    if med_h <= 0:
        med_h = 1.0

    words.sort(key=lambda w: w["yc"])
    lines = []
    for w in words:
        placed = False
        for line in lines:
            if abs(w["yc"] - line["yc"]) < y_center_ratio * med_h:
                line["boxes"].append(w)
                n = len(line["boxes"])
                line["yc"] = (line["yc"] * (n - 1) + w["yc"]) / n
                line["xmin"] = min(line["xmin"], w["xmin"])
                line["xmax"] = max(line["xmax"], w["xmax"])
                line["ymin"] = min(line["ymin"], w["yc"] - 0.5 * w["h"])
                line["ymax"] = max(line["ymax"], w["yc"] + 0.5 * w["h"])
                placed = True
                break
        if not placed:
            lines.append({
                "boxes": [w], "yc": w["yc"],
                "xmin": w["xmin"], "xmax": w["xmax"],
                "ymin": w["yc"] - 0.5 * w["h"], "ymax": w["yc"] + 0.5 * w["h"],
            })

    merged = []
    for line in lines:
        merged.append(np.array([
            [line["xmin"], line["ymin"]],
            [line["xmax"], line["ymin"]],
            [line["xmax"], line["ymax"]],
            [line["xmin"], line["ymax"]],
        ], dtype=np.float32))
    return merged
