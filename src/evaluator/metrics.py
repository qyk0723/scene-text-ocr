"""评估指标与几何工具（纯函数，不加载模型）。"""

from __future__ import annotations

from typing import Dict, List

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


def match_boxes_dnc(
    gt_boxes: List[np.ndarray],
    gt_is_dnc: np.ndarray,
    pred_boxes: List[np.ndarray],
    iou_thresh: float,
) -> Dict[str, int]:
    """don't-care 感知的一对一匹配（ICDAR2015 官方 deteval 语义）。

    与 :func:`match_boxes` 的区别在于它区分两类 GT：

    - 预测命中**真实文本框** → TP
    - 预测命中 **do-not-care 框** → ignored（既不计 TP 也不计 FP）
    - 未参与匹配、但与任一 do-not-care 框重叠 ≥ 阈值 → ignored
    - 其余未参与匹配 → FP
    - 召回率分母只含真实文本框（FN 只统计未被匹配的真实文本框）

    匹配本身仍复用项目原有的贪心策略，以保证与旧口径可比。

    返回 ``{"tp","fp","ignored","fn","matched_dnc"}``。
    """
    gt_is_dnc = np.asarray(gt_is_dnc, dtype=bool)
    used_gt = set()
    matched_pred: Dict[int, int] = {}

    for pi, p in enumerate(pred_boxes):
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gt_boxes):
            if j in used_gt:
                continue
            iou = quad_iou(p, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thresh:
            used_gt.add(best_j)
            matched_pred[pi] = best_j

    tp = sum(1 for j in matched_pred.values() if not gt_is_dnc[j])
    matched_dnc = sum(1 for j in matched_pred.values() if gt_is_dnc[j])

    dnc_idx = [j for j in range(len(gt_boxes)) if gt_is_dnc[j]]
    ignored = matched_dnc
    fp = 0
    for pi, p in enumerate(pred_boxes):
        if pi in matched_pred:
            continue
        if any(quad_iou(p, gt_boxes[j]) >= iou_thresh for j in dnc_idx):
            ignored += 1
        else:
            fp += 1

    fn = sum(
        1 for j in range(len(gt_boxes)) if not gt_is_dnc[j] and j not in used_gt
    )
    return {"tp": tp, "fp": fp, "ignored": ignored, "fn": fn, "matched_dnc": matched_dnc}


def prf_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """由 TP/FP/FN 计算精确率、召回率、F1。"""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


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


def cluster_line_groups(
    quads: List[np.ndarray], y_center_ratio: float = 0.5
) -> List[List[int]]:
    """把词级四边形按 y 中心聚类成行，返回**每行由哪些输入四边形组成**（索引分组）。

    与 :func:`merge_to_lines` 共用同一套聚类规则（后者由本函数实现），
    用于需要知道"哪些词属于同一行"的场景——例如端到端评测要把一行的词拼成行文本。
    """
    if not quads:
        return []

    words = []
    for i, q in enumerate(quads):
        ys = q[:, 1]
        xs = q[:, 0]
        yc = float((ys.min() + ys.max()) / 2.0)
        h = float(ys.max() - ys.min())
        words.append({
            "i": i, "yc": yc, "h": h,
            "xmin": float(xs.min()), "xmax": float(xs.max()),
        })

    med_h = float(np.median([w["h"] for w in words]))
    if med_h <= 0:
        med_h = 1.0

    words.sort(key=lambda w: w["yc"])
    lines: List[dict] = []
    for w in words:
        placed = False
        for line in lines:
            if abs(w["yc"] - line["yc"]) < y_center_ratio * med_h:
                line["items"].append(w["i"])
                n = len(line["items"])
                line["yc"] = (line["yc"] * (n - 1) + w["yc"]) / n
                line["xmin"] = min(line["xmin"], w["xmin"])
                line["xmax"] = max(line["xmax"], w["xmax"])
                line["ymin"] = min(line["ymin"], w["yc"] - 0.5 * w["h"])
                line["ymax"] = max(line["ymax"], w["yc"] + 0.5 * w["h"])
                placed = True
                break
        if not placed:
            lines.append({
                "items": [w["i"]], "yc": w["yc"],
                "xmin": w["xmin"], "xmax": w["xmax"],
                "ymin": w["yc"] - 0.5 * w["h"], "ymax": w["yc"] + 0.5 * w["h"],
            })
    return [line["items"] for line in lines]


def merge_to_lines(quads: List[np.ndarray], y_center_ratio: float = 0.5) -> List[np.ndarray]:
    """把词级四边形按 y 中心聚类成行，返回每行的贴合合并框。

    用 y 中心距离聚类（阈值 = 0.5 × 中位字高），容忍同行错落、
    不并相邻行；合并框 y 取 y 中心 ± 半字高，避免跨度膨胀。
    """
    merged = []
    for idxs in cluster_line_groups(quads, y_center_ratio):
        sub = [quads[i] for i in idxs]
        xs = [float(q[:, 0].min()) for q in sub] + [float(q[:, 0].max()) for q in sub]
        ys = [float(q[:, 1].min()) for q in sub] + [float(q[:, 1].max()) for q in sub]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        merged.append(np.array([
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ], dtype=np.float32))
    return merged
