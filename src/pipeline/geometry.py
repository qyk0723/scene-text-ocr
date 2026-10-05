"""OCR 几何工具：检测框排序 + 透视裁剪。

复刻 PaddleOCR 管线内部逻辑（SortQuadBoxes / get_minarea_rect_crop），
保证与一体化管线行为一致。
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def sort_boxes(boxes: List[np.ndarray]) -> List[np.ndarray]:
    """检测框按从上到下、从左到右排序。"""
    if len(boxes) <= 1:
        return boxes
    arr = [np.asarray(b, dtype=np.float32) for b in boxes]
    sorted_boxes = sorted(arr, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(len(_boxes) - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                _boxes[j], _boxes[j + 1] = _boxes[j + 1], _boxes[j]
            else:
                break
    return _boxes


def crop_box(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """按四点框做最小外接矩形透视裁剪，旋转为水平条。"""
    pts = np.asarray(points, dtype=np.int32)
    bounding_box = cv2.minAreaRect(pts)
    corners = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    idx_a, idx_b, idx_c, idx_d = 0, 1, 2, 3
    if corners[1][1] > corners[0][1]:
        idx_a, idx_d = 0, 1
    else:
        idx_a, idx_d = 1, 0
    if corners[3][1] > corners[2][1]:
        idx_b, idx_c = 2, 3
    else:
        idx_b, idx_c = 3, 2

    box = np.float32([
        corners[idx_a], corners[idx_b], corners[idx_c], corners[idx_d],
    ])

    crop_w = int(max(
        np.linalg.norm(box[0] - box[1]),
        np.linalg.norm(box[2] - box[3]),
    ))
    crop_h = int(max(
        np.linalg.norm(box[0] - box[3]),
        np.linalg.norm(box[1] - box[2]),
    ))
    pts_std = np.float32([
        [0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h],
    ])
    m = cv2.getPerspectiveTransform(box, pts_std)
    dst = cv2.warpPerspective(
        img, m, (crop_w, crop_h),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC,
    )
    h, w = dst.shape[:2]
    if h * 1.0 / w >= 1.5:
        dst = np.rot90(dst)
    return dst
