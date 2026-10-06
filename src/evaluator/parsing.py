"""标注文件解析。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_det_gt(gt_path: Path) -> List[np.ndarray]:
    """解析检测标注文件，每行前 8 个数字为一个四边形。"""
    quads: List[np.ndarray] = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) < 8:
            continue
        try:
            vals = [float(v) for v in fields[:8]]
        except ValueError:
            continue
        quads.append(np.asarray(vals, dtype=np.float32).reshape(4, 2))
    return quads


def parse_rec_gt(gt_txt: Path, img_dir: Path) -> List[Tuple[Path, str]]:
    """解析识别标注：路径 \\t 文本 \\t 语言，路径改写为项目内相对路径。"""
    pairs: List[Tuple[Path, str]] = []
    for line in gt_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        fname = Path(parts[0]).name
        img_path = img_dir / fname
        pairs.append((img_path, parts[1]))
    return pairs


def parse_det_gt_full(gt_path: Path) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """解析检测标注，返回 ``(quads, is_dont_care, texts)``。

    同时保留 do-not-care 标记与**转写原文**：

    - ICDAR2015 官方评测把转写为 ``###`` 的区域视为 don't care（不计入召回分母，
      命中它的预测被忽略）——``parse_det_gt`` 丢弃了该字段，所以那套口径无法实现；
    - 端到端评测还需要转写原文，才能把同一行的词拼成"行文本"再与识别结果比对。
    """
    quads: List[np.ndarray] = []
    flags: List[bool] = []
    texts: List[str] = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) < 8:
            continue
        try:
            vals = [float(v) for v in fields[:8]]
        except ValueError:
            continue
        text = fields[8].strip() if len(fields) > 8 else ""
        quads.append(np.asarray(vals, dtype=np.float32).reshape(4, 2))
        flags.append(text == "###")
        texts.append(text)
    return quads, np.asarray(flags, dtype=bool), texts


def parse_det_gt_flagged(gt_path: Path) -> Tuple[List[np.ndarray], np.ndarray]:
    """解析检测标注，**同时保留** do-not-care 标记（转写字段为 ``###``）。

    返回 ``(quads, is_dont_care)``；需要转写原文时用 :func:`parse_det_gt_full`。
    """
    quads, flags, _ = parse_det_gt_full(gt_path)
    return quads, flags
