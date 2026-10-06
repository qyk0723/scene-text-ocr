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


def parse_det_gt_flagged(gt_path: Path) -> Tuple[List[np.ndarray], np.ndarray]:
    """解析检测标注，**同时保留** do-not-care 标记（转写字段为 ``###``）。

    ICDAR2015 官方评测把转写为 ``###`` 的区域视为 don't care：既不计入召回率分母，
    命中它的预测也应被忽略。``parse_det_gt`` 只取前 8 个数字、丢弃了该字段，
    所以那套口径无法实现——本函数把它保留下来。

    返回 ``(quads, is_dont_care)``，``is_dont_care`` 是与 quads 等长的布尔数组。
    """
    quads: List[np.ndarray] = []
    flags: List[bool] = []
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
        flags.append(len(fields) > 8 and fields[8].strip() == "###")
    return quads, np.asarray(flags, dtype=bool)
