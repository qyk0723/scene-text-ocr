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
