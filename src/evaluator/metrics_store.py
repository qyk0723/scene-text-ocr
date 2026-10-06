"""评测指标的单一数据源（审计 P1-16 / P3-29）。

**为什么需要这个模块**：`make_figures.py` 曾把每个数字硬编码成字面量，与
`data/results/` 里的产物没有任何程序化关联。结果是图表与真实结果**已经漂移过**
——同一个 small 检测 F1 在图表里是 30.62、在另一个产物文件里是 23.53，而没有任何
机制能发现这种分歧。

现在所有图表数值统一从 `metrics/*.json` 读取；缺键会立刻抛出**带路径**的错误，
而不是静默画出一个错的柱子。各文件的来源与口径见 `metrics/README.md`。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

METRICS_DIR = Path("metrics")


def load(name: str, metrics_dir: Any = None) -> Dict[str, Any]:
    """读取 ``metrics/<name>.json``；不存在时给出可操作的报错。"""
    d = Path(metrics_dir) if metrics_dir else METRICS_DIR
    p = d / f"{name}.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"缺少指标文件 {p}。图表数值必须来自这里、不得硬编码——"
            f"请先按 metrics/README.md 的说明生成它。"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def get(data: Dict[str, Any], path: str) -> Any:
    """按点分路径取值，例如 ``get(det, "small.deteval.f1")``。

    缺失时抛出包含**具体断点**的错误，便于定位是哪个数字没落盘。
    """
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"指标缺少键「{path}」（在「{part}」处断开）")
        cur = cur[part]
    return cur


def pct(value: float) -> float:
    """0.6137 -> 61.37，便于直接喂给现有的百分比绘图函数。"""
    return round(float(value) * 100, 2)


def by_iou(entries, iou: float) -> Dict[str, Any]:
    """从行级评估的列表里按 IoU 取值（列表是因为键里的小数点会破坏点分路径）。"""
    for e in entries:
        if abs(float(e["iou"]) - float(iou)) < 1e-9:
            return e
    raise KeyError(f"行级指标里没有 IoU={iou} 的记录（现有：{[e['iou'] for e in entries]}）")
