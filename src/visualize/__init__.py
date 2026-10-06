"""可视化工具：绘制检测框、标注，以及预处理图 → 原图的坐标映射。

单独成子包，是为了让"画图"这件事与"界面/推理"解耦：
`make_figures.py`、`evaluate*.py` 与测试只需 `src.visualize.draw`，
**不必 import `app`**（`app` 顶部 import gradio，实测约 7.7s，且只有界面才需要）。
"""

from src.visualize.draw import (
    DEFAULT_FONT_SIZE,
    _load_font,
    annotate_results,
    draw_results,
    map_boxes_to_original,
)

__all__ = [
    "DEFAULT_FONT_SIZE",
    "_load_font",
    "annotate_results",
    "draw_results",
    "map_boxes_to_original",
]
