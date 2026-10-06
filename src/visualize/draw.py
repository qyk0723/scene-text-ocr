"""可视化绘制：把检测框与中文标注画到图上。

**单独成模块的原因**（审计 P2-21）：`draw_results()` 原本只住在 `app.py` 里，
而 `app.py` 顶部 import 了 `gradio` 与模型管线。于是 `make_figures.py` 仅仅为了
画标注就要 `import app` → **拉起整个 Gradio GUI 依赖**（实测 import gradio 约 7.7s），
生成的图表也因此被绑在界面模块的健康状况上。

本模块只依赖 cv2 / numpy / PIL，可被脚本与测试直接引用。
界面侧（`app.py`）与图表侧（`make_figures.py`）都从这里取函数，避免两处实现漂移。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.preprocess.enhancer import ProcessInfo

# 标注字号：界面与图表共用一个默认值
DEFAULT_FONT_SIZE = 16


@lru_cache(maxsize=16)
def _load_font(size: int = DEFAULT_FONT_SIZE) -> Optional[ImageFont.FreeTypeFont]:
    """加载 Windows 中文字体，找不到则返回 None。

    结果缓存：未缓存时**每次调用都从磁盘重载字体**，
    而每张图的每一行标注都会调用一次。
    """
    candidates = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=0)
            except Exception:
                continue
    return None


def draw_results(
    img_bgr: np.ndarray,
    boxes: List[np.ndarray],
    texts: List[str],
    font_size: int = DEFAULT_FONT_SIZE,
) -> np.ndarray:
    """在原图上用 OpenCV 画检测框，用 PIL 叠加中文标注。

    **坐标契约**：``img_bgr`` 与 ``boxes`` 必须属于**同一个坐标系**。
    检测若跑在预处理图（如勾选「小字放大」后的放大图）上，框属于预处理图的坐标系，
    **不能**直接传进来——必须先用 `ProcessInfo.map_box` 映射回原图，
    否则框会整体错位（见 `app.annotate_results` 与 `docs/PROJECT_AUDIT.md` 第 13 轮）。
    """
    img = img_bgr.copy()
    for box in boxes:
        pts = box.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    font = _load_font(font_size)
    if font is not None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        for box, text in zip(boxes, texts):
            x = int(min(p[0] for p in box))
            y = int(min(p[1] for p in box)) - 22
            y = max(y, 0)
            # 底色宽度按**实际渲染尺寸**量；按字符数估算在中英混排时不准
            try:
                left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                width = right - left + 4
                box_h = bottom - top + 4
            except Exception:  # 度量失败时退回字符数估算
                width, box_h = len(text) * 16, 20
            draw.rectangle([x, y, x + width, y + box_h], fill=(0, 0, 0))
            draw.text((x + 2, y + 2), text, font=font, fill=(0, 255, 0))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    return img


def map_boxes_to_original(
    boxes: List[np.ndarray], info: ProcessInfo
) -> List[np.ndarray]:
    """把**预处理图坐标系**的检测框映射回原图坐标系。

    预处理未改变尺寸时原样返回（``info.resized`` 为假）。映射逻辑（含按原图边界裁剪）
    在 :meth:`ProcessInfo.map_box`，此处只做列表遍历。
    """
    if not info.resized:
        return list(boxes)
    return [info.map_box(b) for b in boxes]


def annotate_results(
    img_bgr: np.ndarray,
    proc: np.ndarray,
    boxes: List[np.ndarray],
    texts: List[str],
    font_size: int = DEFAULT_FONT_SIZE,
) -> np.ndarray:
    """在原图上画检测结果（自动处理预处理图与原图之间的坐标映射）。

    ``boxes`` 是 **``proc``（预处理图）坐标系**下的检测框——检测跑在 ``proc`` 上，
    所以 PaddleX 返回的框属于 ``proc`` 的坐标系（其 ``target_sizes`` 是**输入图**尺寸）。
    这里先按 ``proc`` 与原图的实际尺寸比映射回原图再绘制。

    （原实现直接 ``draw_results(img_bgr, boxes, texts)``，勾选「小字放大」且图长边 <800
    时框整体错位、大半跑到画布外——审计第 13 轮的真实缺陷。）
    """
    if img_bgr.shape[:2] == proc.shape[:2]:
        return draw_results(img_bgr, boxes, texts, font_size=font_size)
    info = ProcessInfo.from_shapes(img_bgr.shape[:2], proc.shape[:2])
    return draw_results(img_bgr, map_boxes_to_original(boxes, info), texts,
                        font_size=font_size)
