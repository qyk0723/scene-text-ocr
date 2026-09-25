"""Gradio 图形界面：上传图片，检测并识别文字，框出结果。

运行::

    python app.py

启动后在浏览器打开终端打印的本地地址（默认 http://127.0.0.1:7860）。
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.pipeline.ocr_pipeline import SceneTextOCR
from src.preprocess.enhancer import ImageEnhancer

# 全局单例：模型只加载一次
_OCR = SceneTextOCR()
_ENHANCER = ImageEnhancer()


def _load_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    """加载 Windows 中文字体，找不到则返回 None。"""
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


def draw_results(img_bgr: np.ndarray, boxes: List[np.ndarray], texts: List[str]) -> np.ndarray:
    """在原图上用 OpenCV 画检测框，用 PIL 叠加中文标注。"""
    img = img_bgr.copy()
    for box in boxes:
        pts = box.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    font = _load_font(16)
    if font is not None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        for box, text in zip(boxes, texts):
            x = int(min(p[0] for p in box))
            y = int(min(p[1] for p in box)) - 22
            y = max(y, 0)
            width = len(text) * 16
            draw.rectangle([x, y, x + width, y + 20], fill=(0, 0, 0))
            draw.text((x + 2, y + 2), text, font=font, fill=(0, 255, 0))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    return img


def _make_table(texts: List[str], scores: List[float]) -> pd.DataFrame:
    rows = [
        [i, text, f"{score:.4f}"]
        for i, (text, score) in enumerate(zip(texts, scores), start=1)
    ]
    return pd.DataFrame(rows, columns=["序号", "识别文字", "置信度"])


def _status_html(n_boxes: int, ocr_elapsed: float, prep_elapsed: float) -> str:
    style = (
        "margin-top:16px;padding:12px 16px;background:#eef3ff;"
        "border:1px solid #dbe4ff;border-radius:12px;color:#374151;font-size:.9rem;"
    )
    return (
        f'<div style="{style}">模型 PP-OCRv6 &nbsp;|&nbsp; 检测框数 {n_boxes} '
        f"&nbsp;|&nbsp; 识别耗时 {ocr_elapsed:.1f}s &nbsp;|&nbsp; 预处理 {prep_elapsed:.1f}s</div>"
    )


def predict(image: np.ndarray, preprocess: bool) -> Tuple[np.ndarray, pd.DataFrame, str]:
    """处理一张上传图片，返回 (标注图, 文本表格, 状态栏 HTML)。"""
    if image is None:
        empty = pd.DataFrame(columns=["序号", "识别文字", "置信度"])
        return np.zeros((1, 1, 3), dtype=np.uint8), empty, _status_html(0, 0.0, 0.0)

    # Gradio 传入 RGB，OCR / OpenCV 用 BGR
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    prep_elapsed = 0.0
    if preprocess:
        start = time.perf_counter()
        proc = _ENHANCER.process(img_bgr)
        prep_elapsed = time.perf_counter() - start
    else:
        proc = img_bgr

    boxes, texts, scores, ocr_elapsed = _OCR.run_detailed(proc)

    annotated_bgr = draw_results(img_bgr, boxes, texts)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    table = _make_table(texts, scores)
    status = _status_html(len(boxes), ocr_elapsed, prep_elapsed)
    return annotated_rgb, table, status


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="slate",
        font=[
            "system-ui",
            "-apple-system",
            "Segoe UI",
            "Microsoft YaHei",
            "PingFang SC",
            "sans-serif",
        ],
    )
    css = """
    .gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
    footer { display: none !important; }
    .main-row {
        gap: 16px !important;
        flex-wrap: nowrap !important;
        flex-direction: row !important;
    }
    .main-row > * { min-width: 0 !important; }
    .card {
        background: #ffffff;
        border: 1px solid #eef0f4;
        border-radius: 12px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
        padding: 16px;
    }
    #upload-box { border: 2px dashed #4F6EF7 !important; border-radius: 12px !important; padding: 6px; }
    .primary-btn button {
        background: #4F6EF7 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        font-size: 1rem !important;
        width: 100%;
    }
    .primary-btn button:hover { background: #3f5be0 !important; }
    """

    with gr.Blocks(title="场景文字检测与识别系统", theme=theme, css=css) as demo:
        # 顶部标题
        gr.HTML(
            """
            <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:8px 4px 20px;">
                <h1 style="color:#1f2937;margin:0;font-size:1.7rem;font-weight:700;">
                    场景文字检测与识别系统
                </h1>
                <span style="color:#6b7280;font-size:.9rem;">基于 PaddleOCR PP-OCRv6</span>
            </div>
            """
        )

        with gr.Row(elem_classes="main-row", equal_height=True):
            # 左栏：控制区
            with gr.Column(scale=38, elem_classes="card"):
                input_img = gr.Image(
                    type="numpy", label="上传图片", height=320, elem_id="upload-box"
                )
                preprocess = gr.Checkbox(
                    label="预处理：去噪 + 对比度增强 + 锐化", value=True
                )
                btn = gr.Button("开始识别", variant="primary", elem_classes="primary-btn")
                gr.Examples(
                    examples=[["data/samples/test.jpg"]],
                    inputs=[input_img],
                    label="示例图",
                )

            # 右栏：结果区
            with gr.Column(scale=62, elem_classes="card"):
                output_img = gr.Image(type="numpy", label="检测结果", height=400)
                table = gr.Dataframe(
                    value=pd.DataFrame(columns=["序号", "识别文字", "置信度"]),
                    headers=["序号", "识别文字", "置信度"],
                    interactive=False,
                    wrap=True,
                )

        # 底部状态栏
        status = gr.HTML(_status_html(0, 0.0, 0.0))

        btn.click(
            predict,
            inputs=[input_img, preprocess],
            outputs=[output_img, table, status],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
