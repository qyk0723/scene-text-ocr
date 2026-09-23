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


def predict(image: np.ndarray, preprocess: bool) -> Tuple[np.ndarray, str, str]:
    """处理一张上传图片，返回 (标注图, 文本列表, 耗时信息)。"""
    if image is None:
        return np.zeros((1, 1, 3), dtype=np.uint8), "", "未上传图片"

    # Gradio 传入 RGB，OCR / OpenCV 用 BGR
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    prep_elapsed = 0.0
    if preprocess:
        start = time.perf_counter()
        proc = _ENHANCER.process(img_bgr)
        prep_elapsed = time.perf_counter() - start
    else:
        proc = img_bgr

    boxes, texts, ocr_elapsed = _OCR.run(proc)

    annotated_bgr = draw_results(img_bgr, boxes, texts)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    if texts:
        text_block = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, start=1))
    else:
        text_block = "（未检测到文字）"

    info = (
        f"检出 {len(boxes)} 个文本框\n"
        f"预处理耗时：{prep_elapsed:.3f}s\n"
        f"识别耗时：{ocr_elapsed:.3f}s"
    )
    return annotated_rgb, text_block, info


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="面向生活场景的文字检测与识别") as demo:
        gr.Markdown("# 面向生活场景的文字检测与识别")
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="numpy", label="上传图片")
                preprocess = gr.Checkbox(
                    label="预处理（去噪 + 对比度增强 + 锐化）", value=True
                )
                btn = gr.Button("识别", variant="primary")
            with gr.Column():
                output_img = gr.Image(type="numpy", label="检测结果")
                info_box = gr.Textbox(label="耗时", lines=3)
        text_box = gr.Textbox(label="识别文本", lines=15)

        btn.click(
            predict,
            inputs=[input_img, preprocess],
            outputs=[output_img, text_box, info_box],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
