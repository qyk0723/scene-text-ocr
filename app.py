"""Gradio 图形界面：上传图片，检测并识别文字，框出结果。

运行::

    python app.py

启动后在浏览器打开终端打印的本地地址（默认 http://127.0.0.1:7860）。
"""

from __future__ import annotations

import base64
import html
import os
import time
from typing import List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.pipeline.ocr_pipeline import SceneTextOCR
from src.preprocess.enhancer import ImageEnhancer

# 全局单例：模型只加载一次（small 模型：整图 ~22s，精度代价见 docs/evaluation_report.md）
_OCR = SceneTextOCR(
    det_model_name="PP-OCRv6_small_det",
    rec_model_name="PP-OCRv6_small_rec",
)


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


def _status_html(n_boxes: int, ocr_elapsed: float, prep_elapsed: float) -> str:
    style = (
        "margin-top:4px;height:36px;display:flex;align-items:center;box-sizing:border-box;"
        "padding:0 16px;background:#eef3ff;border:1px solid #dbe4ff;border-radius:10px;"
        "color:#374151;font-size:.85rem;"
    )
    return (
        f'<div style="{style}">模型 PP-OCRv6 &nbsp;|&nbsp; 检测框数 {n_boxes} '
        f"&nbsp;|&nbsp; 识别耗时 {ocr_elapsed:.1f}s &nbsp;|&nbsp; 预处理 {prep_elapsed:.1f}s</div>"
    )


def _img_to_data_uri(rgb: np.ndarray) -> str:
    """RGB 数组 -> PNG base64 data URI。"""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/png;base64,{b64}"


# 放大 / 复制的内联 JS（gr.HTML 经 innerHTML 注入，<script> 不执行，用内联 onclick）
_ZOOM_JS = "var o=document.getElementById('zoom-overlay');if(!o){o=document.createElement('div');o.id='zoom-overlay';o.style.cssText='position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:9999;display:flex;align-items:center;justify-content:center;cursor:zoom-out;';o.onclick=function(){this.style.display='none';};var img=document.createElement('img');img.src=document.getElementById('result-img').src;img.style.cssText='max-width:90%;max-height:90%;';o.appendChild(img);document.body.appendChild(o);}o.style.display='flex';"
_COPY_JS = (
    "var i=document.getElementById('result-img');"
    "var p=i.src.split(',');"
    "var b=atob(p[1]);"
    "var a=new Uint8Array(b.length);"
    "for(var j=0;j<b.length;j++)a[j]=b.charCodeAt(j);"
    "var blob=new Blob([a],{type:'image/png'});"
    "navigator.clipboard.write([new ClipboardItem({'image/png':blob})])"
    ".then(function(){alert('已复制到剪贴板')})"
    ".catch(function(e){alert('复制失败：'+e)})"
)

_ZOOM_SVG = '🔍'
_COPY_SVG = '📋'
_BTN_FONT = 'font-size:14px;line-height:1;'

_BTN_STYLE = (
    "width:30px;height:30px;background:#ffffff;border:1px solid #cbd5e1;"
    "border-radius:8px;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:14px;"
)

_RESULT_AREA_STYLE = (
    "height:140px;background:#f7f8fa;border-radius:8px;"
)
_TEXT_AREA_STYLE = (
    "height:500px;overflow-y:auto;width:100%;box-sizing:border-box;"
    "border:1px solid #eef0f4;border-radius:10px;"
)


def _result_placeholder() -> str:
    """识别前的占位框：与结果区同高，保持界面稳定。"""
    return (
        f'<div style="{_RESULT_AREA_STYLE}display:flex;align-items:center;'
        'justify-content:center;color:#9ca3af;">识别结果将显示在这里</div>'
    )


def _result_html(data_uri: str) -> str:
    """检测结果图（占满区域、object-fit 不溢出）+ 右下角放大/复制图标按钮。"""
    return (
        f'<div style="position:relative;{_RESULT_AREA_STYLE}display:flex;'
        'align-items:center;justify-content:center;overflow:hidden;">'
        f'<img id="result-img" src="{data_uri}" '
        'style="width:100%;height:100%;object-fit:contain;">'
        '<div style="position:absolute;bottom:10px;right:10px;display:flex;gap:8px;">'
        f'<button onclick="{_ZOOM_JS}" title="放大" style="{_BTN_STYLE}">{_ZOOM_SVG}</button>'
        f'<button onclick="{_COPY_JS}" title="复制" style="{_BTN_STYLE}">{_COPY_SVG}</button>'
        '</div></div>'
    )


def _text_placeholder() -> str:
    return (
        f'<div style="{_TEXT_AREA_STYLE}display:flex;align-items:center;'
        'justify-content:center;color:#9ca3af;">识别文字将显示在这里</div>'
    )


def _text_html(texts: List[str], scores: List[float]) -> str:
    """识别文字逐行列表：灰色序号 + 深色文字 + 行尾灰色置信度。"""
    if not texts:
        return (
            f'<div style="{_TEXT_AREA_STYLE}display:flex;align-items:center;'
            'justify-content:center;color:#9ca3af;">（未检测到文字）</div>'
        )
    rows = []
    for i, (t, s) in enumerate(zip(texts, scores), start=1):
        rows.append(
            '<div style="padding:7px 10px;border-bottom:1px solid #f0f0f0;'
            'display:flex;justify-content:space-between;align-items:baseline;">'
            f'<span><span style="color:#9ca3af;display:inline-block;width:30px;">{i}</span>'
            f'<span style="color:#1f2937;">{html.escape(t)}</span></span>'
            f'<span style="color:#9ca3af;font-size:.8rem;">{s:.4f}</span>'
            '</div>'
        )
    return f'<div style="{_TEXT_AREA_STYLE}">{"".join(rows)}</div>'


def predict(
    image: np.ndarray, use_denoise: bool, use_sharpen: bool, use_upscale: bool
) -> Tuple[str, str, str]:
    """处理一张上传图片，返回 (结果图 HTML, 识别文字 HTML, 状态栏 HTML)。"""
    if image is None:
        return _result_placeholder(), _text_placeholder(), _status_html(0, 0.0, 0.0)

    try:
        # Gradio 传入 RGB，OCR / OpenCV 用 BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        prep_elapsed = 0.0
        if use_denoise or use_sharpen or use_upscale:
            start = time.perf_counter()
            # 按勾选组合启用对应预处理（CLAHE/倾斜校正实测无益不进 UI）
            proc = ImageEnhancer(
                denoise=use_denoise,
                sharpen=use_sharpen,
                upscale=use_upscale,
                contrast=False,
            ).process(img_bgr)
            prep_elapsed = time.perf_counter() - start
        else:
            proc = img_bgr

        boxes, texts, scores, ocr_elapsed = _OCR.run_detailed(proc)

        annotated_bgr = draw_results(img_bgr, boxes, texts)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        data_uri = _img_to_data_uri(annotated_rgb)

        return _result_html(data_uri), _text_html(texts, scores), _status_html(
            len(boxes), ocr_elapsed, prep_elapsed
        )
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print("PREDICT ERROR:", err, flush=True)
        return (
            f'<div style="{_RESULT_AREA_STYLE}display:flex;align-items:center;'
            f'justify-content:center;color:#dc2626;padding:10px;">识别出错：{html.escape(str(e))}</div>',
            f'<div style="{_TEXT_AREA_STYLE}display:flex;align-items:center;'
            f'justify-content:center;color:#dc2626;">（出错了，见左侧）</div>',
            _status_html(0, 0.0, 0.0),
        )


_THEME = gr.themes.Soft(
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

_CSS = """
.gradio-container { max-width: 100% !important; }
.gradio-container .wrap, .gradio-container main.contain {
    max-width: 100% !important;
    width: 100% !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}
footer { display: none !important; }
.main-row { gap: 16px !important; flex-wrap: nowrap !important; flex-direction: row !important; }
.main-row > * { min-width: 0 !important; }
.card {
    background: #ffffff;
    border: 1px solid #eef0f4;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
    padding: 12px;
}
#upload-box { border: 2px dashed #4F6EF7 !important; border-radius: 12px !important; padding: 6px; background: #fafbff !important; }
.primary-btn button {
    background: #4F6EF7 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    height: 52px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100%;
}
.primary-btn button:hover { background: #3f5be0 !important; }
.gradio-container .form { margin-top: 2px !important; margin-bottom: 2px !important; }
.gradio-container .form .block.padded { padding: 2px 12px !important; min-height: 0 !important; }
.gradio-container .form { height: 44px !important; overflow: hidden; }
.upload-container .wrap, .upload-container .or {
    font-size: 0.85rem !important;
}
#upload-box .upload-container span, #upload-box .upload-container p {
    font-size: 0.85rem !important;
}
.right-col { display: flex !important; flex-direction: column !important; overflow: hidden !important; }
.right-col > .block:first-child { flex: 0 0 auto !important; }
.right-col > .block:last-child { flex: 1 1 auto !important; min-height: 0 !important; overflow: hidden !important; }
.right-col > .block:last-child .wrap,
.right-col > .block:last-child .html-container,
.right-col > .block:last-child .prose {
    height: 100% !important;
    overflow: hidden !important;
    max-height: 100% !important;
}
.right-col > .block:last-child .prose > div {
    overflow-y: auto !important;
}
.prep-row input[type="checkbox"] {
    appearance: none !important;
    -webkit-appearance: none !important;
    width: 16px !important;
    height: 16px !important;
    border: 1.5px solid #4F6EF7 !important;
    border-radius: 4px !important;
    background: #ffffff !important;
    cursor: pointer;
    position: relative;
}
.prep-row input[type="checkbox"]:checked {
    background: #4F6EF7 !important;
}
.prep-row input[type="checkbox"]:checked::after {
    content: "";
    position: absolute;
    left: 4px;
    top: 1px;
    width: 4px;
    height: 8px;
    border: solid #ffffff;
    border-width: 0 2px 2px 0;
    transform: rotate(45deg);
}
.prep-tip { position: relative; display: inline-flex; align-items: center; height: 100%; }
.prep-tip .tip-q { cursor: help; color: #6b7280; font-size: .8rem; border: 1px solid #cbd5e1; border-radius: 50%; width: 18px; height: 18px; display: inline-flex; align-items: center; justify-content: center; line-height: 1; background: #ffffff; }
.prep-tip .tip-text { display: none; position: absolute; bottom: 140%; right: 0; background: #1f2937; color: #ffffff; padding: 8px 12px; border-radius: 8px; font-size: .75rem; white-space: nowrap; z-index: 60; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2); }
.prep-tip:hover .tip-text { display: block; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="场景文字检测与识别系统") as demo:
        gr.HTML(f"<style>{_CSS}</style>")
        # 顶部标题栏（约 60px）
        gr.HTML(
            """
            <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:2px 4px 6px;">
                <h1 style="color:#1f2937;margin:0;font-size:28px;font-weight:700;">
                    场景文字检测与识别系统
                </h1>
                <span style="color:#6b7280;font-size:15px;">基于 PaddleOCR PP-OCRv6</span>
            </div>
            """
        )

        with gr.Row(elem_classes="main-row", equal_height=True):
            # 左栏 37%：上传 / 预处理 / 按钮 / 检测结果
            with gr.Column(scale=37, elem_classes="card left-col"):
                # 不传 sources，保留 Gradio 默认来源（上传 / 摄像头 / 剪贴板）
                input_img = gr.Image(
                    type="numpy", label="上传图片", height=150, elem_id="upload-box"
                )
                with gr.Row(equal_height=True, elem_classes="prep-row"):
                    use_denoise = gr.Checkbox(label="去噪", value=False, scale=1)
                    use_sharpen = gr.Checkbox(label="锐化", value=False, scale=1)
                    use_upscale = gr.Checkbox(label="小字放大", value=False, scale=1)
                    gr.HTML(
                        '<div class="prep-tip"><span class="tip-q">?</span>'
                        '<div class="tip-text">去噪→噪点多/低光 · 锐化→模糊/失焦 · '
                        '小字放大→文字偏小；清晰图建议全不勾</div></div>',
                        scale=0,
                        min_width=28,
                    )
                btn = gr.Button("开始识别", variant="primary", elem_classes="primary-btn")
                gr.HTML(
                    '<div style="font-size:.9rem;font-weight:600;color:#374151;'
                    'margin-top:14px;margin-bottom:8px;">检测结果</div>'
                )
                result_box = gr.HTML(value=_result_placeholder(), sanitize_html=False)

            # 右栏 63%：识别文字列表
            with gr.Column(scale=63, elem_classes="card right-col"):
                gr.HTML(
                    '<div style="font-size:.9rem;font-weight:600;color:#374151;'
                    'margin-bottom:8px;">识别文字</div>'
                )
                text_box = gr.HTML(value=_text_placeholder(), sanitize_html=False)

        # 底部状态栏（约 44px）
        status = gr.HTML(_status_html(0, 0.0, 0.0), sanitize_html=False)

        btn.click(
            predict,
            inputs=[input_img, use_denoise, use_sharpen, use_upscale],
            outputs=[result_box, text_box, status],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=_THEME,
        css=_CSS,
    )
