"""生成论文图表与可视化样例（阶段 4）。

产出到 docs/figures/：
- recognition_metrics.png  识别性能柱状图
- detection_metrics.png    检测性能柱状图
- timing.png               耗时对比柱状图（对数刻度）
- preprocess_compare.png   预处理前后对比
- sample_*.png             检测可视化样例
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from app import _load_font, draw_results
from src.pipeline.ocr_pipeline import SceneTextOCR
from src.preprocess.enhancer import ImageEnhancer

FIG_DIR = Path("docs/figures")
DATA = Path("data")

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"


def _h_bar(labels, values, title, xlabel, fmt, out_name, log=False, xlim=None):
    fig, ax = plt.subplots(figsize=(7.2, 1.1 * len(labels) + 1.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(labels))

    if log:
        ax.set_xscale("log")
        left = 0.03
        ax.barh(y, values, height=0.55, color=BLUE, left=left)
        ax.set_xlim(left, xlim if xlim else 60)
        for yi, v in zip(y, values):
            ax.text(left + v, yi, f" {fmt.format(v)}", va="center", ha="left",
                    color=INK, fontsize=11)
    else:
        ax.barh(y, values, height=0.55, color=BLUE)
        ax.set_xlim(0, xlim if xlim else max(values) * 1.15)
        for yi, v in zip(y, values):
            ax.text(v, yi, f" {fmt.format(v)}", va="center", ha="left",
                    color=INK, fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, color=INK)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=SECONDARY, labelsize=10)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=SECONDARY, fontsize=10)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=12)

    plt.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def make_metric_charts():
    _h_bar(
        ["字符准确率", "行级准确率"],
        [82.89, 71.50],
        "识别性能（ICDAR2015 recognition/test，2074 张）",
        "准确率（%）",
        "{:.2f}%",
        "recognition_metrics.png",
        xlim=100,
    )
    _h_bar(
        ["精确率", "召回率", "F1"],
        [61.86, 26.23, 36.84],
        "检测性能（ICDAR2015 detection/test，500 张）",
        "指标值（%）",
        "{:.2f}%",
        "detection_metrics.png",
        xlim=100,
    )
    _h_bar(
        ["预处理（整图）", "识别（单行图）", "检测 + 识别（整图）"],
        [0.10, 1.757, 27.13],
        "平均耗时对比（CPU）",
        "耗时（秒，对数刻度）",
        "{:.3g}s",
        "timing.png",
        log=True,
    )


def make_preprocess_comparison():
    img = cv2.imread(str(DATA / "samples" / "test.jpg"))
    enhanced = ImageEnhancer().process(img)

    def to_rgb(bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def fit_height(arr, target_h):
        h, w = arr.shape[:2]
        scale = target_h / h
        return cv2.resize(arr, (int(w * scale), target_h))

    target_h = 560
    left = fit_height(to_rgb(img), target_h)
    right = fit_height(to_rgb(enhanced), target_h)

    gap = 30
    h = left.shape[0]
    canvas = np.ones((h, left.shape[1] + gap + right.shape[1], 3), dtype=np.uint8) * 255
    canvas[:, : left.shape[1]] = left
    canvas[:, left.shape[1] + gap :] = right

    pil = PILImage.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font = _load_font(26)
    if font is not None:
        draw.text((10, 8), "原图", font=font, fill=(0, 0, 0))
        draw.text((left.shape[1] + gap + 10, 8), "预处理后", font=font, fill=(0, 0, 0))
    pil.save(FIG_DIR / "preprocess_compare.png")


def make_detection_samples():
    ocr = SceneTextOCR()
    imgs = [DATA / "samples" / "test.jpg"]
    icdar = sorted((DATA / "icdar2015" / "detection" / "test" / "imgs").glob("img_*.jpg"))
    if len(icdar) >= 3:
        picks = [0, len(icdar) // 2, len(icdar) - 1]
        imgs += [icdar[i] for i in picks]

    for p in imgs:
        print(f"OCR {p.name} ...")
        img = cv2.imread(str(p))
        boxes, texts, _elapsed = ocr.run(str(p))
        annotated = draw_results(img, boxes, texts)
        cv2.imwrite(str(FIG_DIR / f"sample_{p.stem}.png"), annotated)
        print(f"  检出 {len(boxes)} 框")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_metric_charts()
    print("图表 4/5/6 完成")
    make_preprocess_comparison()
    print("图表 3 完成")
    make_detection_samples()
    print("图表 2 完成")


if __name__ == "__main__":
    main()
