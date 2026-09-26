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
ORANGE = "#eb6834"


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
        [0.10, 0.162, 3.86],
        "平均耗时对比（CPU，PP-OCRv6 small）",
        "耗时（秒，对数刻度）",
        "{:.3g}s",
        "timing.png",
        log=True,
    )


def _grouped_style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=SECONDARY, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def make_model_comparison():
    """medium vs small 对比图：精度 + 速度。"""
    # 精度对比（分组柱状图，全量口径）
    categories = ["识别字符准确率", "检测 F1"]
    medium = [82.89, 36.84]
    small = [75.86, 32.70]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(categories))
    width = 0.34
    bars_m = ax.bar(x - width / 2, medium, width, color=BLUE, label="medium")
    bars_s = ax.bar(x + width / 2, small, width, color=ORANGE, label="small")
    for bars in (bars_m, bars_s):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{b.get_height():.1f}%", ha="center", va="bottom",
                color=INK, fontsize=10,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, color=INK)
    ax.set_ylim(0, 100)
    ax.set_ylabel("数值（%）", color=SECONDARY, fontsize=10)
    _grouped_style(ax)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_title(
        "模型精度对比（识别 2074 张 / 检测 500 张，全量）",
        color=INK, fontsize=13, fontweight="bold", loc="left", pad=12,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "model_compare_accuracy.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # 速度对比（test.jpg 整图耗时）
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(1)
    bars_m = ax.bar(x - width / 2, [115.6], width, color=BLUE, label="medium")
    bars_s = ax.bar(x + width / 2, [21.7], width, color=ORANGE, label="small")
    for bars in (bars_m, bars_s):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{b.get_height():.1f}s", ha="center", va="bottom",
                color=INK, fontsize=11,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(["整图识别耗时（test.jpg，39 行）"], fontsize=11, color=INK)
    ax.set_ylim(0, 135)
    ax.set_ylabel("耗时（秒）", color=SECONDARY, fontsize=10)
    _grouped_style(ax)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_title(
        "整图识别速度对比（CPU）",
        color=INK, fontsize=13, fontweight="bold", loc="left", pad=12,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "model_compare_speed.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def make_operator_chart():
    """各算子在适用场景下的效果：基线 vs 算子开启。"""
    categories = ["去噪\n（噪声图）", "锐化\n（模糊图）", "CLAHE\n（低对比图）",
                  "倾斜校正\n（±12°图）", "小字放大\n（小字图）"]
    baseline = [17.51, 25.58, 68.04, 71.40, 32.07]
    with_op = [33.16, 29.32, 68.13, 24.81, 76.43]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(categories))
    width = 0.34
    bars1 = ax.bar(x - width / 2, baseline, width, color=BLUE, label="无预处理")
    bars2 = ax.bar(x + width / 2, with_op, width, color=ORANGE, label="算子开启")
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{b.get_height():.1f}", ha="center", va="bottom",
                color=INK, fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, color=INK)
    ax.set_ylim(0, 100)
    ax.set_ylabel("字符准确率（%）", color=SECONDARY, fontsize=10)
    _grouped_style(ax)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_title(
        "各算子适用场景消融（200 张单行图，PP-OCRv6 small）",
        color=INK, fontsize=13, fontweight="bold", loc="left", pad=12,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "ablation_operators.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def make_ablation_chart():
    """预处理消融柱状图（按准确率降序）。"""
    _h_bar(
        ["无预处理", "仅去噪", "仅锐化", "仅CLAHE", "全开（默认）"],
        [73.51, 72.42, 67.31, 60.01, 53.65],
        "预处理消融（清晰图）：识别字符准确率（200 张单行图，PP-OCRv6 small）",
        "字符准确率（%）",
        "{:.2f}%",
        "ablation.png",
        xlim=100,
    )
    _h_bar(
        ["仅去噪", "无预处理", "仅CLAHE", "全开（默认）"],
        [33.16, 17.51, 17.11, 14.69],
        "预处理消融（退化图）：识别字符准确率（200 张单行图，PP-OCRv6 small）",
        "字符准确率（%）",
        "{:.2f}%",
        "ablation_degraded.png",
        xlim=100,
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
        draw.text(
            (left.shape[1] + gap + 10, 8),
            "预处理后（去噪+CLAHE+锐化）",
            font=font,
            fill=(0, 0, 0),
        )
    pil.save(FIG_DIR / "preprocess_compare.png")


def make_detection_samples():
    # 与部署系统一致：PP-OCRv6 small
    ocr = SceneTextOCR(
        det_model_name="PP-OCRv6_small_det",
        rec_model_name="PP-OCRv6_small_rec",
    )
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
    make_model_comparison()
    print("模型对比图完成")
    make_ablation_chart()
    print("消融图完成")
    make_operator_chart()
    print("算子适用域图完成")
    make_preprocess_comparison()
    print("图表 3 完成")
    make_detection_samples()
    print("图表 2 完成")


if __name__ == "__main__":
    main()
