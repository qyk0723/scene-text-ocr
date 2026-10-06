"""生成论文图表与可视化样例（阶段 4）。

产出到 docs/figures/：
- recognition_metrics.png / detection_metrics.png  识别、检测性能柱状图（medium 全量）
- timing.png                耗时对比柱状图（small，对数刻度）
- model_compare_accuracy.png / model_compare_speed.png  medium vs small 精度、速度对比
- ablation.png / ablation_degraded.png / ablation_operators.png  预处理消融三图
- iou_sensitivity.png / wordline_compare.png  检测 IoU 敏感性与口径对照
- preprocess_compare.png    预处理前后对比
- sample_*.png              检测可视化样例（small 模型）
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

from src.visualize.draw import _load_font, draw_results
from src.evaluator.metrics_store import MissingMetric, by_iou, get, load, pct
from src.pipeline.ocr_pipeline import SceneTextOCR, imread_unicode
from src.preprocess.enhancer import ImageEnhancer

FIG_DIR = Path("docs/figures")
DATA = Path("data")

# ---- 图表数值的唯一来源（审计 P1-16）：全部从 metrics/*.json 读取，不得硬编码 ----
# 这些数字曾经以字面量形式写在绘图代码里，与结果产物没有程序化关联，已经漂移过。
_DET = load("detection")
_REC = load("recognition")
_LL = load("linelevel")
_ABL = load("ablation")
_TIMING = load("timing")
_MANUAL = load("manual")

# 检测指标口径：'deteval'（ICDAR2015 官方口径，**论文主口径**，do-not-care 不计入）
# 或 'legacy'（历史口径，全部 GT 计入，不可与已发表结果比较）。
# 论文按"两套都报、deteval 为主"处理；此开关决定**图表**用哪一套。
DETECTION_PROTOCOL = "deteval"

# medium 的 deteval 口径需要一次全量重跑才会存在（约 3.7 小时）。缺失时下面
# 会抛出 MissingMetric 并**打印这条命令**，而不是回退到 legacy 静默混用口径。
_MEDIUM_DET_RUN = (
    "python evaluate.py --task det --model medium "
    "--dump-pred data/results/pred_medium_det_500.jsonl "
    "--report data/results/eval_medium_det_deteval.md"
)


def _det(model: str, key: str) -> float:
    """取某模型某口径的检测指标，转成百分数。

    该口径尚未测量时抛 :class:`MissingMetric`，并给出需要运行的命令——
    **绝不回退到另一套口径**，否则图表会在无人察觉的情况下混用口径。
    """
    block = get(_DET, f"{model}.{DETECTION_PROTOCOL}")
    if block is None:
        raise MissingMetric(
            f"{model} 的 {DETECTION_PROTOCOL} 口径检测指标尚未测量。\n"
            f"    请先运行（约 3.7 小时）：\n        {_MEDIUM_DET_RUN}"
        )
    return pct(block[key])


def _abl(mode: str, config: str) -> float:
    """取某图像条件下某配置的字符准确率（百分数）。"""
    return pct(get(_ABL, f"{mode}.configs.{config}.char_acc"))

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


def make_recognition_chart():
    _h_bar(
        ["字符准确率", "行级准确率"],
        [pct(get(_REC, "medium.char_acc")), pct(get(_REC, "medium.line_acc"))],
        "识别性能（ICDAR2015 recognition/test，2074 张）",
        "准确率（%）",
        "{:.2f}%",
        "recognition_metrics.png",
        xlim=100,
    )


def make_detection_chart():
    _h_bar(
        ["精确率", "召回率", "F1"],
        [_det("medium", "precision"), _det("medium", "recall"), _det("medium", "f1")],
        f"检测性能（ICDAR2015 detection/test，500 张，{DETECTION_PROTOCOL} 口径）",
        "指标值（%）",
        "{:.2f}%",
        "detection_metrics.png",
        xlim=100,
    )


def make_timing_chart():
    _h_bar(
        ["预处理（整图）", "识别（单行图）", "检测（整图）"],
        [get(_MANUAL, "preprocess_whole_image_s.value"),
         get(_TIMING, "recognition_single_line_s.small"),
         get(_TIMING, "detection_whole_image_s.small")],
        "平均耗时对比（CPU，PP-OCRv6 small）",
        "耗时（秒，对数刻度）",
        "{:.3g}s",
        "timing.png",
        log=True,
    )


def make_metric_charts():
    """兼容旧调用：依次生成识别 / 检测 / 耗时三张图。"""
    make_recognition_chart()
    make_detection_chart()
    make_timing_chart()


def _grouped_style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=SECONDARY, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def make_model_accuracy_chart():
    """medium vs small 精度对比（依赖检测口径，故单独成函数）。"""
    categories = ["识别字符准确率", "检测 F1"]
    medium = [pct(get(_REC, "medium.char_acc")), _det("medium", "f1")]
    small = [pct(get(_REC, "small.char_acc")), _det("small", "f1")]
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


def make_model_speed_chart():
    """medium vs small 速度对比（只用 test.jpg 耗时，与检测口径无关）。"""
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(1)
    bars_m = ax.bar(x - width / 2, [get(_MANUAL, "test_jpg_end_to_end_s.medium.value")],
                    width, color=BLUE, label="medium")
    bars_s = ax.bar(x + width / 2, [get(_MANUAL, "test_jpg_end_to_end_s.small.value")],
                    width, color=ORANGE, label="small")
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


def make_model_comparison():
    """兼容旧调用：精度图 + 速度图。"""
    make_model_accuracy_chart()
    make_model_speed_chart()


def make_wordline_chart():
    """检测指标口径对照：词级 vs 行级 F1。"""
    categories = ["词级口径", "行级口径"]
    medium = [pct(by_iou(get(_LL, "medium.iou"), 0.5)["word"]["f1"]),
              pct(by_iou(get(_LL, "medium.iou"), 0.5)["line"]["f1"])]
    small = [pct(by_iou(get(_LL, "small.iou"), 0.5)["word"]["f1"]),
             pct(by_iou(get(_LL, "small.iou"), 0.5)["line"]["f1"])]
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(categories))
    width = 0.34
    bars1 = ax.bar(x - width / 2, medium, width, color=BLUE, label="medium")
    bars2 = ax.bar(x + width / 2, small, width, color=ORANGE, label="small")
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{b.get_height():.1f}%", ha="center", va="bottom",
                color=INK, fontsize=11,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, color=INK)
    ax.set_ylim(0, 60)
    ax.set_ylabel("检测 F1（%）", color=SECONDARY, fontsize=10)
    _grouped_style(ax)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_title(
        "检测指标口径对照（30 张子集，IoU 0.5）",
        color=INK, fontsize=13, fontweight="bold", loc="left", pad=12,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "wordline_compare.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def make_iou_sensitivity_chart():
    """检测 IoU 敏感性：词级 vs 行级 F1（medium，30 张）。"""
    categories = ["IoU 0.3", "IoU 0.5", "IoU 0.7"]
    _med_iou = get(_LL, "medium.iou")
    word_f1 = [pct(by_iou(_med_iou, i)["word"]["f1"]) for i in (0.3, 0.5, 0.7)]
    line_f1 = [pct(by_iou(_med_iou, i)["line"]["f1"]) for i in (0.3, 0.5, 0.7)]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(categories))
    width = 0.34
    bars1 = ax.bar(x - width / 2, word_f1, width, color=BLUE, label="词级口径")
    bars2 = ax.bar(x + width / 2, line_f1, width, color=ORANGE, label="行级口径")
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{b.get_height():.1f}%", ha="center", va="bottom",
                color=INK, fontsize=10,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, color=INK)
    ax.set_ylim(0, 70)
    ax.set_ylabel("检测 F1（%）", color=SECONDARY, fontsize=10)
    _grouped_style(ax)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_title(
        "检测 IoU 敏感性：词级 vs 行级（medium，30 张）",
        color=INK, fontsize=13, fontweight="bold", loc="left", pad=12,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "iou_sensitivity.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def make_operator_chart():
    """各算子在适用场景下的效果：基线 vs 算子开启。

    ⚠️ **不含「小字放大」**：该算子的旧消融（`ablation_small.md`，单行裁剪图上 ÷2 再放大到 400px）
    已被审计 P1-3 推翻——两个对照臂像素高度差约 15 倍，收益无法与"就是喂了张大图"分离；
    按整图 + 四臂 + 部署值 800 重做后，它只部分恢复损失、且不如直接还原原尺寸。
    把一个已被推翻的 +55.4 留在论文图里，正是这次审计一直在清理的"图文不一致"，
    所以这里只画设计有效的四个算子；小字放大的正确证据见
    `data/results/small_text_ablation.md`。
    """
    categories = ["去噪\n（噪声图）", "锐化\n（模糊图）", "CLAHE\n（低对比图）",
                  "倾斜校正\n（±12°图）"]
    baseline = [_abl("degraded", "无预处理"), _abl("blur", "无预处理"),
                _abl("lowcontrast", "无预处理"), _abl("skew", "无预处理")]
    with_op = [_abl("degraded", "仅去噪"), _abl("blur", "仅锐化"),
               _abl("lowcontrast", "仅CLAHE"), _abl("skew", "仅倾斜校正")]
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
        "各算子适用场景消融（200 张单行图，PP-OCRv6 small）\n"
        "注：「小字放大」旧结果已推翻、不在此图中，见 data/results/small_text_ablation.md",
        color=INK, fontsize=12, fontweight="bold", loc="left", pad=12,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "ablation_operators.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def make_ablation_chart():
    """预处理消融柱状图（按准确率降序）。"""
    _h_bar(
        ["无预处理", "仅去噪", "仅锐化", "仅CLAHE", "全开（默认）"],
        [_abl("clean", c) for c in ["无预处理", "仅去噪", "仅锐化", "仅CLAHE", "全开（默认）"]],
        "预处理消融（清晰图）：识别字符准确率（200 张单行图，PP-OCRv6 small）",
        "字符准确率（%）",
        "{:.2f}%",
        "ablation.png",
        xlim=100,
    )
    _h_bar(
        ["仅去噪", "无预处理", "仅CLAHE", "全开（默认）"],
        [_abl("degraded", c) for c in ["仅去噪", "无预处理", "仅CLAHE", "全开（默认）"]],
        "预处理消融（退化图）：识别字符准确率（200 张单行图，PP-OCRv6 small）",
        "字符准确率（%）",
        "{:.2f}%",
        "ablation_degraded.png",
        xlim=100,
    )


def make_preprocess_comparison():
    img = imread_unicode(DATA / "samples" / "test.jpg")
    # 必须显式开启算子：ImageEnhancer() 的默认是**全部关闭**，process() 是 no-op，
    # 之前这里写成 ImageEnhancer().process(img)，导致"预处理后"面板与"原图"逐像素相同。
    enhancer = ImageEnhancer(denoise=True, contrast=True, sharpen=True)
    enhanced = enhancer.process(img)
    if np.array_equal(img, enhanced):
        raise RuntimeError(
            "preprocess_compare 两张面板完全相同：预处理算子未生效，请检查 ImageEnhancer 开关"
        )

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
        img = imread_unicode(p)
        boxes, texts, _elapsed = ocr.run(str(p))
        annotated = draw_results(img, boxes, texts)
        cv2.imwrite(str(FIG_DIR / f"sample_{p.stem}.png"), annotated)
        print(f"  检出 {len(boxes)} 框")


def _run_chart(fn, label, outputs=()):
    """生成一张图；若所依赖的指标尚未测量，**跳过、删掉旧图并说明要跑什么**。

    删掉旧图是有意的：留着上一套口径的 PNG，会让"图"与"代码声明的口径"静默不一致
    ——这正是审计里反复出现的那类问题。宁可暂时没有图，也不要一张口径错误的图。
    """
    try:
        fn()
        print(f"{label} 完成")
    except MissingMetric as e:
        for name in outputs:
            stale = FIG_DIR / name
            if stale.is_file():
                stale.unlink()
                print(f"[删除] 旧图 {name}（口径已不一致）")
        print(f"[跳过] {label}：{e}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _run_chart(make_recognition_chart, "识别性能图", ["recognition_metrics.png"])
    _run_chart(make_timing_chart, "耗时对比图", ["timing.png"])
    _run_chart(make_detection_chart, "检测性能图", ["detection_metrics.png"])
    _run_chart(make_model_accuracy_chart, "模型精度对比图", ["model_compare_accuracy.png"])
    _run_chart(make_model_speed_chart, "模型速度对比图", ["model_compare_speed.png"])
    _run_chart(make_wordline_chart, "词级/行级口径对照图", ["wordline_compare.png"])
    _run_chart(make_iou_sensitivity_chart, "IoU 敏感性图", ["iou_sensitivity.png"])
    _run_chart(make_ablation_chart, "消融图", ["ablation.png", "ablation_degraded.png"])
    _run_chart(make_operator_chart, "算子适用域图", ["ablation_operators.png"])
    _run_chart(make_preprocess_comparison, "预处理对比图", ["preprocess_compare.png"])
    _run_chart(make_detection_samples, "检测可视化样例")


if __name__ == "__main__":
    main()
