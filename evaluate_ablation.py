"""预处理消融实验：各预处理开关组合对识别准确率的影响。

用法::

    python evaluate_ablation.py [--limit 200] [--mode clean]

--mode 图像条件：clean 原图 / degraded 噪声图 / lowcontrast 低对比 /
blur 重模糊 / skew 倾斜 / small 小字图。每种模式跑对应预处理配置，
统计字符准确率 / 行级准确率 / 耗时（中位数·平均·p95），结果写入
data/results/ablation_<mode>.md。

耗时口径（2026-10-06 修正，见 docs/PROJECT_AUDIT.md 的 P1-6）：
- 计入**预处理本身**的开销（原实现只累计 OCR 耗时，于是算子越多反而"越快"）；
- 每个配置丢弃首张作为预热，但**仍计入准确率**，保证与历史准确率可比；
- 配置之间清空 OCR 结果缓存，避免缓存命中（elapsed≈0）被当成耗时统计进去。
- 报中位数与 p95，而不是只报平均——实测同一操作两次测量可差 35%。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.config import get_config
from src.evaluator import levenshtein, parse_rec_gt
from src.pipeline.ocr_pipeline import SceneTextOCR, imread_unicode
from src.preprocess.enhancer import ImageEnhancer

# 配置名 -> ImageEnhancer 构造参数（其余开关默认关）
#
# ⚠️ 命名注意（审计 P2-25）：这组"三个算子一起开"的配置曾叫「三算子组合（去噪+CLAHE+锐化）」，
# 两个词都是错的——它只开了 denoise+contrast+sharpen（8 个算子里还关着 5 个），
# 而"默认"在出厂配置里是**全关**。名字会直接进产物表格与图表，所以改成自解释的名字。
CONFIGS: Dict[str, dict] = {
    "无预处理": dict(denoise=False, contrast=False, sharpen=False),
    "仅去噪": dict(denoise=True, contrast=False, sharpen=False),
    "仅CLAHE": dict(denoise=False, contrast=True, sharpen=False),
    "仅锐化": dict(denoise=False, contrast=False, sharpen=True),
    "三算子组合（去噪+CLAHE+锐化）": dict(denoise=True, contrast=True, sharpen=True),
}

# 退化图消融用（去掉仅锐化——噪声图上锐化必然有害）
DEGRADED_CONFIGS: Dict[str, dict] = {
    "无预处理": dict(denoise=False, contrast=False, sharpen=False),
    "三算子组合（去噪+CLAHE+锐化）": dict(denoise=True, contrast=True, sharpen=True),
    "仅去噪": dict(denoise=True, contrast=False, sharpen=False),
    "仅CLAHE": dict(denoise=False, contrast=True, sharpen=False),
}

LOWCONTRAST_CONFIGS: Dict[str, dict] = {
    "无预处理": dict(denoise=False, contrast=False, sharpen=False),
    "仅CLAHE": dict(denoise=False, contrast=True, sharpen=False),
}

BLUR_CONFIGS: Dict[str, dict] = {
    "无预处理": dict(denoise=False, contrast=False, sharpen=False),
    "仅锐化": dict(denoise=False, contrast=False, sharpen=True),
}

SKEW_CONFIGS: Dict[str, dict] = {
    "无预处理": dict(denoise=False, contrast=False, sharpen=False),
    "仅倾斜校正": dict(denoise=False, contrast=False, sharpen=False, deskew=True),
}

SMALL_CONFIGS: Dict[str, dict] = {
    "无预处理": dict(denoise=False, contrast=False, sharpen=False),
    "仅放大": dict(
        denoise=False, contrast=False, sharpen=False,
        upscale=True, upscale_min_long_side=400,
    ),
}

MODES: Dict[str, Tuple[Dict[str, dict], str, str]] = {
    "clean": (CONFIGS, "原图", "ablation_results.md"),
    "degraded": (DEGRADED_CONFIGS, "退化图（模糊+噪声+低对比）", "ablation_degraded.md"),
    "lowcontrast": (LOWCONTRAST_CONFIGS, "低对比图（×0.45 -30）", "ablation_lowcontrast.md"),
    "blur": (BLUR_CONFIGS, "重模糊图（9×9 σ=2.5）", "ablation_blur.md"),
    "skew": (SKEW_CONFIGS, "倾斜图（±12°）", "ablation_skew.md"),
    "small": (SMALL_CONFIGS, "小字图（缩小 2 倍）", "ablation_small.md"),
}


def transform(img: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """按模式施加合成退化。"""
    if mode == "degraded":
        img = cv2.GaussianBlur(img, (5, 5), 1.2)
        img = cv2.convertScaleAbs(img, alpha=0.7, beta=-15)
        noise = rng.normal(0, 15, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if mode == "lowcontrast":
        return cv2.convertScaleAbs(img, alpha=0.45, beta=-30)
    if mode == "blur":
        return cv2.GaussianBlur(img, (9, 9), 2.5)
    if mode == "skew":
        h, w = img.shape[:2]
        angle = float(rng.uniform(-12, 12))
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            img, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    if mode == "small":
        h, w = img.shape[:2]
        return cv2.resize(
            img, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA
        )
    return img


def run_config(
    ocr: SceneTextOCR,
    inputs: List[Tuple[np.ndarray, str]],
    cfg: dict,
    warmup: int = 1,
) -> Dict[str, float]:
    """跑一个预处理配置，返回准确率与可信的耗时统计。

    返回 ``{"n","char_acc","line_acc","mean","median","p95"}``；耗时单位为秒/张，
    并且**包含预处理本身的开销**。
    """
    enhancer = ImageEnhancer(**cfg)
    ocr._cache.clear()  # 防止缓存命中（elapsed≈0）污染耗时统计

    char_accs: List[float] = []
    times: List[float] = []
    line_ok = 0
    n_boxes = 0

    for img, gt_text in inputs:
        t0 = time.perf_counter()
        proc = enhancer.process(img)
        prep_t = time.perf_counter() - t0

        boxes, texts, _scores, ocr_t = ocr.run_detailed(proc)
        n_boxes += len(boxes)

        pred = "".join(texts).upper()
        gt = gt_text.upper()
        char_accs.append(1.0 - levenshtein(pred, gt) / max(len(pred), len(gt), 1))
        if pred == gt:
            line_ok += 1
        times.append(prep_t + ocr_t)

    n = len(char_accs)
    timed = times[warmup:] if len(times) > warmup else times
    return {
        "n": float(n),
        "char_acc": sum(char_accs) / n if n else 0.0,
        "line_acc": line_ok / n if n else 0.0,
        "mean": float(np.mean(timed)) if timed else 0.0,
        "median": float(np.median(timed)) if timed else 0.0,
        "p95": float(np.percentile(timed, 95)) if timed else 0.0,
        # 每张图平均检出的文本框数。**这是耗时的主要驱动因素**：每个框都要跑一次识别，
        # 所以一个把检测"弄坏"的配置会因为框变少而显得很快——必须与准确率一起看。
        "avg_boxes": n_boxes / n if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="预处理消融实验")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--mode",
        choices=list(MODES),
        default="clean",
        help="图像条件：clean / degraded / lowcontrast / blur / skew / small",
    )
    parser.add_argument(
        "--out", default=None,
        help="输出 md 路径，默认 data/results/ablation_<mode>.md",
    )
    args = parser.parse_args()

    configs, mode_label, out_name = MODES[args.mode]

    data_root = Path(get_config().get("data", {}).get("root", "data/icdar2015"))
    img_dir = data_root / "recognition" / "test"
    pairs = parse_rec_gt(data_root / "recognition" / "test.txt", img_dir)[: args.limit]

    ocr = SceneTextOCR(
        det_model_name="PP-OCRv6_small_det",
        rec_model_name="PP-OCRv6_small_rec",
    )

    # 预载图像；固定种子，保证各配置看到同一张图
    inputs: List[Tuple[np.ndarray, str]] = []
    rng = np.random.default_rng(42)
    for img_path, gt_text in pairs:
        img = imread_unicode(img_path)
        if img is None:
            continue
        inputs.append((transform(img, args.mode, rng), gt_text))

    lines = ["# 预处理消融实验结果", ""]
    lines.append(f"图片数：{len(inputs)}（ICDAR2015 recognition/test 单行图，{mode_label}）")
    lines.append("模型：PP-OCRv6 small（det + rec），忽略大小写")
    lines.append("")
    lines.append(
        "耗时口径：**包含预处理本身**的秒/张；每配置丢弃首张作预热（仍计入准确率）；"
        "配置间清空结果缓存。报中位数而非平均值——实测同一操作两次测量可差 35%。"
    )
    lines.append("")
    lines.append(
        "> ⚠️ **耗时必须与「检出框数」一起看**：每个检出框都要跑一次识别，所以耗时主要由"
        "**检出框数**决定。一个把检测弄坏的配置会因为框变少而显得很快（例如清晰图下的"
        "「三算子组合」），把它解读为「算法更快」是错的。"
    )
    if args.mode == "degraded":
        lines.append("")
        lines.append("退化参数：GaussianBlur 5x5 σ=1.2 + 对比度×0.7 -15 + 高斯噪声 σ=15（种子 42）")
    lines.append("")
    lines.append("| 配置 | 字符准确率 | 行级准确率 | 检出框数/张 | 耗时中位数/张 | 平均 | p95 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    lines.append("")

    print(f"共 {len(inputs)} 张图 × {len(configs)} 配置（{mode_label}）")
    for name, cfg in configs.items():
        r = run_config(ocr, inputs, cfg)
        print(
            f"{name}: char={r['char_acc']:.4f} line={r['line_acc']:.4f} "
            f"boxes={r['avg_boxes']:.2f} median={r['median']:.3f}s "
            f"mean={r['mean']:.3f}s p95={r['p95']:.3f}s"
        )
        lines.append(
            f"| {name} | {r['char_acc']:.4f} | {r['line_acc']:.4f} | {r['avg_boxes']:.2f} | "
            f"{r['median']:.3f}s | {r['mean']:.3f}s | {r['p95']:.3f}s |"
        )

    out = Path(args.out) if args.out else Path("data/results") / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
