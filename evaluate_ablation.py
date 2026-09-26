"""预处理消融实验：各预处理开关组合对识别准确率的影响。

用法::

    python evaluate_ablation.py [--limit 200]

对 ICDAR2015 recognition/test 的单行图，分别用 5 种预处理配置跑完整
OCR，统计字符准确率 / 行级准确率 / 平均耗时，结果写入
data/results/ablation_results.md。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from evaluate import levenshtein, parse_rec_gt
from src.pipeline.ocr_pipeline import SceneTextOCR
from src.preprocess.enhancer import ImageEnhancer

# 配置名 -> ImageEnhancer 构造参数（其余开关默认关）
CONFIGS: Dict[str, dict] = {
    "无预处理": dict(denoise=False, contrast=False, sharpen=False),
    "仅去噪": dict(denoise=True, contrast=False, sharpen=False),
    "仅CLAHE": dict(denoise=False, contrast=True, sharpen=False),
    "仅锐化": dict(denoise=False, contrast=False, sharpen=True),
    "全开（默认）": dict(denoise=True, contrast=True, sharpen=True),
}

# 退化图消融用（去掉仅锐化——噪声图上锐化必然有害）
DEGRADED_CONFIGS: Dict[str, dict] = {
    "无预处理": dict(denoise=False, contrast=False, sharpen=False),
    "全开（默认）": dict(denoise=True, contrast=True, sharpen=True),
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
) -> Tuple[float, float, float]:
    enhancer = ImageEnhancer(**cfg)
    char_accs: List[float] = []
    line_ok = 0
    times: List[float] = []

    for img, gt_text in inputs:
        proc = enhancer.process(img)
        _boxes, texts, _scores, t = ocr.run_detailed(proc)
        pred = "".join(texts).upper()
        gt = gt_text.upper()
        char_accs.append(1.0 - levenshtein(pred, gt) / max(len(pred), len(gt), 1))
        if pred == gt:
            line_ok += 1
        times.append(t)

    n = len(char_accs)
    char_acc = sum(char_accs) / n if n else 0.0
    line_acc = line_ok / n if n else 0.0
    avg_t = sum(times) / n if n else 0.0
    return char_acc, line_acc, avg_t


def main() -> None:
    parser = argparse.ArgumentParser(description="预处理消融实验")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--mode",
        choices=list(MODES),
        default="clean",
        help="图像条件：clean / degraded / lowcontrast / blur / skew / small",
    )
    args = parser.parse_args()

    configs, mode_label, out_name = MODES[args.mode]

    data_root = Path("data/icdar2015")
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
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        inputs.append((transform(img, args.mode, rng), gt_text))

    lines = ["# 预处理消融实验结果", ""]
    lines.append(f"图片数：{len(inputs)}（ICDAR2015 recognition/test 单行图，{mode_label}）")
    lines.append("模型：PP-OCRv6 small（det + rec），忽略大小写")
    if args.mode == "degraded":
        lines.append("退化参数：GaussianBlur 5x5 σ=1.2 + 对比度×0.7 -15 + 高斯噪声 σ=15（种子 42）")
    lines.append("")
    lines.append("| 配置 | 字符准确率 | 行级准确率 | 平均耗时/张 |")
    lines.append("| --- | --- | --- | --- |")

    print(f"共 {len(inputs)} 张图 × {len(configs)} 配置（{mode_label}）")
    for name, cfg in configs.items():
        char_acc, line_acc, avg_t = run_config(ocr, inputs, cfg)
        print(f"{name}: char={char_acc:.4f} line={line_acc:.4f} avg={avg_t:.3f}s")
        lines.append(f"| {name} | {char_acc:.4f} | {line_acc:.4f} | {avg_t:.3f}s |")

    out = Path("data/results") / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
