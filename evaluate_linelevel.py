"""行级口径检测评估：把词级 GT 合并成行级框后重算检测 P/R/F1。

用途：说明 ICDAR2015 检测指标偏低是「词级标注 vs 行级检测」的粒度错配，
而非检测质量差。对同一批图同时报「词级口径」与「行级口径」两组指标。

用法::

    python evaluate_linelevel.py --limit 30 --model medium
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from evaluate import match_boxes, parse_det_gt
from src.pipeline.ocr_pipeline import SceneTextOCR


def merge_to_lines(quads: List[np.ndarray], y_center_ratio: float = 0.5) -> List[np.ndarray]:
    """把词级四边形按 y 中心聚类成行，返回每行的贴合合并框。

    用 y 中心距离聚类（阈值 = 0.5 × 中位字高），容忍同行错落、
    不并相邻行；合并框 y 取 y 中心 ± 半字高，避免跨度膨胀。
    """
    words = []
    for q in quads:
        ys = q[:, 1]
        xs = q[:, 0]
        yc = float((ys.min() + ys.max()) / 2.0)
        h = float(ys.max() - ys.min())
        words.append({
            "q": q, "yc": yc, "h": h,
            "xmin": float(xs.min()), "xmax": float(xs.max()),
        })
    if not words:
        return []

    med_h = float(np.median([w["h"] for w in words]))
    if med_h <= 0:
        med_h = 1.0

    words.sort(key=lambda w: w["yc"])
    lines = []
    for w in words:
        placed = False
        for line in lines:
            if abs(w["yc"] - line["yc"]) < y_center_ratio * med_h:
                line["boxes"].append(w)
                n = len(line["boxes"])
                line["yc"] = (line["yc"] * (n - 1) + w["yc"]) / n
                line["xmin"] = min(line["xmin"], w["xmin"])
                line["xmax"] = max(line["xmax"], w["xmax"])
                line["ymin"] = min(line["ymin"], w["yc"] - 0.5 * w["h"])
                line["ymax"] = max(line["ymax"], w["yc"] + 0.5 * w["h"])
                placed = True
                break
        if not placed:
            lines.append({
                "boxes": [w], "yc": w["yc"],
                "xmin": w["xmin"], "xmax": w["xmax"],
                "ymin": w["yc"] - 0.5 * w["h"], "ymax": w["yc"] + 0.5 * w["h"],
            })

    merged = []
    for line in lines:
        merged.append(np.array([
            [line["xmin"], line["ymin"]],
            [line["xmax"], line["ymin"]],
            [line["xmax"], line["ymax"]],
            [line["xmin"], line["ymax"]],
        ], dtype=np.float32))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="行级口径检测评估")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--model", choices=["medium", "small"], default="medium")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    data_root = Path("data/icdar2015")
    imgs_dir = data_root / "detection" / "test" / "imgs"
    gt_dir = data_root / "detection" / "test" / "gt"
    img_paths = sorted(imgs_dir.glob("img_*.jpg"))[: args.limit]

    if args.model == "small":
        ocr = SceneTextOCR(
            det_model_name="PP-OCRv6_small_det",
            rec_model_name="PP-OCRv6_small_rec",
        )
    else:
        ocr = SceneTextOCR()

    gt_word = gt_line = pred = matched_word = matched_line = 0
    print(f"[{args.model}] 共 {len(img_paths)} 张图，IoU {args.iou}")

    for idx, img_path in enumerate(img_paths, start=1):
        gt_path = gt_dir / f"gt_{img_path.stem}.txt"
        if not gt_path.is_file():
            continue
        gt_words = parse_det_gt(gt_path)
        gt_lines = merge_to_lines(gt_words)

        boxes, _texts, _t = ocr.run(str(img_path))

        gt_word += len(gt_words)
        gt_line += len(gt_lines)
        pred += len(boxes)
        matched_word += match_boxes(gt_words, boxes, args.iou)
        matched_line += match_boxes(gt_lines, boxes, args.iou)

        if idx % 10 == 0 or idx == len(img_paths):
            print(f"  {idx}/{len(img_paths)} 累计 词级GT={gt_word} 行级GT={gt_line} Pred={pred}")

    def prf(matched, gt_total, pred_total):
        p = matched / pred_total if pred_total else 0.0
        r = matched / gt_total if gt_total else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    p_word, r_word, f1_word = prf(matched_word, gt_word, pred)
    p_line, r_line, f1_line = prf(matched_line, gt_line, pred)

    print("=" * 50)
    print(f"[{args.model}] 词级口径：P {p_word:.4f} / R {r_word:.4f} / F1 {f1_word:.4f}（GT {gt_word}）")
    print(f"[{args.model}] 行级口径：P {p_line:.4f} / R {r_line:.4f} / F1 {f1_line:.4f}（GT {gt_line}）")

    out = Path(f"data/results/linelevel_{args.model}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# 行级口径检测评估（{args.model}）\n\n"
        f"图片数：{len(img_paths)}（ICDAR2015 detection/test），IoU {args.iou}\n\n"
        f"| 口径 | 精确率 | 召回率 | F1 | GT 框数 |\n"
        f"| --- | --- | --- | --- | --- |\n"
        f"| 词级（原始标注） | {p_word:.4f} | {r_word:.4f} | {f1_word:.4f} | {gt_word} |\n"
        f"| 行级（合并后） | {p_line:.4f} | {r_line:.4f} | {f1_line:.4f} | {gt_line} |\n\n"
        f"说明：预测框不变，仅将词级 GT 按垂直重叠聚合成行级后重算匹配。\n",
        encoding="utf-8",
    )
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
