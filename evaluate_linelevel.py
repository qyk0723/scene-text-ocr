"""行级口径检测评估：把词级 GT 合并成行级框后重算检测 P/R/F1。

对同一批图同时报「词级口径」与「行级口径」两组指标，并在
IoU 0.3 / 0.5 / 0.7 三档阈值下做敏感性分析，用于定位检测指标
偏低的成因（粒度错位 vs 难例漏检）。

用法::

    python evaluate_linelevel.py --limit 30 --model medium
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import get_config
from src.evaluator import match_boxes, merge_to_lines, parse_det_gt
from src.pipeline.ocr_pipeline import SceneTextOCR


def main() -> None:
    parser = argparse.ArgumentParser(description="行级口径检测评估")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--model", choices=["medium", "small"], default="medium")
    args = parser.parse_args()

    data_root = Path(get_config().get("data", {}).get("root", "data/icdar2015"))
    imgs_dir = data_root / "detection" / "test" / "imgs"
    gt_dir = data_root / "detection" / "test" / "gt"
    img_paths = sorted(imgs_dir.glob("img_*.jpg"))[: args.limit]

    if args.model == "small":
        ocr = SceneTextOCR(
            det_model_name="PP-OCRv6_small_det",
            rec_model_name="PP-OCRv6_small_rec",
        )
    else:
        ocr = SceneTextOCR(
            det_model_name="PP-OCRv6_medium_det",
            rec_model_name="PP-OCRv6_medium_rec",
        )

    ious = [0.3, 0.5, 0.7]
    gt_word = gt_line = pred = 0
    matched_word = {i: 0 for i in ious}
    matched_line = {i: 0 for i in ious}
    print(f"[{args.model}] 共 {len(img_paths)} 张图，IoU {ious}")

    for idx, img_path in enumerate(img_paths, start=1):
        gt_path = gt_dir / f"gt_{img_path.stem}.txt"
        if not gt_path.is_file():
            continue
        gt_words = parse_det_gt(gt_path)
        gt_lines = merge_to_lines(gt_words)

        boxes = ocr.detect(str(img_path))

        gt_word += len(gt_words)
        gt_line += len(gt_lines)
        pred += len(boxes)
        for i in ious:
            matched_word[i] += match_boxes(gt_words, boxes, i)
            matched_line[i] += match_boxes(gt_lines, boxes, i)

        if idx % 10 == 0 or idx == len(img_paths):
            print(f"  {idx}/{len(img_paths)} 累计 词级GT={gt_word} 行级GT={gt_line} Pred={pred}")

    def prf(matched, gt_total, pred_total):
        p = matched / pred_total if pred_total else 0.0
        r = matched / gt_total if gt_total else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    print("=" * 60)
    print(f"[{args.model}] 词级GT={gt_word} 行级GT={gt_line} Pred={pred}")
    table_rows = []
    for i in ious:
        pw, rw, fw = prf(matched_word[i], gt_word, pred)
        pl, rl, fl = prf(matched_line[i], gt_line, pred)
        print(f"IoU {i}: 词级 P {pw:.4f}/R {rw:.4f}/F1 {fw:.4f} | "
              f"行级 P {pl:.4f}/R {rl:.4f}/F1 {fl:.4f}")
        table_rows.append(f"| {i} | {pw:.4f} | {rw:.4f} | {fw:.4f} | {pl:.4f} | {rl:.4f} | {fl:.4f} |")

    out = Path(f"data/results/linelevel_{args.model}_iou.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# 行级口径检测评估 + IoU 敏感性（{args.model}）\n\n"
        f"图片数：{len(img_paths)}（ICDAR2015 detection/test）\n\n"
        f"| IoU | 词级 P | 词级 R | 词级 F1 | 行级 P | 行级 R | 行级 F1 |\n"
        f"| --- | --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(table_rows)
        + "\n\n说明：预测框不变，词级 GT 按 y 中心聚合成行级后重算匹配。\n",
        encoding="utf-8",
    )
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
