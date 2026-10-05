"""评估执行与报告生成。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np

from src.evaluator.metrics import levenshtein, match_boxes
from src.evaluator.parsing import parse_det_gt, parse_rec_gt
from src.pipeline.ocr_pipeline import SceneTextOCR


def evaluate_detection(ocr: SceneTextOCR, data_root: Path, iou_thresh: float, limit: int):
    imgs_dir = data_root / "detection" / "test" / "imgs"
    gt_dir = data_root / "detection" / "test" / "gt"
    img_paths = sorted(imgs_dir.glob("img_*.jpg"))
    if limit > 0:
        img_paths = img_paths[:limit]

    total_gt = total_pred = matched = 0
    times: List[float] = []
    print(f"[det] 共 {len(img_paths)} 张图，IoU 阈值 {iou_thresh}")

    for idx, img_path in enumerate(img_paths, start=1):
        gt_path = gt_dir / f"gt_{img_path.stem}.txt"
        if not gt_path.is_file():
            print(f"  skip {img_path.name}: 缺标注")
            continue
        gt_boxes = parse_det_gt(gt_path)

        start = time.perf_counter()
        boxes, _texts, elapsed = ocr.run(str(img_path))
        times.append(elapsed)

        m = match_boxes(gt_boxes, boxes, iou_thresh)
        total_gt += len(gt_boxes)
        total_pred += len(boxes)
        matched += m

        if idx % 20 == 0 or idx == len(img_paths):
            print(f"  {idx}/{len(img_paths)} 累计 GT={total_gt} Pred={total_pred} 匹配={matched}")

    precision = matched / total_pred if total_pred else 0.0
    recall = matched / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_time = float(np.mean(times)) if times else 0.0

    return {
        "task": "检测",
        "images": len(img_paths),
        "gt_boxes": total_gt,
        "pred_boxes": total_pred,
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_time": avg_time,
    }


def evaluate_recognition(ocr: SceneTextOCR, data_root: Path, limit: int, ignore_case: bool = True):
    img_dir = data_root / "recognition" / "test"
    gt_txt = data_root / "recognition" / "test.txt"
    pairs = parse_rec_gt(gt_txt, img_dir)
    if limit > 0:
        pairs = pairs[:limit]

    char_accs: List[float] = []
    line_ok = 0
    times: List[float] = []
    print(f"[rec] 共 {len(pairs)} 张单行图（ignore_case={ignore_case}）")

    for idx, (img_path, gt_text) in enumerate(pairs, start=1):
        if not img_path.is_file():
            print(f"  skip {img_path.name}: 文件缺失")
            continue

        start = time.perf_counter()
        _boxes, texts, elapsed = ocr.run(str(img_path))
        times.append(elapsed)
        pred = "".join(texts)

        if ignore_case:
            pred = pred.upper()
            gt_text = gt_text.upper()

        char_accs.append(1.0 - levenshtein(pred, gt_text) / max(len(pred), len(gt_text), 1))
        if pred == gt_text:
            line_ok += 1

        if idx % 100 == 0 or idx == len(pairs):
            print(f"  {idx}/{len(pairs)}")

    n = len(char_accs)
    char_acc = float(np.mean(char_accs)) if n else 0.0
    line_acc = line_ok / n if n else 0.0
    avg_time = float(np.mean(times)) if times else 0.0

    return {
        "task": "识别",
        "images": n,
        "char_acc": char_acc,
        "line_acc": line_acc,
        "avg_time": avg_time,
    }


def render_report(results: List[dict], model_label: str = "PP-OCRv6 medium（det + rec）") -> str:
    lines = ["# ICDAR2015 评估报告", ""]
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"模型：{model_label}")
    lines.append("")
    lines.append("| 任务 | 图片数 | 指标 | 平均耗时/图 |")
    lines.append("| --- | --- | --- | --- |")

    for r in results:
        if r["task"] == "检测":
            metric = (
                f"精确率 {r['precision']:.4f} / 召回率 {r['recall']:.4f} / "
                f"F1 {r['f1']:.4f}（GT {r['gt_boxes']} 框，Pred {r['pred_boxes']} 框，匹配 {r['matched']}）"
            )
        else:
            metric = (
                f"字符准确率 {r['char_acc']:.4f} / 行级准确率 {r['line_acc']:.4f}"
            )
        lines.append(
            f"| {r['task']} | {r['images']} | {metric} | {r['avg_time']:.3f}s |"
        )

    lines.append("")
    lines.append("说明：检测框匹配采用 IoU 阈值 0.5 贪心一对一；识别字符准确率按编辑距离计算，默认忽略大小写。")
    return "\n".join(lines)
