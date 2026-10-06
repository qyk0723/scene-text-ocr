"""评估执行与报告生成。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import numpy as np

from src.evaluator.metrics import (
    levenshtein,
    match_boxes,
    match_boxes_dnc,
    prf_from_counts,
)
from src.evaluator.parsing import parse_det_gt_flagged, parse_rec_gt
from src.pipeline.ocr_pipeline import SceneTextOCR


def evaluate_detection(
    ocr: SceneTextOCR,
    data_root: Path,
    iou_thresh: float,
    limit: int,
    sample_seed: int = None,
    dump_pred=None,
):
    """检测评估。**一次推理同时算出两套口径**：

    - **legacy（现状口径）**：全部 GT 框计入召回分母，命中任何 GT 都算 TP。
      这是仓库历史结果使用的口径，保留以保持可比与可复现。
    - **deteval（ICDAR2015 官方口径）**：转写为 ``###`` 的 GT 视为 do-not-care，
      不计入召回分母，命中它的预测被忽略（既非 TP 也非 FP）。

    ``sample_seed`` 非空且 ``limit`` 小于全量时，改为**带种子的随机抽样**而非取前 N 张
    （原实现取字典序前 N 张，实测该子集文本密度偏高约 20%，不具代表性）。
    """
    imgs_dir = data_root / "detection" / "test" / "imgs"
    gt_dir = data_root / "detection" / "test" / "gt"
    all_paths = sorted(imgs_dir.glob("img_*.jpg"))
    img_paths = all_paths
    if limit > 0 and limit < len(all_paths):
        if sample_seed is not None:
            rng = np.random.default_rng(sample_seed)
            idx = np.sort(rng.choice(len(all_paths), limit, replace=False))
            img_paths = [all_paths[i] for i in idx]
            print(f"[det] 随机抽样 {limit}/{len(all_paths)} 张（种子 {sample_seed}）")
        else:
            img_paths = all_paths[:limit]

    total_gt = total_pred = matched = 0
    total_care = 0
    dtp = dfp = dignored = dfn = 0
    times: List[float] = []
    print(f"[det] 共 {len(img_paths)} 张图，IoU 阈值 {iou_thresh}")

    dump_handle = None
    if dump_pred is not None:
        dump_path = Path(dump_pred)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_handle = open(dump_path, "w", encoding="utf-8")
        print(f"[det] 预测框（含置信度）写入 {dump_path}")

    try:
        for idx, img_path in enumerate(img_paths, start=1):
            gt_path = gt_dir / f"gt_{img_path.stem}.txt"
            if not gt_path.is_file():
                print(f"  skip {img_path.name}: 缺标注")
                continue
            gt_boxes, gt_dnc = parse_det_gt_flagged(gt_path)

            start = time.perf_counter()
            boxes, scores = ocr.detect_detailed(str(img_path))
            times.append(time.perf_counter() - start)

            if dump_handle is not None:
                dump_handle.write(
                    json.dumps(
                        {
                            "image": img_path.stem,
                            "boxes": [
                                np.asarray(b, dtype=float).round(2).tolist() for b in boxes
                            ],
                            "scores": [round(float(s), 6) for s in scores],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # legacy 口径
            m = match_boxes(gt_boxes, boxes, iou_thresh)
            total_gt += len(gt_boxes)
            total_pred += len(boxes)
            matched += m

            # deteval 口径
            r = match_boxes_dnc(gt_boxes, gt_dnc, boxes, iou_thresh)
            dtp += r["tp"]
            dfp += r["fp"]
            dignored += r["ignored"]
            dfn += r["fn"]
            total_care += int((~gt_dnc).sum())

            if idx % 20 == 0 or idx == len(img_paths):
                print(
                    f"  {idx}/{len(img_paths)} 累计 GT={total_gt}(有效 {total_care}) "
                    f"Pred={total_pred} legacy匹配={matched} deteval TP={dtp}"
                )
    finally:
        if dump_handle is not None:
            dump_handle.close()

    legacy = prf_from_counts(matched, total_pred - matched, total_gt - matched)
    deteval = prf_from_counts(dtp, dfp, dfn)
    avg_time = float(np.mean(times)) if times else 0.0

    return {
        "task": "检测",
        "images": len(img_paths),
        # ---- legacy（保持原有键名与含义，勿改）----
        "gt_boxes": total_gt,
        "pred_boxes": total_pred,
        "matched": matched,
        "precision": legacy["precision"],
        "recall": legacy["recall"],
        "f1": legacy["f1"],
        # ---- deteval（ICDAR2015 官方口径）----
        "deteval_gt_boxes": total_care,
        "deteval_tp": dtp,
        "deteval_fp": dfp,
        "deteval_fn": dfn,
        "deteval_ignored": dignored,
        "deteval_precision": deteval["precision"],
        "deteval_recall": deteval["recall"],
        "deteval_f1": deteval["f1"],
        "avg_time": avg_time,
    }


def evaluate_recognition(
    ocr: SceneTextOCR,
    data_root: Path,
    limit: int,
    ignore_case: bool = True,
    sample_seed: int = None,
):
    img_dir = data_root / "recognition" / "test"
    gt_txt = data_root / "recognition" / "test.txt"
    pairs = parse_rec_gt(gt_txt, img_dir)
    if limit > 0 and limit < len(pairs):
        if sample_seed is not None:
            rng = np.random.default_rng(sample_seed)
            idx = np.sort(rng.choice(len(pairs), limit, replace=False))
            pairs = [pairs[i] for i in idx]
        else:
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
        text, _score = ocr.recognize(str(img_path))
        times.append(time.perf_counter() - start)
        pred = text

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


def render_report(
    results: List[dict],
    model_label: str = "PP-OCRv6 medium（det + rec）",
    iou_thresh: float = 0.5,
) -> str:
    lines = ["# ICDAR2015 评估报告", ""]
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"模型：{model_label}")
    lines.append("")
    lines.append("| 任务 | 图片数 | 指标 | 平均耗时/图 |")
    lines.append("| --- | --- | --- | --- |")

    for r in results:
        if r["task"] == "检测":
            # 主行用 ICDAR2015 官方口径；紧接一行给出 legacy 口径以便与历史结果对照
            if "deteval_f1" in r:
                metric = (
                    f"**deteval 口径**（do-not-care 不计入）精确率 {r['deteval_precision']:.4f} / "
                    f"召回率 {r['deteval_recall']:.4f} / F1 {r['deteval_f1']:.4f}"
                    f"（有效 GT {r['deteval_gt_boxes']} 框，Pred {r['pred_boxes']} 框，"
                    f"TP {r['deteval_tp']}，FP {r['deteval_fp']}，FN {r['deteval_fn']}，"
                    f"忽略 {r['deteval_ignored']}）"
                )
                lines.append(
                    f"| {r['task']} | {r['images']} | {metric} | {r['avg_time']:.3f}s |"
                )
                metric = (
                    f"legacy 口径（全部 GT 计入，历史口径）精确率 {r['precision']:.4f} / "
                    f"召回率 {r['recall']:.4f} / F1 {r['f1']:.4f}"
                    f"（GT {r['gt_boxes']} 框，匹配 {r['matched']}）"
                )
            else:
                metric = (
                    f"精确率 {r['precision']:.4f} / 召回率 {r['recall']:.4f} / "
                    f"F1 {r['f1']:.4f}（GT {r['gt_boxes']} 框，Pred {r['pred_boxes']} 框，"
                    f"匹配 {r['matched']}）"
                )
        else:
            metric = (
                f"字符准确率 {r['char_acc']:.4f} / 行级准确率 {r['line_acc']:.4f}"
            )
        lines.append(
            f"| {r['task']} | {r['images']} | {metric} | {r['avg_time']:.3f}s |"
        )

    lines.append("")
    lines.append(
        f"说明：检测框匹配采用 IoU 阈值 {iou_thresh:g} 贪心一对一；"
        "识别字符准确率按编辑距离计算，默认忽略大小写。"
    )
    lines.append("")
    lines.append(
        "> 两套检测口径：**deteval** 按 ICDAR2015 官方做法把转写为 `###` 的 GT 视为 "
        "do-not-care（不计入召回分母，命中它的预测被忽略）；**legacy** 为仓库历史口径，"
        "把全部 GT 计入分母，数值偏低且不可与已发表结果比较，仅用于与旧结果对照。"
    )
    return "\n".join(lines)
