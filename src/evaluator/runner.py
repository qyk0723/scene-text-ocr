"""评估执行与报告生成。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import numpy as np

from src.evaluator.metrics import (
    cluster_line_groups,
    levenshtein,
    match_boxes,
    match_boxes_dnc,
    merge_to_lines,
    prf_from_counts,
    quad_iou,
)
from src.evaluator.parsing import parse_det_gt_flagged, parse_det_gt_full, parse_rec_gt
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


def _norm_text(s) -> str:
    """行文本规范化：去掉全部空白并大写。

    ICDAR2015 是词级标注，词与词之间的空格语义并不稳定，模型输出是否带空格也不稳定；
    因此比较时统一去掉空白，只比对字符内容（大小写同样忽略，与识别评测口径一致）。
    """
    return "".join(str(s).split()).upper()


def _line_text(quads, texts, idxs) -> str:
    """把一个行分组内的词按 x 排序后拼成行文本（规范化）。"""
    ordered = sorted(idxs, key=lambda i: float(quads[i][:, 0].min()))
    return _norm_text(" ".join(texts[i] for i in ordered))


def _score_end2end_set(gt_boxes, gt_texts, pred_boxes, pred_norm, dnc_quads, iou_thresh):
    """对**一组** GT（词级或行级）计分，返回 ``(tp, fp, fn, ignored, char_accs)``。

    ``gt_texts`` 需已规范化。匹配策略与项目其它评测一致（IoU 阈值 + 贪婪一对一），
    以保证同一份预测在不同口径下可比。
    """
    used, pairs = set(), {}
    for pi, b in enumerate(pred_boxes):
        bj, biou = -1, 0.0
        for j, g in enumerate(gt_boxes):
            if j in used:
                continue
            iou = quad_iou(b, g)
            if iou > biou:
                biou, bj = iou, j
        if bj >= 0 and biou >= iou_thresh:
            used.add(bj)
            pairs[pi] = bj

    cands: dict = {}
    for pi, j in pairs.items():
        cands.setdefault(j, []).append(pred_norm[pi])

    tp = 0
    char_accs: List[float] = []
    for j, gt in enumerate(gt_texts):
        got = cands.get(j, [])
        if gt and gt in got:
            tp += 1
        best = min(got, key=lambda c: levenshtein(c, gt)) if got else ""
        char_accs.append(1.0 - levenshtein(best, gt) / max(len(best), len(gt), 1))
    fn = len(gt_texts) - tp

    # FP：匹配上但读错的预测；以及未匹配且不落在 do-not-care 区域的预测
    fp = 0
    for pi, j in pairs.items():
        if gt_texts[j] and pred_norm[pi] == gt_texts[j]:
            continue
        fp += 1
    ignored = 0
    for pi, b in enumerate(pred_boxes):
        if pi in pairs:
            continue
        if any(quad_iou(b, d) >= iou_thresh for d in dnc_quads):
            ignored += 1
        else:
            fp += 1

    return tp, fp, fn, ignored, char_accs


def evaluate_end2end(
    ocr: SceneTextOCR,
    data_root: Path,
    iou_thresh: float = 0.5,
    limit: int = 0,
    sample_seed: int = None,
):
    """**端到端（整图）系统评测**：det → crop → rec 全流程，判定读对与否。

    审计 P1-13：项目曾把唯一的端到端数字当作"口径不准"删掉（82.89% → rec-only 91.28%），
    但对**系统**类论文而言，端到端才是用户实际体验的指标。本函数把它恢复成规范指标。

    同时给出**两个粒度**，因为二者回答不同问题、且对检测粒度敏感度不同：

    - **词级**：GT 是**有真实转写的词框**，一个词判为读对 = 存在 IoU≥阈值的预测框且其文本
      完全等于该词。⚠️ 本实现用的是**严格 1:1 IoU 匹配**，而官方 IC15 端到端评测允许
      **一对多**匹配（一个预测覆盖多个 GT 词时按面积重叠判定）。因此本口径**会惩罚
      "一个框覆盖多个词"的预测**，数值偏低，**不可与文献直接比较**——与检测指标的情况相同。
    - **行级**：GT 是有效词按 y 中心聚成的**行框**，行文本 = 行内词按 x 拼接。要求整行由
      **一个**预测框读对，因此对"模型把一行切成多个框"同样敏感。这一口径更贴近
      「用户能不能一次读对一行」，但**也不是官方方案**。

    **两个口径都不是官方 IC15 端到端方案**，都只能作自我对照与趋势分析，
    不得当作可与文献对比的绝对水平。

    两套都只把**有真实转写**的 GT 计入分母（``###`` 是 do-not-care），
    命中 ``###`` 区域的预测被忽略。行文本与词文本比较时均**去掉空白、忽略大小写**。
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
            print(f"[e2e] 随机抽样 {limit}/{len(all_paths)} 张（种子 {sample_seed}）")
        else:
            img_paths = all_paths[:limit]

    acc = {"word": [0, 0, 0, 0, []], "line": [0, 0, 0, 0, []]}  # tp, fp, fn, ignored, accs
    times: List[float] = []
    print(f"[e2e] 共 {len(img_paths)} 张整图，IoU 阈值 {iou_thresh}（端到端 det→crop→rec）")

    for idx, img_path in enumerate(img_paths, start=1):
        gt_path = gt_dir / f"gt_{img_path.stem}.txt"
        if not gt_path.is_file():
            print(f"  skip {img_path.name}: 缺标注")
            continue
        quads, dnc, gt_texts = parse_det_gt_full(gt_path)
        care = [i for i in range(len(quads)) if not dnc[i]]
        care_quads = [quads[i] for i in care]
        care_texts = [gt_texts[i] for i in care]
        dnc_quads = [quads[i] for i in range(len(quads)) if dnc[i]]

        groups = cluster_line_groups(care_quads)
        line_boxes = merge_to_lines(care_quads)  # 与 groups 同序、同规则
        line_texts = [_line_text(care_quads, care_texts, g) for g in groups]

        start = time.perf_counter()
        boxes, pred_texts, _scores, _t = ocr.run_detailed(str(img_path))
        times.append(time.perf_counter() - start)
        pred_norm = {pi: _norm_text(t) for pi, t in enumerate(pred_texts)}

        for level, gt_boxes, texts in (
            ("word", care_quads, [_norm_text(t) for t in care_texts]),
            ("line", line_boxes, line_texts),
        ):
            tp, fp, fn, ign, accs = _score_end2end_set(
                gt_boxes, texts, boxes, pred_norm, dnc_quads, iou_thresh
            )
            a = acc[level]
            a[0] += tp
            a[1] += fp
            a[2] += fn
            a[3] += ign
            a[4].extend(accs)

        if idx % 20 == 0 or idx == len(img_paths):
            print(f"  {idx}/{len(img_paths)} 累计 词级 TP={acc['word'][0]} 行级 TP={acc['line'][0]}")

    out = {"task": "端到端", "images": len(img_paths),
           "avg_time": float(np.mean(times)) if times else 0.0}
    for level in ("word", "line"):
        tp, fp, fn, ign, accs = acc[level]
        m = prf_from_counts(tp, fp, fn)
        out[level] = {"total": tp + fn, "tp": tp, "fp": fp, "fn": fn, "ignored": ign,
                      "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                      "char_acc": float(np.mean(accs)) if accs else 0.0}
    return out


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
