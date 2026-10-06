"""行级口径检测评估：把词级 GT 合并成行级框后重算检测 P/R/F1。

对同一批图同时报「词级口径」与「行级口径」两套指标，并在 IoU 0.3 / 0.5 / 0.7
三档阈值下做敏感性分析，用于定位检测指标偏低的成因（粒度错位 vs 难例漏检）。

**同时给出两套 do-not-care 口径**（2026-10-06 增补）：

- ``legacy``：全部 GT 计入召回分母（仓库历史口径，不可与已发表结果比较）；
- ``deteval``：转写为 ``###`` 的 GT 视为 do-not-care，不计入分母、命中它的预测被忽略。

这一点对本脚本尤其关键：实测现状口径下行级 GT 里约 **74%** 是由 do-not-care
区域合并出来的"行"，把它们当作"文字行"来考核会严重夸大漏检。

用法::

    python evaluate_linelevel.py --limit 30 --model medium
    python evaluate_linelevel.py --limit 30 --model medium --sample-seed 20261006
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import get_config
from src.evaluator import (
    match_boxes,
    match_boxes_dnc,
    merge_to_lines,
    parse_det_gt_flagged,
)
from src.pipeline.ocr_pipeline import SceneTextOCR


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _line_gt_with_flags(gt_words, gt_dnc):
    """行级 GT：有效行与无效区行分别合并后拼在一起，并给出标记。

    匹配采用官方约定——预测先与**全部** GT 求最优匹配，再按匹配到的 GT 类型
    判定（有效→TP，无效→忽略）。
    """
    care = merge_to_lines([b for b, d in zip(gt_words, gt_dnc) if not d])
    dnc = merge_to_lines([b for b, d in zip(gt_words, gt_dnc) if d])
    lines = care + dnc
    flags = np.array([False] * len(care) + [True] * len(dnc), dtype=bool)
    return lines, flags, len(care)


def main() -> None:
    parser = argparse.ArgumentParser(description="行级口径检测评估")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--model", choices=["medium", "small"], default="medium")
    parser.add_argument(
        "--sample-seed", type=int, default=None,
        help="用该种子随机抽样，而不是取字典序前 N 张（推荐，避免子集偏差）",
    )
    parser.add_argument("--out", default=None, help="输出 md 路径，默认 data/results/linelevel_<model>_iou.md")
    args = parser.parse_args()

    data_root = Path(get_config().get("data", {}).get("root", "data/icdar2015"))
    imgs_dir = data_root / "detection" / "test" / "imgs"
    gt_dir = data_root / "detection" / "test" / "gt"
    all_paths = sorted(imgs_dir.glob("img_*.jpg"))
    if args.limit > 0 and args.limit < len(all_paths):
        if args.sample_seed is not None:
            rng = np.random.default_rng(args.sample_seed)
            idx = np.sort(rng.choice(len(all_paths), args.limit, replace=False))
            img_paths = [all_paths[i] for i in idx]
            print(f"随机抽样 {args.limit}/{len(all_paths)} 张（种子 {args.sample_seed}）")
        else:
            img_paths = all_paths[:args.limit]
    else:
        img_paths = all_paths

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
    # legacy 累计
    gt_word = gt_line = pred = 0
    matched_word = {i: 0 for i in ious}
    matched_line = {i: 0 for i in ious}
    # deteval 累计
    care_word = care_line = 0
    de_word = {i: [0, 0, 0] for i in ious}   # [tp, fp, fn]
    de_line = {i: [0, 0, 0] for i in ious}
    ignored_word = {i: 0 for i in ious}
    ignored_line = {i: 0 for i in ious}
    print(f"[{args.model}] 共 {len(img_paths)} 张图，IoU {ious}")

    for idx, img_path in enumerate(img_paths, start=1):
        gt_path = gt_dir / f"gt_{img_path.stem}.txt"
        if not gt_path.is_file():
            continue
        gt_words, gt_dnc = parse_det_gt_flagged(gt_path)
        gt_lines_all = merge_to_lines(gt_words)
        lines, line_flags, n_care_lines = _line_gt_with_flags(gt_words, gt_dnc)

        boxes = ocr.detect(str(img_path))

        gt_word += len(gt_words)
        gt_line += len(gt_lines_all)
        pred += len(boxes)
        care_word += int((~gt_dnc).sum())
        care_line += n_care_lines

        for i in ious:
            matched_word[i] += match_boxes(gt_words, boxes, i)
            matched_line[i] += match_boxes(gt_lines_all, boxes, i)

            r = match_boxes_dnc(gt_words, gt_dnc, boxes, i)
            de_word[i][0] += r["tp"]; de_word[i][1] += r["fp"]; de_word[i][2] += r["fn"]
            ignored_word[i] += r["ignored"]

            r = match_boxes_dnc(lines, line_flags, boxes, i)
            de_line[i][0] += r["tp"]; de_line[i][1] += r["fp"]; de_line[i][2] += r["fn"]
            ignored_line[i] += r["ignored"]

        if idx % 10 == 0 or idx == len(img_paths):
            print(f"  {idx}/{len(img_paths)} 累计 词级GT={gt_word}(有效{care_word}) "
                  f"行级GT={gt_line}(有效{care_line}) Pred={pred}")

    print("=" * 72)
    print(f"[{args.model}] 词级GT={gt_word}(有效 {care_word}) 行级GT={gt_line}(有效 {care_line}) Pred={pred}")

    legacy_rows, deteval_rows = [], []
    for i in ious:
        pw, rw, fw = _prf(matched_word[i], pred - matched_word[i], gt_word - matched_word[i])
        pl, rl, fl = _prf(matched_line[i], pred - matched_line[i], gt_line - matched_line[i])
        print(f"IoU {i} legacy : 词级 {pw:.4f}/{rw:.4f}/{fw:.4f} | 行级 {pl:.4f}/{rl:.4f}/{fl:.4f}")
        legacy_rows.append(f"| {i} | {pw:.4f} | {rw:.4f} | {fw:.4f} | {pl:.4f} | {rl:.4f} | {fl:.4f} |")

        dwp, dwr, dwf = _prf(*de_word[i])
        dlp, dlr, dlf = _prf(*de_line[i])
        print(f"IoU {i} deteval: 词级 {dwp:.4f}/{dwr:.4f}/{dwf:.4f} | 行级 {dlp:.4f}/{dlr:.4f}/{dlf:.4f}"
              f"  (忽略 词级 {ignored_word[i]} 行级 {ignored_line[i]})")
        deteval_rows.append(
            f"| {i} | {dwp:.4f} | {dwr:.4f} | {dwf:.4f} | {dlp:.4f} | {dlr:.4f} | {dlf:.4f} "
            f"| {ignored_word[i]} | {ignored_line[i]} |"
        )

    out = Path(args.out) if args.out else Path(f"data/results/linelevel_{args.model}_iou.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# 行级口径检测评估 + IoU 敏感性（{args.model}）\n\n"
        f"图片数：{len(img_paths)}（ICDAR2015 detection/test）"
        + (f"，随机种子 {args.sample_seed}" if args.sample_seed is not None else "，字典序前 N 张")
        + f"\n\n词级 GT={gt_word}（有效 {care_word}）  行级 GT={gt_line}（有效 {care_line}）  "
        f"预测框={pred}\n\n"
        f"## legacy 口径（全部 GT 计入，历史口径）\n\n"
        f"| IoU | 词级 P | 词级 R | 词级 F1 | 行级 P | 行级 R | 行级 F1 |\n"
        f"| --- | --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(legacy_rows)
        + f"\n\n## deteval 口径（do-not-care 不计入，ICDAR2015 官方做法）\n\n"
        f"| IoU | 词级 P | 词级 R | 词级 F1 | 行级 P | 行级 R | 行级 F1 | 忽略(词) | 忽略(行) |\n"
        f"| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(deteval_rows)
        + "\n\n说明：预测框不变；行级 GT 由词级 GT 按 y 中心聚类合并。deteval 口径下，\n"
        f"现状口径的行级 GT 中有大量「行」其实来自 do-not-care 区域，被排除后行级召回会明显上升。\n",
        encoding="utf-8",
    )
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
