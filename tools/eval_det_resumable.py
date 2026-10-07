"""可续跑的分档检测评估：一次加载模型，逐张写检查点，崩溃也不丢进度。

**为什么需要它**（2026-10-07 的教训）：审计 P1-8 的 medium 分辨率复测用
`evaluate.py --limit 100 --det-limit-side-len 1280` 跑到第 44 张时**被外部杀掉**
（medium@1280 约 70 s/张，全程约 2 小时；`evaluate.py` 只在**全部跑完**时写报告，
所以 53 分钟的工作连一个数字都没留下）。本脚本把粒度降到**单张**：

- 每张图算完立刻追加一行到 JSONL 检查点（`flush`），中断后重跑会**跳过已完成项**；
- 只用**同一套**评测原语（`parse_det_gt_flagged` + `match_boxes` + `match_boxes_dnc` +
  `prf_from_counts`），与 `evaluate.py` 口径严格一致；
- 复用同一批样本的**权威清单**（与 A/B/C 档同种子同顺序），保证可合并。

用法::

    python tools/eval_det_resumable.py --images .tmp_tests/b1/images.txt \\
        --out .tmp_tests/b1/M_1280max_100.resume.jsonl \\
        --model medium --det-limit-side-len 1280 --det-limit-type max
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.evaluator import (  # noqa: E402
    match_boxes,
    match_boxes_dnc,
    parse_det_gt_flagged,
    prf_from_counts,
)
from src.evaluator.metrics import quad_iou  # noqa: E402,F401  (保持导入面与原评估一致)
from src.pipeline.ocr_pipeline import SceneTextOCR  # noqa: E402

GT_DIR = PROJ / "data" / "icdar2015" / "detection" / "test" / "gt"


def load_done(path: Path) -> dict:
    """已完成的检查点：``{image_stem: 指标dict}``。"""
    done: dict = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # 末行可能被截断
            done[rec["image"]] = rec
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description="可续跑的分档检测评估")
    ap.add_argument("--images", required=True, help="图片路径清单（每行一个）")
    ap.add_argument("--out", required=True, help="检查点 JSONL（追加、可续跑）")
    ap.add_argument("--model", choices=["medium", "small"], default="medium")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--det-limit-side-len", type=int, default=None)
    ap.add_argument("--det-limit-type", choices=["max", "min"], default=None)
    args = ap.parse_args()

    paths = [Path(l.strip()) for l in Path(args.images).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    todo = [p for p in paths if p.stem not in done]
    print(f"[resume] 清单 {len(paths)} 张，已完成 {len(done)}，本次待跑 {len(todo)}")

    if not todo:
        print("[resume] 无需再跑")
        return 0

    kwargs = dict(
        det_model_name="PP-OCRv6_medium_det" if args.model == "medium" else "PP-OCRv6_small_det",
        rec_model_name="PP-OCRv6_medium_rec" if args.model == "medium" else "PP-OCRv6_small_rec",
    )
    if args.det_limit_side_len is not None:
        kwargs["det_limit_side_len"] = args.det_limit_side_len
    if args.det_limit_type is not None:
        kwargs["det_limit_type"] = args.det_limit_type
    ocr = SceneTextOCR(**kwargs)
    print(f"[resume] 生效参数：limit_side_len={ocr.det_limit_side_len} "
          f"limit_type={ocr.det_limit_type}")

    with open(out, "a", encoding="utf-8") as fh:
        t_start = time.perf_counter()
        for i, p in enumerate(todo, 1):
            gt_boxes, gt_dnc = parse_det_gt_flagged(GT_DIR / f"gt_{p.stem}.txt")
            t0 = time.perf_counter()
            boxes, _scores = ocr.detect_detailed(str(p))
            dt = time.perf_counter() - t0

            m = match_boxes(gt_boxes, boxes, args.iou)              # legacy
            r = match_boxes_dnc(gt_boxes, gt_dnc, boxes, args.iou)  # deteval
            rec = {
                "image": p.stem,
                "seconds": dt,
                "legacy": {"matched": m, "gt": len(gt_boxes), "pred": len(boxes)},
                "deteval": {"tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                            "ignored": r["ignored"], "care": int((~gt_dnc).sum())},
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()  # 每张都落盘：被杀也只损失当前这一张
            if i % 25 == 0 or i == len(todo):
                el = time.perf_counter() - t_start
                print(f"  {i}/{len(todo)} 完成（累计 {len(done)+i}/{len(paths)}）"
                      f" 已用 {el/60:.1f} min  预计总 {el/i*len(todo)/60:.1f} min", flush=True)

    # 汇总
    recs = list(load_done(out).values())
    dtp = sum(r["deteval"]["tp"] for r in recs)
    dfp = sum(r["deteval"]["fp"] for r in recs)
    dfn = sum(r["deteval"]["fn"] for r in recs)
    d = prf_from_counts(dtp, dfp, dfn)
    secs = float(np.mean([r["seconds"] for r in recs]))
    print(f"\n[deteval] 图 {len(recs)} 张  P {d['precision']:.4f} / R {d['recall']:.4f} / "
          f"F1 {d['f1']:.4f}   平均 {secs:.3f} s/张   (TP {dtp} FP {dfp} FN {dfn})")
    summary = {"images": len(recs), "deteval": d, "seconds_per_image": secs,
               "tp": dtp, "fp": dfp, "fn": dfn}
    Path(out).with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[resume] 汇总已写入 {Path(out).with_suffix('.summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
