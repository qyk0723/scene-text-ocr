"""ICDAR2015 评估脚本：检测 + 识别指标。

用法::

    # 只跑检测（先取 50 张验证速度）
    python evaluate.py --task det --limit 50

    # 只跑识别
    python evaluate.py --task rec --limit 100

    # 全量
    python evaluate.py --task all

检测：对整图跑 OCR，预测框与 GT 框按 IoU 阈值贪心一对一匹配，
算精确率 / 召回率 / F1。
识别：对单行裁剪图跑 OCR，对比文字，算字符准确率 / 行级完全匹配率。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.pipeline.ocr_pipeline import SceneTextOCR


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------

def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """两个凸四边形的 IoU。"""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
    except cv2.error:
        return 0.0
    if inter <= 0:
        return 0.0
    area_a = cv2.contourArea(a)
    area_b = cv2.contourArea(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt_boxes: List[np.ndarray], pred_boxes: List[np.ndarray], iou_thresh: float) -> int:
    """贪心一对一匹配，返回匹配上的对数。"""
    matched_gt = set()
    matched = 0
    for p in pred_boxes:
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = quad_iou(p, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thresh:
            matched_gt.add(best_j)
            matched += 1
    return matched


def levenshtein(a: str, b: str) -> int:
    """编辑距离。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# 标注解析
# ---------------------------------------------------------------------------

def parse_det_gt(gt_path: Path) -> List[np.ndarray]:
    """解析检测标注文件，每行前 8 个数字为一个四边形。"""
    quads: List[np.ndarray] = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) < 8:
            continue
        try:
            vals = [float(v) for v in fields[:8]]
        except ValueError:
            continue
        quads.append(np.asarray(vals, dtype=np.float32).reshape(4, 2))
    return quads


def parse_rec_gt(gt_txt: Path, img_dir: Path) -> List[Tuple[Path, str]]:
    """解析识别标注：路径 \t 文本 \t 语言，路径改写为项目内相对路径。"""
    pairs: List[Tuple[Path, str]] = []
    for line in gt_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        fname = Path(parts[0]).name
        img_path = img_dir / fname
        pairs.append((img_path, parts[1]))
    return pairs


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def render_report(results: List[dict]) -> str:
    lines = ["# ICDAR2015 评估报告", ""]
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="ICDAR2015 检测与识别评估")
    parser.add_argument("--data-root", default="data/icdar2015", help="数据集根目录")
    parser.add_argument("--task", choices=["det", "rec", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="每任务最多评估图片数，0 为全量")
    parser.add_argument("--iou", type=float, default=0.5, help="检测 IoU 阈值")
    parser.add_argument(
        "--ignore-case", action=argparse.BooleanOptionalAction, default=True,
        help="识别比对忽略大小写（默认开，ICDAR2015 标准）",
    )
    parser.add_argument("--det-thresh", type=float, default=None, help="检测过滤阈值，默认用 PaddleOCR 内置值")
    parser.add_argument("--det-box-thresh", type=float, default=None, help="检测框阈值，默认用 PaddleOCR 内置值")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", default="data/results/evaluation_report.md")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ocr = SceneTextOCR(
        device=args.device,
        det_thresh=args.det_thresh,
        det_box_thresh=args.det_box_thresh,
    )

    results: List[dict] = []
    if args.task in ("det", "all"):
        results.append(evaluate_detection(ocr, data_root, args.iou, args.limit))
    if args.task in ("rec", "all"):
        results.append(evaluate_recognition(ocr, data_root, args.limit, args.ignore_case))

    report = render_report(results)
    print("\n" + report)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已写入 {report_path}")


if __name__ == "__main__":
    main()
