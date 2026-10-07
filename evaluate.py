"""ICDAR2015 评估命令行入口（核心逻辑在 src/evaluator/）。

用法::

    python evaluate.py --task det --limit 50
    python evaluate.py --task rec --limit 100
    python evaluate.py --task all
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.config import get_config
from src.evaluator import evaluate_detection, evaluate_recognition, render_report
from src.pipeline.ocr_pipeline import SceneTextOCR


def main() -> None:
    parser = argparse.ArgumentParser(description="ICDAR2015 检测与识别评估")
    parser.add_argument("--data-root", default=None, help="数据集根目录，默认读 config.yaml")
    parser.add_argument("--task", choices=["det", "rec", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="每任务最多评估图片数，0 为全量")
    parser.add_argument(
        "--sample-seed", type=int, default=None,
        help="与 --limit 配合：用该种子随机抽样，而不是取字典序前 N 张（推荐）",
    )
    parser.add_argument(
        "--dump-pred", default=None,
        help="把逐图检测框与置信度写入该 JSONL 文件，供离线阈值/口径分析",
    )
    parser.add_argument("--iou", type=float, default=0.5, help="检测 IoU 阈值")
    parser.add_argument(
        "--ignore-case", action=argparse.BooleanOptionalAction, default=True,
        help="识别比对忽略大小写（默认开，ICDAR2015 标准）",
    )
    parser.add_argument("--det-thresh", type=float, default=None, help="检测过滤阈值，默认用 PaddleOCR 内置值")
    parser.add_argument("--det-box-thresh", type=float, default=None, help="检测框阈值，默认用 PaddleOCR 内置值")
    parser.add_argument(
        "--det-unclip-ratio", type=float, default=None,
        help="检测框外扩比例，默认用模型内置值（PP-OCRv6 为 1.4）。它直接决定框的大小，"
             "可用于检验「输出粒度介于词/行之间」的假设",
    )
    parser.add_argument(
        "--det-limit-side-len", type=int, default=None,
        help="检测输入的长边限制，默认用模型内置值（PP-OCRv6 为 960）。"
             "配合 --det-limit-type 可检验审计 P1-8 的「0.75 倍下采样损失小字召回」："
             "1280/max = 不下采样；960/min = 反而放大到短边 960。注意耗时约按面积增长",
    )
    parser.add_argument(
        "--det-limit-type", choices=["max", "min"], default=None,
        help="长边限制的方向：max 只缩不放（模型默认），min 会放大过小的图",
    )
    parser.add_argument(
        "--model", choices=["medium", "small"], default="medium",
        help="模型档位：评估基准用 medium，部署系统用 small",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", default="data/results/evaluation_report.md")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else Path(
        get_config().get("data", {}).get("root", "data/icdar2015")
    )
    kwargs = dict(
        device=args.device,
        det_thresh=args.det_thresh,
        det_box_thresh=args.det_box_thresh,
        det_unclip_ratio=args.det_unclip_ratio,
        det_limit_side_len=args.det_limit_side_len,
        det_limit_type=args.det_limit_type,
    )
    if args.model == "small":
        kwargs["det_model_name"] = "PP-OCRv6_small_det"
        kwargs["rec_model_name"] = "PP-OCRv6_small_rec"
    else:
        kwargs["det_model_name"] = "PP-OCRv6_medium_det"
        kwargs["rec_model_name"] = "PP-OCRv6_medium_rec"
    ocr = SceneTextOCR(**kwargs)

    results: List[dict] = []
    if args.task in ("det", "all"):
        results.append(
            evaluate_detection(
                ocr, data_root, args.iou, args.limit, args.sample_seed, args.dump_pred
            )
        )
    if args.task in ("rec", "all"):
        results.append(
            evaluate_recognition(
                ocr, data_root, args.limit, args.ignore_case, args.sample_seed
            )
        )

    # 报告标签按**实际执行的任务**生成，避免 --task det 的产物声称跑过 rec
    _task_label = {"检测": "det", "识别": "rec"}
    tasks = "+".join(_task_label.get(r["task"], r["task"]) for r in results)
    report = render_report(results, f"PP-OCRv6 {args.model}（{tasks}）", args.iou)
    print("\n" + report)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已写入 {report_path}")


if __name__ == "__main__":
    main()
