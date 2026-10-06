"""评估模块：指标、标注解析、评估执行与报告。"""

from src.evaluator.metrics import (
    cluster_line_groups,
    levenshtein,
    match_boxes,
    match_boxes_dnc,
    merge_to_lines,
    prf_from_counts,
    quad_iou,
)
from src.evaluator.parsing import (
    parse_det_gt,
    parse_det_gt_flagged,
    parse_det_gt_full,
    parse_rec_gt,
)
from src.evaluator.runner import evaluate_detection, evaluate_recognition, render_report

__all__ = [
    "cluster_line_groups",
    "levenshtein",
    "match_boxes",
    "match_boxes_dnc",
    "merge_to_lines",
    "prf_from_counts",
    "quad_iou",
    "parse_det_gt",
    "parse_det_gt_flagged",
    "parse_det_gt_full",
    "parse_rec_gt",
    "evaluate_detection",
    "evaluate_recognition",
    "render_report",
]
