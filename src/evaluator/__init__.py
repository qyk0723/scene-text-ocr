"""评估模块：指标、标注解析、评估执行与报告。"""

from src.evaluator.metrics import levenshtein, match_boxes, merge_to_lines, quad_iou
from src.evaluator.parsing import parse_det_gt, parse_rec_gt
from src.evaluator.runner import evaluate_detection, evaluate_recognition, render_report

__all__ = [
    "levenshtein",
    "match_boxes",
    "merge_to_lines",
    "quad_iou",
    "parse_det_gt",
    "parse_rec_gt",
    "evaluate_detection",
    "evaluate_recognition",
    "render_report",
]
