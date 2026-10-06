"""端到端（整图）系统评测 CLI：det → crop → rec 全流程，按"行"判定读对与否。

审计 P1-13 指出：项目把唯一的端到端数字（识别 82.89%）当作"口径不准"删掉了，
而它对"系统"类论文恰恰是**用户实际体验**的指标。本脚本把它作为规范指标补回来。

判定口径见 :func:`src.evaluator.runner.evaluate_end2end`：
GT 行只取有真实转写的词级框（排除 ``###``），行文本按 x 拼接并去掉空白、忽略大小写；
一行判为**读对**要求存在 IoU≥阈值的预测框且其文本完全等于行文本；
命中 ``###`` 区域的预测被忽略。

用法::

    python evaluate_end2end.py --model small --limit 100 --sample-seed 20261006
    python evaluate_end2end.py --model small            # 全量 500 张（约 50 分钟）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import get_config
from src.evaluator.runner import evaluate_end2end
from src.pipeline.ocr_pipeline import SceneTextOCR


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端（整图）系统评测")
    parser.add_argument("--limit", type=int, default=0, help="最多评估图片数，0 为全量")
    parser.add_argument("--model", choices=["medium", "small"], default="small")
    parser.add_argument("--sample-seed", type=int, default=None,
                        help="与 --limit 配合：随机抽样而非取字典序前 N 张（推荐）")
    parser.add_argument("--iou", type=float, default=0.5, help="行匹配 IoU 阈值")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", default="data/results/e2e_report.md")
    parser.add_argument("--dump-lines", default=None,
                        help="把行级明细写入 JSONL（读对/读错/漏读/多余/忽略），供错误分类")
    parser.add_argument("--metrics-json", default="metrics/e2e.json",
                        help="把结果写成机器可读 JSON，供图表使用（单一数据源）")
    args = parser.parse_args()

    data_root = Path(get_config().get("data", {}).get("root", "data/icdar2015"))

    if args.model == "small":
        ocr = SceneTextOCR(det_model_name="PP-OCRv6_small_det",
                           rec_model_name="PP-OCRv6_small_rec", device=args.device)
    else:
        ocr = SceneTextOCR(det_model_name="PP-OCRv6_medium_det",
                           rec_model_name="PP-OCRv6_medium_rec", device=args.device)

    r = evaluate_end2end(ocr, data_root, args.iou, args.limit, args.sample_seed,
                         args.dump_lines)
    w, ln = r["word"], r["line"]

    body = (
        f"# 端到端（整图）系统评测\n\n"
        f"模型：PP-OCRv6 {args.model}；图片数：{r['images']}"
        + (f"（随机种子 {args.sample_seed}）" if args.sample_seed is not None else "（字典序前 N 张）")
        + f"；匹配 IoU 阈值：{args.iou:g}\n\n"
        f"**整图走完 det→crop→rec 全流程**，与「单行裁剪图直接识别」的 rec-only 指标口径不同，"
        f"两者不可互相替代。判定要求识别文本与 GT 完全一致（去空白、忽略大小写）。\n\n"
        f"## 词级（严格 1:1 匹配；**不可与文献直接比较**，原因见下）\n\n"
        f"| 指标 | 数值 |\n| --- | --- |\n"
        f"| 有效词数 | {w['total']} |\n"
        f"| 读对（TP） | {w['tp']} |\n"
        f"| 读错/多余（FP） | {w['fp']} |\n"
        f"| 漏读（FN） | {w['fn']} |\n"
        f"| 忽略（落在 do-not-care） | {w['ignored']} |\n"
        f"| 精确率 / 召回率 / **F1** | {w['precision']:.4f} / {w['recall']:.4f} / **{w['f1']:.4f}** |\n"
        f"| 字符准确率 | {w['char_acc']:.4f} |\n\n"
        f"## 行级（严格口径，**对「一行被切成多个框」很敏感**）\n\n"
        f"| 指标 | 数值 |\n| --- | --- |\n"
        f"| 有效行数 | {ln['total']} |\n"
        f"| 读对（TP） | {ln['tp']} |\n"
        f"| 读错/多余（FP） | {ln['fp']} |\n"
        f"| 漏读（FN） | {ln['fn']} |\n"
        f"| 忽略（落在 do-not-care） | {ln['ignored']} |\n"
        f"| 精确率 / 召回率 / **F1** | {ln['precision']:.4f} / {ln['recall']:.4f} / **{ln['f1']:.4f}** |\n"
        f"| 字符准确率 | {ln['char_acc']:.4f} |\n\n"
        f"平均耗时：{r['avg_time']:.3f} s/张（CPU，整图）。\n\n"
        f"> **口径说明**：两套都只把**有真实转写**的 GT 计入分母（`###` 是 do-not-care，"
        f"不计入、命中它的预测被忽略）——即 ICDAR2015 官方立场。\n"
        f">\n"
        f"> ⚠️ **但两个粒度的匹配方案都不是官方 IC15 端到端方案**：官方允许**一对多**匹配"
        f"（一个预测覆盖多个 GT 词时按面积重叠判定），本实现是**严格 1:1 IoU 匹配**，"
        f"因此会惩罚「一个框覆盖多个词」的预测，**数值偏低且不可与文献直接比较**"
        f"（与检测指标遇到的是同一个问题）。只能用于自我对照与趋势分析。\n"
        f">\n"
        f"> 词级与行级分别回答「模型认得出多少词」与「用户能不能一次读对一行」，"
        f"需一并解读，且**不要**把其中任一当作可与文献对比的绝对水平。\n"
    )
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print("\n" + body)
    print(f"报告已写入 {out}")

    if args.metrics_json:
        mj = Path(args.metrics_json)
        mj.parent.mkdir(parents=True, exist_ok=True)
        payload = {args.model: {**{k: v for k, v in r.items()},
                                "protocol": "deteval（GT 仅含有效文本转写；命中 ### 的预测被忽略）",
                                "source": str(out)}}
        mj.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"指标已写入 {mj}")


if __name__ == "__main__":
    main()
