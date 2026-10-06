"""从 data/results/ 的规范产物生成 metrics/*.json（图表数值的单一数据源）。

**为什么存在**：`make_figures.py` 曾把每个数字硬编码成字面量，与结果产物没有任何
程序化关联，导致图表与真实结果漂移过（同一个 small 检测 F1，图表里 30.62、另一个
产物里 23.53，无人发现）。现在图表数值一律从 `metrics/*.json` 读取。

本脚本把所有可解析的数值**从产物中解出**（不手工转录），把没有产物支撑的历史数值
放进 `manual.json` 并显式标注"无产物支撑"。字段口径见 `metrics/README.md`。

用法（在项目根目录）::

    python tools/gen_metrics.py

只写 metrics/ 目录，不改动其它文件。
"""
import json
import re
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
RES = PROJ / "data" / "results"
OUT = PROJ / "metrics"


def read(name):
    p = RES / name
    if not p.is_file():
        raise FileNotFoundError(f"缺少产物 {p}")
    return p.read_text(encoding="utf-8")


def find_det_row(text, label):
    """匹配形如：| 检测 | 500 | 精确率 0.6137 / 召回率 0.2591 / F1 0.3643（...） | 11.264s |"""
    m = re.search(
        r"精确率\s*([\d.]+)\s*/\s*召回率\s*([\d.]+)\s*/\s*F1\s*([\d.]+)", text)
    if not m:
        raise ValueError(f"{label}: 未找到检测指标")
    return {"precision": float(m.group(1)), "recall": float(m.group(2)), "f1": float(m.group(3))}


def _row_with(text, marker, label):
    """在**含指定标记的那一行**里取 P/R/F1。用于区分同一文件中的两套口径。"""
    for line in text.splitlines():
        if marker in line:
            m = re.search(
                r"精确率\s*([\d.]+)\s*/\s*召回率\s*([\d.]+)\s*/\s*F1\s*([\d.]+)", line)
            if m:
                return {"precision": float(m.group(1)), "recall": float(m.group(2)),
                        "f1": float(m.group(3))}
    raise ValueError(f"{label}: 未找到含「{marker}」的检测行")


def legacy_row(text, label):
    """取 legacy 口径那一行；这类文件里 legacy 行**排在 deteval 行之后**，
    不能简单取第一个匹配（否则会把 deteval 当成 legacy——已踩过这个坑）。"""
    for line in text.splitlines():
        if "legacy 口径" in line:
            m = re.search(
                r"精确率\s*([\d.]+)\s*/\s*召回率\s*([\d.]+)\s*/\s*F1\s*([\d.]+)", line)
            if m:
                return {"precision": float(m.group(1)), "recall": float(m.group(2)),
                        "f1": float(m.group(3))}
    # 没有两套口径标注的旧格式文件：只有一行，直接取
    return find_det_row(text, label)


def deteval_row(text):
    try:
        return _row_with(text, "deteval 口径", "deteval")
    except ValueError:
        return None


def rec_row(text):
    m = re.search(r"字符准确率\s*([\d.]+)\s*/\s*行级准确率\s*([\d.]+)", text)
    if not m:
        raise ValueError("未找到识别指标")
    return {"char_acc": float(m.group(1)), "line_acc": float(m.group(2))}


def rec_time(text):
    m = re.search(r"平均耗时/图\s*\|\s*\n\|\s*识别\s*\|\s*\d+\s*\|[^|]*\|\s*([\d.]+)s", text)
    if m:
        return float(m.group(1))
    # 回退：取识别行的最后一个耗时
    for line in text.splitlines():
        if line.startswith("| 识别"):
            m2 = re.findall(r"([\d.]+)s\s*\|", line)
            if m2:
                return float(m2[-1])
    return None


def det_time(text):
    for line in text.splitlines():
        if line.startswith("| 检测"):
            m2 = re.findall(r"([\d.]+)s\s*\|", line)
            if m2:
                return float(m2[-1])
    return None


def linelevel(name):
    """旧格式：| IoU | 词级P | 词级R | 词级F1 | 行级P | 行级R | 行级F1 |

    输出为**列表**而非以 IoU 为键的字典——键里带小数点会破坏点分路径访问。
    """
    out = []
    for line in read(name).splitlines():
        m = re.match(
            r"\|\s*(0\.\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
            r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            out.append({
                "iou": float(m.group(1)),
                "word": {"p": float(m.group(2)), "r": float(m.group(3)), "f1": float(m.group(4))},
                "line": {"p": float(m.group(5)), "r": float(m.group(6)), "f1": float(m.group(7))},
            })
    if not out:
        raise ValueError(f"{name}: 未解析到 IoU 表")
    return out


def ablation(name):
    """新格式：| 配置 | 字符准确率 | 行级准确率 | 检出框数 | 中位数 | 平均 | p95 |"""
    out = {}
    for line in read(name).splitlines():
        m = re.match(
            r"\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
            r"\s*([\d.]+)s\s*\|\s*([\d.]+)s\s*\|\s*([\d.]+)s\s*\|", line)
        if m:
            out[m.group(1)] = {
                "char_acc": float(m.group(2)), "line_acc": float(m.group(3)),
                "boxes_per_image": float(m.group(4)),
                "median_s": float(m.group(5)), "mean_s": float(m.group(6)),
                "p95_s": float(m.group(7)),
            }
    if not out:
        raise ValueError(f"{name}: 未解析到消融表")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    med_det_final = read("eval_medium_det_final.md")
    sml_det = read("eval_small_det_deteval.md")
    med_final = read("eval_medium_final.md")
    sml_v2 = read("eval_small_v2.md")

    detection = {
        "medium": {
            "source": "data/results/eval_medium_det_final.md",
            "legacy": find_det_row(med_det_final, "medium"),
            "deteval": deteval_row(med_det_final),
            "seconds_per_image": det_time(med_det_final),
        },
        "small": {
            "source": "data/results/eval_small_det_deteval.md",
            "legacy": legacy_row(sml_det, "small"),
            "deteval": deteval_row(sml_det),
            # 耗时**沿用 eval_small_v2.md 的值**——报告 §六 引用的就是它，换掉会让图文不一致。
            # 新跑的 500 张实测另记为 fresh：两者相差 41%，正是 P1-5 记录的"耗时不可复现"。
            "seconds_per_image": det_time(sml_v2),
            "seconds_per_image_fresh": det_time(sml_det),
            "legacy_crosscheck": find_det_row(sml_v2, "small-v2"),
        },
    }
    recognition = {
        "medium": {"source": "data/results/eval_medium_final.md",
                   **rec_row(med_final), "seconds_per_image": rec_time(med_final)},
        "small": {"source": "data/results/eval_small_v2.md",
                  **rec_row(sml_v2), "seconds_per_image": rec_time(sml_v2)},
    }
    linelevel_data = {
        "medium": {"source": "data/results/linelevel_medium_iou.md",
                   "subset": "字典序前 30 张（非随机）", "iou": linelevel("linelevel_medium_iou.md")},
        "small": {"source": "data/results/linelevel_small_iou.md",
                  "subset": "字典序前 30 张（非随机）", "iou": linelevel("linelevel_small_iou.md")},
    }
    ablation_data = {
        "clean": {"source": "data/results/ablation_results.md", "configs": ablation("ablation_results.md")},
        "degraded": {"source": "data/results/ablation_degraded.md", "configs": ablation("ablation_degraded.md")},
        "lowcontrast": {"source": "data/results/ablation_lowcontrast.md", "configs": ablation("ablation_lowcontrast.md")},
        "blur": {"source": "data/results/ablation_blur.md", "configs": ablation("ablation_blur.md")},
        "skew": {"source": "data/results/ablation_skew.md", "configs": ablation("ablation_skew.md")},
        "small": {"source": "data/results/ablation_small.md", "configs": ablation("ablation_small.md")},
    }

    # 有产物支撑的耗时
    timing = {
        "preprocess_whole_image_s": None,      # 无产物，见 manual.json
        "recognition_single_line_s": {"small": recognition["small"]["seconds_per_image"],
                                      "medium": recognition["medium"]["seconds_per_image"]},
        "detection_whole_image_s": {"small": detection["small"]["seconds_per_image"],
                                    "medium": detection["medium"]["seconds_per_image"]},
    }
    manual = {
        "_note": "以下数值**没有任何产物支撑**，是历史手工记录。审计（P1-5）已指出它们不可复现；"
                 "重测前请勿在论文中作为精确值引用。",
        "preprocess_whole_image_s": {"value": 0.10, "provenance": "make_figures.py 历史字面量，无产物"},
        "test_jpg_end_to_end_s": {"medium": {"value": 82.0, "provenance": "历史字面量，无产物"},
                                  "small": {"value": 11.1, "provenance": "历史字面量，无产物"},
                                  "note": "同一操作实测见过 11.1s 与 29.0s，波动极大"},
    }

    for name, payload in (("detection", detection), ("recognition", recognition),
                          ("linelevel", linelevel_data), ("ablation", ablation_data),
                          ("timing", timing), ("manual", manual)):
        p = OUT / f"{name}.json"
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  写入 {p.relative_to(PROJ)}")


if __name__ == "__main__":
    main()
