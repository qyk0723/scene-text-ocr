"""解析 `evaluate.py` 生成的检测报告，输出并排对比表（离线，不跑模型）。

用于审计 P1-8 的分辨率扫描：对同一批样本、同一个模型、不同的
`limit_side_len` / `limit_type` 配置，把 deteval 口径的 P/R/F1 与耗时并排比较。

用法::

    python tools/compare_det_reports.py A=data/results/x.md B=... C=...
    python tools/compare_det_reports.py .tmp_tests/b1/A.md .tmp_tests/b1/B.md

第一列是标签（`标签=路径`），省略标签时用文件名。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# 匹配报告里的一行：| 检测 | 100 | **deteval 口径**… 精确率 0.4706 / 召回率 0.3828 / F1 0.4222
#                  （有效 GT 418 框，Pred 395 框，TP 160，FP 180，FN 258，忽略 55） | 3.997s |
ROW_RE = re.compile(
    r"\|\s*检测\s*\|\s*(\d+)\s*\|\s*(?P<proto>\*\*deteval|legacy)[^|]*?"
    r"精确率\s*(?P<p>[\d.]+)\s*/\s*召回率\s*(?P<r>[\d.]+)\s*/\s*F1\s*(?P<f1>[\d.]+)"
    r"[^|]*?\|\s*(?P<t>[\d.]+)s\s*\|"
)


def parse(path: Path) -> dict:
    """解析一份报告，返回 ``{'deteval': {...}, 'legacy': {...}}``。"""
    out: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.search(line)
        if not m:
            continue
        key = "deteval" if "deteval" in m.group("proto") else "legacy"
        out[key] = {
            "images": int(m.group(1)),
            "precision": float(m.group("p")),
            "recall": float(m.group("r")),
            "f1": float(m.group("f1")),
            "seconds": float(m.group("t")),
        }
    if not out:
        raise ValueError(f"{path}: 未解析到检测行")
    return out


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 2
    rows = []
    for spec in argv:
        label, _, raw = spec.partition("=")
        if not raw:
            raw, label = label, Path(label).stem
        path = Path(raw)
        if not path.is_file():
            print(f"跳过（不存在）: {path}")
            continue
        rows.append((label, path, parse(path)))

    if not rows:
        print("没有可解析的报告")
        return 1

    base = rows[0][2]["deteval"]
    print(f"\n{'配置':<22}{'张数':>5}{'deteval P':>11}{'R':>9}{'F1':>9}"
          f"{'ΔF1(点)':>10}{'耗时/图':>10}{'相对':>8}")
    print("-" * 86)
    for label, _path, data in rows:
        d = data["deteval"]
        df1 = (d["f1"] - base["f1"]) * 100
        ratio = d["seconds"] / base["seconds"]
        print(f"{label:<22}{d['images']:>5}{d['precision']:>11.4f}{d['recall']:>9.4f}"
              f"{d['f1']:>9.4f}{df1:>+10.2f}{d['seconds']:>9.3f}s{ratio:>7.2f}x")
    print()
    print("（ΔF1 相对第一行；deteval 口径 = ICDAR2015 官方做法，do-not-care 不计入召回分母）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
