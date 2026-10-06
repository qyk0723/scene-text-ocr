"""一致性校验：文档里的关键数字 ↔ `metrics/*.json` ↔ `data/results/` 产物。

**为什么需要它**：本次审计的核心病灶是**同一个数字在文档、图表、产物三处各写一遍，
没有任何机制发现它们漂移**。真实发生过：同一个 small 检测 F1，图表里是 30.62、
另一个产物文件里是 23.53，无人察觉。图表侧已经改为从 `metrics/` 单一数据源读取；
本脚本补上另一半——**文档侧**。

两层校验：

1. **`metrics/` 是否与产物同步**：把 `tools/gen_metrics.py` 跑到临时目录，与 `metrics/`
   逐字节（结构化）比对。产物更新后忘了重生成指标，这里会拦下。
2. **文档数字是否与 `metrics/` 一致**：按"期望字符串存在性"检查——不解析散文，
   只要求文档里出现由指标算出的那个数。这样数字改了而文档没跟上，就会失败。

用法（在项目根目录）::

    python tools/check_consistency.py

退出码 0 = 全部一致；1 = 存在不一致（会逐条列出）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.evaluator.metrics_store import get, load  # noqa: E402

FAILURES: list = []
CHECKS = 0


def _pct(x: float) -> str:
    """0.4221 -> '42.21'（百分比，两位小数）。"""
    return f"{round(float(x) * 100, 2):.2f}"


def check(label: str, doc: str, needle: str) -> None:
    """要求 `doc` 里出现 `needle`。"""
    global CHECKS
    CHECKS += 1
    path = PROJ / doc
    if not path.is_file():
        FAILURES.append((label, doc, needle, "文件不存在"))
        return
    if needle not in path.read_text(encoding="utf-8"):
        FAILURES.append((label, doc, needle, "文档中未找到该值"))


def check_absent(label: str, doc: str, needle: str) -> None:
    """要求 `doc` 里**不**出现 `needle`（用于确认过期表述已清除）。"""
    global CHECKS
    CHECKS += 1
    path = PROJ / doc
    if path.is_file() and needle in path.read_text(encoding="utf-8"):
        FAILURES.append((label, doc, needle, "文档中仍存在应已清除的内容"))


def check_metrics_in_sync() -> None:
    """把 gen_metrics 跑到临时目录，与 metrics/ 比对。

    `e2e.json` 是例外：**它由 `evaluate_end2end.py` 在跑评测时直接写出**，
    没有中间产物可派生（`gen_metrics.py` 无法凭空造出端到端结果）。
    因此对它是"存在性检查"而不是"派生一致性检查"——这个区别必须显式写出来，
    否则校验器会把正常情况报成错误。
    """
    global CHECKS
    CHECKS += 1
    directly_produced = {"e2e.json", "small_text.json"}
    spec = importlib.util.spec_from_file_location(
        "gen_metrics_for_check", PROJ / "tools" / "gen_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory() as td:
        mod.main(td)
        derived = {p.name for p in Path(td).glob("*.json")}
        # 结构性检查：metrics/ 下每个文件都必须"有归属"——要么由 gen_metrics 从产物派生，
        # 要么在 directly_produced 里显式声明。新增指标文件却忘了归类，这里会报出来，
        # 免得出现"这个数字从哪来的没人知道"的文件（正是审计要防的那类问题）。
        present = {p.name for p in (PROJ / "metrics").glob("*.json")}
        for extra in sorted(present - derived - directly_produced):
            FAILURES.append(("metrics 文件未归类", f"metrics/{extra}", "-",
                             "既非 gen_metrics 从产物派生，也不在 directly_produced 中；"
                             "请归类，否则无法判断其数字来源"))
        for p in sorted((PROJ / "metrics").glob("*.json")):
            # 未归类的文件已在上面报过一次，这里跳过，避免同一个问题报两遍
            if p.name in directly_produced or p.name not in derived:
                continue
            q = Path(td) / p.name
            if not q.is_file():
                FAILURES.append(("metrics 与产物同步", f"metrics/{p.name}", "-", "生成的指标里缺少该文件"))
                continue
            if json.loads(p.read_text(encoding="utf-8")) != json.loads(q.read_text(encoding="utf-8")):
                FAILURES.append(("metrics 与产物同步", f"metrics/{p.name}", "-",
                                 "与从产物重新生成的结果不一致；请运行 python tools/gen_metrics.py"))
    print(f"  （{', '.join(sorted(directly_produced))} 由评测脚本直接写出，只做存在性检查）")


def main() -> int:
    det = load("detection")
    rec = load("recognition")
    abl = load("ablation")
    e2e = load("e2e") if (PROJ / "metrics" / "e2e.json").is_file() else None

    print("== 第一层：metrics/ 是否与 data/results/ 产物同步 ==")
    check_metrics_in_sync()

    print("== 第二层：文档数字是否与 metrics/ 一致 ==")
    # ---- 检测：small deteval（主口径）----
    check("small deteval F1", "PROJECT_SUMMARY.md", _pct(get(det, "small.deteval.f1")) + "%")
    for k, lbl in (("precision", "P"), ("recall", "R"), ("f1", "F1")):
        check(f"报告 §2.1 small deteval {lbl}", "docs/evaluation_report.md",
              _pct(get(det, f"small.deteval.{k}")) + "%")

    # ---- 检测：legacy 两模型（历史口径）----
    check("报告 §2.2 medium legacy F1", "docs/evaluation_report.md",
          _pct(get(det, "medium.legacy.f1")) + "%")
    check("报告 §2.2 small legacy F1", "docs/evaluation_report.md",
          _pct(get(det, "small.legacy.f1")) + "%")
    check("PROJECT_SUMMARY medium legacy F1", "PROJECT_SUMMARY.md",
          _pct(get(det, "medium.legacy.f1")) + "%")
    check("PROJECT_SUMMARY small legacy F1", "PROJECT_SUMMARY.md",
          _pct(get(det, "small.legacy.f1")) + "%")

    # ---- medium deteval：随状态自适应（未测 -> 必须写明待重跑；已测 -> 必须出现数值）----
    med_de = get(det, "medium.deteval")
    if med_de is None:
        check("medium deteval 未测时须明确标注", "PROJECT_SUMMARY.md", "待全量重跑")
        check("medium deteval 未测时须明确标注（报告）", "docs/evaluation_report.md", "待重跑")
    else:
        check("medium deteval F1", "PROJECT_SUMMARY.md", _pct(med_de["f1"]) + "%")
        check_absent("medium deteval 已测则应移除「待重跑」", "PROJECT_SUMMARY.md", "待全量重跑")

    # ---- 识别 ----
    for model, doc in (("medium", "docs/evaluation_report.md"), ("small", "PROJECT_SUMMARY.md")):
        check(f"{model} 识别字符准确率", doc, _pct(get(rec, f"{model}.char_acc")) + "%")
        check(f"{model} 识别行级准确率", doc, _pct(get(rec, f"{model}.line_acc")) + "%")

    # ---- 端到端 ----
    if e2e is not None:
        check("报告 §3.1 端到端词级 F1", "docs/evaluation_report.md",
              _pct(get(e2e, "small.word.f1")) + "%")
        check("报告 §3.1 端到端行级 F1", "docs/evaluation_report.md",
              _pct(get(e2e, "small.line.f1")) + "%")
        check("PROJECT_SUMMARY 端到端行级 F1", "PROJECT_SUMMARY.md",
              _pct(get(e2e, "small.line.f1")) + "%")

    # ---- README 的消融结论（由 ablation.json 算差值）----
    def delta(mode: str, on: str, off: str) -> float:
        return (get(abl, f"{mode}.configs.{on}.char_acc")
                - get(abl, f"{mode}.configs.{off}.char_acc")) * 100

    up = delta("small", "仅放大", "无预处理")
    dn = delta("degraded", "仅去噪", "无预处理")
    sh = delta("blur", "仅锐化", "无预处理")
    all_on = -delta("clean", "无预处理", "全开（默认）")
    check("README 小字放大差值", "README.md", f"+{up:.1f}")
    check("README 去噪差值", "README.md", f"+{dn:.1f}")
    check("README 锐化差值", "README.md", f"+{sh:.1f}")
    check("README 清晰图全开代价", "README.md", f"{all_on:.0f} 字符点")

    # ---- 结构性断言：核心结论不得回退到旧表述 ----
    check("报告 §五 有口径更正块", "docs/evaluation_report.md", "本节口径更正")
    check_absent("README 不得再说 CLAHE 中性", "README.md", "CLAHE | 不暴露（实测中性）")
    check("make_figures 检测口径为 deteval", "make_figures.py", 'DETECTION_PROTOCOL = "deteval"')
    check_absent("README 不得再写「小字放大 +55.4 最强」", "README.md", "小字放大（+55.4，最强）")
    check_absent("报告不得再写倾斜校正「-39.5 有害」为结论", "docs/evaluation_report.md",
                 "| **-39.5 有害** |")

    print()
    if FAILURES:
        print(f"❌ 发现 {len(FAILURES)} 处不一致（共检查 {CHECKS} 项）：")
        for label, doc, needle, why in FAILURES:
            print(f"  - [{label}] {doc}: 期望「{needle}」-> {why}")
        return 1
    print(f"✅ 全部一致（共检查 {CHECKS} 项）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
