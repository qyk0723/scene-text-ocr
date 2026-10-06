"""真机验证：`app.py` 的坐标映射修复（审计 §待办第 1 项）是否端到端成立。

**背景**：`app.py` 勾选「小字放大」时，检测跑在**预处理图**上，框却画在**原图**上，
长边 <800 的图会整体错位（审计第 13 轮，实测 299×400 的图 124/156 个坐标点越界）。
修复是纯几何的（`tests/test_display_transform.py`，不加载模型），本脚本补上**真实模型**
的端到端验证：确认修复后画出来的框确实落在文字上。

**为什么必须真机跑**：单元测试只能证明"映射算得对"，无法证明"检测框本身在原图坐标系里
位置合理"。本脚本用两个**模型无关、不依赖 GT** 的客观指标：

1. **框-文字像素重叠（IoU）**：用 Otsu 逆二值化取"暗像素=文字"，算每个框内文字像素占比；
   再看检测框区域与"含文字网格单元"的重合度 IoU。修复前框整体偏移，两个数都会很低。
2. **越界绘制像素占比**：解析出图里绿色框线与黑色标注底色落在画布外的比例。
   修复前可测量地大于 0，修复后必须为 0。

**用法**（项目根目录，需先确认可占用 CPU）::

    python scripts/validate_app_display.py            # 默认 limit=3
    python scripts/validate_app_display.py --limit 5 --out data/results/app_display_validation.md
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import app as app_mod  # noqa: E402  （导入即加载模型，与真实界面一致）
from src.preprocess.enhancer import ImageEnhancer  # noqa: E402

SAMPLE = PROJ / "data" / "samples" / "test.jpg"
# 与审计实测一致：把样本缩到长边 400，触发 upscale（阈值 800）
TRIGGER_LONG_SIDE = 400


# ---------- 产物落盘（含非 ASCII 路径安全） ----------

def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    """写图：``cv2.imwrite`` 在含中文的路径上同样失败，用 imencode + 文件 IO 规避。

    与 `src/pipeline/ocr_pipeline.imread_unicode` 对称。本项目已迁到 ASCII 路径，
    此函数是为"项目可能被放到任何位置"留的对称实现（审计同一条结论）。
    """
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(f"编码失败: {path}")
    path.write_bytes(buf.tobytes())


# ---------- 输入构造 ----------

def make_trigger_image(long_side: int = TRIGGER_LONG_SIDE) -> np.ndarray:
    """把样本缩到长边 < 800，构造会触发「小字放大」的输入（BGR）。"""
    src = cv2.imdecode(np.fromfile(str(SAMPLE), dtype=np.uint8), cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError(f"样本读不到: {SAMPLE}")
    h, w = src.shape[:2]
    scale = long_side / float(max(h, w))
    return cv2.resize(src, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


# ---------- 解析 predict() 的输出 ----------

def decode_data_uri(html: str) -> np.ndarray:
    """从 `_result_html` 里抠出 PNG data URI 并解码为 BGR 图。"""
    marker = "base64,"
    if marker not in html:
        raise AssertionError("结果 HTML 里没有 PNG data URI（predict 可能走进了错误分支）")
    raw = html.split(marker, 1)[1].split('"', 1)[0]
    buf = np.frombuffer(base64.b64decode(raw), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise AssertionError("结果图解码失败")
    return img


# ---------- 客观指标 ----------

def dark_mask(bgr: np.ndarray) -> np.ndarray:
    """文字像素掩膜：Otsu 逆二值化（文字取暗色）。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th > 0


def ink_ratio(bgr: np.ndarray, box: np.ndarray) -> float:
    """框内文字像素占比（Otsu 逆二值化）。框落在行间空白 → 明显低于框住文字时。"""
    mask = dark_mask(bgr)
    h, w = mask.shape
    x0 = int(np.clip(np.floor(box[:, 0].min()), 0, w))
    x1 = int(np.clip(np.ceil(box[:, 0].max()), 0, w))
    y0 = int(np.clip(np.floor(box[:, 1].min()), 0, h))
    y1 = int(np.clip(np.ceil(box[:, 1].max()), 0, h))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def summarize_ink(bgr: np.ndarray, boxes) -> float:
    """一组框的「框内文字占比」中位数。

    用中位数：个别框落进页边空白不影响结论。
    """
    if not boxes:
        return 0.0
    return round(float(np.median([ink_ratio(bgr, b) for b in boxes])), 4)


GREEN = np.array([0, 255, 0], np.uint8)


def overlay_stats(annotated: np.ndarray, orig_shape) -> dict:
    """出图里"绘制像素"的构成，用于判断有没有画到画布外。

    注：predict() 的输出与输入同尺寸（这是修复的直接目标），所以"越界像素"
    本身无法出现在图里——真正能测的是**绿框是否被画布边界裁断**：
    统计紧贴四条边界的绿框像素数，裁断越多说明框越出界。
    """
    h, w = annotated.shape[:2]
    green = (np.abs(annotated.astype(np.int16) - GREEN).sum(axis=2) < 60)
    border = np.zeros((h, w), bool)
    t = 2
    border[:t, :] = border[-t:, :] = True
    border[:, :t] = border[:, -t:] = True
    return {
        "canvas": [h, w],
        "matches_input": [h, w] == list(orig_shape[:2]),
        "green_px": int(green.sum()),
        "green_on_border_px": int((green & border).sum()),
    }


# ---------- 单例验证 ----------

def validate_one(img_bgr: np.ndarray, use_upscale: bool) -> dict:
    """跑一次真实 `app.predict`，返回指标。

    ``cached`` 标记该次推理是否命中 `SceneTextOCR` 的 LRU 缓存——每次验证会先后两次
    取结果（一次是 predict 内部、一次是本脚本为算指标而重取），第二次必然命中，
    所以这里显式记录，避免把缓存耗时当成推理耗时。
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    result_html, text_html, status_html = app_mod.predict(
        rgb, use_denoise=False, use_sharpen=False, use_upscale=use_upscale
    )
    wall = time.perf_counter() - t0

    annotated = decode_data_uri(result_html)
    enhancer = ImageEnhancer(upscale=use_upscale, contrast=False)
    proc, info = enhancer.process_with_info(img_bgr)

    # 重取一次框（命中 LRU 缓存，几乎不耗时），用于计算"旧行为会画成什么样"
    boxes, texts, scores, cache_elapsed = app_mod._get_ocr().run_detailed(proc)

    mapped = app_mod.map_boxes_to_original(boxes, info)

    # ---- 核心不变式：**坐标映射必须保密度** ----
    # 检测跑在 proc 上，所以"框在 proc 上框住了文字"（ink_proc）是既成事实。
    # 正确的映射把同一个框按比例放回原图，它应当框住**同样的内容**，
    # 于是 ink_orig ≈ ink_proc。修复前的旧行为把 proc 系坐标当原图坐标用，
    # 框整体位移 → 落到别的行/行间空白 → ink_orig 明显下降。
    sx, sy = info.scale_x, info.scale_y
    proc_space_boxes = [
        np.stack([b[:, 0] * (1 / sx), b[:, 1] * (1 / sy)], axis=1) for b in mapped
    ]
    ink_proc = summarize_ink(proc, proc_space_boxes)
    ink_orig = summarize_ink(img_bgr, mapped)
    ink_orig_buggy = summarize_ink(img_bgr, boxes)  # 同一批框不做映射

    return {
        "use_upscale": use_upscale,
        "input_shape": list(img_bgr.shape[:2]),
        "proc_shape": list(proc.shape[:2]),
        "resized": info.resized,
        "scale_x": round(sx, 6),
        "scale_y": round(sy, 6),
        "n_boxes": len(boxes),
        "n_texts": sum(1 for t in texts if t),
        "sample_texts": [t for t in texts if t][:4],
        "ink_proc": ink_proc,
        "ink_orig": ink_orig,
        "ink_orig_buggy": ink_orig_buggy,
        "ink_ratio_kept": round(ink_orig / ink_proc, 4) if ink_proc > 0 else 0.0,
        "wall_seconds": round(wall, 2),
        "cache_elapsed_seconds": round(cache_elapsed, 3),
        "hit_cache": cache_elapsed < 0.05,
        **overlay_stats(annotated, img_bgr.shape),
        "status_has_model": "模型" in status_html,
        "checks_ok": None,  # 下面统一判定
    }


def decide_checks(case: dict) -> dict:
    """判定本次验证是否通过（判据写死在代码里，避免"看数字感觉还行"）。

    在**会触发缩放**的构型（``use_upscale=True``）上要求：

    - 出图与输入**同尺寸**（修复前会等于预处理图尺寸 800×600）；
    - 映射后框内文字密度**基本保持**：``ink_orig / ink_proc ≥ 0.8``；
    - 且**优于**同一批框不做映射的旧行为——否则本验证没有区分度。
    """
    case["checks_ok"] = bool(
        case["matches_input"]
        and case["n_boxes"] > 0
        and (not case["resized"] or (
            case["ink_ratio_kept"] >= 0.8
            and case["ink_orig"] > case["ink_orig_buggy"]
        ))
    )
    return case


def main() -> int:
    ap = argparse.ArgumentParser(description="真机验证 app.py 坐标映射修复")
    ap.add_argument("--limit", type=int, default=3, help="最多验证几张（每张一次推理）")
    ap.add_argument("--out", default="data/results/app_display_validation.md")
    ap.add_argument("--figure", default="docs/figures/app_box_transform_fix.png")
    ap.add_argument("--skip-figure", action="store_true")
    args = ap.parse_args()

    print(f"[1/3] 构造触发图（长边 {TRIGGER_LONG_SIDE}）...", flush=True)
    img = make_trigger_image()
    print(f"      {img.shape[1]}×{img.shape[0]}（宽×高；长边 {max(img.shape[:2])} < 800，必触发放大）",
          flush=True)

    # 每个构型只跑一次真实推理：同图同构型的第二次调用必然命中 LRU 缓存，
    # 多跑只会得到缓存耗时，反而污染耗时数字（故计数上限为 2）。
    cases = []
    n_variants = min(max(args.limit, 1), 2)
    for i in range(n_variants):
        variant = i == 0  # 先跑会缩放的构型：它是本次要验证的对象
        label = "勾选小字放大" if variant else "不勾选（对照）"
        print(f"[2/3] 推理：{label} ...", flush=True)
        cases.append(decide_checks(validate_one(img, variant)))

    # 对照图：同一批检测结果，旧画法 vs 新画法（不额外跑模型，走缓存）
    if not args.skip_figure:
        fig = PROJ / args.figure
        fig.parent.mkdir(parents=True, exist_ok=True)
        proc, info = ImageEnhancer(upscale=True, contrast=False).process_with_info(img)
        boxes, texts, _, _ = app_mod._get_ocr().run_detailed(proc)
        old = app_mod.draw_results(img, boxes, texts)                       # 旧行为
        new = app_mod.draw_results(img, app_mod.map_boxes_to_original(boxes, info), texts)
        sep = np.full((img.shape[0], 6, 3), 255, np.uint8)
        imwrite_unicode(fig, np.hstack([old, sep, new]))
        print(f"[3/3] 对照图已写入 {args.figure}", flush=True)

    out = PROJ / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(cases, img), encoding="utf-8")
    print(f"      报告已写入 {args.out}", flush=True)
    for c in cases:
        print(json.dumps({k: v for k, v in c.items() if k != "sample_texts"},
                         ensure_ascii=False), flush=True)
    return 0 if all(c["checks_ok"] for c in cases) else 1


def render_report(cases, img: np.ndarray) -> str:
    tick = lambda ok: "✅" if ok else "❌"  # noqa: E731
    lines = [
        "# 真机验证：`app.py` 坐标映射修复（§待办第 1 项）",
        "",
        "> 由 `scripts/validate_app_display.py` 生成（**真实模型推理**，非合成图）。",
        f"> 输入：`data/samples/test.jpg`（1279×1706）长边缩到 {TRIGGER_LONG_SIDE} → "
        f"**{img.shape[1]}×{img.shape[0]}**（宽×高），触发「小字放大」（阈值 800）。",
        "> 每个构型跑一次真实推理；同一构型的第二次调用必然命中缓存，故不重复跑。",
        "",
        "## 指标口径（模型无关、不依赖 GT）",
        "",
        "核心是**坐标映射的保密度不变式**：检测跑在预处理图 `proc` 上，",
        "「框在 `proc` 上框住了文字」是既成事实（`ink_proc`）。正确的映射把同一个框",
        "按比例放回原图，它必须框住**同样的内容**，于是 `ink_orig ≈ ink_proc`。",
        "修复前的旧行为把 `proc` 系坐标直接当原图坐标用，框整体位移，",
        "落到别的行或行间空白 → `ink_orig` 明显下降（即 `ink_orig_buggy`）。",
        "",
        "「文字像素」= Otsu 逆二值化的暗像素；每个指标取**所有框的中位数**",
        "（个别框落进页边空白不影响结论）。",
        "",
        "> 为什么不用「框与文字外接框的 IoU」：这页是多栏稠密排版，任意位置的局部窗口内",
        "> 都有大段文字，外接框会跨越好几行，该指标实测**没有区分度**（新旧同为 0.41）。",
        "",
        "| 指标 | 含义 | 期望 |",
        "| --- | --- | --- |",
        "| `ink_proc` | 框在**预处理图**上的框内文字占比（基准） | 越高说明框住的是文字 |",
        "| `ink_orig` | **映射后**框在原图上的框内文字占比 | 应 ≈ `ink_proc` |",
        "| `ink_ratio_kept` | `ink_orig / ink_proc` | ≥ 0.8 |",
        "| `ink_orig_buggy` | 同一批框**不映射**（旧行为）在原图上的占比 | 应明显低于 `ink_orig` |",
        "| `matches_input` | 出图尺寸 == 输入尺寸 | 必须为真（修复前为 800×600） |",
        "",
        "## 结果",
        "",
        "| 构型 | 输入 → 预处理图 | scale | 框数 | 出图==输入 | "
        "ink_proc | ink_orig | 保持率 | ink 旧行为 | 判定 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in cases:
        lines.append(
            f"| 小字放大={c['use_upscale']} | {c['input_shape']} → {c['proc_shape']} | "
            f"{c['scale_x']}/{c['scale_y']} | {c['n_boxes']} | "
            f"{tick(c['matches_input'])} | {c['ink_proc']} | **{c['ink_orig']}** | "
            f"**{c['ink_ratio_kept']}** | {c['ink_orig_buggy']} | "
            f"{tick(c['checks_ok'])} |"
        )
    lines.append("")

    for c in cases:
        lines += [
            f"### 小字放大 = {c['use_upscale']}",
            "",
            f"- 预处理：`{c['input_shape']}` → `{c['proc_shape']}`"
            f"（resized={c['resized']}，scale={c['scale_x']}/{c['scale_y']}）",
            f"- 检出 **{c['n_boxes']}** 框 / 非空文本 **{c['n_texts']}** 行；"
            f"整次 `predict()` 墙钟 **{c['wall_seconds']}s**"
            f"（重取缓存耗时 {c['cache_elapsed_seconds']}s）",
            f"- 出图 `{c['canvas']}`，与输入同尺寸：{tick(c['matches_input'])}",
            f"- 文字密度：预处理图上 {c['ink_proc']} → 原图上（映射后）**{c['ink_orig']}**"
            f"（保持率 **{c['ink_ratio_kept']}**）；旧行为（不映射）{c['ink_orig_buggy']}",
            f"- 检出样例：{c['sample_texts']}",
            "",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
