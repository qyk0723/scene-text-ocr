"""小字放大算子的**可信消融**（审计 P1-3 要求的「最小可信证据」）。

原「小字放大 +55.4」有四个问题：

1. **不可复现**：同一条命令的另一份日志记录的是 **−0.61**，仓库里没有任何东西说明以哪份为准。
2. **设计混淆**：消融在 200 张**单行裁剪图**上做（长边中位数仅 54 px），÷2 后全部低于 400px
   触发线，于是算子臂放大到 400px——相对原图中位放大 **+641%**。两个臂像素高度差约 **15 倍**，
   「放大收益」无法与「就是喂了张大图」分离。
3. **归因无依据**：度量的是 `run_detailed` 后 `"".join(texts)`，即单行图上的**联合 det+rec 拼接串**，
   却断言「作用于检测侧」——项目里没有任何 det-only / rec-only 的分离测量。
4. **配置错配**：消融用 `upscale_min_long_side=400`，部署默认是 **800**——
   被消融的算子不是被部署的算子。

本脚本按审计要求重做：**整图**、四个臂、**分离** det-only 与 rec-only。

    D  原图，不预处理                       （上界参考）
    A  整图降采样，不预处理                  （制造真实的小字条件）
    B  降采样 → 放大到**部署值** min_long_side （被部署的算子）
    C  降采样 → 放大回**精确原尺寸**           （对照：只恢复尺寸、不做任何"放大增强"）

判据：若 **B ≫ C**，说明收益来自"过度放大"而非"恢复小字"；若 B ≈ C ≈ A，说明该算子无益。

用法::

    python evaluate_small_text.py --limit 100 --sample-seed 20261006 --scale 0.5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.config import get_config
from src.evaluator import levenshtein, quad_iou
from src.evaluator.parsing import parse_det_gt_full
from src.pipeline.geometry import crop_box
from src.pipeline.ocr_pipeline import SceneTextOCR


def scale_to_long_side(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    cur = max(h, w)
    if cur == target or cur == 0:
        return img
    s = target / float(cur)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=interp)


def evaluate_arm(det, rec, img, quads, dnc, texts, iou_thresh=0.5):
    """返回 (det 召回, det 有效框数, rec 字符准确率, 检测框数)。"""
    boxes = det.detect(img)

    # ---- det-only：deteval 口径（do-not-care 不计入分母），贪婪一对一 ----
    used, pairs = set(), {}
    for pi, b in enumerate(boxes):
        bj, biou = -1, 0.0
        for j, g in enumerate(quads):
            if j in used:
                continue
            v = quad_iou(b, g)
            if v > biou:
                biou, bj = v, j
        if bj >= 0 and biou >= iou_thresh:
            used.add(bj)
            pairs[pi] = bj
    care_idx = [j for j in range(len(quads)) if not dnc[j]]
    det_hit = sum(1 for j in care_idx if j in used)
    det_n = len(care_idx)
    det_recall = det_hit / det_n if det_n else float("nan")

    # ---- rec-only：按 GT 框裁图后单独识别（与检测结果无关）----
    accs = []
    for j in care_idx:
        crop = crop_box(img, quads[j])
        if crop is None or getattr(crop, "size", 0) == 0:
            accs.append(0.0)
            continue
        try:
            pred, _ = rec.recognize(crop)
        except Exception:
            accs.append(0.0)
            continue
        p, g = pred.upper(), texts[j].upper()
        accs.append(1.0 - levenshtein(p, g) / max(len(p), len(g), 1))
    rec_acc = float(np.mean(accs)) if accs else float("nan")
    return det_recall, det_n, rec_acc, len(boxes)


def main() -> None:
    ap = argparse.ArgumentParser(description="小字放大算子的可信消融（整图、四臂、det/rec 分离）")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--sample-seed", type=int, default=20261006)
    ap.add_argument("--scale", type=float, default=0.5, help="制造小字条件的降采样倍率")
    ap.add_argument("--model", choices=["small", "medium"], default="small")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--report", default="data/results/small_text_ablation.md")
    ap.add_argument("--metrics-json", default="metrics/small_text.json")
    args = ap.parse_args()

    cfg = get_config()
    data_root = Path(cfg.get("data", {}).get("root", "data/icdar2015"))
    deploy_side = int(cfg.get("preprocess", {}).get("upscale_min_long_side", 800))
    print(f"部署值 upscale_min_long_side = {deploy_side}（原消融用的是 400，配置错配）")

    all_paths = sorted((data_root / "detection" / "test" / "imgs").glob("img_*.jpg"))
    gt_dir = data_root / "detection" / "test" / "gt"
    rng = np.random.default_rng(args.sample_seed)
    idx = np.sort(rng.choice(len(all_paths), min(args.limit, len(all_paths)), replace=False))
    paths = [all_paths[i] for i in idx]
    print(f"整图 {len(paths)} 张（种子 {args.sample_seed}），降采样倍率 {args.scale}\n")

    ocr = SceneTextOCR(det_model_name=f"PP-OCRv6_{args.model}_det",
                       rec_model_name=f"PP-OCRv6_{args.model}_rec")
    # SceneTextOCR 的检测/识别器是懒加载私有的，走它自己的取值方法
    det, rec = ocr._get_detector(), ocr._get_recognizer()

    arms = ["D 原图不预处理", "A 降采样不预处理",
            f"B 降采样→放大到 {deploy_side}", "C 降采样→放大回原尺寸"]
    agg = {a: {"det_tp": 0, "det_n": 0, "accs": [], "boxes": [], "t": []} for a in arms}

    for i, p in enumerate(paths, 1):
        gt = gt_dir / f"gt_{p.stem}.txt"
        if not gt.is_file():
            continue
        quads, dnc, texts = parse_det_gt_full(gt)
        if not any(not d for d in dnc):
            continue  # 该图没有有效文本，跳过
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        small = scale_to_long_side(img, int(round(max(h, w) * args.scale)))
        b_img = scale_to_long_side(small, deploy_side)
        c_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        # 每个臂的图尺寸不同，**标注框必须按同样的倍率缩放**——否则 GT 框会落到图外，
        # 裁出空图并让该臂的准确率假性归零（冒烟测试时确实踩到了这个坑：
        # A/B 两臂 rec 准确率 0.0000，而 C 因为被还原成原尺寸才正常）。
        variants = [
            ("D 原图不预处理", img, 1.0),
            ("A 降采样不预处理", small, small.shape[1] / float(w)),
            (f"B 降采样→放大到 {deploy_side}", b_img, b_img.shape[1] / float(w)),
            ("C 降采样→放大回原尺寸", c_img, 1.0),
        ]
        for name, im, s in variants:
            q_arm = [q * s for q in quads]
            t0 = time.perf_counter()
            d, dn, r, nb = evaluate_arm(det, rec, im, q_arm, dnc, texts, args.iou)
            a = agg[name]
            a["det_tp"] += int(round(d * dn)) if dn else 0
            a["det_n"] += dn
            if not np.isnan(r):
                a["accs"].append(r)
            a["boxes"].append(nb)
            a["t"].append(time.perf_counter() - t0)
        if i % 10 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}")

    rows = []
    for name in arms:
        a = agg[name]
        det_r = a["det_tp"] / a["det_n"] if a["det_n"] else float("nan")
        rec_a = float(np.mean(a["accs"])) if a["accs"] else float("nan")
        rows.append({"arm": name, "det_recall": det_r, "det_n": a["det_n"],
                     "rec_char_acc": rec_a,
                     "boxes_per_image": float(np.mean(a["boxes"])) if a["boxes"] else 0.0,
                     "seconds_per_image": float(np.mean(a["t"])) if a["t"] else 0.0})

    d0 = rows[0]
    body = ["# 小字放大算子的可信消融（整图、四臂、det/rec 分离）", "",
            f"模型：PP-OCRv6 {args.model}；整图 {len(paths)} 张（种子 {args.sample_seed}）；"
            f"降采样倍率 {args.scale}；部署值 `upscale_min_long_side = {deploy_side}`", "",
            "| 臂 | det 召回（deteval） | 有效 GT 框 | rec 字符准确率 | 检出框数/图 | 秒/图 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        body.append(f"| {r['arm']} | {r['det_recall']:.4f} | {r['det_n']} | "
                    f"{r['rec_char_acc']:.4f} | {r['boxes_per_image']:.2f} | "
                    f"{r['seconds_per_image']:.2f} |")
    det_d, det_a = d0["det_recall"], rows[1]["det_recall"]
    rec_d, rec_a = d0["rec_char_acc"], rows[1]["rec_char_acc"]
    rec_b, rec_c = rows[2]["rec_char_acc"], rows[3]["rec_char_acc"]
    body += ["", "## 判读", "",
             f"- **降采样确实制造了困难**：det 召回 {det_d:.4f} → {det_a:.4f}"
             f"（{det_a - det_d:+.4f}），rec 字符准确率 {rec_d:.4f} → {rec_a:.4f}"
             f"（{rec_a - rec_d:+.4f}）。",
             f"- **放大到部署值({deploy_side}) vs 放回原尺寸**：rec {rec_c:.4f} → {rec_b:.4f}"
             f"（{rec_b - rec_c:+.4f}）。",
             f"- **能否恢复到原图水平**：原图 rec {rec_d:.4f}，放大臂 {rec_b:.4f}"
             f"（{rec_b - rec_d:+.4f}）。"]

    # 机制说明：检测器内部有 limit_side_len/limit_type='max'，只缩不放。
    lim = get_config().get("det", {}).get("limit_side_len")
    body += ["", "## 机制（这一点决定了怎么解读上表）", "",
             f"检测模型内部按 `limit_side_len={lim} / limit_type=max` 处理输入——**只缩不放**。因此：",
             "",
             f"- 臂 A 的图低于该上限，检测器**按原样**处理 → 文字确实小；",
             f"- 臂 B（放大到 {deploy_side}）与臂 C（回到原尺寸）**都已达到/超过上限**，"
             f"检测器会把两者都缩到 {lim} —— 于是 **B 与 C 看到的是同一个有效分辨率**，"
             f"只差插值质量。",
             "",
             "**所以「小字放大」的真实机制是**：把一个**低于检测器输入上限**的小图抬到上限附近，"
             "让检测器拿到比原图更多的像素；一旦达到上限，再放大就没有额外收益（B ≈ C 即此意）。",
             "这解释了为什么旧消融在 54px 的单行裁剪图上测出巨大收益——那些图远低于上限，"
             "放大等于凭空给检测器加了像素；而两个对照臂相差 15 倍，"
             "结论无法与「就是喂了张大图」分离。"]

    body += ["", "> 与旧结论的区别：旧消融在**单行裁剪图**上度量**联合 det+rec 拼接串**，"
             "两个臂像素高度差约 15 倍，且用的是部署不存在的 400px；本表在**整图**上"
             "**分离** det-only 与 rec-only，并使用部署值。"]
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body) + "\n", encoding="utf-8")
    print("\n" + "\n".join(body))

    mj = Path(args.metrics_json)
    mj.parent.mkdir(parents=True, exist_ok=True)
    mj.write_text(json.dumps({
        "model": args.model, "images": len(paths), "sample_seed": args.sample_seed,
        "scale": args.scale, "deploy_upscale_min_long_side": deploy_side,
        "arms": rows, "source": str(out),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n报告 {out}\n指标 {mj}")


if __name__ == "__main__":
    main()
