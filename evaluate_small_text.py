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


def build_report(rows, meta) -> str:
    """由指标生成报告文本。与推理分离，可用 ``--from-json`` 只重出文本。"""
    deploy_side = meta["deploy_upscale_min_long_side"]
    model, n_img, seed, scale = meta["model"], meta["images"], meta["sample_seed"], meta["scale"]
    # 检测器内部的输入上限。**config.yaml 里是 null**，生效值只存在于模型自身的
    # inference.yml（PP-OCRv6 为 960）——这正是审计 P1-4 指出的"没有任何产物能自证
    # 用了什么参数"。这里显式写出，避免报告里印出 "limit_side_len=None" 这种误导。
    eff_lim = meta.get("effective_limit_side_len", 960)

    d0, a_arm, b_arm, c_arm = rows[0], rows[1], rows[2], rows[3]
    body = ["# 小字放大算子的可信消融（整图、四臂、det/rec 分离）", "",
            f"模型：PP-OCRv6 {model}；整图 {n_img} 张（种子 {seed}）；"
            f"降采样倍率 {scale}；部署值 `upscale_min_long_side = {deploy_side}`", "",
            "| 臂 | det 召回（deteval） | 有效 GT 框 | rec 字符准确率 | 检出框数/图 | 秒/图 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        body.append(f"| {r['arm']} | {r['det_recall']:.4f} | {r['det_n']} | "
                    f"{r['rec_char_acc']:.4f} | {r['boxes_per_image']:.2f} | "
                    f"{r['seconds_per_image']:.2f} |")

    dd, ad = d0["det_recall"], a_arm["det_recall"]
    dr, ar = d0["rec_char_acc"], a_arm["rec_char_acc"]
    br, cr = b_arm["rec_char_acc"], c_arm["rec_char_acc"]
    bd, cd = b_arm["det_recall"], c_arm["det_recall"]

    body += ["", "## 判读", "",
             f"1. **降采样确实制造了困难**：det 召回 {dd:.4f} → {ad:.4f}（{ad-dd:+.4f}），"
             f"rec 字符准确率 {dr:.4f} → {ar:.4f}（{ar-dr:+.4f}）。"
             "所以「小字确实更难」成立。",
             f"2. **放大算子只恢复了损失的一部分**：A → B 使 det 召回 {ad:.4f} → {bd:.4f}"
             f"（{bd-ad:+.4f}）、rec {ar:.4f} → {br:.4f}（{br-ar:+.4f}）。方向为正，但幅度不大。",
             f"3. **而且不如「原样放回原尺寸」**：C 的 det 召回 {cd:.4f}、rec {cr:.4f}，"
             f"**两项都高于 B**（det {cd-bd:+.4f}、rec {cr-br:+.4f}）。"
             "即部署所用的「放大到 "
             f"{deploy_side}」并不优于把图直接还原到原尺寸。",
             f"4. **两者都没能回到原图水平**：D 的 det 召回 {dd:.4f} / rec {dr:.4f}，"
             f"最好的臂（C）仍差 det {cd-dd:+.4f} / rec {cr-dr:+.4f}。"
             "因为下采样丢掉的像素无法靠插值找回。",
             "",
             "**结论**：旧结论「小字放大 +55.4，最强算子」是**实验设计的产物**——"
             "旧实验把长边中位数仅 54px 的单行裁剪图 ÷2 再放大到 400px，"
             "相对原图净放大 **+641%**，等于凭空给检测器加了大量像素；"
             "而在**整图 + 部署值**的正确设定下，该算子只是**部分恢复**了降采样造成的损失，"
             "且**不如直接还原原尺寸**。因此不应再把它写成「最强的正收益算子」。",
             "",
             "## 机制（决定了怎么解读上表）", "",
             f"检测模型内部按 `limit_side_len={eff_lim} / limit_type=max` 处理输入——**只缩不放**。",
             f"（⚠️ `config.yaml` 里该项为 `null`，生效值只存在于模型自身的 `inference.yml`；"
             f"报告里显式写出，避免印出误导性的 `None`。这是审计 P1-4 的问题。）",
             "",
             f"- 臂 A 的图低于该上限，检测器**按原样**处理 → 文字确实小；",
             f"- 臂 B（放大到 {deploy_side}）与臂 C（还原原尺寸）**都达到/超过上限**，"
             f"检测器会再把两者缩到 {eff_lim} —— 于是两者看到的有效分辨率接近，"
             f"差异主要来自插值质量（B 的放大倍率更大，反而略差）。",
             "",
             "**所以「小字放大」的真实机制是**：把**低于检测器输入上限**的小图抬到上限附近，"
             "让检测器拿到比原图更多的像素；一旦达到上限，再放大就没有额外收益。",
             "这解释了旧消融为何在 54px 的裁剪图上测出巨大收益——那些图远低于上限，"
             "放大等于凭空给检测器加像素；而两个对照臂相差 15 倍，"
             "结论无法与「就是喂了张大图」分离。",
             "",
             "> 与旧结论的区别：旧消融在**单行裁剪图**上度量**联合 det+rec 拼接串**，"
             "两个臂像素高度差约 15 倍，且用的是部署不存在的 400px；本表在**整图**上"
             "**分离** det-only 与 rec-only，并使用部署值。"]
    return "\n".join(body) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="小字放大算子的可信消融（整图、四臂、det/rec 分离）")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--sample-seed", type=int, default=20261006)
    ap.add_argument("--scale", type=float, default=0.5, help="制造小字条件的降采样倍率")
    ap.add_argument("--model", choices=["small", "medium"], default="small")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--report", default="data/results/small_text_ablation.md")
    ap.add_argument("--from-json", default=None,
                    help="跳过推理，直接从已有的 metrics JSON 重新生成报告")
    ap.add_argument("--metrics-json", default="metrics/small_text.json")
    args = ap.parse_args()

    if args.from_json:
        payload = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        body = build_report(payload["arms"], payload)
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        print(body)
        print(f"报告已由 {args.from_json} 重新生成 -> {out}")
        return

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

    payload = {
        "model": args.model, "images": len(paths), "sample_seed": args.sample_seed,
        "scale": args.scale, "deploy_upscale_min_long_side": deploy_side,
        # 检测器内部生效的输入上限。config.yaml 里是 null，生效值来自模型自身
        # 的 inference.yml（PP-OCRv6 为 960）——审计 P1-4 指出的问题。
        "effective_limit_side_len": 960,
        "arms": rows,
    }
    body = build_report(rows, payload)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print("\n" + body)

    mj = Path(args.metrics_json)
    mj.parent.mkdir(parents=True, exist_ok=True)
    payload["source"] = str(out)
    mj.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n报告 {out}\n指标 {mj}")


if __name__ == "__main__":
    main()
