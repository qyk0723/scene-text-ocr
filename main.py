"""命令行入口：对单张图片执行场景文字检测与识别。

用法::

    python main.py path/to/image.jpg [--lang ch] [--device cpu] [--preprocess]
"""

from __future__ import annotations

import argparse
import time

import cv2

from src.pipeline.ocr_pipeline import SceneTextOCR
from src.preprocess.enhancer import ImageEnhancer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="场景文字检测与识别（PP-OCR 预训练模型）"
    )
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--lang", default="ch", help="识别语言，默认 ch")
    parser.add_argument(
        "--device", default="cpu", help="推理设备，默认 cpu（可传 gpu / gpu:0）"
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="识别前先做图像预处理"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 默认 PP-OCRv6 small，与 Gradio 界面一致；评估基准 medium 由 evaluate.py 承担
    ocr = SceneTextOCR(
        lang=args.lang,
        device=args.device,
        det_model_name="PP-OCRv6_small_det",
        rec_model_name="PP-OCRv6_small_rec",
    )

    if args.preprocess:
        image = cv2.imread(args.image)
        if image is None:
            raise FileNotFoundError(f"图片读取失败: {args.image}")

        # 显式全开：仅作预处理对比实验用（消融显示清晰图下有害）
        enhancer = ImageEnhancer(denoise=True, contrast=True, sharpen=True)
        start = time.perf_counter()
        processed = enhancer.process(image)
        prep_elapsed = time.perf_counter() - start

        boxes, texts, ocr_elapsed = ocr.run(processed)

        print(f"预处理耗时 {prep_elapsed:.3f}s，识别耗时 {ocr_elapsed:.3f}s")
    else:
        boxes, texts, ocr_elapsed = ocr.run(args.image)
        print(f"识别耗时 {ocr_elapsed:.3f}s")

    print(f"检出 {len(boxes)} 个文本框")
    print("-" * 40)
    for i, (box, text) in enumerate(zip(boxes, texts), start=1):
        pts = [(round(x, 1), round(y, 1)) for x, y in box]
        print(f"[{i}] {text}")
        print(f"    box: {pts}")


if __name__ == "__main__":
    main()
