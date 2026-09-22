"""命令行入口：对单张图片执行场景文字检测与识别。

用法::

    python main.py path/to/image.jpg [--lang ch] [--device cpu]
"""

from __future__ import annotations

import argparse

from src.pipeline.ocr_pipeline import SceneTextOCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="场景文字检测与识别（PP-OCR 预训练模型）"
    )
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--lang", default="ch", help="识别语言，默认 ch")
    parser.add_argument(
        "--device", default="cpu", help="推理设备，默认 cpu（可传 gpu / gpu:0）"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ocr = SceneTextOCR(lang=args.lang, device=args.device)
    boxes, texts, elapsed = ocr.run(args.image)

    print(f"检出 {len(boxes)} 个文本框，耗时 {elapsed:.3f}s")
    print("-" * 40)
    for i, (box, text) in enumerate(zip(boxes, texts), start=1):
        pts = [(round(x, 1), round(y, 1)) for x, y in box]
        print(f"[{i}] {text}")
        print(f"    box: {pts}")


if __name__ == "__main__":
    main()
