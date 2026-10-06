"""命令行入口：对单张图片执行场景文字检测与识别。

用法::

    python main.py path/to/image.jpg [--device cpu] [--preprocess]
"""

from __future__ import annotations

import argparse
import time

from src.pipeline.ocr_pipeline import SceneTextOCR, imread_unicode
from src.preprocess.enhancer import ImageEnhancer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="场景文字检测与识别（PP-OCR 预训练模型）"
    )
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument(
        "--device", default="cpu", help="推理设备，默认 cpu（可传 gpu / gpu:0）"
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="识别前先做图像预处理"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 模型默认读 config.yaml（small），与 Gradio 界面一致；评估基准 medium 由 evaluate.py --model 指定
    ocr = SceneTextOCR(device=args.device)

    if args.preprocess:
        # 必须用 imread_unicode：cv2.imread 在 Windows 上以 ANSI 代码页打开文件，
        # **路径含非 ASCII 字符时静默返回 None**（审计记录：项目原先位于 毕业设计
        # 目录下，任何绝对路径都含中文，必然踩到）。项目已迁到纯 ASCII 路径，
        # 但代码不应依赖"路径恰好没有中文"。
        image = imread_unicode(args.image)
        if image is None:
            raise FileNotFoundError(
                f"图片读取失败: {args.image}\n"
                "  请确认路径存在、且是可读的图片格式（jpg/png/bmp 等）。"
            )

        # 显式全开：仅作预处理对比实验用（消融显示清晰图下有害）。
        # 注：这里只开 denoise/contrast/sharpen —— 都**不改变图像尺寸**，所以
        # 检测框与原图坐标天然一致。若将来加入 resize/upscale 等改尺寸算子，
        # 输出的 box 坐标会变成预处理图坐标系，必须按
        # `ImageEnhancer.process_with_info()` 返回的 ProcessInfo 映射回原图
        # （app.py 的坐标缺陷就是这么来的，见 docs/PROJECT_AUDIT.md 第 13 轮）。
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
