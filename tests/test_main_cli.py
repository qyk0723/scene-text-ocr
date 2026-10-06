"""守护测试：CLI（`main.py`）的读图路径与 `--preprocess` 分支。

**对应缺陷**（审计 §待办第 2 项）：`main.py --preprocess` 用的是裸 `cv2.imread`，
是**全仓库最后一处漏网**——`cv2.imread` 在 Windows 上以 ANSI 代码页打开文件，
路径含非 ASCII 字符时**静默返回 None**。项目当年位于 `毕业设计` 目录下，
任何绝对路径都含中文，所以 `python main.py "<绝对路径>" --preprocess` 必然失败；
迁到 ASCII 路径后不再触发，但代码缺陷仍在（项目可能被放到任何位置）。

**不加载模型**：`SceneTextOCR` 被替换为假实现，因此本文件是秒级的。
"""

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

import main as main_mod

FAKE_IMG = np.full((40, 60, 3), 200, np.uint8)


class _FakeOCR:
    """假管线：记录收到的图，返回固定结果。不加载任何模型。"""

    instances: list = []

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.seen = []          # 每次 run() 收到的输入（原样记录）
        self.seen_shapes = []   # 若是数组，记录其形状
        _FakeOCR.instances.append(self)

    def run(self, image):
        self.seen.append(image)
        self.seen_shapes.append(
            np.asarray(image).shape if isinstance(image, np.ndarray) else None
        )
        return ([np.array([[1, 2], [5, 2], [5, 6], [1, 6]], np.float32)],
                ["测试"], 0.01)


class _CliCase(unittest.TestCase):
    def setUp(self):
        _FakeOCR.instances = []
        self.img = FAKE_IMG

    def _run_cli(self, argv, read_returns):
        """跑一次 main.main()，返回 (stdout, 假 OCR 实例)。

        用真实的 `parse_args()` 解析受控的 argv（临时替换 `sys.argv`），
        这样参数定义本身的改动也会被这些测试覆盖到。
        """
        buf = io.StringIO()
        with mock.patch.object(main_mod, "SceneTextOCR", _FakeOCR), \
             mock.patch.object(main_mod, "imread_unicode", return_value=read_returns) as rd, \
             mock.patch("sys.argv", ["main.py"] + argv), \
             redirect_stdout(buf):
            main_mod.main()
        self.reader = rd
        return buf.getvalue(), _FakeOCR.instances[-1]


class TestPreprocessReadsWithUnicodeHelper(_CliCase):
    """`--preprocess` 必须走 `imread_unicode`，而不是裸 `cv2.imread`。"""

    def test_uses_imread_unicode(self):
        out, ocr = self._run_cli(["某图.jpg", "--preprocess"], self.img)

        self.reader.assert_called_once_with("某图.jpg")
        self.assertIn("预处理耗时", out)
        self.assertIn("检出 1 个文本框", out)

    def test_ocr_receives_preprocessed_image_not_none(self):
        """回归：裸 cv2.imread 在非 ASCII 路径下返回 None，会让下游炸在别处。"""
        out, ocr = self._run_cli(["毕业设计/样本.jpg", "--preprocess"], self.img)

        self.assertEqual(ocr.seen_shapes, [self.img.shape],
                         "OCR 应收到预处理后的图（尺寸不变，因为三个算子不改尺寸）")
        self.assertIsInstance(ocr.seen[0], np.ndarray)

    def test_bare_cv2_imread_is_not_used_as_an_attribute(self):
        """更强的守护：main 模块里不应再出现 cv2 依赖。"""
        self.assertFalse(hasattr(main_mod, "cv2"),
                         "main.py 不应再 import cv2：读图统一走 imread_unicode")

    def test_read_failure_raises_readable_error(self):
        """读图失败（None）必须抛出带提示的错误，而不是让 None 流到下游。"""
        with self.assertRaises(FileNotFoundError) as ctx:
            self._run_cli(["坏路径.jpg", "--preprocess"], None)

        msg = str(ctx.exception)
        self.assertIn("图片读取失败", msg)
        self.assertIn("坏路径.jpg", msg)
        self.assertIn("请确认路径存在", msg)

    def test_default_path_does_not_read_image_in_main(self):
        """不加 --preprocess 时，读图由 SceneTextOCR 负责（传路径而非数组）。"""
        out, ocr = self._run_cli(["某图.jpg"], self.img)

        self.reader.assert_not_called()
        self.assertEqual(ocr.seen, ["某图.jpg"], "应把路径直接交给 OCR，而不是自己读图")
        self.assertIn("识别耗时", out)
        self.assertNotIn("预处理耗时", out)


class TestPreprocessOperatorsDoNotResize(_CliCase):
    """`main.py --preprocess` 开的三个算子都**不改变尺寸**。

    这是"输出的 box 坐标仍是原图坐标系"的前提。若将来有人往这里加
    `resize`/`upscale`，这条测试会失败，提醒必须改用 `process_with_info()`
    的 `ProcessInfo` 做坐标映射（app.py 的坐标缺陷即源于此）。
    """

    def test_enhancer_used_by_cli_keeps_image_size(self):
        from src.preprocess.enhancer import ImageEnhancer

        enhancer = ImageEnhancer(denoise=True, contrast=True, sharpen=True)
        out = enhancer.process(self.img)

        self.assertEqual(out.shape, self.img.shape)

    def test_cli_enhancer_config_matches_expectation(self):
        """用真实 enhancer 跑一遍，确认开关组合与进程里用的一致。"""
        import inspect

        src = inspect.getsource(main_mod.main)
        self.assertIn("denoise=True", src)
        self.assertIn("contrast=True", src)
        self.assertIn("sharpen=True", src)
        for resizing in ("resize=True", "upscale=True"):
            self.assertNotIn(
                resizing, src,
                f"main.py 的预处理开了 {resizing}，会改变尺寸，"
                "必须改用 process_with_info() + ProcessInfo 映射坐标",
            )


if __name__ == "__main__":
    unittest.main()
