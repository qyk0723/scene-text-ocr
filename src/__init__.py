"""场景文字检测与识别系统 —— 公共包。

这里只做一件事：把标准输出/标准错误切到 UTF-8。

审计发现 `data/results/` 下有 10 个 `.log` **无法按 UTF-8 解码**——脚本自身输出是
UTF-8，但交错的 Windows/PowerShell 提示文本用的是 OEM 代码页，单文件内混编码会让
任何 UTF-8 读取器在中途损坏（这些"原始证据"因此对工具链不可读）。

放在包初始化里，可以保证**所有入口**（`main.py` / `app.py` / `evaluate*.py` /
`make_figures.py`）都生效，不会有人忘记调用。原做法是每条命令前手动加
`PYTHONIOENCODING=utf-8`，没有固化进仓库。
"""

from __future__ import annotations

import sys


def _ensure_utf8_streams() -> None:
    """尽力把 stdout/stderr 切成 UTF-8；对象不支持时静默跳过。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):
            pass


_ensure_utf8_streams()
