"""配置加载：从项目根目录 config.yaml 读取，返回嵌套 dict。

用法::

    from src.config import get_config
    cfg = get_config()
    det_model = cfg["model"]["det"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_config: Dict[str, Any] | None = None


def load_config(path=None) -> Dict[str, Any]:
    """读取 YAML 配置（不缓存）。"""
    p = Path(path) if path else _CONFIG_PATH
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def get_config() -> Dict[str, Any]:
    """读取配置并缓存为模块级单例。"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
