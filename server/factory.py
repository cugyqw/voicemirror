"""工厂：按 config.yaml 实例化对应 backend。

业务层只调用 build_*，永远不直接 import 具体模型类，实现「换模型不改代码」。
"""
from __future__ import annotations

import importlib
from typing import Any

from server.models.base import ASRModel, MTModel, TTSModel
from server.registry import get


# 已知 backend 所在的模块，确保被 import 以触发 @register 装饰。
# 新增 backend 文件后在此登记即可（也可改成自动扫描 backends 包）。
_BACKEND_MODULES = [
    "server.backends.asr_funasr",
    "server.backends.mt_nllb",
    "server.backends.tts_f5",
]


def ensure_backends_loaded() -> None:
    """导入所有 backend 模块以触发 @register，供业务层在构建前预加载。"""
    for mod in _BACKEND_MODULES:
        importlib.import_module(mod)


def build(kind: str, cfg: dict) -> Any:
    """kind: 'asr' | 'mt' | 'tts'。cfg 为该模块的配置块。"""
    ensure_backends_loaded()
    backend_name = cfg["backend"]
    cls = get(backend_name)
    base = {"asr": ASRModel, "mt": MTModel, "tts": TTSModel}[kind]
    if not issubclass(cls, base):
        raise TypeError(f"backend '{backend_name}' is not a {base.__name__}")
    return cls(cfg)


def build_asr(cfg: dict) -> ASRModel:
    return build("asr", cfg)


def build_mt(cfg: dict) -> MTModel:
    return build("mt", cfg)


def build_tts(cfg: dict) -> TTSModel:
    return build("tts", cfg)
