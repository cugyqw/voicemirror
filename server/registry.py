"""后端注册表：名称 -> 实现类。

新模型只需在自己的 backend 模块里加：
    from server.registry import register
    @register("asr.funasr.paraformer")
    class ParaformerStreamingBackend(ASRModel): ...
并确保在 factory 使用前 import 该模块（见 factory._ensure_imports）。
"""
from __future__ import annotations

from typing import Dict, Type

from server.models.base import ASRModel, MTModel, TTSModel

_REGISTRY: Dict[str, Type] = {}


def register(name: str):
    """类装饰器：把一个 backend 类登记到全局注册表。"""
    def _wrap(cls):
        if name in _REGISTRY:
            raise ValueError(f"backend name '{name}' already registered by {_REGISTRY[name]}")
        _REGISTRY[name] = cls
        return cls
    return _wrap


def get(name: str) -> Type:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend '{name}'. available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
