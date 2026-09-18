"""模型能力抽象基类（与具体模型无关）。

所有 backend 实现这些接口，业务编排层只依赖本文件，不依赖任何具体模型。
换模型 = 新增 backend 实现 + 在 registry 注册 + 改 config.yaml，编排层零改动。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 数据契约：统一音频格式
# ---------------------------------------------------------------------------
# 约定：进入 pipeline 的音频统一为 16kHz / 单声道 / float32，范围 [-1, 1]。
# TTS 输出的音频统一为 16kHz / 单声道 / float32，由基类负责重采样与格式归一化。

@dataclass
class ASRResult:
    """一次识别结果。流式模式下 partial=True 表示中间增量。"""
    text: str
    is_final: bool = True
    # 可选：语种、情感等，由具体 backend 填充
    info: dict = field(default_factory=dict)


@dataclass
class MTResult:
    text: str
    src_lang: str = ""
    tgt_lang: str = ""


@dataclass
class TTSResult:
    """合成音频，统一 16kHz float32。"""
    audio: np.ndarray          # shape (n_samples,), float32 in [-1, 1]
    sample_rate: int = 16000


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------
class ASRModel(ABC):
    """语音识别能力。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sample_rate: int = int(cfg.get("sample_rate", 16000))

    @abstractmethod
    def load(self) -> None:
        """加载权重到设备。具体 backend 处理 cuda / 量化 / vad 等。"""

    @abstractmethod
    def transcribe(self, wav16k: np.ndarray) -> ASRResult:
        """整段识别。wav16k: float32, 16kHz, 单声道。"""

    def transcribe_stream(self, wav16k: np.ndarray) -> Iterator[ASRResult]:
        """流式识别（可选）。默认退化为整段识别。"""
        yield self.transcribe(wav16k)


class MTModel(ABC):
    """机器翻译能力。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.src_lang: str = cfg.get("src_lang", "cmn_Hans")
        self.tgt_lang: str = cfg.get("tgt_lang", "eng_Latn")

    @abstractmethod
    def load(self) -> None:
        """加载权重到设备。"""

    @abstractmethod
    def translate(self, text: str, src: Optional[str] = None, tgt: Optional[str] = None) -> MTResult:
        """翻译文本。src/tgt 省略时用 config 默认值（FLORES code）。"""


class TTSModel(ABC):
    """语音合成（含音色克隆）能力。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out_sample_rate: int = int(cfg.get("out_sample_rate", 16000))

    @abstractmethod
    def load(self) -> None:
        """加载权重与声码器到设备。"""

    @abstractmethod
    def synthesize(self, text: str, ref_audio: Optional[str] = None,
                   ref_text: Optional[str] = None) -> TTSResult:
        """合成语音。返回统一 16kHz float32 音频（基类不强制重采样，
        backend 应主动重采样到 self.out_sample_rate）。"""
