"""同传流水线编排：ASR -> MT -> TTS。

只依赖 server.models.base 的抽象接口与 server.factory，不 import 任何具体模型。
因此换模型完全由 config.yaml 控制。
"""
from __future__ import annotations

import time

import numpy as np

from server.models.base import ASRModel, MTModel, TTSModel
from server.factory import build_asr, build_mt, build_tts


class TranslationPipeline:
    def __init__(self, cfg: dict, load: bool = True):
        self.asr: ASRModel = build_asr(cfg["asr"])
        self.mt: MTModel = build_mt(cfg["mt"])
        self.tts: TTSModel = build_tts(cfg["tts"])
        if load:
            self.load()

    def load(self) -> None:
        t0 = time.time()
        self.asr.load()
        self.mt.load()
        self.tts.load()
        print(f"[pipeline] 三模型加载完成，耗时 {time.time()-t0:.1f}s")

    def process(self, wav16k: np.ndarray, ref_audio: str | None = None) -> dict:
        """端到端处理一段 16kHz 音频，返回各阶段文本与合成音频。"""
        # 1) ASR
        t = time.time()
        asr_res = self.asr.transcribe(wav16k)
        t_asr = time.time() - t

        # 2) MT（句级；后续可改成增量翻译压感知延迟）
        t = time.time()
        mt_res = self.mt.translate(asr_res.text)
        t_mt = time.time() - t

        # 3) TTS
        t = time.time()
        tts_res = self.tts.synthesize(mt_res.text, ref_audio=ref_audio)
        t_tts = time.time() - t

        return {
            "src": asr_res.text,
            "tgt": mt_res.text,
            "audio": tts_res.audio,
            "sample_rate": tts_res.sample_rate,
            "latency": {"asr": t_asr, "mt": t_mt, "tts": t_tts},
        }
