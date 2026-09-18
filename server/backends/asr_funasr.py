"""ASR backend：FunASR 流式 Paraformer + FSMN-VAD。

支持通过 config 切换：
  backend: asr.funasr.paraformer   -> 流式 Paraformer（600ms 级延迟）
  backend: asr.funasr.sensevoice    -> SenseVoiceSmall（非流式，多语种+情感）

两者共用同一 AutoModel 接口，仅 model_id / 后处理不同，故放同一文件两个类。
"""
from __future__ import annotations

import numpy as np

from server.models.base import ASRModel, ASRResult
from server.registry import register


def _to_pcm16(wav32: np.ndarray) -> np.ndarray:
    """FunASR 接受 int16 PCM 或 float32。这里给 int16。"""
    wav32 = np.asarray(wav32, dtype=np.float32)
    wav32 = np.clip(wav32, -1.0, 1.0)
    return (wav32 * 32767.0).astype(np.int16)


@register("asr.funasr.paraformer")
class ParaformerStreamingBackend(ASRModel):
    """流式 Paraformer。chunk 由 fsmn-vad 自动切分，流式靠 FunASR 的 cache。"""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model_id = cfg.get("model_id", "funasr/paraformer-zh-streaming")
        self.vad_model = cfg.get("vad_model", "funasr/fsmn-vad")
        # FunASR 默认从 ModelScope 拉，国内常 404；统一走 HF（hf-mirror 已配）
        self.model_hub = cfg.get("model_hub", "hf")
        self._model = None

    def load(self) -> None:
        from funasr import AutoModel
        self._model = AutoModel(
            model=self.model_id,
            model_hub=self.model_hub,
            vad_model=self.vad_model,
            vad_model_hub=self.model_hub,
            vad_kwargs={"max_single_segment_time": 30000},
            disable_update=True,
            device="cuda:0" if _cuda_ok() else "cpu",
        )

    def transcribe(self, wav16k: np.ndarray) -> ASRResult:
        pcm = _to_pcm16(wav16k)
        res = self._model.generate(input=pcm, cache={}, cache_chunk_size=16,
                                   encoder_chunk_look_back=4, decoder_chunk_look_back=1,
                                   return_spk_res=False)
        text = res[0]["text"] if res and "text" in res[0] else ""
        return ASRResult(text=text.strip(), is_final=True)


@register("asr.funasr.sensevoice")
class SenseVoiceBackend(ASRModel):
    """SenseVoiceSmall：整段非自回归，需自切 chunk；支持情感/事件标签。"""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model_id = cfg.get("model_id", "funasr/SenseVoiceSmall")
        self.model_hub = cfg.get("model_hub", "hf")
        self._model = None
        self._post = None

    def load(self) -> None:
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        self._post = rich_transcription_postprocess
        self._model = AutoModel(
            model=self.model_id,
            model_hub=self.model_hub,
            vad_model="fsmn-vad", vad_model_hub=self.model_hub,
            disable_update=True,
            device="cuda:0" if _cuda_ok() else "cpu",
            trust_remote_code=True,
        )

    def transcribe(self, wav16k: np.ndarray) -> ASRResult:
        pcm = _to_pcm16(wav16k)
        res = self._model.generate(input=pcm, cache={}, language="auto", use_itn=True)
        raw = res[0]["text"] if res and "text" in res[0] else ""
        text = self._post(raw)
        return ASRResult(text=text.strip(), is_final=True, info={"raw": raw})


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
