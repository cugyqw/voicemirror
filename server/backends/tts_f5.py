"""TTS backend：F5-TTS（纯 PyTorch + Vocos）。

特点：最轻（335M）、MIT 代码许可、跨语种克隆可用。
注意：
  - 预训练权重 CC-BY-NC（非商用），商用需自训或切 IndexTTS-2。
  - 非流式，输出 24kHz，基类统一重采样到 16kHz（见 _resample）。
  - 参考音色走 config：ref_audio / ref_text。
"""
from __future__ import annotations

import numpy as np

from server.models.base import TTSModel, TTSResult
from server.registry import register


@register("tts.f5")
class F5TTSBackend(TTSModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # 本地权重目录（F5 预训练 ckpt 与 vocab 放这里）
        self.model_dir = cfg.get("model_dir", "models/F5-TTS")
        self.ref_audio = cfg.get("ref_audio", "models/voice_ref/ref.wav")
        self.ref_text = cfg.get("ref_text", "")
        self.ckpt_name = cfg.get("ckpt_name", "F5TTS_v1_Base")
        self.vocab_file = cfg.get("vocab_file",
                                  f"{self.model_dir}/F5TTS_v1_Base/vocab.txt")
        self._pipe = None

    def load(self) -> None:
        # 延迟 import，避免无 torch 环境报错
        from f5_tts.api import F5TTS
        # 用显式 ckpt_file 绝对路径最稳，避免 F5TTS 在当前目录查找
        ckpt_file = f"{self.model_dir}/{self.ckpt_name}/model_1250000.safetensors"
        self._pipe = F5TTS(
            model=self.ckpt_name,
            ckpt_file=ckpt_file,
            vocab_file=self.vocab_file,
            device="cuda:0" if _cuda_ok() else "cpu",
        )

    def synthesize(self, text: str, ref_audio: str | None = None,
                   ref_text: str | None = None) -> TTSResult:
        ref_audio = ref_audio or self.ref_audio
        ref_text = ref_text if ref_text is not None else self.ref_text
        sr, wav = self._pipe.infer(
            ref_file=ref_audio, ref_text=ref_text, gen_text=text,
            speed=1.0, nfe_step=32,
        )
        audio = np.asarray(wav, dtype=np.float32)
        if sr != self.out_sample_rate:
            audio = _resample(audio, sr, self.out_sample_rate)
        return TTSResult(audio=audio, sample_rate=self.out_sample_rate)


def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """轻量重采样：优先 librosa，回退到 torchaudio。"""
    if src_sr == dst_sr:
        return wav
    try:
        import librosa
        return librosa.resample(wav.astype(np.float32), orig_sr=src_sr, target_sr=dst_sr)
    except Exception:
        import torchaudio.functional as F
        import torch
        return F.resample(torch.from_numpy(wav), src_sr, dst_sr).numpy()


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
