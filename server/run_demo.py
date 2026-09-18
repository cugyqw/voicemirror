"""最小可跑 demo：验证  ASR -> MT -> TTS 全链路。

用法（在推理环境 ~/inference-env 下）：
    cd /home/yqw/voicemirror
    python -m server.run_demo            # 用内置中文测试音频
    python -m server.run_demo path/to/audio.wav   # 指定 16k wav

输出：打印源/译文、各阶段延迟、显存峰值，并把合成音频存到 outputs/demo_out.wav。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import soundfile as sf
import yaml

# 允许 `python server/run_demo.py` 直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.core.pipeline import TranslationPipeline
from server.registry import available
from server.factory import ensure_backends_loaded


def _make_test_audio() -> tuple[np.ndarray, int]:
    """没有测试音频时，合成 3 秒 16kHz 中文普通话风格的占位音频（实际应放真人录音）。
    这里只保证链路能跑，识别结果会是噪声，仅供验证 pipeline 不死。"""
    sr = 16000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    # 多频混合，模拟语音包络
    sig = (np.sin(2 * np.pi * 200 * t) * 0.3 +
           np.sin(2 * np.pi * 400 * t) * 0.2) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
    return sig.astype(np.float32), sr


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, always_2d=False)
    wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    if np.max(np.abs(wav)) > 1.0:
        wav = wav / np.max(np.abs(wav))
    return wav.astype(np.float32), sr


def main() -> None:
    audio_arg = sys.argv[1] if len(sys.argv) > 1 else None
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, "server", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ensure_backends_loaded()
    print("[demo] 已注册 backend:", available())

    if audio_arg and os.path.exists(audio_arg):
        wav, sr = _load_audio(audio_arg)
    else:
        print("[demo] 未提供音频，使用内置占位音频（识别结果无语义，仅供验证链路）")
        wav, sr = _make_test_audio()
    assert sr == 16000, f"需 16kHz，实际 {sr}"

    pipe = TranslationPipeline(cfg)
    out = pipe.process(wav)

    print("\n===== 结果 =====")
    print("源文:", out["src"])
    print("译文:", out["tgt"])
    print("延迟:", {k: f"{v*1000:.0f}ms" for k, v in out["latency"].items()})

    # 显存峰值（若用 cuda）
    try:
        import torch
        if torch.cuda.is_available():
            mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            print(f"显存峰值: {mb:.1f} MB / 8192 MB")
    except Exception:
        pass

    os.makedirs(os.path.join(root, "outputs"), exist_ok=True)
    out_wav = os.path.join(root, "outputs", "demo_out.wav")
    sf.write(out_wav, (out["audio"] * 32767).astype(np.int16), out["sample_rate"])
    print(f"合成音频已保存: {out_wav}")


if __name__ == "__main__":
    main()
