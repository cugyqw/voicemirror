"""准备 TTS 参考音色样本。

F5-TTS（及其他克隆 TTS）需要一个参考音频 + 其文本来克隆音色。
本脚本把 F5 仓库自带的测试中文参考音频拷贝到 models/voice_ref/ref.wav，
并写入对应文本到 models/voice_ref/ref.txt，供 config.yaml 的 tts.ref_audio/ref_text 使用。

用法：
    python tools/prepare_ref_voice.py
    python tools/prepare_ref_voice.py /path/to/your_voice.wav "参考文本"

注意：要把中文音色样本用于说英文（VoiceMirror 核心场景），参考音频用中文即可，
F5 支持跨语种克隆。参考音频建议 3~10 秒、清晰、少噪声、16kHz 单声道。
"""
from __future__ import annotations

import os
import shutil
import sys

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_f5_ref() -> str | None:
    """在 HF 缓存里找 F5 自带的中文参考音频。"""
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.isdir(cache):
        return None
    for root, _, files in os.walk(cache):
        if "F5-TTS" in root and "ref_audio" in root:
            for f in files:
                if f.endswith(".wav") and "zh" in f:
                    return os.path.join(root, f)
    return None


def main() -> None:
    dst_dir = os.path.join(ROOT, "models", "voice_ref")
    os.makedirs(dst_dir, exist_ok=True)

    if len(sys.argv) > 1:
        src = sys.argv[1]
        ref_text = sys.argv[2] if len(sys.argv) > 2 else ""
    else:
        src = _find_f5_ref()
        ref_text = "对，这就是我，万人敬仰的太乙真人。"
        if src is None:
            print("[prepare_ref] 未找到 F5 自带参考音频，且未指定路径。")
            print("  请先确保 F5-TTS 权重已下载，或手动指定：")
            print("  python tools/prepare_ref_voice.py /path/to/voice.wav \"参考文本\"")
            return

    wav, sr = sf.read(src, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
        sr = 16000
    if abs(wav).max() > 1.0:
        wav = wav / abs(wav).max()

    dst = os.path.join(dst_dir, "ref.wav")
    sf.write(dst, wav.astype("float32"), sr)
    with open(os.path.join(dst_dir, "ref.txt"), "w", encoding="utf-8") as f:
        f.write(ref_text)
    print(f"[prepare_ref] 参考音色已写入 {dst} (16kHz, {len(wav)/sr:.1f}s)")
    print(f"[prepare_ref] 参考文本: {ref_text}")
    print(f"[prepare_ref] 请在 server/config.yaml 设置 tts.ref_audio: {dst} 与 tts.ref_text")


if __name__ == "__main__":
    main()
