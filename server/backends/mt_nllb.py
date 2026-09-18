"""MT backend：NLLB-200-distilled-600M。

语种用 FLORES-200 code（如 cmn_Hans / eng_Latn）。
新增翻译模型：在同目录加一个文件并 @register 一个名字，再改 config.mt.backend。
"""
from __future__ import annotations

from server.models.base import MTModel, MTResult
from server.registry import register


@register("mt.nllb.transformers")
class NLLBTransformersBackend(MTModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model_id = cfg.get("model_id", "facebook/nllb-200-distilled-600M")
        self.device = cfg.get("device", "cuda" if _cuda_ok() else "cpu")
        self._model = None
        self._tok = None

    def load(self) -> None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self._tok = AutoTokenizer.from_pretrained(self.model_id, src_lang=self.src_lang)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, torch_dtype="auto"
        ).to(self.device).eval()

    def translate(self, text: str, src: str | None = None, tgt: str | None = None) -> MTResult:
        src = src or self.src_lang
        tgt = tgt or self.tgt_lang
        self._tok.src_lang = src
        inputs = self._tok(text, return_tensors="pt", truncation=True,
                           max_length=512).to(self.device)
        # NLLB 强制解码到目标语种
        tgt_id = self._tok.convert_tokens_to_ids(tgt)
        out = self._model.generate(**inputs, forced_bos_token_id=tgt_id,
                                   max_length=512, num_beams=4)
        text_out = self._tok.batch_decode(out, skip_special_tokens=True)[0]
        return MTResult(text=text_out.strip(), src_lang=src, tgt_lang=tgt)


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
