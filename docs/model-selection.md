# 模型选型调研

> 开始时间：2026-09-06
> 更新：2026-09-19（落地语音翻译全链路：ASR + MT + TTS 已跑通；清理硬件代码与冗余权重）

## 一、先定死的三条硬约束

选型前必须记住，否则选回来跑不起来：

### 1. 显存总量 8GB（RTX 5060 Ti）

三个模型（ASR / MT / TTS）在**同一次会话里都要用到**。建议总预算 **≤ 6GB**，留 2GB 给 CUDA context、cuDNN workspace、PyTorch 缓存。

如果放不下，方案是：

- **串行加载 + 显存池复用**（推荐）：哪个阶段用哪个，用完 `del` + `torch.cuda.empty_cache()`
- **量化**：int8 / int4 / GGUF
- **CTranslate2**：NLLB 有现成的 CT2 int8 版本，显存减半、速度翻倍

### 2. 音频采样率统一 16kHz

ASR（FunASR 系）原生 16kHz。让 TTS 也输出 16kHz 可以省掉一路重采样。

> 待确认：CosyVoice / F5-TTS 的原生输出采样率（常见 22.05k / 24k / 44.1k）

### 3. Blackwell（sm_120）兼容性

5060 Ti 是 Blackwell 架构，compute capability 12.0。**很多模型依赖的算子没有 sm_120 预编译包**：

| 依赖 | 风险 |
|---|---|
| flash-attn | 需确认有 sm_120 wheel，否则只能走 SDPA |
| xformers | 同上 |
| 自定义 CUDA 算子（部分 TTS） | 可能直接编译失败 |
| ONNX Runtime / TensorRT | 需对应版本支持 |

**优先选纯 PyTorch 或 ONNX 就能跑的模型**，把 flash-attn 当成可选项而非必需。

## 二、ASR（语音识别）候选

| 模型 | 参数 | 流式 | 特点 | 状态 |
|---|---|---|---|---|
| `iic/SenseVoiceSmall` | ~300M | 否（需配 VAD 切 chunk） | 非自回归，10s 音频约 70ms；中/英/日/韩/粤 + 情感 + 事件检测 | 待测 |
| `iic/paraformer-zh-streaming` | 220M | 原生流式 | FunASR 亲儿子，配 `fsmn-vad` 开箱即用 | 待测 |
| `iic/paraformer-zh` | 220M | 否 | 非流式高精度，10s 约 150ms | 待测 |
| `Qwen/Qwen3-ASR-1.7B` | 1.7B | ? | 方言支持好，但显存压力大 | 待测 |

**调研要点**

- [ ] 是否**原生支持流式**？SenseVoice 是整段非自回归，做同传必须自己切 chunk + VAD，体验上是本质差别
- [ ] 中文 + 英文的 WER（重点看带口音、带噪声的场景）
- [ ] 是否自带 VAD / 标点模型
- [ ] 输入采样率、chunk 大小与延迟的关系
- [ ] License

> 备注：`requirements.txt` 里已装 `funasr` + `modelscope`，FunASR 全家桶是阻力最小的路径。

## 三、MT（机器翻译）候选

| 模型 | 参数 | 显存(fp16) | 特点 | 状态 |
|---|---|---|---|---|
| `facebook/nllb-200-distilled-600M` | 600M | ~1.2GB | 200 语种，中英质量中等，速度最快 | 待测 |
| `JustFrederik/nllb-200-distilled-600M-ct2-int8` | 600M | **~0.6GB** | CTranslate2 int8，显存减半速度翻倍 | 待测 |
| `facebook/nllb-200-distilled-1.3B` | 1.3B | ~2.7GB | 质量更好 | 待测 |
| `facebook/m2m100-418M` | 418M | ~0.9GB | 老模型但成熟稳定 | 待测 |
| `tencent/HY-MT1.5-1.8B` | 1.8B | ~3.6GB | 腾讯混元翻译，**中文强**，有 GGUF 量化 | 待测 |
| `Qwen2.5-1.5B-Instruct` + prompt | 1.5B | ~3GB | 通用 LLM 当翻译用，质量好但自回归慢 | 待测 |

**调研要点**

- [ ] **是否支持前缀/增量翻译**（ASR 出前几个字就开始翻）—— 这是把感知延迟压到 1s 内的唯一杠杆
- [ ] 中→英、英→中 双向质量（很多模型两个方向差距很大）
- [ ] 句级翻译延迟（不是吞吐，是单句延迟）
- [ ] 是否有 CT2 / GGUF / ONNX 量化版本
- [ ] 术语/专有名词处理能力
- [ ] License

## 四、TTS + 音色克隆候选（项目差异化点，优先级最高）

| 模型 | 参数 | 显存 | 克隆 | 特点 | 状态 |
|---|---|---|---|---|---|
| `CosyVoice3`（阿里通义） | ? | ~3~5GB | 3 秒样本 | 中文最强，支持流式合成 | 待测 |
| `CosyVoice2-0.5B` | 0.5B | ~2~3GB | 支持 | 上一代，更轻量 | 待测 |
| `IndexTTS-2` | ? | ~4~6GB | 支持 | 2026 热门，中文效果好 | 待测 |
| `F5-TTS` | 335M | ~1.2~2GB | 支持 | 最轻量，中文略逊，NFE 步数可调 | 待测 |
| `fish-speech / OpenAudio` | ? | ? | 支持 | 多语言 | 待测 |

**调研要点**

- [ ] **参考音频最短时长**（3 秒 vs 10 秒，直接影响产品体验）
- [ ] **首包延迟**（合成第一个音频块的时间，目标 <500ms）
- [ ] **RTF**（实时率，必须 <1 才能实时；目标 <0.5）
- [ ] 是否支持**流式输出**（边合成边播，不等整句）
- [ ] 输出采样率（能否直接出 16kHz）
- [ ] 显存峰值（含 vocoder）
- [ ] 跨语种克隆：中文音色样本 → 说英文，效果如何（**这是 VoiceMirror 的核心场景，务必测**）
- [ ] License（CosyVoice 系列商用需仔细看）

## 五、选型结论（轻量级测试版 · 已落地）

> 原则：**先用最轻、纯 PyTorch、许可友好的组合跑通全链路**；所有模型通过统一接口封装，换模型只改配置文件、不动业务代码（详见「六之二、普适化部署架构」）。
> 场景：**语音翻译全链路**（中文语音 → 英文语音）。

| 模块 | 选定模型 | 框架 | 显存 | 权重位置 | 备注 |
|---|---|---|---|---|---|
| ASR | `funasr/paraformer-zh-streaming` + `funasr/fsmn-vad` | FunASR | ~0.5GB | `models/funasr_paraformer`、`models/funasr_vad` | 原生流式；FunASR 默认从 ModelScope 拉，国内常 404，改走 HF mirror |
| MT | `facebook/nllb-200-distilled-600M` | transformers | ~1.2GB | HF 缓存 `~/.cache/huggingface` | 200 语种；中英用 `cmn_Hans`/`eng_Latn` FLORES code；实测中→英句级 0.6s |
| TTS | `SWivid/F5-TTS`（F5TTS_v1_Base，335M） | 纯 PyTorch + Vocos | ~1.2~2GB | `models/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors` | 最轻；输出 24kHz → 重采样 16kHz；参考音色 `models/voice_ref/ref.wav` |
| **合计** | | | **~3~4GB / 8GB** | | 留 2GB+ 余量给 CUDA context |

> 注：F5 各版本权重文件名为 `model_1250000.safetensors`（v1 Base）。仓库内原有 4 份权重（Base / v1_Base / bigvgan / no_zero_init），已裁剪只留 v1 Base，省 3.9G。

**备选/升级路径（改配置即可切换，无需改代码）：**
- ASR：`SenseVoiceSmall`（多语种+情感，非流式需自切 chunk）、`Qwen3-ASR-1.7B`（方言好，显存压力大）
- MT：`nllb-200-distilled-1.3B`（质量更好）、`HY-MT1.5-1.8B`（腾讯中文强，GGUF）、`m2m100-418M`（老牌稳定）
- TTS：`IndexTTS-2`（1.5B，中文最强、情感/时长可控）、`CosyVoice2-0.5B`（阿里，流式但商用许可严）

**关键现实约束（与初版调研修正）：**
- F5-TTS **非流式**（基于固定 NFE 的 flow matching），同传场景只能「整句合成完再下发」，靠 VAD 断句 + MT 增量翻译压感知延迟；若后续要边合成边播，再切 IndexTTS-2 / CosyVoice。
- F5-TTS 预训练权重 **CC-BY-NC**（非商用），商用需自训或换 IndexTTS-2（看其许可）。
- 所有 3 个候选均**纯 PyTorch 路径可跑**，不依赖 flash-attn / xformers，规避 Blackwell sm_120 编译风险。
- TTS 输出统一重采样到 **16kHz** 与 ASR 对齐，省一路重采样。

## 六、下载与环境

```bash
# 国内镜像（HF 直连慢就加这个）
export HF_ENDPOINT=https://hf-mirror.com

# 注：FunASR 默认从 ModelScope 拉权重，国内常 404；本项目的 ASR 权重
# 已从 HF 的 funasr/* 仓库下到本地 models/ ，config 指向本地目录。
```

模型统一放 `models/`（已在 `.gitignore` 中排除，不入库）。

## 六之二、普适化部署架构（换模型不改代码）

核心思路：**业务编排层只依赖「能力接口」，不依赖具体模型**。每个模块（ASR/MT/TTS）定义抽象基类 + 配置驱动的工厂，新模型只需写一个实现类并在配置里登记。

### 1. 三层结构（实际落地）

```
server/
  core/pipeline.py    # 与模型无关：流水线编排（ASR -> MT -> TTS）
  models/base.py      # 模型能力抽象：ASRModel / MTModel / TTSModel + 结果 dataclass
  backends/           # 具体模型实现（每个文件 = 一个可插拔后端）
    asr_funasr.py     # ParaformerStreamingBackend / SenseVoiceBackend
    mt_nllb.py        # NLLBTransformersBackend
    tts_f5.py         # F5TTSBackend
  registry.py         # 后端注册表：name -> class（装饰器 @register("asr.funasr.paraformer")）
  factory.py          # build_asr/build_mt/build_tts，按 config 实例化对应后端
  config.yaml         # 模型选型集中在这里
  run_demo.py         # 最小可跑 demo（音频 -> ASR -> MT -> TTS -> wav）
```

### 2. 配置驱动（唯一需要改的地方）

```yaml
# server/config.yaml —— 换模型只改 backend 字段
asr:
  backend: asr.funasr.paraformer      # 改 "asr.funasr.sensevoice" 即换模型
  model_id: models/funasr_paraformer  # 本地目录
  vad_model: models/funasr_vad
  model_hub: hf
  sample_rate: 16000
mt:
  backend: mt.nllb.transformers       # 换翻译模型只改这里
  model_id: facebook/nllb-200-distilled-600M
  src_lang: cmn_Hans
  tgt_lang: eng_Latn
  device: cuda
tts:
  backend: tts.f5                     # 换 TTS 只改这里
  model_dir: models/F5-TTS
  ckpt_name: F5TTS_v1_Base
  ref_audio: models/voice_ref/ref.wav
  ref_text: "参考音频文本"
  out_sample_rate: 16000              # 24k -> 重采样
```

业务代码永远只写：`asr = factory.build(config.asr); text = asr.transcribe(audio)`。
加新模型 = 新增一个 `backends/xxx.py` + 在 `registry` 注册 + 改 `config.yaml`，**流水线编排零改动**。

### 3. 为什么这套组合最契合普适化

| 模型 | 服务化/接口成熟度 | 对普适化的价值 |
|---|---|---|
| FunASR | 自带 `funasr-wss-server`（WebSocket 流式）+ `openai_api`（HTTP，OpenAI 兼容） | 不用自己写 socket 服务；换 Paraformer/SenseVoice 仅是 config 字段不同 |
| NLLB | transformers 加载，FLORES 语种 code 统一 | 抽象出 `translate(src,tgt,text)`，权重源是配置项 |
| F5-TTS | 纯 PyTorch，`F5TTS.infer(ref_file, ref_text, gen_text)` 三参数 | TTS 后端统一为 `synthesize(text)`，参考音色走配置，重采样在 backend 内做 |

### 4. 接口契约（保证可替换）

- ASR：`transcribe(wav16k: np.ndarray) -> ASRResult`，流式变体 `transcribe_stream()` 返回增量 `ASRResult`。
- MT：`translate(text, src=None, tgt=None) -> MTResult`。
- TTS：`synthesize(text, ref_audio=None, ref_text=None) -> TTSResult`（统一 16kHz float32）。
- 所有后端 `load()` 内部处理设备/cuda/量化；采样率归一化在 backend 内做，业务层无感。

## 七、Benchmark 计划

选定候选后，统一测以下四项，脚本放 `tools/bench_*.py`：

| 指标 | 说明 | 目标 |
|---|---|---|
| 显存峰值 | `torch.cuda.max_memory_allocated()` | 三模型合计 ≤ 6GB |
| 首包延迟 | 输入到第一块输出的时间 | ASR <300ms / MT <300ms / TTS <500ms |
| RTF | 处理时长 / 音频时长 | 全部 <1，目标 <0.5 |
| 质量 | ASR 看 WER，MT 看人工评估，TTS 听感 + 音色相似度 | — |
