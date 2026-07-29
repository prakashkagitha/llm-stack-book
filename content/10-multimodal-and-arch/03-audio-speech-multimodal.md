# 10.3 Audio, Speech & Multimodal Fusion

Audio is the oldest natural interface between humans and machines, yet it only recently became a first-class citizen in the LLM stack. The shift happened for a simple reason: once you can convert raw audio waveforms into a stream of discrete tokens, a standard transformer can treat speech, music, and environmental sound exactly like text. This chapter traces that conversion path end to end — from raw PCM samples to codec tokens, from Whisper-style encoders to native speech LLMs, and finally to the any-to-any multimodal systems that can receive audio in and emit audio out within a single forward pass.

Related chapters you should keep open while reading this one:
- [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html) — the text-side analogue of audio tokenization.
- [Vision Transformers & Image Encoders](../10-multimodal-and-arch/01-vision-transformers.html) — vision uses the same "patch embedding then attention" skeleton.
- [Vision-Language Models](../10-multimodal-and-arch/02-vision-language-models.html) — cross-modal projection strategies directly apply to audio-language models.
- [Unified & Any-to-Any Models](../10-multimodal-and-arch/05-unified-any-to-any.html) — where speech, vision, and text converge.
- [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html) — streaming latency constraints matter acutely for speech output.

## Audio Fundamentals: What the Model Actually Sees

Before we discuss tokenization strategies, we need to understand the raw signal and the two classical representations that bridge waveforms to machine learning.

### Pulse-Code Modulation and the Waveform

A digital audio signal is a sequence of amplitude samples taken at a fixed sample rate $f_s$. Common values are 8 kHz (telephony), 16 kHz (speech models), and 44.1 kHz (music). A 10-second clip at 16 kHz produces 160,000 scalar values — one float per sample. Passing these directly to an attention layer would be impractical: even at 16 kHz the sequence length dwarfs typical text contexts.

### The Mel Spectrogram

The standard preprocessing step compresses a waveform into a 2-D time-frequency representation. Given a short-time Fourier transform (STFT) with window size $N$ and hop $H$:

$$
X[k, t] = \sum_{n=0}^{N-1} x[n + tH]\, w[n]\, e^{-j 2\pi k n / N}
$$

The power spectrogram $|X[k,t]|^2$ is then mapped through a bank of $M$ triangular Mel-scale filter banks. Mel frequency approximates the logarithmic frequency resolution of the human cochlea:

$$
m = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right)
$$

The output is an $M \times T$ matrix — typically $M=80$ or $M=128$ Mel bins and $T$ frames at roughly 10 ms per frame. A 10-second clip at 16 kHz with 25 ms windows, 10 ms hop, and 80 Mel bins yields an $80 \times 1000$ matrix: 200$\times$ shorter than the raw waveform while retaining virtually all speech-discriminative information.

```python
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

def compute_mel_spectrogram(
    waveform: torch.Tensor,   # shape: (1, num_samples)
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,          # 25 ms window at 16 kHz
    hop_length: int = 160,     # 10 ms hop at 16 kHz
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> torch.Tensor:
    """
    Returns a log-Mel spectrogram of shape (n_mels, time_frames).

    At 16 kHz and hop_length=160, each frame = 10 ms, so
    a 10-second clip produces ~1000 frames.
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,               # power spectrogram
    )
    mel = mel_transform(waveform)          # (1, n_mels, T)
    # Log-compress: clamp avoids log(0)
    log_mel = torch.log(mel.clamp(min=1e-9))
    return log_mel.squeeze(0)              # (n_mels, T)

# Quick sanity check
wav = torch.randn(1, 160000)   # 10 seconds, 16 kHz
spec = compute_mel_spectrogram(wav)
print(spec.shape)  # torch.Size([80, 1001])
```

## Audio Tokenization: Discrete Codes from Continuous Sound

The Mel spectrogram is continuous — it is a suitable input for a convolutional or transformer encoder, but not for autoregressive next-token prediction over a vocabulary. Audio tokenization solves this by learning a finite codebook of audio "words."

### Neural Audio Codecs

A neural audio codec uses a convolutional encoder, a residual vector quantizer (RVQ), and a decoder trained end-to-end to reconstruct audio from discrete codes. The landmark systems are EnCodec (Défossez et al., 2022) and SoundStream (Zeghidour et al., 2021).

**Residual Vector Quantization (RVQ).** A single vector quantizer replaces each encoder frame $\mathbf{z}_t \in \mathbb{R}^D$ with its nearest codebook entry $\mathbf{e}_{i^*}$ where $i^* = \arg\min_i \|\mathbf{z}_t - \mathbf{e}_i\|_2$. The residual $\mathbf{r}_t = \mathbf{z}_t - \mathbf{e}_{i^*}$ is then quantized by a second codebook, and so on for $K$ levels:

$$
\hat{\mathbf{z}}_t = \sum_{k=1}^{K} \mathbf{e}^{(k)}_{i^*_k}, \quad \text{with } \mathbf{r}^{(1)}_t = \mathbf{z}_t, \; \mathbf{r}^{(k+1)}_t = \mathbf{r}^{(k)}_t - \mathbf{e}^{(k)}_{i^{*(k)}}
$$

The result is $K$ integer indices per time step, each drawn from a codebook of size $C$ (typically $C=1024$). EnCodec at 24 kHz with 8 RVQ levels and a stride of 320 samples per frame produces approximately 75 frames/second, with each frame represented as 8 integers — 600 tokens/second.

```python
# Minimal RVQ forward pass (illustrative, not the full EnCodec architecture)
import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        # Codebook entries are learnable embeddings
        self.codebook = nn.Embedding(codebook_size, dim)

    def forward(self, z: torch.Tensor):
        """
        z: (B, T, D)
        Returns quantized tensor and indices.
        """
        B, T, D = z.shape
        # Flatten to (B*T, D) for distance computation
        flat = z.reshape(-1, D)
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)            # (B*T,)
        quantized = self.codebook(indices).reshape(B, T, D)
        # Straight-through estimator: gradients flow through z unchanged
        quantized_st = z + (quantized - z).detach()
        return quantized_st, indices.reshape(B, T)


class ResidualVQ(nn.Module):
    def __init__(self, num_levels: int, codebook_size: int, dim: int):
        super().__init__()
        self.levels = nn.ModuleList([
            VectorQuantizer(codebook_size, dim) for _ in range(num_levels)
        ])

    def forward(self, z: torch.Tensor):
        """
        z: (B, T, D)
        Returns: list of index tensors, one per RVQ level.
        """
        residual = z
        all_indices = []
        for vq in self.levels:
            quantized, indices = vq(residual)
            residual = residual - quantized.detach()  # compute residual
            all_indices.append(indices)              # (B, T) per level
        return all_indices  # K tensors each (B, T)
```

{{fig:rvq-residual-cascade}}

You will essentially never train your own codec. The pretrained ones ship in **HuggingFace `transformers`** (`EncodecModel`, `MimiModel`, `DacModel`), in Descript's **`descript-audio-codec`** package (DAC), and in **`kyutai-labs/moshi`** (Mimi). A full encode/decode round trip is a dozen lines, and it is the first thing to run when you are debugging an audio LM: if the codec round trip already sounds wrong, nothing downstream can be right.

```python
# Real EnCodec round trip: waveform -> integer codes -> waveform
import torch
from transformers import EncodecModel, AutoProcessor

model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

wav = torch.randn(24000 * 2)                     # 2 s of (fake) 24 kHz mono audio
inputs = processor(raw_audio=wav.numpy(),
                   sampling_rate=processor.sampling_rate,
                   return_tensors="pt")

with torch.no_grad():
    enc = model.encode(inputs["input_values"], inputs["padding_mask"],
                       bandwidth=6.0)            # 6 kbps -> 8 RVQ levels
    # audio_codes: (n_chunks, batch, num_quantizers, num_frames) integers
    print(enc.audio_codes.shape, enc.audio_codes.dtype)   # ... torch.int64
    recon = model.decode(enc.audio_codes, enc.audio_scales,
                         inputs["padding_mask"])[0]       # (batch, 1, samples)

# 2 s at 75 frames/s = 150 frames; 8 quantizers -> 1200 integers total.
print(recon.shape)
```

Those integers are the whole trick: from here on, `audio_codes` is just a `(K, T)` block of vocabulary indices, and every technique from [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html) applies unchanged.

### Token Rate and Sequence Length

!!! example "Worked example: token budget for 30 s of speech"

    EnCodec at 24 kHz with stride 320 and 8 RVQ levels:

    - Frames per second: $24000 / 320 = 75$ frames/s
    - Tokens per second (flat interleave): $75 \times 8 = 600$ tokens/s
    - 30-second utterance: $30 \times 600 = 18{,}000$ tokens

    Compare to text: a typical spoken utterance at 130 words/min for 30 s is $\approx 65$ words, or roughly 90 BPE tokens.

    The codec produces **200× more tokens than the transcript**. This is the core engineering tension: faithful audio reconstruction demands high token density, but LLM context windows are finite. Real systems use one of three mitigations:
    1. Use only the coarsest 1–2 RVQ levels for modeling semantics (the rest are predicted in parallel or recovered by a separate decoder).
    2. Use a **low-frame-rate codec**. This is where the field moved after 2024: Kyutai's Mimi (the codec inside Moshi) runs at **12.5 frames/s** rather than 75, so 8 quantizers cost 100 tokens/s instead of 600 — a 6$\times$ reduction with speech quality preserved by distilling semantic features into the first codebook. Descript's DAC, SNAC's multi-scale hierarchy, and single-codebook designs like WavTokenizer push in the same direction, typically landing on the order of 40–100 tokens/s for speech.
    3. Encode audio as a continuous vector sequence (Whisper-style) instead of discrete tokens, and quantize only for generation.

    Redo the arithmetic with Mimi: 30 s $\times$ 100 tokens/s $= 3{,}000$ tokens, roughly 33$\times$ the transcript instead of 200$\times$. That single change is what made real-time full-duplex dialogue affordable on one GPU.

### Codec Token Interleaving Patterns

When flattening $K$ RVQ levels into a 1-D token stream, there are two main strategies:

{{fig:audio-rvq-interleave-patterns}}

The delay pattern (introduced by MusicGen and adopted by most later codec LMs, e.g. Parler-TTS) allows the autoregressive model to condition each level on all previous time steps, limiting temporal lookahead. Level 1 alone captures coarse semantics; levels 2–8 refine acoustic detail. Systems that instead separate the levels into distinct *stages* — AudioLM's coarse-then-fine acoustic models, VALL-E's AR level-1 plus NAR levels 2–8 — are solving the same problem with a different factorization: both refuse to pay the $K\times$ sequence-length cost of a flat interleave.

## Whisper: The Encoder-Only Path to ASR

Whisper (Radford et al., OpenAI, 2022) remains the de facto open ASR baseline — recognition of speech as text. It takes a different philosophy from codec-based models: rather than discretizing audio into tokens for generation, it encodes a fixed 30-second window of log-Mel features into continuous hidden states and decodes with a standard text autoregressive decoder. OpenAI's 2024 `large-v3-turbo` variant (809 M params) prunes the decoder for substantially faster transcription at near-identical accuracy, and is a common default for latency-sensitive pipelines.

### Architecture

{{fig:whisper-encoder-decoder-pipeline}}

The convolutional front-end halves the temporal resolution from 3000 to 1500 frames. Each encoder block applies self-attention over these 1500 positions — note that this is always a fixed-length context regardless of actual utterance duration (silence is padded/masked). The decoder autoregressively generates transcript tokens with cross-attention back to encoder states.

```python
# Using OpenAI's whisper library for transcription
import whisper
import torch

def transcribe_with_whisper(audio_path: str, model_size: str = "large-v3"):
    """
    Load Whisper and transcribe an audio file.
    model_size: tiny | base | small | medium | large-v3
    """
    model = whisper.load_model(model_size)
    model.eval()

    # whisper.load_audio handles resampling to 16 kHz mono
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)  # pad/trim to exactly 30 s

    # Compute log-Mel spectrogram on the same device as the model.
    # IMPORTANT: the number of Mel bins is a per-checkpoint property —
    # 80 for tiny..medium, 128 for large-v3 and large-v3-turbo. Hard-coding
    # 80 will feed a large-v3 encoder the wrong input shape.
    n_mels = model.dims.n_mels
    mel = whisper.log_mel_spectrogram(audio, n_mels).to(model.device)  # (n_mels, 3000)

    # Detect language (optional)
    _, probs = model.detect_language(mel.unsqueeze(0))
    lang = max(probs, key=probs.get)
    print(f"Detected language: {lang}")

    # Decode with greedy or beam search
    options = whisper.DecodingOptions(language=lang, fp16=True)
    result = whisper.decode(model, mel, options)
    return result.text

# Example:
# text = transcribe_with_whisper("interview.wav", model_size="base")
```

In production almost nobody uses the reference `openai/whisper` package. The two libraries that matter are **HuggingFace `transformers`** (`WhisperForConditionalGeneration` + `AutoProcessor`, which handles long-form audio by chunking with overlap and gives you batching, `torch.compile`, and Flash-Attention kernels for free) and **`faster-whisper`** (a CTranslate2 reimplementation with INT8/FP16 weights that is several times faster at the same word error rate, and the usual choice for CPU or single-GPU real-time transcription). `WhisperX` adds forced alignment for word-level timestamps and `pyannote.audio` diarization on top.

```python
# The mainstream path: transformers ASR pipeline (handles >30 s audio by chunking)
import torch
from transformers import pipeline

asr = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device="cuda:0",
    chunk_length_s=30,      # sliding 30 s windows for long-form audio
    stride_length_s=5,      # overlap so words at chunk seams are not lost
)
out = asr("interview.wav", return_timestamps=True, batch_size=8)
# out["text"] -> full transcript; out["chunks"] -> [{"timestamp": (t0, t1), "text": ...}, ...]
```

### CTC: The Encoder-Only Alternative

Whisper is a sequence-to-sequence model, but the other major ASR family — wav2vec 2.0, HuBERT fine-tuned for recognition, and most streaming production recognizers — uses **Connectionist Temporal Classification (CTC)**. CTC keeps only the encoder: it emits one distribution over characters (plus a special blank symbol $\varnothing$) per acoustic frame, and defines the probability of a transcript $y$ as the sum over every frame-level alignment $a$ that collapses to it:

$$
p(y \mid x) = \sum_{a \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} p(a_t \mid x)
$$

where $\mathcal{B}$ deletes blanks and merges repeated symbols ("h h $\varnothing$ i" $\to$ "hi"). The sum has exponentially many terms but is computed in $O(T \cdot |y|)$ by a forward–backward dynamic program — the same algorithm as an HMM forward pass. PyTorch ships it as `torch.nn.functional.ctc_loss`, so fine-tuning `Wav2Vec2ForCTC` on your own labelled audio is a handful of lines.

CTC's conditional independence across frames (no autoregressive decoder) is exactly why it is fast, streamable, and immune to the hallucinated-text failure mode that seq2seq decoders like Whisper exhibit on silence or music; it is also why plain CTC has no language model and needs an external one (beam search with a KenLM n-gram, or a shallow-fused neural LM) to reach competitive accuracy.

```python
# Measuring what actually matters: word error rate
import jiwer

reference  = "the quick brown fox jumps over the lazy dog"
hypothesis = "the quick brown fox jumped over a lazy dog"

# Normalize before scoring — casing and punctuation otherwise dominate WER
transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])
wer = jiwer.wer(reference, hypothesis,
                reference_transform=transform, hypothesis_transform=transform)
print(f"WER = {wer:.3f}")   # 2 substitutions / 9 reference words = 0.222
```

WER is edit distance (substitutions + insertions + deletions) divided by reference length, so it is unbounded above; the normalization step matters more than people expect, which is why HuggingFace's Open ASR Leaderboard pins a single text normalizer across all submitted models. Audio has no `lm-evaluation-harness` equivalent with the same gravitational pull — the community standards are that leaderboard for ASR, and toolkits like **ESPnet**, **SpeechBrain**, and **NVIDIA NeMo** for training and scoring recipes end to end.

### Whisper as a Feature Extractor

For downstream tasks — speech LLMs, speaker diarization, voice cloning — we often want encoder hidden states rather than the text output. The encoder's final hidden states are rich acoustic representations that have been used in place of (or alongside) codec tokens.

```python
import torch
import whisper
from whisper.model import AudioEncoder

def extract_whisper_features(
    mel: torch.Tensor,   # (batch, model.dims.n_mels, 3000) on GPU
    model: whisper.Whisper,
) -> torch.Tensor:
    """
    Return encoder hidden states: (batch, 1500, encoder_dim).
    For whisper-large-v3, n_mels = 128 and encoder_dim = 1280;
    for whisper-base, n_mels = 80 and encoder_dim = 512.
    """
    with torch.no_grad():
        # model.encoder is a standard TransformerEncoder
        hidden = model.encoder(mel)   # (B, 1500, D)
    return hidden
```

## Text-to-Speech: Neural Vocoders and Codec-Based TTS

Text-to-Speech (TTS) synthesis has converged on two paradigms in the LLM era: (1) neural vocoder pipelines and (2) codec LM pipelines.

### Neural Vocoder Pipelines

Classic neural TTS decomposes the problem:

{{fig:tts-vocoder-pipeline}}

HiFi-GAN (Kong et al., 2020) is a GAN-based vocoder trained to invert Mel spectrograms to waveforms with high perceptual quality. The generator is a series of transposed convolutions with multi-receptive-field fusion (MRF) blocks. Training uses a combination of multi-period and multi-scale discriminators.

This paradigm is far from obsolete, and it is the one to reach for on a small budget: end-to-end non-autoregressive systems in the VITS/StyleTTS family run in the tens of millions of parameters and synthesize far faster than real time on a CPU. `transformers` serves VITS and Meta's MMS-TTS checkpoints (`VitsModel`) directly, `coqui-ai/TTS` and `speechbrain` package the full train/finetune recipes, and the 2025 open-weight Kokoro model is roughly 82 M parameters — the same order as the capstone's Stack-100M. Codec LMs win on zero-shot voice cloning and expressive prosody; vocoder pipelines win on latency, footprint, and determinism.

### Codec LM Pipelines: VALL-E and Relatives

VALL-E (Wang et al., Microsoft, 2023) reframes TTS as a language modeling problem over codec tokens. Given a 3-second acoustic prompt and a text transcript, VALL-E:

1. **Predicts coarse tokens (AR stage):** Autoregressively models EnCodec level-1 tokens conditioned on text BPE tokens and the acoustic prompt. This captures prosody and speaker identity.
2. **Predicts fine tokens (NAR stage):** Non-autoregressively predicts RVQ levels 2–8 conditioned on level 1 and all other context. This fills in acoustic detail in $O(1)$ parallel steps.

The conceptual architecture:

{{fig:valle-ar-nar-architecture}}

The key insight: level-1 tokens determine *what* is said and *how* (prosody, speaker style); higher levels determine the acoustic rendering quality. This decomposition separates semantic control from acoustic fidelity.

## Speech Language Models: Audio-In, Audio-Out

The logical endpoint of audio tokenization is a model that accepts and emits audio tokens natively — no ASR transcription step, no TTS synthesis step, just raw audio tokens flowing through a standard transformer. We call these "speech LMs" or "audio LLMs."

### SpeechTokenizer and Semantic-Acoustic Disentanglement

A pure codec tokenizer conflates semantic content with acoustic style in its first RVQ level. SpeechTokenizer (Zhang et al., 2023) addresses this by training the first codebook to align with HuBERT semantic features (see below), forcing it to capture *what was said* while higher levels capture *how it was said*.

The training objective adds a distillation term:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reconstruct}} + \lambda \cdot \mathcal{L}_{\text{semantic}}
$$

where $\mathcal{L}_{\text{semantic}}$ encourages VQ level-1 outputs to match HuBERT's discrete pseudo-labels. This disentanglement makes level-1 tokens a drop-in replacement for text tokens in a speech LM.

### HuBERT: Self-Supervised Acoustic Units

HuBERT (Hsu et al., Facebook AI Research, 2021) is a BERT-style masked prediction model for speech. It uses k-means clusters of MFCC or previous-iteration HuBERT features as "pseudo-labels" and trains a transformer encoder to predict the cluster assignments of masked frames. The result is a rich contextual representation of speech that has been widely used as:
- Discrete token targets for speech LMs
- Features for speech-text joint embedding spaces
- Semantic units for zero-shot TTS

### The AudioPaLM and Moshi Architectures

**AudioPaLM** (Rubenstein et al., Google, 2023) interleaves audio tokens and text tokens in the same token stream fed to a pre-trained PaLM language model. Audio tokens use a separate embedding table; text and audio share the same positional encoding and transformer blocks. This enables a single model to perform ASR, TTS, and speech-to-speech translation in a unified framework.

**Moshi** (Défossez et al., Kyutai, 2024) goes further: it is designed for real-time full-duplex spoken dialogue, meaning the model continuously emits audio while simultaneously listening. Three ideas make it work:

- **Mimi, a 12.5 Hz streaming codec.** One frame every 80 ms, 8 quantizers, with the first codebook distilled from a self-supervised speech model so it carries semantics. The low frame rate is what puts a full conversation inside a normal context window.
- **A two-transformer hierarchy.** A large *temporal* transformer (Moshi's 7 B-parameter Helium text LM backbone) advances one step per 80 ms frame; a small *depth* transformer then autoregresses over the $K$ codebooks *within* that frame. Cost therefore scales with frames, not with $K \times$ frames.
- **Two parallel audio streams plus "Inner Monologue."** Moshi models its own voice and the user's voice as separate token streams at every step — that is what "full duplex" means mechanically: there is no turn variable, both streams always exist, so barge-in and backchannels are just ordinary predictions. Alongside its own audio stream it also predicts time-aligned *text* tokens as a prefix to the audio of each frame, which measurably improves linguistic quality — the model quite literally thinks in text a beat before it speaks.

{{fig:moshi-temporal-hierarchy}}

The crucial engineering decision in Moshi: **the temporal transformer runs causally over frames** — it only attends to past codec frames, enabling true streaming with no lookahead. The depth transformer runs $K$ tiny steps inside each frame, keeping per-step latency bounded; the reported theoretical latency is around 160 ms (one 80 ms frame of codec delay plus one frame of compute).

By 2025–2026 this native-audio approach scaled into full any-to-any "omni" backbones. Qwen3-Omni (Alibaba, 2025), for instance, ingests text, audio, image, and video and emits text plus streaming speech from a single model — using a Thinker–Talker split and time-aligned position embeddings — reaching open-source SOTA on most audio benchmarks. Speech dialogue is increasingly a capability folded into one multimodal model rather than a standalone speech LM (see [Unified & Any-to-Any Models](../10-multimodal-and-arch/05-unified-any-to-any.html)).

## Real-Time Streaming and Latency Budgets

Speech interfaces have tight latency requirements. A phone conversation feels natural below about 200 ms round-trip delay. For a voice assistant the target is typically under 500 ms from end-of-speech to first audio output.

### The Latency Decomposition

{{fig:latency-budget-decomposition}}

For speech LMs that skip the ASR/TTS steps, the budget collapses to:

```text
VAD + LLM audio-token TTFT + codec decode ≈ 150–300 ms
```

### Streaming Architectures

The standard approach to low-latency TTS uses chunk-by-chunk generation: the LM produces audio codec tokens in small batches (e.g., 25 tokens = ~333 ms of audio at 75 fps), which are decoded and streamed to the audio output device while generation continues.

```python
import asyncio
import queue
import threading
from typing import Iterator, AsyncIterator

class StreamingTTSPipeline:
    """
    Minimal sketch of a streaming codec-LM TTS pipeline.
    The LM generates codec tokens; a separate thread decodes and plays.
    """

    def __init__(self, lm, codec_decoder, chunk_size: int = 25):
        """
        lm: a language model that yields codec token ids one at a time
        codec_decoder: converts a buffer of codec tokens → waveform chunk
        chunk_size: number of codec frames per audio chunk (25 ≈ 333 ms)
        """
        self.lm = lm
        self.codec_decoder = codec_decoder
        self.chunk_size = chunk_size

    def generate_tokens(self, text_tokens: list[int]) -> Iterator[list[int]]:
        """
        Yields one codec frame (a list of K RVQ token ids) at a time.
        This is a generator — the LM samples one step at a time.
        """
        # (In practice: call lm.generate() with streaming enabled)
        for frame_tokens in self.lm.stream(text_tokens):
            yield frame_tokens  # list of K ints, one per RVQ level

    async def run(self, text: str):
        audio_queue = asyncio.Queue()
        text_tokens = self.lm.tokenize(text)

        async def producer():
            buffer = []
            for frame in self.generate_tokens(text_tokens):
                buffer.append(frame)
                if len(buffer) >= self.chunk_size:
                    # Decode chunk_size codec frames to waveform
                    waveform = self.codec_decoder.decode(buffer)
                    await audio_queue.put(waveform)
                    buffer = []
            # Flush remainder
            if buffer:
                waveform = self.codec_decoder.decode(buffer)
                await audio_queue.put(waveform)
            await audio_queue.put(None)  # sentinel

        async def consumer():
            while True:
                waveform = await audio_queue.get()
                if waveform is None:
                    break
                # In a real system: write to audio output device
                print(f"[audio] playing {len(waveform)/24000:.3f}s chunk")

        await asyncio.gather(producer(), consumer())
```

!!! interview "Interview Corner"

    **Q:** You are designing a voice assistant that must respond within 400 ms. The LLM backbone typically has a 300 ms time-to-first-token. How do you architect the audio pipeline to meet the latency target, and what tradeoffs are involved?

    **A:** The core idea is to overlap as much work as possible:

    1. **End-of-speech detection (VAD):** Use a lightweight model (e.g., Silero VAD, ~1 ms) running continuously. Begin the LLM prefill as soon as VAD fires — do not wait for the ASR transcript.
    2. **Streaming ASR or direct audio input:** If using ASR + LLM, pipeline them: send partial ASR hypotheses to the LLM as a speculative prefix (a "beam speculation" trick). If using an audio-LLM (Whisper encoder → LLM), the encoder runs in ~50 ms on a GPU for short utterances.
    3. **Speculative first-chunk TTS:** As soon as the LLM emits its first ~20 tokens (a fraction of a second), start TTS synthesis of that prefix while the rest generates. Many responses start with short filler tokens ("Sure," "Of course,") which can be pre-synthesized.
    4. **On-device vs. server:** Moving VAD and even the TTS vocoder on-device eliminates the network round-trip for the audio output path (~30–80 ms saved).
    5. **Tradeoffs:** Speculative prefixes can be wrong (the LLM may back-track), causing an awkward audio gap. Pre-synthesized fillers can feel robotic. Using direct audio-in/audio-out avoids ASR/TTS latency but requires a larger, more expensive model.

    For a typical deployment: VAD (10 ms) + audio encoder (50 ms) + LLM TTFT (200 ms, optimized with continuous batching and FlashAttention) + first-chunk codec decode (30 ms) = 290 ms — achievable on a single A100.

## The Multimodal Token-Stream View

The cleanest conceptual model of audio-language fusion treats all modalities as **token sequences** feeding into a shared autoregressive transformer. Different modalities get different embedding tables and, potentially, different positional encodings, but share all transformer weights.

{{fig:multimodal-shared-token-stream}}

### Projection and Alignment Strategies

For *encoder-decoder* style audio fusion (e.g., Whisper encoder + LLM decoder):

1. **Linear projection:** A single learnable $D_{\text{audio}} \to D_{\text{LLM}}$ matrix maps audio encoder states directly to the LLM's embedding space. Fast and effective for well-aligned modalities.
2. **Q-Former (borrowed from BLIP-2):** A small cross-attention module with $N$ learnable query tokens extracts a compressed, fixed-length summary of the audio, then passes these $N$ vectors to the LLM. Reduces the token count regardless of audio length.
3. **Perceiver resampler:** A more flexible version of Q-Former with a learned set of latent vectors attending to audio frames, used in Flamingo-style architectures.

```python
import torch
import torch.nn as nn

class AudioQFormer(nn.Module):
    """
    Minimal Q-Former: compress variable-length audio features to
    a fixed set of N query tokens via cross-attention.
    Used to bridge an audio encoder (e.g., Whisper) to an LLM.
    """

    def __init__(
        self,
        n_queries: int = 32,    # number of output tokens fed to LLM
        audio_dim: int = 1280,  # Whisper-large encoder dim
        llm_dim: int = 4096,    # LLM embedding dim
        n_heads: int = 8,
        n_layers: int = 2,
    ):
        super().__init__()
        # Learnable query tokens — these become the LLM input
        self.queries = nn.Parameter(torch.randn(1, n_queries, llm_dim))
        # Cross-attention layers: queries attend to audio features
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=llm_dim,
                num_heads=n_heads,
                kdim=audio_dim,
                vdim=audio_dim,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(llm_dim) for _ in range(n_layers)])

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        audio_features: (B, T_audio, audio_dim) — e.g., Whisper encoder output
        Returns: (B, n_queries, llm_dim) — fixed-length token sequence for LLM
        """
        B = audio_features.shape[0]
        q = self.queries.expand(B, -1, -1)   # (B, N, llm_dim)

        for cross_attn, norm in zip(self.cross_attn_layers, self.norms):
            # Cross-attend: queries (Q) attend to audio features (K, V)
            attended, _ = cross_attn(q, audio_features, audio_features)
            q = norm(q + attended)           # residual + layer norm

        return q  # (B, 32, 4096) — drop into LLM as prefix tokens
```

!!! tip "Practitioner tip: giving a small text LM ears"

    This is the cheapest real multimodal project in the book, and it is entirely within reach of the capstone model. Take the frozen Stack-100M decoder from [The Stack-100M Architecture](../14-capstone/04-architecture.html), freeze a `whisper-small` encoder next to it, and train *only* a two-layer MLP projector from the encoder's 768-dim states into the decoder's `d_model` — on the order of a million trainable parameters. Average-pool the encoder output by 4 to keep the audio prefix near 375 tokens, format each example as `<audio prefix> + "Transcribe:" + transcript`, and mask the loss to the transcript tokens exactly as in the SFT recipe of [Post-Training Stack-100M](../14-capstone/09-post-training.html). A few thousand hours of LibriSpeech/Common Voice gets a usable, if unimpressive, speech-in model; only then is it worth unfreezing the encoder's top blocks at a 10$\times$ lower learning rate.

### Handling Variable-Length Audio in a Batch

One practical challenge is batching audio inputs of different lengths. There are two strategies:

- **Fixed-length window (Whisper style):** Pad or truncate every input to 30 s. Simple and avoids dynamic shapes. The downside: short utterances waste compute; long ones are truncated.
- **Dynamic chunking:** Split long audio into overlapping 30-second chunks, process each independently, then concatenate encoder outputs before passing to the LLM. Must handle boundary artifacts.

## Multimodal Fusion: Audio + Vision + Text

Full multimodal systems must fuse audio, visual, and text streams. The architectures converge on a unified pattern: each modality has a dedicated encoder, a modality-specific projection/adapter, and a shared LLM backbone that receives all modalities as token sequences.

{{fig:unified-avl-fusion-architecture}}

The LLM backbone sees a prefix of "soft tokens" from each modality adapter, followed by text tokens forming the instruction or dialogue. Self-attention is unrestricted across all tokens, allowing the LLM to cross-attend between audio and visual context.

### AudioCaps, LibriSpeech, and Evaluation

Multimodal audio models are evaluated on:

| Task | Dataset | Metric |
|------|---------|--------|
| ASR | LibriSpeech | WER (word error rate) |
| Audio captioning | AudioCaps | CIDEr, METEOR |
| Speech translation | CoVoST-2 | BLEU |
| Zero-shot TTS similarity | LibriSpeech test-clean | SECS (speaker cosine sim) |
| Speech emotion recognition | IEMOCAP | Weighted accuracy |

All of these are loadable through HuggingFace `datasets`, whose `Audio` feature decodes and resamples lazily — `load_dataset("librispeech_asr", "clean", split="test", streaming=True).cast_column("audio", Audio(sampling_rate=16_000))` gives you 16 kHz `numpy` arrays without ever materializing the corpus on disk. That streaming path matters: audio corpora are one to two orders of magnitude larger per hour of content than the text corpora of [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html).

!!! warning "Common pitfall"

    When fine-tuning an audio-LLM end-to-end, the audio encoder's early layers can catastrophically forget their pre-trained acoustic representations if the learning rate is too high. Standard practice: use a much lower learning rate for the encoder (e.g., 1e-5) than for the projection and LLM layers (e.g., 1e-4), or freeze the encoder entirely for the first few thousand steps.

### Audio Tokens in a Shared Vocabulary

When audio codec tokens and text BPE tokens share a single vocabulary table, the LLM must simultaneously learn to predict audio continuation ("what sound comes next?") and text continuation ("what word comes next?"). A critical design choice is **vocabulary partitioning**: audio tokens occupy a contiguous slice of the vocab (e.g., indices 50265–51288 for a 1024-code codec), so the output head's softmax operates over a larger space.

The cross-entropy loss during training typically applies different weights to audio vs. text tokens, since audio token sequences are much longer and can dominate the gradient signal:

```python
def multimodal_cross_entropy(
    logits: torch.Tensor,    # (B, T, vocab_size)
    targets: torch.Tensor,   # (B, T) integer targets
    token_type: torch.Tensor, # (B, T) — 0=text, 1=audio
    audio_weight: float = 0.1,
) -> torch.Tensor:
    """
    Weighted cross-entropy that down-weights audio tokens.
    Without weighting, 600 audio tokens/s vs ~3 text tokens/s
    means audio dominates the loss 200:1, degrading text quality.
    """
    B, T, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    flat_type = token_type.reshape(-1)

    # Per-token loss (no reduction)
    loss_per_token = torch.nn.functional.cross_entropy(
        flat_logits, flat_targets, reduction="none"
    )  # (B*T,)

    # Weight mask: text tokens get weight 1.0, audio tokens get audio_weight
    weights = torch.where(flat_type == 0,
                          torch.ones_like(loss_per_token),
                          torch.full_like(loss_per_token, audio_weight))

    return (loss_per_token * weights).sum() / weights.sum()
```

## Any-to-Any Audio Generation: Music, Environmental Sound, and Voice

Beyond speech, audio LLMs can model music and environmental sound by replacing the speech-specific encoder with a general-purpose codec (EnCodec or DAC) and training on diverse audio datasets.

### MusicGen

MusicGen (Copet et al., Meta AI, 2023) is a single autoregressive transformer trained on EnCodec tokens from licensed music. Its contribution is a systematic study of **codebook interleaving patterns** — flat, parallel, delayed, and coarse-first — and the finding that the *delay* pattern is the best quality/compute trade-off. One transformer step emits all $K$ codebooks for the current column through $K$ separate output heads; because level $k$ is shifted right by $k$ frames, each head still gets to condition on the coarser levels of the same acoustic instant, without the $K\times$ longer sequence a flat interleave would require. (The alternative — a small depth transformer autoregressing within the frame, as in Moshi and UniAudio — buys exact within-frame conditioning at the cost of an extra network.)

```python
from audiocraft.models import MusicGen
import soundfile as sf
import numpy as np

def generate_music(
    description: str,
    duration_seconds: float = 10.0,
    model_name: str = "facebook/musicgen-small"   # small | medium | large | melody
) -> np.ndarray:
    """
    Generate music conditioned on a text description.
    Returns float32 audio at 32 kHz.
    """
    model = MusicGen.get_pretrained(model_name)
    model.set_generation_params(duration=duration_seconds)

    # Condition on text description
    wav = model.generate([description])   # (1, 1, T) float tensor
    audio = wav[0, 0].cpu().numpy()       # (T,) float32 at 32 kHz
    return audio

# Example (requires audiocraft installed):
# audio = generate_music("upbeat jazz piano with drums, 120 BPM", duration_seconds=15)
# sf.write("output.wav", audio, 32000)
```

### AudioLM: The Hierarchical Approach

AudioLM (Borsos et al., Google, 2022) pioneered the hierarchical two-stage approach specifically for long-form audio generation:

1. **Semantic modeling:** An autoregressive LM over k-means clusters of w2v-BERT features (semantic tokens). This captures long-range structure — melody, prosody, content — with a compact ~50 token/s rate.
2. **Acoustic modeling:** Two coarse-to-fine codec LMs that condition on semantic tokens and progressively generate EnCodec tokens at increasing bitrate.

The key insight: semantic tokens are far more compressible than acoustic tokens. A 30-second clip requires only ~1,500 semantic tokens but ~18,000 EnCodec tokens. By modeling semantics first, the LM can plan global structure before committing to acoustic details.

!!! note "Connection to language modeling"

    The AudioLM hierarchy mirrors the byte-pair encoding intuition in [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html): coarser representations capture structure; finer representations capture surface form. The same trade-off appears throughout the stack — from BPE merges to RVQ levels.

## Key Design Patterns and Engineering Considerations

### Positional Encoding for Audio

Audio frames have a meaningful temporal ordering at multiple time scales: within-word phoneme sequences (10–100 ms), prosodic phrases (0.5–5 s), and discourse structure (>5 s). Standard RoPE or learned absolute positional embeddings work at the token level. When mixing audio and text tokens in the same sequence, the positional encoding must remain monotone across modality boundaries — a common source of subtle bugs.

### Speaker and Style Conditioning

Zero-shot TTS requires the model to clone a speaker's voice from a brief prompt without fine-tuning. The two dominant approaches are:

1. **Acoustic prompt prefix:** Prepend the acoustic prompt's codec tokens directly to the context (VALL-E style). The LM learns to match the style of its prefix.
2. **Speaker embedding injection:** Encode the prompt with an independent speaker encoder (e.g., d-vector or x-vector), and add the speaker embedding to every token's residual stream.

### Data Efficiency and Pretraining Strategies

High-quality paired audio-text data (e.g., studio-recorded audiobooks) is scarce. State-of-the-art systems use:

- **Semi-supervised pretraining:** Pretrain the audio encoder on large unlabeled audio corpora (HuBERT, wav2vec 2.0), then fine-tune the full pipeline on smaller paired data.
- **Pseudo-labeling:** Run a large ASR system on unlabeled audio to generate transcript "labels," then train a smaller model on the pseudo-labeled corpus.
- **Cross-modal transfer:** Initialize the audio LM from a strong text LLM checkpoint, freeze the LLM, and train only the audio adapter initially. The LLM's linguistic priors transfer even without audio pretraining.

{{fig:audio-llm-transfer-curriculum}}

!!! tip "Practitioner tip"

    When debugging a speech LM, always check word error rate on a standard ASR benchmark (e.g., LibriSpeech test-clean) throughout training. A sudden spike in WER indicates that the model has lost its speech understanding capability — often caused by a learning rate that is too high for the audio encoder, or by an accidental change in the audio preprocessing pipeline (normalization, resampling).

!!! key "Key Takeaways"

    - Audio is made machine-learnable as either continuous Mel spectrograms (for encoder models like Whisper) or discrete codec tokens via Residual Vector Quantization (for generative models like VALL-E, AudioLM, MusicGen).
    - Neural audio codecs (EnCodec, SoundStream) compress audio to ~75 frames/second with 8 RVQ levels, producing approximately 600 tokens/second — far more than text, creating a fundamental token-budget tension. Low-frame-rate codecs (Mimi at 12.5 frames/s, ~100 tokens/s) are the 2024–2026 answer, and are what make full-duplex dialogue fit in a context window.
    - Whisper's encoder-decoder architecture treats ASR as a standard seq2seq problem over log-Mel features, making it easy to reuse as an audio feature extractor for downstream LLMs via linear projection or Q-Former adapters.
    - Speech LMs (AudioPaLM, Moshi) bypass the ASR/TTS pipeline by treating audio tokens and text tokens as interchangeable elements of a unified sequence, enabling lower latency and richer parallelism.
    - Real-time full-duplex dialogue (Moshi-style) requires a hierarchical temporal architecture: a slow inner LM operating on coarse semantic tokens, and fast depth transformers producing fine acoustic tokens per step.
    - The multimodal token-stream view — each modality contributing tokens to a shared sequence — is the dominant abstraction, requiring only modality-specific encoders and projection adapters on top of a frozen or lightly fine-tuned LLM.
    - Audio token sequences are 200× longer than equivalent text, necessitating weighted loss, hierarchical generation, or compressed representations (Q-Former) to prevent the LLM from being overwhelmed by acoustic detail.
    - For production voice systems, the latency budget is roughly VAD (50 ms) + audio encoder (50 ms) + LLM TTFT (200 ms) + first-chunk decode (30 ms) ≈ 330 ms — achievable on a single modern GPU with optimized inference.
    - Transfer learning and semi-supervised pretraining (HuBERT, wav2vec 2.0, pseudo-labeling) are essential because high-quality paired audio-text data is scarce relative to the scale of text-only corpora.

!!! sota "State of the Art & Resources (2026)"
    Audio-language research has converged on a unified paradigm: neural audio codecs (RVQ-based) tokenize sound into discrete sequences that a shared transformer processes alongside text, enabling end-to-end speech-in/speech-out systems with sub-200 ms latency. The field moved rapidly from cascade pipelines (ASR → LLM → TTS) to native audio LLMs like Moshi and VALL-E 2 that operate directly on codec tokens, and by 2026 into any-to-any "omni" backbones (e.g., Qwen3-Omni) that fold speech dialogue into a single text/audio/image/video model.

    **Foundational work**

    - [Défossez et al., *High Fidelity Neural Audio Compression* (2022)](https://arxiv.org/abs/2210.13438) — EnCodec: the RVQ-based neural codec that became the standard tokenizer for audio LLMs.
    - [Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (2022)](https://arxiv.org/abs/2212.04356) — Whisper: weak-supervision ASR trained on 680 k hours, the default encoder for audio-LLM pipelines.
    - [Hsu et al., *HuBERT: Self-Supervised Speech Representation Learning* (2021)](https://arxiv.org/abs/2106.07447) — masked-prediction pre-training that yields semantic speech units used widely as discrete targets.
    - [Borsos et al., *AudioLM: a Language Modeling Approach to Audio Generation* (2022)](https://arxiv.org/abs/2209.03143) — introduced the hierarchical semantic-then-acoustic two-stage generation framework.

    **Recent advances (2023–2026)**

    - [Wang et al., *VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers* (2023)](https://arxiv.org/abs/2301.02111) — reframes TTS as AR + NAR codec-token prediction with only a 3-second speaker prompt.
    - [Chen et al., *VALL-E 2: Human Parity Zero-Shot TTS* (2024)](https://arxiv.org/abs/2406.05370) — repetition-aware sampling and grouped code modeling push TTS to claimed human parity on LibriSpeech.
    - [Défossez et al., *Moshi: a speech-text foundation model for real-time dialogue* (2024)](https://arxiv.org/abs/2410.00037) — first full-duplex spoken dialogue LLM with ~160 ms theoretical latency via dual audio streams.
    - [Rubenstein et al., *AudioPaLM: A Large Language Model That Can Speak and Listen* (2023)](https://arxiv.org/abs/2306.12925) — interleaves audio and text tokens in a single PaLM-2 backbone for ASR, TTS, and speech translation.
    - [Xu et al., *Qwen3-Omni Technical Report* (2025)](https://arxiv.org/abs/2509.17765) — any-to-any omni model (text/audio/image/video in, text + streaming speech out) reaching open-source SOTA on most audio benchmarks; the 2026 frontier for unified speech dialogue.

    **Open-source & tools**

    - [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) — Meta's library for MusicGen, AudioGen, and EnCodec; includes training code and pretrained checkpoints.
    - [openai/whisper](https://github.com/openai/whisper) — official Whisper inference library with all model sizes (tiny → large-v3, plus the faster 2024 `large-v3-turbo`) under MIT license.
    - [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) — full-duplex spoken dialogue framework with the Mimi streaming codec and pretrained Moshi weights.
    - [huggingface/transformers](https://github.com/huggingface/transformers) — the practical home of most of this chapter: `WhisperForConditionalGeneration`, `Wav2Vec2ForCTC`, `EncodecModel`, `MimiModel`, `DacModel`, `VitsModel`, and the `automatic-speech-recognition` pipeline with long-form chunking.
    - [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 Whisper runtime with INT8/FP16 inference; the usual choice for real-time or CPU transcription.
    - [jitsi/jiwer](https://github.com/jitsi/jiwer) and the HuggingFace Open ASR Leaderboard — WER computation with a pinned text normalizer, the only fair way to compare ASR systems.
    - [espnet/espnet](https://github.com/espnet/espnet), [speechbrain/speechbrain](https://github.com/speechbrain/speechbrain), and NVIDIA NeMo — end-to-end training recipes for ASR, TTS, and speaker tasks.

## Further Reading

- **SoundStream:** Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2022.
- **EnCodec:** Défossez et al., "High Fidelity Neural Audio Compression," arXiv 2210.13438, 2022.
- **Whisper:** Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," OpenAI technical report, 2022.
- **HuBERT:** Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units," *TASLP*, 2021.
- **CTC:** Graves et al., "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks," *ICML*, 2006.
- **wav2vec 2.0:** Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," *NeurIPS*, 2020.
- **VALL-E:** Wang et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers," arXiv 2301.02111, 2023.
- **AudioLM:** Borsos et al., "AudioLM: a Language Modeling Approach to Audio Generation," *TASLP*, 2023.
- **MusicGen:** Copet et al., "Simple and Controllable Music Generation," *NeurIPS*, 2023.
- **AudioPaLM:** Rubenstein et al., "AudioPaLM: A Large Language Model That Can Speak and Listen," arXiv 2306.12925, 2023.
- **Moshi:** Défossez et al., "Moshi: a speech-text foundation model for real-time dialogue," Kyutai technical report, 2024.
- **SpeechTokenizer:** Zhang et al., "SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models," arXiv 2308.16692, 2023.
- **HiFi-GAN:** Kong et al., "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis," *NeurIPS*, 2020.

## Exercises

**1.** A team builds a neural audio codec that runs at $f_s = 16$ kHz with a stride of $320$ samples per frame and $K = 4$ RVQ levels. (a) How many frames per second does it emit? (b) With a flat interleave of all levels, how many tokens does a 60-second clip cost? (c) If the model instead uses only the coarsest RVQ level for semantic modeling, how many tokens does the same clip cost? (d) Roughly how many BPE tokens would the 60-second transcript occupy at 130 words/minute, and what is the audio-to-text token ratio for the full-codec case? Use the chapter's rule of thumb that $65$ words is about $90$ BPE tokens.

??? note "Solution"
    (a) Frames per second is the sample rate divided by the stride:
    $$
    \frac{f_s}{\text{stride}} = \frac{16000}{320} = 50 \text{ frames/s}.
    $$

    (b) With a flat interleave, every frame contributes one token per RVQ level, so the token rate is $50 \times 4 = 200$ tokens/s. Over 60 seconds:
    $$
    60 \times 200 = 12{,}000 \text{ tokens}.
    $$

    (c) Using only the coarsest level drops the per-frame cost to a single token: $50$ tokens/s, so $60 \times 50 = 3{,}000$ tokens. This is a $4\times$ reduction, exactly the number of RVQ levels $K$ — the same lever the chapter's worked example describes as "use only the coarsest 1-2 RVQ levels."

    (d) At 130 words/min, a 60-second clip is $130$ words. Scaling the chapter's rule ($65$ words $\approx 90$ tokens, i.e. $\approx 1.38$ tokens/word) gives $130 \times 1.38 \approx 180$ BPE tokens. The audio-to-text ratio for the full codec is
    $$
    \frac{12{,}000}{180} \approx 67\times.
    $$
    Even at this modest 4-level, 16 kHz configuration the codec stream is dozens of times denser than the transcript, which is the core token-budget tension of the chapter.

**2.** The chapter's `multimodal_cross_entropy` down-weights audio tokens because "600 audio tokens/s vs ~3 text tokens/s means audio dominates the loss 200:1." (a) Derive the 200:1 figure. (b) The default `audio_weight=0.1`: after weighting, what is the ratio of total audio-token loss weight to total text-token loss weight over one second? (c) What value of `audio_weight` would make audio and text contribute *equally* to the (weight-normalized) loss over equal wall-clock time?

??? note "Solution"
    (a) Over one second the model sees about $600$ audio tokens and about $3$ text tokens. With no weighting every token carries weight $1$, so the audio share of the summed loss weight is proportional to the token counts:
    $$
    \frac{600}{3} = 200 \quad \Rightarrow \quad 200{:}1.
    $$

    (b) The weighted scheme gives audio tokens weight $0.1$ and text tokens weight $1.0$. Total audio weight over one second is $0.1 \times 600 = 60$; total text weight is $1.0 \times 3 = 3$. The ratio is
    $$
    \frac{60}{3} = 20 \quad \Rightarrow \quad 20{:}1.
    $$
    So `audio_weight=0.1` shrinks the imbalance by $10\times$ (from 200:1 to 20:1) but audio still dominates.

    (c) Equal contribution means the summed weights are equal:
    $$
    w_{\text{audio}} \times 600 = 1.0 \times 3
    \quad \Rightarrow \quad
    w_{\text{audio}} = \frac{3}{600} = 0.005.
    $$
    An `audio_weight` of $0.005$ balances the two modalities per second. In practice teams pick something between this and $1.0$ so audio still receives a meaningful gradient rather than being nearly ignored.

**3.** Whisper's encoder always processes exactly 1500 positions, "regardless of actual utterance duration." (a) Trace how a raw 30-second clip becomes 1500 encoder positions given the chapter's numbers (16 kHz input, 10 ms hop, and a convolutional front-end that halves the temporal resolution). (b) What does the pipeline do with a 5-second clip, and what is the compute cost consequence? (c) What happens to a 40-second clip, and why can this silently hurt transcription quality?

??? note "Solution"
    (a) At 16 kHz with a 10 ms hop, each frame covers $160$ samples, so a 30-second clip yields $30 \text{ s} \times 100 \text{ frames/s} = 3000$ log-Mel frames (the `(n_mels, 3000)` tensor in the transcription code — $n_{\text{mels}} = 80$ for `base`, $128$ for `large-v3`). The convolutional front-end halves the temporal resolution, $3000 \to 1500$, which is the fixed number of positions each self-attention block attends over.

    (b) `whisper.pad_or_trim` pads the 5-second clip with silence up to the full 30-second window before feature extraction, so it still becomes a `(n_mels, 3000)` input and 1500 encoder positions. The padded/silent region is masked, but the encoder self-attention still runs over all 1500 positions — so a 5-second utterance costs the same encoder compute as a 30-second one. Short utterances waste compute, exactly the downside the chapter lists for the fixed-length-window strategy.

    (c) A 40-second clip is *trimmed* to the first 30 seconds. The final 10 seconds are simply discarded, so any speech there is never transcribed. Because the API returns a fluent transcript for the portion it did see, the truncation is silent — there is no error, just missing words. The chapter's remedy is dynamic chunking: split long audio into overlapping 30-second windows, encode each, and concatenate.

**4.** AudioLM and SpeechTokenizer both hinge on separating *semantic* from *acoustic* information. (a) Using the chapter's figures, contrast the token rate of semantic tokens versus EnCodec acoustic tokens for a 30-second clip. (b) Explain why AudioLM models semantic tokens *first* and only then generates acoustic tokens. (c) SpeechTokenizer reaches a similar goal differently — how, and what practical property does that give its level-1 tokens?

??? note "Solution"
    (a) The chapter states a 30-second clip needs only about $1{,}500$ semantic tokens (roughly $50$ tokens/s) but about $18{,}000$ EnCodec acoustic tokens ($600$ tokens/s across 8 RVQ levels). Semantic tokens are therefore about $12\times$ more compact.

    (b) Semantic tokens capture long-range structure — content, melody, prosody — in a compact stream, so an autoregressive LM can plan the *global* shape of the audio over a short, tractable sequence before committing to detail. Acoustic tokens are far denser and mostly encode surface fidelity; generating them first would force the model to decide fine acoustic detail before it has settled what is even being said. Modeling semantics first, then conditioning acoustic generation on those tokens, mirrors the coarse-to-fine intuition the chapter draws to BPE merges and RVQ levels.

    (c) SpeechTokenizer keeps a single RVQ codec but adds a distillation loss $\mathcal{L}_{\text{semantic}}$ that forces VQ level-1 outputs to match HuBERT's discrete pseudo-labels, so
    $$
    \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reconstruct}} + \lambda \cdot \mathcal{L}_{\text{semantic}}.
    $$
    This makes level-1 tokens carry *what was said* while higher levels carry *how it was said*. The practical payoff: level-1 tokens become a drop-in replacement for text tokens in a speech LM, so the disentanglement is built into the tokenizer rather than needing a separate semantic model.

**5.** The chapter's `ResidualVQ` returns only the per-level indices, never a reconstruction. Extend it with a `decode` method that turns a list of per-level index tensors back into the reconstructed vector $\hat{\mathbf{z}}_t = \sum_{k=1}^{K} \mathbf{e}^{(k)}_{i^{*}_k}$, then write a short check that reconstruction error decreases as more RVQ levels are used.

??? note "Solution"
    Decoding just looks up each level's codebook entry for the stored indices and sums them, following the RVQ reconstruction formula. Adding a `decode` (and a small `reconstruct_prefix` helper that sums only the first $j$ levels) lets us measure the error curve:

    ```python
    import torch
    import torch.nn as nn

    class ResidualVQ(nn.Module):
        def __init__(self, num_levels: int, codebook_size: int, dim: int):
            super().__init__()
            self.levels = nn.ModuleList([
                VectorQuantizer(codebook_size, dim) for _ in range(num_levels)
            ])

        def forward(self, z: torch.Tensor):
            residual = z
            all_indices = []
            for vq in self.levels:
                quantized, indices = vq(residual)
                residual = residual - quantized.detach()
                all_indices.append(indices)          # (B, T) per level
            return all_indices

        def decode(self, all_indices: list[torch.Tensor]) -> torch.Tensor:
            """Sum codebook lookups across all provided levels -> (B, T, D)."""
            recon = None
            for vq, idx in zip(self.levels, all_indices):
                e = vq.codebook(idx)                 # (B, T, D)
                recon = e if recon is None else recon + e
            return recon

    # --- check: error falls monotonically as levels are added ---
    torch.manual_seed(0)
    B, T, D, C, K = 2, 50, 16, 256, 4
    rvq = ResidualVQ(num_levels=K, codebook_size=C, dim=D)
    z = torch.randn(B, T, D)
    all_indices = rvq(z)

    for j in range(1, K + 1):
        recon_j = rvq.decode(all_indices[:j])        # use first j levels only
        mse = (z - recon_j).pow(2).mean().item()
        print(f"levels=1..{j}  reconstruction MSE = {mse:.4f}")
    ```

    Because each level quantizes the residual left by the previous ones, `decode(all_indices[:j])` reconstructs $\sum_{k=1}^{j}\mathbf{e}^{(k)}$, and the MSE decreases as $j$ grows — the numeric embodiment of the chapter's statement that "level 1 alone captures coarse semantics; levels 2-8 refine acoustic detail." (The codebooks here are randomly initialized, not trained, so the absolute MSE is large; the point is the monotone decrease.)

**6.** The chapter contrasts flat interleaving with the *delay pattern* introduced by MusicGen. Implement `apply_delay_pattern(codes, pad_id)` that takes a codec tensor of shape `(B, K, T)` (K RVQ levels over T frames) and returns the delayed layout where level $k$ (0-indexed) is shifted right by $k$ steps, with `pad_id` filling the exposed positions. State the output shape and explain, in one sentence, why this pattern helps a causal autoregressive model.

??? note "Solution"
    Each level $k$ is written into the output starting at column $k$, so higher levels lag behind level 0. The output has $K - 1$ extra columns to hold the staggered tail:

    ```python
    import torch

    def apply_delay_pattern(codes: torch.Tensor, pad_id: int) -> torch.Tensor:
        """
        codes:  (B, K, T) integer codec tokens, K RVQ levels over T frames.
        Returns (B, K, T + K - 1): level k is delayed by k steps; exposed
        positions are filled with pad_id.
        """
        B, K, T = codes.shape
        out = torch.full((B, K, T + K - 1), pad_id,
                         dtype=codes.dtype, device=codes.device)
        for k in range(K):
            out[:, k, k:k + T] = codes[:, k, :]
        return out

    # --- tiny illustration ---
    codes = torch.arange(1, 1 + 3 * 4).reshape(1, 3, 4)  # (B=1, K=3, T=4)
    print(codes[0])
    print(apply_delay_pattern(codes, pad_id=0)[0])
    ```

    Running it shows level 0 unshifted, level 1 shifted right by one, level 2 by two, with zeros padding the exposed corners:

    ```text
    codes (K=3, T=4):
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]

    delayed (K=3, T+K-1=6):
    [[ 1  2  3  4  0  0]
     [ 0  5  6  7  8  0]
     [ 0  0  9 10 11 12]]
    ```

    The output shape is `(B, K, T + K - 1)`. When the model predicts one column at a time, the delay guarantees that a fine level's token for frame $t$ is emitted only after the coarser levels for that frame (and all earlier frames) are already in context — so each level can condition on the coarser structure while the causal LM only ever looks at the past, which is exactly the "limiting temporal lookahead" property the chapter attributes to the delay pattern.
