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

### Token Rate and Sequence Length

!!! example "Worked example: token budget for 30 s of speech"

    EnCodec at 24 kHz with stride 320 and 8 RVQ levels:

    - Frames per second: $24000 / 320 = 75$ frames/s
    - Tokens per second (flat interleave): $75 \times 8 = 600$ tokens/s
    - 30-second utterance: $30 \times 600 = 18{,}000$ tokens

    Compare to text: a typical spoken utterance at 130 words/min for 30 s is $\approx 65$ words, or roughly 90 BPE tokens.

    The codec produces **200× more tokens than the transcript**. This is the core engineering tension: faithful audio reconstruction demands high token density, but LLM context windows are finite. Real systems use one of three mitigations:
    1. Use only the coarsest 1–2 RVQ levels for modeling semantics (the rest are predicted in parallel or recovered by a separate decoder).
    2. Use a higher compression codec (e.g., DAC at lower bitrates).
    3. Encode audio as a continuous vector sequence (Whisper-style) instead of discrete tokens, and quantize only for generation.

### Codec Token Interleaving Patterns

When flattening $K$ RVQ levels into a 1-D token stream, there are two main strategies:

{{fig:audio-rvq-interleave-patterns}}

The delay pattern (used in AudioLM, VALL-E, and related systems) allows the autoregressive model to condition each level on all previous time steps, limiting temporal lookahead. Level 1 alone captures coarse semantics; levels 2–8 refine acoustic detail.

## Whisper: The Encoder-Only Path to ASR

Whisper (Radford et al., OpenAI, 2022) is the de facto standard for Automatic Speech Recognition (ASR) — recognition of speech as text. It takes a different philosophy from codec-based models: rather than discretizing audio into tokens for generation, it encodes a fixed 30-second window of log-Mel features into continuous hidden states and decodes with a standard text autoregressive decoder.

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

    # Compute log-Mel spectrogram on the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)  # (80, 3000)

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

### Whisper as a Feature Extractor

For downstream tasks — speech LLMs, speaker diarization, voice cloning — we often want encoder hidden states rather than the text output. The encoder's final hidden states are rich acoustic representations that have been used in place of (or alongside) codec tokens.

```python
import torch
import whisper
from whisper.model import AudioEncoder

def extract_whisper_features(
    mel: torch.Tensor,   # (batch, 80, 3000) on GPU
    model: whisper.Whisper,
) -> torch.Tensor:
    """
    Return encoder hidden states: (batch, 1500, encoder_dim).
    For whisper-large-v3, encoder_dim = 1280.
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

### Codec LM Pipelines: VALL-E and Relatives

VALL-E (Wang et al., Microsoft, 2023) reframes TTS as a language modeling problem over codec tokens. Given a 3-second acoustic prompt and a text transcript, VALL-E:

1. **Predicts coarse tokens (AR stage):** Autoregressively models EnCodec level-1 tokens conditioned on text BPE tokens and the acoustic prompt. This captures prosody and speaker identity.
2. **Predicts fine tokens (NAR stage):** Non-autoregressively predicts RVQ levels 2–8 conditioned on level 1 and all other context. This fills in acoustic detail in $O(1)$ parallel steps.

The conceptual architecture:

{{fig:valle-ar-nar-architecture}}

The key insight: level-1 tokens determine *what* is said and *how* (prosody, speaker style); higher levels determine the acoustic rendering quality. This decomposition separates semantic control from acoustic fidelity.

## Speech Language Models: Audio-In, Audio-Out

The logical endpoint of audio tokenization is a model that accepts and emits audio tokens natively — no ASR transcription step, no TTS synthesis step, just raw audio tokens flowing through a standard transformer. We call these "speech LMs" or "audio LLMs."

### SpeechTokenizer, dSPIN, and Semantic-Acoustic Disentanglement

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

**Moshi** (Défossez et al., Kyutai, 2024) goes further: it is designed for real-time full-duplex spoken dialogue, meaning the model continuously emits audio while simultaneously listening. It uses two parallel audio token streams — one for the system's voice and one for the user's voice — processed by separate depth-transformers at each time step, plus a shared inner "language model" operating at a slower rate. This hierarchical temporal structure enables sub-200 ms response latency in practice.

{{fig:moshi-temporal-hierarchy}}

The crucial engineering decision in Moshi: **the inner LM runs causally** — it only attends to past codec frames, enabling real-time streaming. The depth transformer runs separately per frame, keeping per-step latency bounded.

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

MusicGen (Copet et al., Meta AI, 2023) is an autoregressive transformer trained on EnCodec tokens from licensed music. It uses a **codebook interleaving** technique: instead of the delay pattern, it uses a "codebook per time step" approach where all $K$ levels for time step $t$ are modeled jointly in a single forward pass using a small depth transformer, before advancing to step $t+1$. This avoids the exponential search space of fully autoregressive RVQ and keeps generation tractable.

```python
from audiocraft.models import MusicGen
import soundfile as sf
import numpy as np

def generate_music(
    description: str,
    duration_seconds: float = 10.0,
    model_size: str = "small"   # small | medium | large | melody
) -> np.ndarray:
    """
    Generate music conditioned on a text description.
    Returns float32 audio at 32 kHz.
    """
    model = MusicGen.get_pretrained(model_size)
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
    - Neural audio codecs (EnCodec, SoundStream) compress audio to ~75 frames/second with 8 RVQ levels, producing approximately 600 tokens/second — far more than text, creating a fundamental token-budget tension.
    - Whisper's encoder-decoder architecture treats ASR as a standard seq2seq problem over log-Mel features, making it easy to reuse as an audio feature extractor for downstream LLMs via linear projection or Q-Former adapters.
    - Speech LMs (AudioPaLM, Moshi) bypass the ASR/TTS pipeline by treating audio tokens and text tokens as interchangeable elements of a unified sequence, enabling lower latency and richer parallelism.
    - Real-time full-duplex dialogue (Moshi-style) requires a hierarchical temporal architecture: a slow inner LM operating on coarse semantic tokens, and fast depth transformers producing fine acoustic tokens per step.
    - The multimodal token-stream view — each modality contributing tokens to a shared sequence — is the dominant abstraction, requiring only modality-specific encoders and projection adapters on top of a frozen or lightly fine-tuned LLM.
    - Audio token sequences are 200× longer than equivalent text, necessitating weighted loss, hierarchical generation, or compressed representations (Q-Former) to prevent the LLM from being overwhelmed by acoustic detail.
    - For production voice systems, the latency budget is roughly VAD (50 ms) + audio encoder (50 ms) + LLM TTFT (200 ms) + first-chunk decode (30 ms) ≈ 330 ms — achievable on a single modern GPU with optimized inference.
    - Transfer learning and semi-supervised pretraining (HuBERT, wav2vec 2.0, pseudo-labeling) are essential because high-quality paired audio-text data is scarce relative to the scale of text-only corpora.

!!! sota "State of the Art & Resources (2026)"
    Audio-language research has converged on a unified paradigm: neural audio codecs (RVQ-based) tokenize sound into discrete sequences that a shared transformer processes alongside text, enabling end-to-end speech-in/speech-out systems with sub-200 ms latency. The field moved rapidly from cascade pipelines (ASR → LLM → TTS) to native audio LLMs like Moshi and VALL-E 2 that operate directly on codec tokens.

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
    - [Gandhi et al., *Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling* (2023)](https://arxiv.org/abs/2311.00430) — 6× faster, 50% smaller distilled Whisper within 1% WER; practical choice for low-latency pipelines.

    **Open-source & tools**

    - [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) — Meta's library for MusicGen, AudioGen, and EnCodec; includes training code and pretrained checkpoints.
    - [openai/whisper](https://github.com/openai/whisper) — official Whisper inference library with all model sizes (tiny → large-v3) under MIT license.
    - [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) — full-duplex spoken dialogue framework with the Mimi streaming codec and pretrained Moshi weights.

## Further Reading

- **SoundStream:** Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2022.
- **EnCodec:** Défossez et al., "High Fidelity Neural Audio Compression," arXiv 2210.13438, 2022.
- **Whisper:** Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," OpenAI technical report, 2022.
- **HuBERT:** Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units," *TASLP*, 2021.
- **VALL-E:** Wang et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers," arXiv 2301.02111, 2023.
- **AudioLM:** Borsos et al., "AudioLM: a Language Modeling Approach to Audio Generation," *TASLP*, 2023.
- **MusicGen:** Copet et al., "Simple and Controllable Music Generation," *NeurIPS*, 2023.
- **AudioPaLM:** Rubenstein et al., "AudioPaLM: A Large Language Model That Can Speak and Listen," arXiv 2306.12925, 2023.
- **Moshi:** Défossez et al., "Moshi: a speech-text foundation model for real-time dialogue," Kyutai technical report, 2024.
- **SpeechTokenizer:** Zhang et al., "SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models," arXiv 2308.16692, 2023.
- **HiFi-GAN:** Kong et al., "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis," *NeurIPS*, 2020.
