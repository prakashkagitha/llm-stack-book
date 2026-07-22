"""
Executable test for content/10-multimodal-and-arch/03-audio-speech-multimodal.md

Concatenates the chapter's 5 heuristically CPU-runnable Python blocks in order
and exercises each one so the book's actual code runs end to end (torch CPU,
tiny shapes, fixed seeds where relevant).

Blocks covered:
  #0 (line ~36)  compute_mel_spectrogram (torchaudio)
       -- guarded: torchaudio is not in the guaranteed CI dependency set
       (numpy, torch, einops, sklearn, stdlib). Runs for real if importable,
       otherwise honestly SKIPPED.
  #3 (line ~221) extract_whisper_features
       -- guarded: the `whisper` package is not in the guaranteed CI
       dependency set, and even if installed, `whisper.load_model(...)`
       requires a network download. We mock the model boundary with a tiny
       duck-typed stand-in object exposing `.encoder`, so the block's OWN
       logic (no_grad + call model.encoder(mel) + return hidden states)
       actually executes offline, per the "mock the boundary" rule.
  #5 (line ~316) StreamingTTSPipeline
       -- pure stdlib (asyncio/queue/threading). Run for real with a tiny
       fake `lm` (yields codec frames) and `codec_decoder` (sums frames into
       a fake waveform) standing in for the real LM/codec.
  #6 (line ~405) AudioQFormer
       -- pure torch/nn. Instantiated with tiny dims and run forward on
       random audio features.
  #8 (line ~531) generate_music
       -- guarded: `audiocraft` and `soundfile` are not in the guaranteed CI
       dependency set, and `MusicGen.get_pretrained(...)` requires a network
       download of pretrained weights. Import is guarded so the module still
       loads; the function is defined but NOT called (SKIP: network/model
       download required).

Skipped (per the task's block classification, not required to test):
  #1 (line ~93 area, RVQ) -- not in the required 5-block list for this run
       (classified separately by the harness); note the chapter also has a
       ResidualVQ/VectorQuantizer block used purely for exposition elsewhere.
  #2 (whisper CLI transcribe_with_whisper) -- needs-net (model download +
       audio file), not one of the 5 required blocks.
  #4 (```text latency budget) -- non-python.
  #7 (multimodal_cross_entropy) -- fragment per the harness's classification,
       not one of the 5 required blocks for this run.
"""

from __future__ import annotations

import asyncio
import queue  # noqa: F401 -- imported by block #5's own header, kept for fidelity
import threading  # noqa: F401 -- imported by block #5's own header, kept for fidelity
from typing import Iterator, AsyncIterator  # noqa: F401 -- fidelity with block #5's imports

import numpy as np
import torch
import torch.nn as nn

# Optional third-party deps used only by blocks #0, #3, #8. Guard per the hard
# rules so the module always loads even when these aren't installed in CI.
try:
    import torchaudio
    import torchaudio.transforms as T
except Exception:
    torchaudio = None
    T = None

try:
    import whisper
    from whisper.model import AudioEncoder
except Exception:
    whisper = None
    AudioEncoder = None

try:
    from audiocraft.models import MusicGen
except Exception:
    MusicGen = None
try:
    import soundfile as sf
except Exception:
    sf = None


# ============================================================
# Block #0 (line ~36) -- compute_mel_spectrogram
# ============================================================

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


# ============================================================
# Block #3 (line ~221) -- extract_whisper_features
# ============================================================

def extract_whisper_features(
    mel: torch.Tensor,   # (batch, 80, 3000) on GPU
    model: "whisper.Whisper",
) -> torch.Tensor:
    """
    Return encoder hidden states: (batch, 1500, encoder_dim).
    For whisper-large-v3, encoder_dim = 1280.
    """
    with torch.no_grad():
        # model.encoder is a standard TransformerEncoder
        hidden = model.encoder(mel)   # (B, 1500, D)
    return hidden


# ============================================================
# Block #5 (line ~316) -- StreamingTTSPipeline
# ============================================================

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
            chunks_played = []
            while True:
                waveform = await audio_queue.get()
                if waveform is None:
                    break
                # In a real system: write to audio output device
                print(f"[audio] playing {len(waveform)/24000:.3f}s chunk")
                chunks_played.append(waveform)
            return chunks_played

        _, played = await asyncio.gather(producer(), consumer())
        return played


# ============================================================
# Block #6 (line ~405) -- AudioQFormer
# ============================================================

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


# ============================================================
# Block #8 (line ~531) -- generate_music (MusicGen)
# ============================================================

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


# ============================================================
# Test harness
# ============================================================

def main():
    torch.manual_seed(0)

    # --- Block #0: compute_mel_spectrogram -----------------------------
    if T is not None:
        wav = torch.randn(1, 160000)   # 10 seconds, 16 kHz
        spec = compute_mel_spectrogram(wav)
        assert spec.shape == (80, 1001), spec.shape
        assert torch.isfinite(spec).all()
        print(f"[OK] block #0 compute_mel_spectrogram: shape={tuple(spec.shape)}")
    else:
        print("[SKIP] block #0 compute_mel_spectrogram: torchaudio not "
              "available in this environment (not in the guaranteed CI "
              "dependency set: numpy, torch, einops, sklearn, stdlib)")

    # --- Block #3: extract_whisper_features -----------------------------
    # Mock the model boundary: whisper.load_model() would need a network
    # download, and the `whisper` package itself isn't guaranteed in CI.
    # We duck-type a tiny `.encoder` so the block's OWN logic (no_grad +
    # call model.encoder(mel) + return hidden states) runs for real.
    class _FakeWhisperEncoder(nn.Module):
        def __init__(self, n_mels=80, dim=32):
            super().__init__()
            # Mimic Whisper's stride-2 conv front-end that halves T.
            self.conv = nn.Conv1d(n_mels, dim, kernel_size=3, stride=2, padding=1)

        def forward(self, mel):  # (B, n_mels, T) -> (B, T//2, dim)
            x = self.conv(mel)          # (B, dim, T//2)
            return x.transpose(1, 2)    # (B, T//2, dim)

    class _FakeWhisperModel:
        def __init__(self):
            self.encoder = _FakeWhisperEncoder(n_mels=80, dim=32)

    fake_model = _FakeWhisperModel()
    fake_mel = torch.randn(2, 80, 300)   # tiny stand-in for (batch, 80, 3000)
    hidden = extract_whisper_features(fake_mel, fake_model)
    assert hidden.shape == (2, 150, 32), hidden.shape
    assert torch.isfinite(hidden).all()
    print(f"[OK] block #3 extract_whisper_features: hidden={tuple(hidden.shape)}")

    # --- Block #5: StreamingTTSPipeline ---------------------------------
    class _FakeLM:
        def tokenize(self, text: str):
            return [ord(c) for c in text]

        def stream(self, text_tokens):
            # Yield 7 codec frames, each a list of K=4 RVQ token ids.
            for step in range(7):
                yield [step * 4 + k for k in range(4)]

    class _FakeCodecDecoder:
        def decode(self, buffer):
            # Fake waveform: one 24 kHz sample per token across the buffer.
            n_samples = sum(len(frame) for frame in buffer) * 100
            return np.zeros(n_samples, dtype=np.float32)

    pipeline = StreamingTTSPipeline(_FakeLM(), _FakeCodecDecoder(), chunk_size=3)
    played_chunks = asyncio.run(pipeline.run("hello world"))
    # 7 frames at chunk_size=3 -> two full chunks (3,3) + one remainder (1)
    assert len(played_chunks) == 3, played_chunks
    assert all(isinstance(c, np.ndarray) for c in played_chunks)
    print(f"[OK] block #5 StreamingTTSPipeline: {len(played_chunks)} chunks played")

    # --- Block #6: AudioQFormer ------------------------------------------
    qformer = AudioQFormer(n_queries=4, audio_dim=16, llm_dim=32, n_heads=4, n_layers=2)
    audio_features = torch.randn(2, 10, 16)   # (B, T_audio, audio_dim)
    out = qformer(audio_features)
    assert out.shape == (2, 4, 32), out.shape
    assert torch.isfinite(out).all()
    print(f"[OK] block #6 AudioQFormer: out={tuple(out.shape)}")

    # --- Block #8: generate_music -----------------------------------------
    if MusicGen is not None:
        # Even if audiocraft were installed, MusicGen.get_pretrained(...)
        # downloads weights over the network, which is forbidden here.
        print("[SKIP(network)] block #8 generate_music: audiocraft is "
              "installed but MusicGen.get_pretrained() requires a network "
              "weight download -- not invoked.")
    else:
        print("[SKIP(network+optional-dep)] block #8 generate_music: "
              "audiocraft not available in this environment, and "
              "MusicGen.get_pretrained() requires a network weight "
              "download in any case -- function defined but not called.")

    print("\nAll required blocks executed or honestly skipped "
          "(block #0 runs if torchaudio is available; block #8 always "
          "skipped -- needs network weight download).")


if __name__ == "__main__":
    main()
