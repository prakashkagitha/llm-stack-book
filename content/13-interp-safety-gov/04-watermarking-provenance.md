# 13.4 Watermarking, Provenance & AI-Content Detection

As generative models reach human-level fluency, the question "did a machine write this?" becomes legally, commercially, and politically loaded. A journalism outlet needs to know whether a submitted opinion piece is AI-generated. A school needs evidence for an academic-integrity decision. A regulator enforcing the EU AI Act needs a disclosure mechanism that works at scale. This chapter develops three complementary answers to that question: *proactive watermarking* embedded during generation, *content-provenance standards* that cryptographically chain a file to its origin, and *post-hoc detection* applied to content whose origin is unknown. We will see why each approach is necessary, how each works at the algorithmic level, and why none of them is sufficient alone.

Related chapters that establish useful background: [Privacy, Memorization & Differential Privacy for LLMs](../13-interp-safety-gov/03-privacy-memorization-dp.html) covers what information leaks from model outputs and is a natural companion here; [Safety, Guardrails & Content Moderation](../12-production-mlops/04-safety-guardrails.html) covers the runtime enforcement layer; [Mechanistic Interpretability & Model Internals](../13-interp-safety-gov/01-mechanistic-interpretability.html) provides the foundational lens on how models work internally; [AI Governance, Compliance & Regulation](../13-interp-safety-gov/06-governance-compliance.html) treats the legal landscape in depth.

---

## Why Watermarking Exists: The Motivation

Consider a model that can produce text indistinguishable from human writing with essentially zero marginal cost. Without a provenance signal, every downstream consumer of that text must either (a) trust the claimed source, (b) run a statistical detector that is probabilistically unreliable, or (c) give up. The asymmetry is extreme: a determined attacker can polish AI text with trivial paraphrasing, but a defender running a post-hoc classifier faces a distribution shift problem — they cannot enumerate every possible model or every possible attack.

Watermarking inverts this asymmetry. The model operator *embeds* a signal at generation time. Detection is then a verification problem (does this signal match our key?) rather than a classification problem (does this text look AI-generated?). The analogy is a cryptographic signature: the hardness of verification scales with the strength of a secret key, not with the adversary's sophistication at mimicking text.

Three desiderata tension against each other:

1. **Detectability** — the signal must be recoverable with high statistical confidence from a short passage.
2. **Imperceptibility** — watermarked text must be indistinguishable (in quality, style, content) from unwatermarked text.
3. **Robustness** — the signal must survive paraphrase, translation, summarization, or selective editing by an adversary.

These three cannot all be maximized simultaneously, and the tradeoffs define the design space.

{{fig:wmprov-asymmetry-inversion}}

---

## Statistical Text Watermarking: The Green-List Scheme

### The Kirchenbauer et al. Construction

The seminal paper by Kirchenbauer, Geiping, Wen, Kirchenbauer, Goldblum, and Goldstein (2023) — commonly called the *KGW watermark* — embeds a signal by biasing the token sampling distribution at each step.

The construction works as follows. Let $V$ be the vocabulary of size $|V|$. At each generation step $t$, the *preceding context* (or a hash of it) is used to seed a pseudorandom function $f$ that partitions $V$ into a *green list* $G_t \subseteq V$ of size $\gamma |V|$ and a *red list* $R_t = V \setminus G_t$.

During generation, the logit for every token $v \in G_t$ is boosted by a hardness parameter $\delta > 0$ before softmax:

$$
\tilde{\ell}_{t,v} = \ell_{t,v} + \delta \cdot \mathbf{1}[v \in G_t]
$$

The model therefore preferentially samples green tokens without requiring any modification to the model weights. An unmodified greedy or nucleus-sampled text would use $\gamma |V|$ green tokens in expectation. A watermarked text uses them at a much higher rate.

**Detection.** Given a candidate text of $T$ tokens and a secret key $k$, the detector reconstructs each $G_t$ and counts the number of green tokens $g$. Under the null hypothesis (human text), $g \sim \text{Binomial}(T, \gamma)$. The *z-score* is:

$$
z = \frac{g - \gamma T}{\sqrt{T \gamma (1 - \gamma)}}
$$

A threshold $z^* \approx 4$ corresponds to a false-positive rate on the order of $10^{-5}$ per document.

!!! example "Worked numerical example"
    Suppose $\gamma = 0.5$ (half the vocabulary is green at each step), $\delta = 2.0$, and the text is $T = 200$ tokens long.

    - Under the null (human text): expected green tokens $= 200 \times 0.5 = 100$, $\sigma = \sqrt{200 \times 0.5 \times 0.5} \approx 7.07$.
    - A watermarked text might observe $g = 145$ green tokens (heavy green bias from $\delta = 2.0$).
    - $z = (145 - 100) / 7.07 \approx 6.36$.
    - Using the standard normal tail: $P(Z > 6.36) \approx 10^{-10}$, an extremely strong rejection of the null.
    - At a threshold of $z^* = 4.0$, this text is flagged with overwhelming confidence.

    Now suppose an adversary randomly replaces 30% of tokens via paraphrase. Empirically, $\approx 50\%$ of replaced tokens will land in the green list (random chance), so $g_{\text{post}} \approx 145 \times 0.7 + 100 \times 0.3 = 101.5 + 30 = 131.5$. $z \approx (131.5 - 100)/7.07 \approx 4.45$ — still well above threshold. This illustrates why moderate paraphrase attacks are insufficient.

{{fig:wmprov-greenlist-mechanism}}

### Context Hashing and Key Security

The green list $G_t$ is typically derived by hashing the previous $h$ tokens together with a secret key $k$:

$$
G_t = \text{TopGamma}\!\left(\text{Hash}(k, w_{t-h:t-1})\right)
$$

where TopGamma selects the $\lfloor \gamma |V| \rfloor$ tokens with the lowest hash values (a deterministic pseudo-random selection). Setting $h=1$ (hash only the previous token) creates a Markov-1 watermark that is easy to implement but can be approximated by an adversary who observes many samples. Setting $h \geq 4$ dramatically increases the adversary's work because the number of possible contexts is $|V|^h \sim 32000^4 \approx 10^{18}$.

### Distortion-Free Watermarks

A limitation of the KGW scheme is that boosting green-list logits changes the output distribution — it is not *distortion-free*. For tasks where output quality is critical (medical summarization, legal reasoning), this is unacceptable.

Kuditipudi, Thickstun, Hashimoto, and Liang (2023) introduced *distortion-free* watermarking. The key idea: rather than modifying the logit distribution, use the secret key to generate a sequence of random numbers $\{r_t\}$ and then *select* the token whose inverse-CDF transform under the model distribution corresponds to $r_t$:

$$
w_t = F_t^{-1}(r_t), \quad r_t = \text{PRF}(k, t)
$$

where $F_t^{-1}$ is the quantile function of the model's next-token distribution. Because we apply a monotone transformation, the *marginal* distribution of each token is unchanged — the text is statistically identical to unmodified sampling. Yet the sequence of tokens is correlated with the known random sequence, giving a detectable signal.

Detection uses a test based on the rank statistics of the observed tokens under the model's predicted distribution: under watermarking, observed tokens should cluster near $r_t$ in probability space, while human text is uniform.

---

## From-Scratch Green-List Watermark: Code

```python
"""
green_list_watermark.py
From-scratch implementation of a KGW-style token watermark.
We simulate the core algorithm without a full language model,
using a random vocabulary and a toy probability distribution.
"""

import hashlib
import struct
import math
import random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Vocabulary and pseudo-logits (stand-in for a real LM)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32_000
VOCAB = list(range(VOCAB_SIZE))  # tokens are just integers

def fake_lm_logits(prev_token: int, seed: int = 42) -> list[float]:
    """
    Returns random logits for demonstration. A real implementation
    would call model.forward() and extract the next-token logit vector.
    """
    rng = random.Random(seed ^ prev_token)
    return [rng.gauss(0, 1) for _ in VOCAB]


# ---------------------------------------------------------------------------
# Green-list construction
# ---------------------------------------------------------------------------

def get_green_list(
    prev_token: int,
    secret_key: bytes,
    gamma: float = 0.5,
) -> set[int]:
    """
    Derive the green list for position t given the previous token and key.
    Uses HMAC-SHA256 as the PRF; maps each vocabulary token to a hash value
    and selects the lowest gamma*|V| hashes as the green list.
    """
    green_size = int(gamma * VOCAB_SIZE)
    
    # Score every token by hashing (key || prev_token || token_id)
    scores = []
    for tok in VOCAB:
        # Pack prev_token and tok as little-endian 32-bit ints
        data = secret_key + struct.pack("<II", prev_token, tok)
        h = hashlib.sha256(data).digest()
        # Interpret first 8 bytes as a uint64 for a uniform [0, 2^64) score
        score = struct.unpack("<Q", h[:8])[0]
        scores.append((score, tok))
    
    # The green list is the gamma fraction with lowest hash scores
    scores.sort()
    green_set = {tok for _, tok in scores[:green_size]}
    return green_set


# ---------------------------------------------------------------------------
# Watermarked sampler
# ---------------------------------------------------------------------------

def softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax."""
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def sample_token(probs: list[float], rng: random.Random) -> int:
    """Sample a token index from a probability distribution."""
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return i
    return len(probs) - 1  # fallback


def generate_watermarked(
    seed_token: int,
    length: int,
    secret_key: bytes,
    delta: float = 2.0,
    gamma: float = 0.5,
    generation_seed: int = 0,
) -> list[int]:
    """
    Generate a watermarked token sequence of given length.
    
    Args:
        seed_token: The token preceding the generation window (context).
        length: Number of tokens to generate.
        secret_key: The watermark secret key (kept by the operator).
        delta: Green-list logit boost.
        gamma: Fraction of vocabulary in the green list.
        generation_seed: Seed for token sampling (mimics temperature sampling).
    
    Returns:
        List of generated token ids.
    """
    rng = random.Random(generation_seed)
    tokens = []
    prev = seed_token
    
    for step in range(length):
        # Get base logits from the (fake) language model
        logits = fake_lm_logits(prev, seed=step)
        
        # Construct green list for this step
        green = get_green_list(prev, secret_key, gamma=gamma)
        
        # Boost green-list logits
        boosted_logits = [
            l + delta if tok in green else l
            for tok, l in enumerate(logits)
        ]
        
        probs = softmax(boosted_logits)
        tok = sample_token(probs, rng)
        tokens.append(tok)
        prev = tok
    
    return tokens


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def detect_watermark(
    tokens: list[int],
    seed_token: int,
    secret_key: bytes,
    gamma: float = 0.5,
) -> dict:
    """
    Compute the z-score for a sequence of tokens.
    
    Returns a dict with z_score, green_count, total_tokens, and p_value.
    The null hypothesis is that the text is human-written (each token is
    in the green list with probability gamma, independently).
    """
    T = len(tokens)
    if T == 0:
        return {"z_score": 0.0, "green_count": 0, "total": 0, "p_value": 1.0}
    
    green_count = 0
    prev = seed_token
    
    for tok in tokens:
        green = get_green_list(prev, secret_key, gamma=gamma)
        if tok in green:
            green_count += 1
        prev = tok
    
    # z-score under Binomial(T, gamma) null
    mu = gamma * T
    sigma = math.sqrt(T * gamma * (1 - gamma))
    z = (green_count - mu) / sigma
    
    # One-sided p-value (standard normal CDF approximation via erfc)
    p_value = 0.5 * math.erfc(z / math.sqrt(2))
    
    return {
        "z_score": round(z, 4),
        "green_count": green_count,
        "total_tokens": T,
        "gamma_expected": round(mu, 1),
        "p_value": round(p_value, 8),
        "flagged": z > 4.0,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    KEY = b"supersecret-operator-key-2024"
    SEED_TOKEN = 1234
    N = 200  # tokens to generate

    print("=== Watermarked text ===")
    wm_tokens = generate_watermarked(SEED_TOKEN, N, KEY, delta=2.0, gamma=0.5)
    result_wm = detect_watermark(wm_tokens, SEED_TOKEN, KEY)
    print(result_wm)
    # Expected: z_score >> 4, flagged=True

    print("\n=== Human (random) text ===")
    rng = random.Random(99)
    human_tokens = [rng.randint(0, VOCAB_SIZE - 1) for _ in range(N)]
    result_human = detect_watermark(human_tokens, SEED_TOKEN, KEY)
    print(result_human)
    # Expected: z_score near 0, flagged=False

    print("\n=== Paraphrase attack: replace 40% tokens randomly ===")
    attacked = list(wm_tokens)
    rng2 = random.Random(7)
    for i in range(N):
        if rng2.random() < 0.4:
            attacked[i] = rng2.randint(0, VOCAB_SIZE - 1)
    result_atk = detect_watermark(attacked, SEED_TOKEN, KEY)
    print(result_atk)
    # Expected: z_score substantially reduced from the unattacked value;
    # in this synthetic (random-logit) simulation it drops below the z*=4.0
    # threshold, though with a real LM's peakier distributions it typically
    # stays flagged until 70-80% of tokens are replaced (see prose below).
```

Running this demo (with the fixed seeds shown above) produces exactly this output:

```text
=== Watermarked text ===
{'z_score': 11.4551, 'green_count': 181, 'total_tokens': 200, 'gamma_expected': 100.0, 'p_value': 0.0, 'flagged': True}

=== Human (random) text ===
{'z_score': -0.7071, 'green_count': 95, 'total_tokens': 200, 'gamma_expected': 100.0, 'p_value': 0.76024994, 'flagged': False}

=== Paraphrase attack: replace 40% tokens randomly ===
{'z_score': 3.3941, 'green_count': 124, 'total_tokens': 200, 'gamma_expected': 100.0, 'p_value': 0.00034426, 'flagged': False}
```

Here the 40% substitution attack already drops $z$ below the 4.0 threshold for this particular random draw — a reminder that in this toy simulation (random logits standing in for a real LM, so the model has no genuine preference among tokens) the watermark carries less signal than it would in a real deployment, where a language model's confident, low-entropy continuations mean a much larger fraction of tokens must be destroyed before $z$ drops below threshold. With a real model and this $\delta,\gamma$ setting, published results show an adversary typically needs to replace on the order of 70–80% of tokens to reliably evade detection, at which point the original content is largely gone.

---

## Robustness to Attacks

A watermark that fails under modest editing provides only false assurance. The main attack classes are:

| Attack | Description | Effect on $z$ |
|---|---|---|
| Random token substitution | Replace $p$ fraction with random tokens | $z$ scales as $(1-p)$; at $p=0.5$, $z$ halved |
| Paraphrase (LLM rewrite) | Feed text to a second model and rewrite | Moderate; semantic content preserved, tokens changed |
| Translation roundtrip | EN→FR→EN | Moderate; depends on vocabulary overlap |
| Copy-paste splicing | Embed watermarked snippet into human text | Dilutes $z$ by dilution factor |
| Generative attack (DIPPER-style) | Adversarial paraphraser trained to remove signal | Strong; can drop $z$ below threshold if attacker has API access |
| Token insertion/deletion | Insert filler tokens | Context hashing can be disrupted |

The *context-window hash* choice ($h$, the number of preceding tokens hashed) determines how an insertion/deletion attack propagates. With $h=1$ (Markov-1), inserting one token shifts the green list for only one future token. With $h=4$, it shifts four. Sliding-window detection (try all possible insertion offsets) partially recovers.

Adaptive adversaries who can query the model repeatedly — and observe which tokens land in the green list — can, in principle, reconstruct the green list and mask it. This motivates *multi-bit and multi-key* watermarks: rather than a single binary signal, embed a message identifier drawn from a space of $2^{32}$ or more possible keys, so that cracking one key does not transfer.

{{fig:wmprov-detection-separation}}

---

## Multimodal Watermarking: SynthID and Its Relatives

Text watermarking exploits the discrete token distribution. Image, audio, and video watermarking must operate in continuous, high-dimensional pixel/waveform space and must survive JPEG compression, resizing, screen-capture, and codec re-encoding.

### SynthID-Class Image Watermarking

Google DeepMind's SynthID (Fernandez et al., 2023) embeds a learned, imperceptible pattern into image pixel values. Rather than modifying pixel intensities post-hoc (the classical LSB or DWT approach), SynthID integrates a small neural network "watermark decoder" trained jointly with or fine-tuned alongside the image model. The encoder learns to add a near-zero-magnitude steganographic signal that:

- Survives JPEG compression at quality factors as low as 50.
- Survives resizing and cropping.
- Is invisible to the human eye (SSIM > 0.999 on natural images).

Detection uses the same decoder network: the residual embedding is projected onto a learned detection vector, and a threshold test is applied.

**Spectral domain techniques.** Earlier (non-neural) approaches work in the frequency domain. The DWT-DFT watermark embeds a pseudo-random bit sequence into the mid-frequency DCT or DFT coefficients of an image. These are robust to JPEG (which discards high-frequency coefficients) but remain in the preserved mid-frequencies. The trade-off is that neural-network detectors can learn to strip these signals when given enough examples.

### Audio and Video

For speech synthesis (text-to-speech), audio watermarking can be applied in the mel-spectrogram domain before vocoding or via a neural encoder-decoder analogous to SynthID. SynthID-Audio (announced by Google, 2024) claims robustness to mp3 compression, speed changes of ±10%, and additive noise up to 30dB SNR loss.

Video watermarking faces an additional adversary: temporal resampling (frame-rate change, slow-motion). Frame-level image watermarks compound across frames, so detection uses majority voting across sampled frames, which is robust to partial frame replacement.

{{fig:wmprov-image-wm-pipeline}}

---

## Content Provenance: C2PA and Signed Manifests

Watermarking answers "was this generated by model X?" but not "who generated it, when, and with what inputs?". *Content provenance* standards answer the richer question by attaching a cryptographically signed manifest to the file at the moment of creation.

### The C2PA Standard

The Coalition for Content Provenance and Authenticity (C2PA) is a joint effort of Adobe, Microsoft, BBC, Intel, Sony, and others. The specification (C2PA 2.0, released 2024) defines:

- **Content Credentials**: a JSON-LD manifest embedded in the file's XMP/JUMBF metadata. It records the asset's identity, creation tool, timestamp, and a list of *actions* (crop, generate, edit) performed.
- **Hard-binding**: a cryptographic hash of the asset bytes (SHA-256) is included in the manifest, binding the manifest to exactly this file. Any modification breaks the hash.
- **Soft-binding**: a perceptual hash (e.g., pHash) allows approximate matching after lossy transformations. If the hard-binding hash fails but the perceptual hash matches, the tool reports "modified from a C2PA-signed original."
- **X.509 certificate chain**: the manifest is signed with an operator certificate, whose trust roots back to a C2PA-approved CA. This gives a verifiable, non-repudiable binding between the manifest and the signer's identity.

```json
{
  "alg": "sha256",
  "assertions": [
    {
      "label": "c2pa.actions",
      "data": {
        "actions": [
          {
            "action": "c2pa.created",
            "softwareAgent": "MyAIImageGen v2.1",
            "when": "2025-04-10T14:23:00Z",
            "digitalSourceType": "trainedAlgorithmicMedia"
          }
        ]
      }
    },
    {
      "label": "c2pa.hash.data",
      "data": {
        "alg": "sha256",
        "hash": "a3f4...c291"
      }
    }
  ],
  "claim_generator": "MyAIImageGen/2.1 c2pa-rs/0.25.0",
  "signature_info": {
    "issuer": "MyAI Corp",
    "cert_serial_number": "0x4A2F..."
  }
}
```

### Limitations of Provenance

C2PA solves *honest-party verification*: if a trustworthy party generates content and signs it, downstream consumers can verify the claim. It does not prevent a malicious party from generating content with a modified tool that forges the manifest, or from distributing content through channels that strip metadata.

The key insight: C2PA and watermarking are *complementary*. C2PA provides a rich, human-readable, auditable record for compliant workflows. Watermarking provides a signal that survives metadata stripping. Together they cover the main attack surface.

---

## Post-Hoc AI-Text Detection: What Works and What Doesn't

When proactive watermarks are absent, operators sometimes fall back to *post-hoc detectors* that classify text as AI- or human-written based on statistical features. We must be honest about their limitations.

### Likelihood-Based Detectors

The simplest family uses the generating model's own perplexity. If a text has unusually low perplexity under GPT-4, it was probably written by GPT-4. DetectGPT (Mitchell et al., 2023) formalizes this: it computes the model's log-probability on the candidate text, then samples small perturbations (via a masking model) and asks whether the original is a *local maximum* of log-probability — an expected property of model-sampled text.

$$
\Delta(x) = \log p_\theta(x) - \mathbb{E}_{\tilde{x} \sim q(\cdot|x)}[\log p_\theta(\tilde{x})]
$$

If $\Delta(x) > 0$ by a large margin, the text is flagged as AI-generated. The method requires access to the generating model's probability, which is unavailable for closed-source models.

### Trained Classifier Detectors

Fine-tuned classifiers (e.g., OpenAI's Text Classifier, GPTZero) are trained on (human, AI) pairs. They achieve high accuracy on in-distribution examples but degrade severely under:

- **Domain shift**: a classifier trained on Reddit posts misclassifies academic AI writing.
- **Prompt conditioning**: AI text generated with unusual system prompts is less "typical" and evades detection.
- **Short texts**: below roughly 100 words, false-positive rates jump dramatically.
- **Paraphrase**: light editing by a human or a second model is often sufficient to evade commercial detectors.

A rigorous audit by Weber-Wulff et al. (2023) found that widely-used commercial tools had false-positive rates of 2–10% on human text and true-positive rates as low as 30–70% on real AI-generated content — worse than their advertised numbers. This has profound fairness implications: non-native English speakers tend to use more regular, predictable sentence structures that resemble AI text, and their work is flagged at higher rates.

!!! warning "High false-positive rates are a fundamental fairness problem"
    Deploying AI-text detectors for high-stakes decisions (academic misconduct, legal documents) without understanding their false-positive characteristics can cause serious harm. Non-native English speakers and writers in certain domains face systematically elevated false-positive rates. Any deployment of these systems must involve calibration on the specific population and domain, transparency about error rates, and a meaningful appeals process.

### Why Post-Hoc Detection Is Fundamentally Limited

A language model that is fine-tuned post-hoc to evade a specific detector can do so while maintaining utility on the downstream task. This is a *Goodhart's Law* dynamic: as soon as a measure of AI-ness becomes a target, it ceases to be a reliable measure. Watermarking avoids this problem by putting the signal under the operator's private-key control.

!!! interview "Interview Corner"
    **Q:** A product manager proposes integrating a commercial AI-text detector into a hiring pipeline to flag AI-written cover letters. What technical concerns would you raise?

    **A:** Several interlocking issues: (1) *False-positive rates* on this specific population — job applicants, many non-native speakers, covering diverse domains — are likely uncharacterized and could be 5–15%, far too high for a consequential HR decision. (2) *Distribution shift*: the detector was trained on generic AI text; candidates using domain-specific prompts may be systematically under-flagged or over-flagged. (3) *Adversarial ease*: a trivial human edit pass defeats most commercial detectors; determined bad actors are not meaningfully deterred. (4) *Legal risk*: in many jurisdictions, automated adverse employment decisions based on unvalidated systems create liability. A robust alternative is to require a live writing sample or structured interview component rather than relying on any automated detector.

---

## The EU AI Act: Article 50 and Transparency Obligations

The EU AI Act (fully applicable from August 2026) imposes direct obligations on providers and deployers of generative AI. Article 50 specifies:

1. **Machine-generated content must be labeled.** Providers of general-purpose AI systems that generate synthetic audio, image, video, or text "intended to interact with persons" must ensure the output is marked as AI-generated in a machine-readable format.

2. **Text watermarking for public communication.** For AI-generated text used for public communication (news, opinion, political content), Article 50(2) additionally requires that the content be *watermarked* in a technically robust way, "unless the AI-generated content has undergone substantial human review or editorial oversight."

3. **Deepfake disclosure.** AI-generated or manipulated images, audio, or video of real persons must carry a clear disclosure label visible to the end user.

The Act leaves the specific technical implementation to the Commission (via standards bodies, likely ETSI and CEN/CENELEC), but explicitly names C2PA-style content credentials as a compliant approach. Penalties for non-compliance can reach 1.5% of global annual turnover.

Practically, this means:

- LLM API providers serving EU users must expose a watermark API (or embed watermarks by default) for public-facing text generation.
- Image and video generation services must embed C2PA credentials or equivalent.
- The "substantial human review" exemption creates a compliance design choice: a product can choose either watermarking or a mandatory human-in-the-loop review gate.

Cross-reference: [AI Governance, Compliance & Regulation](../13-interp-safety-gov/06-governance-compliance.html) covers the full EU AI Act risk-tier framework and the broader global compliance landscape.

---

## System-Level Architecture for a Production Watermark Pipeline

Deploying watermarking at scale requires integrating it into the inference serving stack without unacceptable latency overhead.

{{fig:wmprov-vllm-logits-processor}}

For high-throughput serving, the hash computation can be parallelized on GPU. The green-list can be represented as a bitmask of $|V|$ bits (32000 / 8 = 4 KB), which fits comfortably in L1 cache. The logit-addition is a vectorized CUDA kernel that adds `delta * mask` to the logit vector, adding negligible latency.

A real implementation also needs:

- **Key management**: the secret key must be stored in a hardware security module (HSM) or equivalent; compromise of the key allows an adversary to construct text that passes detection.
- **Per-request key rotation**: rather than a global key, derive a per-request subkey from the master key and a request ID. This limits the exposure from any single leaked subkey.
- **Audit log**: record the request ID → watermark key mapping so that a flagged document can be traced back to the originating request (with appropriate legal authorization).

---

## Practical Limitations and Open Problems

Despite their elegance, current watermarking schemes face several real-world limitations.

**Low-entropy text.** If the next token is highly predictable — e.g., completing a fixed template, writing code with rigid syntax, or generating a phone number — the model has essentially no degrees of freedom. The green list is irrelevant because the only viable token may be red. The effective green-token rate collapses, and the $z$-score is diluted. This is a fundamental tension: watermarking works best when the model has many plausible continuations.

**Multi-model provenance.** When a document is partly human-written, partly AI-generated, and partly AI-edited, neither a single $z$-score nor a binary classifier can give a nuanced answer. Forensic attribution — "this paragraph is AI-generated; this one is not" — requires token-level rather than document-level watermarking, an active research area.

**Open-weight models.** Any open-weight model (LLaMA, Mistral, etc.) can be deployed without watermarking. The operator can remove the watermark processor trivially. This means watermarking mandates (like EU AI Act Article 50) apply only to API-based closed models; open-weight models present a policy gap. One proposed mitigation is embedding watermarks *in the model weights* (via fine-tuning to prefer green tokens without an explicit processor), but this is easily circumvented by further fine-tuning.

**Semantic watermarks.** An emerging line of work embeds watermarks at the *semantic* level — in the choice of which concepts to mention, which synonyms to use — rather than at the token level. Semantic watermarks are more robust to paraphrase but harder to detect without access to a semantic model, and their statistical properties are less well understood.

---

## Key Takeaways

!!! key "Key Takeaways"
    - The KGW green-list watermark biases sampling toward a secret-key-derived token subset. Detection computes a z-score under the Binomial null; a score above ~4 corresponds to a false-positive rate around $10^{-5}$.
    - Distortion-free watermarks (Kuditipudi et al.) preserve the exact marginal token distribution while still embedding a detectable signal via inverse-CDF coupling; preferred when output quality is paramount.
    - A 40% random token substitution attack roughly halves the z-score; an adversary must destroy 70–80% of the text to reliably evade detection — at which point they have rewritten the content anyway.
    - SynthID-class image watermarks use learned neural encoders trained end-to-end; they survive JPEG, resizing, and cropping far better than classical LSB or DFT approaches.
    - C2PA content credentials provide cryptographically signed manifests with hard and soft asset bindings; they are complementary to watermarking — manifests survive format preservation, watermarks survive metadata stripping.
    - Post-hoc AI-text detectors have fundamental limitations: domain shift, adversarial evasion, and false-positive rates that systematically disadvantage non-native speakers. They should not be used for high-stakes automated decisions.
    - EU AI Act Article 50 mandates machine-readable watermarking or labeling for AI-generated text used in public communication, with penalties up to 1.5% of global turnover for non-compliance.
    - Open-weight models create a policy gap: any operator can remove watermarking infrastructure, making technical mandates applicable only to API-gated services.
    - Key management and per-request key derivation are the production engineering concerns most often overlooked in academic watermarking papers.

---

!!! sota "State of the Art & Resources (2026)"
    Watermarking and provenance for AI-generated content is an active research and standards area: the KGW green-list scheme (2023) anchors the statistical watermarking literature, distortion-free variants have closed the quality gap, Google's SynthID has been deployed at production scale for text and images, and the C2PA 2.x specification is now embedded in major creative tools. EU AI Act Article 50 enforcement from August 2026 is accelerating industry adoption of both watermarking APIs and content-credential pipelines.

    **Foundational work**

    - [Kirchenbauer et al., *A Watermark for Large Language Models* (ICML 2023)](https://arxiv.org/abs/2301.10226) — the KGW green-list scheme; defines the z-score detection framework that remains the field's baseline.
    - [Kuditipudi et al., *Robust Distortion-Free Watermarks for Language Models* (2023)](https://arxiv.org/abs/2307.15593) — inverse-CDF coupling that preserves the exact token distribution; preferred when output quality is paramount.
    - [Mitchell et al., *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature* (ICML 2023)](https://arxiv.org/abs/2301.11305) — post-hoc likelihood-curvature detector; illustrates both the promise and limits of model-based detection.

    **Recent advances (2023–2026)**

    - [Fernandez et al., *The Stable Signature: Rooting Watermarks in Latent Diffusion Models* (ICCV 2023)](https://arxiv.org/abs/2303.15435) — fine-tunes the latent decoder of diffusion models to embed per-user invisible signatures; >90% detection after 90% crop.
    - [Weber-Wulff et al., *Testing of Detection Tools for AI-Generated Text* (2023)](https://arxiv.org/abs/2306.15666) — rigorous audit showing commercial detectors achieve 30–70% true-positive rates with 2–10% false-positive rates; essential reading before deploying any detector.
    - [Google DeepMind, *Watermarking AI-generated text and video with SynthID* (blog, 2024)](https://deepmind.google/blog/watermarking-ai-generated-text-and-video-with-synthid/) — production deployment of tournament-sampling text watermarks and video watermarking at Gemini scale.
    - [EU AI Act Article 50 — Transparency Rules Guide](https://artificialintelligenceact.eu/transparency-rules-article-50/) — plain-language breakdown of the August 2026 mandatory watermarking and labeling obligations for generative AI providers and deployers.

    **Open-source & tools**

    - [jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking) — official KGW reference implementation as a Hugging Face `LogitsProcessor`; drop-in for any model supporting `generate`.
    - [google-deepmind/synthid-text](https://github.com/google-deepmind/synthid-text) — reference implementation for the Nature 2024 SynthID-Text watermark with both weighted-mean and Bayesian detectors.
    - [THU-BPM/MarkLLM](https://github.com/THU-BPM/MarkLLM) — unified toolkit (EMNLP 2024) implementing 24 watermarking algorithms including KGW, SynthID-Text, and SIR, with detection pipelines and robustness evaluation.

    **Go deeper**

    - [C2PA Technical Specification (v2.4)](https://spec.c2pa.org/specifications/) — normative standard for content credentials: hard- and soft-binding, X.509 certificate chains, and the JSON-LD manifest format.
    - [Content Authenticity Initiative — How It Works](https://contentauthenticity.org/how-it-works) — accessible explainer on C2PA deployment across cameras, editing tools, and social platforms; covers the "nutrition label" model for provenance.

## Further Reading

- Kirchenbauer, J., Geiping, J., Wen, Y., Kirchenbauer, K., Goldblum, M., and Goldstein, T. — *A Watermark for Large Language Models* (2023). The foundational green-list watermark paper.
- Kuditipudi, R., Thickstun, J., Hashimoto, T., and Liang, P. — *Robust Distortion-Free Watermarks for Language Models* (2023). Introduces the distortion-free inverse-CDF construction.
- Fernandez, P., Couairon, G., Jégou, H., Douze, M., and Furon, T. — *The Stable Signature: Rooting Watermarks in Latent Diffusion Models* (NeurIPS 2023). Neural watermarking for latent diffusion.
- Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., and Finn, C. — *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature* (ICML 2023).
- Weber-Wulff, D. et al. — *Testing of Detection Tools for AI-Generated Text* (2023). Rigorous empirical audit of commercial detectors.
- C2PA Technical Specification v2.0 — Coalition for Content Provenance and Authenticity (2024). The normative standard for content credentials.
- Google DeepMind — *SynthID: Identifying AI-Generated Content* (Nature, 2024). Details of SynthID image and audio watermarking.
- Zhao, X., Ananth, P., Li, L., and Wang, Y. — *Provably Robust Multi-bit Watermarking for AI-Generated Text* (2023). Multi-bit extensions with information-theoretic robustness proofs.
