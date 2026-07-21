# 7.9 Sampling Strategies & Decoding Algorithms

Every token an LLM produces is chosen by a *decoding algorithm* — a procedure that converts the raw logit vector from the final transformer layer into a discrete token. The choice of algorithm is not a minor implementation detail. It governs the fundamental tradeoff between diversity and fidelity: too deterministic, and the model parrots its training distribution; too stochastic, and it drifts into incoherence. This chapter covers that tradeoff with precision — from the mathematics of each algorithm to production-grade implementation tricks.

We assume you have read about the autoregressive decoding loop in [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html) and that you understand how [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html) interacts with the sampling step. We also touch on connections to RL fine-tuning discussed in [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html).

## The Probability Pipeline: Logits to Token

The model outputs a real-valued vector $\mathbf{z} \in \mathbb{R}^{|V|}$ called *logits*, where $|V|$ is the vocabulary size (32,000 for LLaMA-2, 128,256 for LLaMA-3). The pipeline from logits to a sampled token is:

$$
\mathbf{z} \xrightarrow{\text{processors}} \mathbf{z}' \xrightarrow{\text{softmax}} \mathbf{p} \xrightarrow{\text{sampler}} t \in \{0,\ldots,|V|-1\}
$$

*Logit processors* transform the raw logits before the softmax — applying temperature scaling, vocabulary filtering, or penalty terms. The *sampler* draws from the resulting distribution. This separation is important: processors compose and stack, while the sampler is usually just a multinomial draw.

```python
import torch
import torch.nn.functional as F
from typing import List, Optional

def sample_token(
    logits: torch.Tensor,          # shape: (vocab_size,), raw model outputs
    processors: List["LogitProcessor"],
    do_sample: bool = True,
) -> int:
    """
    Apply a list of logit processors in sequence, then sample one token.
    Returns the integer token id.
    """
    logits = logits.clone().float()  # always upcast to fp32 for numerical safety

    for proc in processors:
        logits = proc(logits)        # each processor modifies logits in-place or returns a new tensor

    if not do_sample:
        # greedy: just the argmax
        return int(logits.argmax())

    # convert to probabilities and sample
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))
```

Everything in this chapter is a specialisation of this loop. Let's build each piece.

## Greedy Decoding and Its Failure Modes

Greedy decoding picks the most probable token at every step:

$$
t_i = \arg\max_{t} P(t \mid t_{<i}, \mathbf{x})
$$

It is deterministic, fast, and *not* equivalent to finding the most probable sequence. The globally most probable sequence — the *maximum a posteriori* (MAP) sequence — requires an exponential search, because token choices interact: the best next token often forecloses the best continuation.

In practice, greedy decoding produces repetitive, low-entropy text. Once the model writes "The cat sat on the", it assigns high probability to "mat", which then makes "mat mat mat" the greedy continuation. This is not a defect in the model; it is a consequence of choosing the locally optimal token at each step without lookahead.

```python
def greedy_decode(
    model,
    input_ids: torch.Tensor,   # (1, seq_len)
    max_new_tokens: int = 100,
    eos_token_id: int = 2,
) -> torch.Tensor:
    """Minimal greedy decode loop."""
    generated = input_ids
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated).logits[:, -1, :]   # (1, vocab)
        next_token = logits.argmax(dim=-1, keepdim=True)  # (1, 1)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == eos_token_id:
            break
    return generated
```

Use greedy when: (a) you need exact reproducibility, (b) the task has a single correct answer and the model is well-calibrated (e.g., code completion with high-temperature training), or (c) you are the draft model in speculative decoding and the verifier handles stochasticity.

## Temperature Scaling

Temperature $T > 0$ is the single most important hyperparameter in decoding. It divides every logit by $T$ before the softmax:

$$
P_T(t) = \frac{\exp(z_t / T)}{\sum_{t'} \exp(z_{t'} / T)}
$$

- $T \to 0$: distribution collapses to a point mass at the argmax (greedy).
- $T = 1$: the model's trained distribution, unchanged.
- $T \to \infty$: distribution becomes uniform over the vocabulary.

Dividing by $T < 1$ *sharpens* the distribution (amplifies differences between logits); dividing by $T > 1$ *flattens* it (suppresses differences).

```python
class TemperatureProcessor:
    def __init__(self, temperature: float):
        assert temperature > 0, "Temperature must be positive"
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        # Dividing logits by T is equivalent to multiplying log-probs by 1/T,
        # which is exactly the "softmax temperature" formulation.
        return logits / self.temperature
```

!!! example "Worked example: feeling the temperature"

    Suppose the top two logits are $z_A = 5.0$ and $z_B = 4.0$ (all others much smaller).

    At **$T = 1$**:
    $$p_A = \frac{e^5}{e^5 + e^4} = \frac{148.4}{148.4 + 54.6} \approx 0.731, \quad p_B \approx 0.269$$

    At **$T = 0.5$** (sharpen):
    $$p_A = \frac{e^{10}}{e^{10} + e^{8}} \approx \frac{22026}{22026 + 2981} \approx 0.881, \quad p_B \approx 0.119$$

    At **$T = 2$** (flatten):
    $$p_A = \frac{e^{2.5}}{e^{2.5} + e^{2}} \approx \frac{12.18}{12.18 + 7.39} \approx 0.622, \quad p_B \approx 0.378$$

    At $T = 0.5$, the model is roughly 7.4× more likely to pick $A$ than $B$; at $T = 2$, only 1.6× more likely. A small change in temperature creates large changes in practice.

{{fig:sampling-temperature-reshapes-distribution}}

### Temperature and the Training Distribution

A crucial subtlety: the model was trained with teacher-forcing at $T = 1$. When you use $T < 1$ at inference, you are sampling from a distribution that is *sharper* than what the model saw during training. This is usually fine — the probabilities just become more concentrated. When you use $T > 1$, you are sampling tokens the model considers unlikely, which can produce creative but also hallucinated or incoherent text.

Temperature interacts deeply with RL fine-tuning (see [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html)). During PPO rollouts, the *policy temperature* determines exploration; a too-cold policy underfits the reward landscape, a too-hot policy produces garbage completions that confuse the reward model. Similarly, distillation objectives (see [Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html)) often match *soft targets* — temperature-scaled probability distributions — from the teacher, where a higher temperature exposes the "dark knowledge" in the teacher's off-peak probabilities.

## Top-k Sampling

Top-k sampling restricts the vocabulary to the $k$ tokens with the highest logits before sampling:

$$
\text{TopK}(\mathbf{z}, k): \text{set } z_t = -\infty \text{ for all } t \notin \text{Top}k(\mathbf{z})
$$

Setting logits to $-\infty$ ensures those tokens get zero probability after softmax. This prevents the long tail of improbable tokens from polluting the distribution.

```python
class TopKProcessor:
    def __init__(self, top_k: int):
        assert top_k >= 1
        self.top_k = top_k

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        # kth_val is the minimum value among the top-k
        kth_val = torch.topk(logits, self.top_k).values[-1]
        # Mask everything below the threshold
        return logits.masked_fill(logits < kth_val, float('-inf'))
```

**Limitation:** $k$ is absolute. With $k = 50$, you always consider 50 tokens whether the distribution is tight (one token dominates) or flat (many tokens are reasonable). Top-p addresses this.

## Top-p / Nucleus Sampling

Holtzman et al. ("The Curious Case of Neural Text Degeneration", 2020) introduced *nucleus sampling*: sample from the smallest set of tokens whose cumulative probability exceeds $p$.

$$
V_p = \min S \subseteq V \;\text{ s.t. }\; \sum_{t \in S} P(t) \geq p
$$

Tokens outside $V_p$ are suppressed. The nucleus adapts: when the model is confident (the top token alone has probability 0.9), the nucleus is tiny; when the model is uncertain (probability spread over many tokens), the nucleus expands.

```python
class TopPProcessor:
    def __init__(self, top_p: float):
        assert 0 < top_p <= 1.0
        self.top_p = top_p

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        # Sort logits descending to build cumulative sum
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Remove tokens once cumulative probability exceeds top_p.
        # We shift by one so the token that crosses the threshold is kept.
        remove_mask = cumulative_probs - probs > self.top_p
        sorted_logits[remove_mask] = float('-inf')

        # Scatter back to original token ordering
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(0, sorted_indices, sorted_logits)
        return logits_filtered
```

A typical production default is `top_p=0.9` with `temperature=0.8`. Note that top-k and top-p *compose*: apply top-k first to cut the long tail cheaply, then apply top-p for adaptive nucleus selection.

## Min-p Sampling

Min-p (Nguyen et al., 2023) takes a different angle: instead of keeping a top fraction by mass, it keeps all tokens whose probability is at least $p_{\min}$ times the *maximum* token probability:

$$
V_{\min\text{-}p} = \{t : P(t) \geq p_{\min} \cdot \max_{t'} P(t')\}
$$

This scales the threshold relative to the peak, so it automatically tightens when the model is confident and loosens when it is uncertain — similar to top-p, but parameterised differently and often producing smoother behaviour at extreme temperatures.

```python
class MinPProcessor:
    def __init__(self, min_p: float):
        assert 0 < min_p < 1.0
        self.min_p = min_p

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        # Absolute threshold: min_p * p_max
        threshold = self.min_p * probs.max()
        logits = logits.masked_fill(probs < threshold, float('-inf'))
        return logits
```

{{fig:sampling-truncation-family-comparison}}

## Typical Sampling

Meister et al. ("Typical Decoding for Natural Language Generation", 2023) approach diversity from an information-theoretic angle. They observe that human text is *typically* drawn from the centre of the entropy distribution: tokens whose surprisal $-\log P(t)$ is close to the conditional entropy $H = -\sum_t P(t) \log P(t)$.

Define the *typicality* of token $t$:

$$
|\!-\!\log P(t) - H| \leq \delta
$$

Typical sampling keeps only tokens satisfying this condition, discarding both the most probable (low surprisal, repetitive) and the least probable (high surprisal, incoherent).

```python
class TypicalProcessor:
    def __init__(self, mass: float = 0.9):
        """Keep the smallest set of 'typical' tokens covering `mass` probability."""
        self.mass = mass

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        neg_log_probs = -torch.log(probs + 1e-10)

        # Conditional entropy (expected surprisal)
        H = (probs * neg_log_probs).sum()

        # Typicality: distance of each token's surprisal from entropy
        typicality = (neg_log_probs - H).abs()

        # Sort by typicality (most typical first), then take the nucleus covering `mass`
        sorted_typicality, sorted_indices = torch.sort(typicality)
        sorted_probs = probs[sorted_indices]
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumsum - sorted_probs > self.mass

        sorted_logits = logits[sorted_indices]
        sorted_logits[remove_mask] = float('-inf')

        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(0, sorted_indices, sorted_logits)
        return logits_filtered
```

## Repetition and Frequency Penalties

Even with good sampling, models can fall into repetitive loops. Two complementary penalties address this.

**Repetition penalty** (Keskar et al., "CTRL", 2019) discounts any token that has already appeared in the context:

$$
z'_t = \begin{cases} z_t / \theta & \text{if } t \in \text{context}, \; z_t > 0 \\ z_t \cdot \theta & \text{if } t \in \text{context}, \; z_t < 0 \end{cases}
$$

for penalty $\theta > 1$. This reduces the probability of repeated tokens without suppressing them entirely.

**Frequency penalty** (used in OpenAI's API) subtracts a penalty proportional to how many times the token has appeared:

$$
z'_t = z_t - \alpha \cdot \text{count}(t, \text{context})
$$

**Presence penalty** is a simpler binary version — subtract a fixed $\beta$ if the token appears at all:

$$
z'_t = z_t - \beta \cdot \mathbf{1}[t \in \text{context}]
$$

```python
class RepetitionPenaltyProcessor:
    """
    Multiplicative repetition penalty (CTRL-style).
    theta > 1 reduces repetition; theta = 1 is no-op.
    """
    def __init__(self, penalty: float, input_ids: torch.Tensor):
        assert penalty >= 1.0
        self.penalty = penalty
        # Track unique tokens seen in the context
        self.seen = set(input_ids.flatten().tolist())

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        for token_id in self.seen:
            if logits[token_id] > 0:
                logits[token_id] /= self.penalty
            else:
                logits[token_id] *= self.penalty
        return logits


class FrequencyPenaltyProcessor:
    """
    Additive frequency penalty: subtract alpha * count(t).
    Also supports presence penalty (binary).
    """
    def __init__(
        self,
        frequency_penalty: float,
        presence_penalty: float,
        input_ids: torch.Tensor,
    ):
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        # Count occurrences of each token in context
        token_list = input_ids.flatten().tolist()
        from collections import Counter
        self.counts = Counter(token_list)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        for token_id, count in self.counts.items():
            logits[token_id] -= (
                self.frequency_penalty * count
                + self.presence_penalty
            )
        return logits
```

!!! warning "Penalty interaction with temperature"

    Repetition penalties are applied to logits *before* temperature scaling in most implementations (HuggingFace `transformers` applies them in the order: repetition penalty → temperature → top-k → top-p). If you change that order, the effective penalty magnitude changes. Always check the order in your stack.

## Beam Search

Beam search maintains a *beam* of $B$ partial hypotheses, expanding each at every step and keeping only the top-$B$ by cumulative log-probability:

$$
\text{score}(t_{1:n}) = \frac{1}{n^\alpha} \sum_{i=1}^n \log P(t_i \mid t_{<i})
$$

The length normalization exponent $\alpha$ (typically 0.6–0.8) prevents the beam from preferring shorter sequences.


{{fig:sampling-beam-search-tree}}


```python
import heapq
from dataclasses import dataclass, field
from typing import Tuple

@dataclass(order=True)
class BeamHypothesis:
    score: float          # negative log-prob (min-heap)
    tokens: list = field(compare=False)

def beam_search(
    model,
    input_ids: torch.Tensor,     # (1, prefix_len)
    beam_size: int = 4,
    max_new_tokens: int = 100,
    eos_id: int = 2,
    length_alpha: float = 0.6,
) -> List[int]:
    """
    Minimal beam search. Returns the best sequence as a list of token ids.
    NB: production implementations use KV-cache for each beam; this toy
    version re-encodes each step for clarity.
    """
    prefix = input_ids[0].tolist()
    # heap items: (neg_score, token_list)
    active_beams: List[Tuple[float, List[int]]] = [(0.0, prefix[:])]
    finished: List[Tuple[float, List[int]]] = []

    for _ in range(max_new_tokens):
        if not active_beams:
            break
        candidates: List[Tuple[float, List[int]]] = []

        for neg_score, tokens in active_beams:
            ids = torch.tensor([tokens], dtype=torch.long)
            with torch.no_grad():
                logits = model(ids).logits[0, -1, :]   # (vocab,)
            log_probs = F.log_softmax(logits, dim=-1)

            # Expand: consider all tokens in the vocabulary
            topk_logprob, topk_ids = torch.topk(log_probs, beam_size)
            for lp, tid in zip(topk_logprob.tolist(), topk_ids.tolist()):
                new_tokens = tokens + [tid]
                new_neg_score = neg_score - lp   # minimise negative log-prob
                if tid == eos_id:
                    # Normalise by length
                    n = len(new_tokens) - len(prefix)
                    normalised = new_neg_score / (n ** length_alpha)
                    finished.append((normalised, new_tokens))
                else:
                    candidates.append((new_neg_score, new_tokens))

        # Keep the best beam_size active hypotheses
        candidates.sort(key=lambda x: x[0])
        active_beams = candidates[:beam_size]

    # Fall back to active beams if none finished
    if not finished:
        n_prefix = len(prefix)
        for neg_score, tokens in active_beams:
            n = len(tokens) - n_prefix
            normalised = neg_score / max(n, 1) ** length_alpha
            finished.append((normalised, tokens))

    finished.sort(key=lambda x: x[0])
    return finished[0][1]   # best hypothesis
```

**When to use beam search:** summarisation, machine translation, and tasks where the output has a clear quality metric. For open-ended generation, beam search amplifies repetition and produces bland text — sampling is better. For structured generation (see [Structured & Constrained Generation](../07-inference-serving/10-structured-generation.html)), beam search is often combined with constraint masks.

**Diverse beam search** (Vijayakumar et al., 2018) adds a dissimilarity penalty between beams, encouraging them to explore different branches — useful when you want $N$-best diverse outputs.

## Contrastive Decoding and DoLa

Two recent methods depart from purely probability-based selection.

### Contrastive Decoding

Li et al. ("Contrastive Decoding", 2023) observe that an *amateur* model (smaller, weaker) and the *expert* model (the LLM) assign high probability to the same fluent but factually incorrect completions. The contrastive score subtracts the amateur's log-probability from the expert's:

$$
\text{CD}(t) = \log P_{\text{expert}}(t) - \log P_{\text{amateur}}(t)
$$

Tokens that the expert prefers over the amateur are amplified; tokens both models agree on (fluency) but the expert is unsure about are suppressed. In practice, the amateur is often the same model at a smaller context window or with its early layers.

### DoLa (Decoding by Contrasting Layers)

Chuang et al. ("DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models", 2023) push this idea inside a single model. The observation: factual knowledge is represented in the later transformer layers, while surface-level fluency is settled earlier. DoLa computes:

$$
J(t) = \text{JSD}\!\left(P_{\text{final}}(\cdot) \;\|\; P_{\text{premature}}(\cdot)\right)
$$

to select the premature layer that diverges most from the final layer, then uses:

$$
\text{DoLa}(t) = \log P_{\text{final}}(t) - \log P_{\text{premature}}(t)
$$

as the decoding score. This amplifies tokens the final (knowledge-rich) layers prefer over intermediate layers, reducing hallucination without an external model.

```python
def dola_logits(
    model,
    input_ids: torch.Tensor,
    premature_layer: int,         # e.g. layer 16 in a 32-layer model
    final_layer: int = -1,        # -1 = last layer
    alpha: float = 0.1,           # mixing coefficient
) -> torch.Tensor:
    """
    Compute DoLa-adjusted logits by contrasting the final layer
    against a premature intermediate layer.

    The model must expose intermediate hidden states, e.g. via
    output_hidden_states=True in HuggingFace.
    """
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
        )
    hidden_states = outputs.hidden_states   # tuple of (batch, seq, d_model)

    # Project intermediate hidden state through the LM head
    lm_head = model.lm_head
    premature_logits = lm_head(hidden_states[premature_layer][:, -1, :])
    final_logits     = lm_head(hidden_states[final_layer][:, -1, :])

    # Contrastive combination
    dola_log_probs = (
        F.log_softmax(final_logits, dim=-1)
        - alpha * F.log_softmax(premature_logits, dim=-1)
    )
    return dola_log_probs.squeeze(0)
```

## The Bias-Diversity Tradeoff

Every decoding choice sits on a single underlying tradeoff surface. Let us make it concrete.

**Diversity** refers to the entropy or variety of the generated text — how many different continuations the sampler would produce for the same prompt. **Bias** refers to the systematic difference between the decoded distribution and the model's true distribution.

Greedy decoding has zero variance (perfectly reproducible) but extreme bias — it always picks the mode, ignoring the full shape of the distribution. Uniform sampling has maximum diversity but ignores everything the model learned. Temperature is the most direct knob on this axis.


{{fig:sampling-bias-diversity-axis}}


The ideal point depends on the task:

| Task | Typical choice |
|------|---------------|
| Code completion, factual QA | Low temperature (0.1–0.5), greedy or small top-p |
| Chat, instruction following | Temperature 0.7–1.0, top-p 0.9 |
| Creative writing, brainstorming | Temperature 1.0–1.3, top-p 0.95 or min-p |
| RL rollouts (exploration) | Temperature 0.9–1.2, no truncation |
| Distillation soft targets | Temperature 2.0–5.0, no truncation |

One important empirical finding: for most instruction-tuned models, the training process implicitly calibrates the output distribution for temperature around 0.7–1.0. Going significantly below 0.3 causes the model to confidently hallucinate because the distribution was not trained to be sharp; going above 1.5 on instruction models often causes grammatical collapse.

## Logit Processor Pipelines in Practice

Production systems (HuggingFace Transformers, vLLM, SGLang) implement logit processing as a composable pipeline. The full processing order typically is:


{{fig:sampling-logit-pipeline}}


Below is a composable, HuggingFace-compatible `LogitsProcessor` implementation:

```python
from transformers import LogitsProcessor, LogitsProcessorList
import torch

class CompositeLogitsProcessor(LogitsProcessor):
    """
    A single LogitsProcessor that applies a pipeline of sub-processors,
    each expecting (input_ids, scores) -> scores.
    This follows the HuggingFace LogitsProcessor protocol.
    """

    def __init__(self, processors: list):
        self.processors = processors

    def __call__(
        self,
        input_ids: torch.LongTensor,     # (batch, seq_len)
        scores: torch.FloatTensor,       # (batch, vocab_size)
    ) -> torch.FloatTensor:
        for proc in self.processors:
            scores = proc(input_ids, scores)
        return scores


# Example: build a typical production pipeline
def build_logits_processor_list(
    temperature: float = 0.8,
    repetition_penalty: float = 1.1,
    top_k: int = 50,
    top_p: float = 0.9,
) -> LogitsProcessorList:
    from transformers import (
        TemperatureLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    procs = LogitsProcessorList()
    if repetition_penalty != 1.0:
        procs.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if temperature != 1.0:
        procs.append(TemperatureLogitsWarper(temperature=temperature))
    if top_k > 0:
        procs.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p < 1.0:
        procs.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
    return procs
```

The `min_tokens_to_keep=1` argument is critical: it prevents cases where the filter masks *all* tokens (which would cause a nan after softmax), by always keeping at least the top token.

### Custom Logit Processors

Custom processors enable powerful behaviours:

- **Grammar enforcement**: mask all tokens incompatible with a context-free grammar at the current parser state (see [Structured & Constrained Generation](../07-inference-serving/10-structured-generation.html)).
- **Watermarking**: John Kirchenbauer et al. ("A Watermark for Large Language Models", 2023) add a small positive bias to a randomly chosen half of the vocabulary, seeded by the previous token. This creates a statistically detectable fingerprint in the output.
- **Token healing**: vLLM's token healing re-processes the last token of the prompt to avoid boundary artefacts when the prompt ends mid-token.
- **Vocabulary bias**: force specific tokens (e.g., "yes"/"no" for classification prompts) by setting all other logits to $-\infty$.

## Temperature and RL/Distillation Interactions

This section connects decoding to training, two topics that interact subtly.

### Policy Temperature in RL Fine-Tuning

During PPO (see [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html)), the policy generates rollouts. The sampling temperature during rollout generation serves as an *exploration* parameter:

- Low $T$: the policy stays close to the current mode, producing similar completions. Good for exploiting known-good strategies.
- High $T$: the policy explores more, which may discover high-reward completions but also floods the reward model with low-quality samples.

The KL penalty term in PPO ($\beta \cdot D_{\text{KL}}(\pi \| \pi_{\text{ref}})$) is computed between the policy logits and reference model logits at $T = 1$. If you sample rollouts at $T \ne 1$ but compute the KL at $T = 1$, you are optimising a different objective than the one you are generating samples from. Most correct implementations compute both log-probabilities at $T = 1$ (or consistently at the same $T$). See [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html) for details.

### Knowledge Distillation and Soft Targets

Hinton et al. ("Distilling the Knowledge in a Neural Network", 2015) showed that training a student on soft targets from a teacher at temperature $T > 1$ transfers more information than one-hot labels. The teacher's distribution at high temperature exposes similarities between classes (or tokens) — its "dark knowledge".

For LLM distillation, we match:

$$
\mathcal{L}_{\text{distill}} = \sum_t D_{\text{KL}}\!\left(P_T^{\text{teacher}}(t) \;\|\; P_T^{\text{student}}(t)\right) \cdot T^2
$$

The $T^2$ factor re-scales the gradient to have the same magnitude regardless of temperature (since softmax gradients scale as $1/T^2$ when divided by $T$). In practice, temperatures of 2–5 are common for token-level distillation.

!!! interview "Interview Corner"

    **Q:** Your LLM is generating repetitive, low-quality text even with `top_p=0.9` and `temperature=0.8`. What levers would you pull, in what order, and why?

    **A:** Start with diagnosis. Check if the repetition is in the prompt (prompt-induced looping) or model-induced (a distributional artifact). Then:

    1. **Add repetition penalty** ($\theta \approx 1.1$–$1.3$): this directly discounts already-seen tokens, breaking loops without changing the overall distribution shape.
    2. **Increase temperature slightly** (to 0.9–1.0): if the model is stuck in a high-probability mode, raising $T$ flattens the distribution and allows it to escape.
    3. **Switch from top-k to top-p** or **add min-p**: top-k with a small $k$ can be too restrictive, locking the model into a few tokens. Top-p adapts to the distribution shape.
    4. **Check for context length issues**: if the context is very long, the model may be anchored by repetition in the context itself. Truncating or re-formatting the prompt often helps.
    5. **Add frequency penalty** on top of repetition penalty to penalise *how often* tokens appear, not just *whether* they appeared.
    6. **Inspect the base model's training data**: some models have been trained on data with repetitive patterns; repetition at inference may be a training artifact that inference-time tuning cannot fully fix — fine-tuning on higher-quality data is the real solution.

## Putting It All Together: A Production Sampler

Here is a complete, self-contained sampler that combines all the techniques above:

```python
import torch
import torch.nn.functional as F
from collections import Counter
from typing import List, Optional


class ProductionSampler:
    """
    A complete, composable sampler implementing:
      - Temperature scaling
      - Repetition penalty (CTRL-style)
      - Frequency + presence penalties (OpenAI-style)
      - Top-k filtering
      - Top-p (nucleus) filtering
      - Min-p filtering
      - Greedy fallback

    Usage:
        sampler = ProductionSampler(temperature=0.8, top_p=0.9, rep_penalty=1.1)
        next_token = sampler(logits, context_ids)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 0,                  # 0 = disabled
        top_p: float = 1.0,              # 1.0 = disabled
        min_p: float = 0.0,              # 0.0 = disabled
        repetition_penalty: float = 1.0, # 1.0 = disabled
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        do_sample: bool = True,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.do_sample = do_sample

    def __call__(
        self,
        logits: torch.Tensor,            # (vocab_size,) raw logits
        context_ids: Optional[List[int]] = None,
    ) -> int:
        logits = logits.clone().float()

        # 1. Repetition penalty (multiplicative)
        if self.repetition_penalty != 1.0 and context_ids:
            for tid in set(context_ids):
                if logits[tid] > 0:
                    logits[tid] /= self.repetition_penalty
                else:
                    logits[tid] *= self.repetition_penalty

        # 2. Frequency + presence penalties (additive)
        if (self.frequency_penalty != 0.0 or self.presence_penalty != 0.0) and context_ids:
            counts = Counter(context_ids)
            for tid, cnt in counts.items():
                logits[tid] -= (
                    self.frequency_penalty * cnt + self.presence_penalty
                )

        # 3. Temperature scaling
        if self.temperature != 1.0:
            logits = logits / self.temperature

        if not self.do_sample:
            return int(logits.argmax())

        # 4. Top-k filter
        if self.top_k > 0:
            kth = torch.topk(logits, min(self.top_k, logits.size(-1))).values[-1]
            logits = logits.masked_fill(logits < kth, float('-inf'))

        # 5. Top-p (nucleus) filter
        if self.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            remove = cumprobs - probs > self.top_p
            sorted_logits[remove] = float('-inf')
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(0, sorted_idx, sorted_logits)

        # 6. Min-p filter
        if self.min_p > 0.0:
            probs = F.softmax(logits, dim=-1)
            threshold = self.min_p * probs.max()
            logits = logits.masked_fill(probs < threshold, float('-inf'))

        # 7. Safety: ensure at least one valid token
        if torch.all(logits == float('-inf')):
            # Fallback: revert to the temperature-scaled greedy token
            return int((logits.clone().fill_(1.0)).argmax())

        # 8. Sample
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1))


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    vocab_size = 32_000

    # Simulate a logit vector with a clear peak at token 7
    logits = torch.randn(vocab_size) * 2.0
    logits[7] = 10.0

    sampler = ProductionSampler(temperature=0.8, top_p=0.9, repetition_penalty=1.1)
    context = [1, 7, 42, 7, 7]        # token 7 has appeared 3 times

    counts = Counter()
    for _ in range(1000):
        t = sampler(logits.clone(), context)
        counts[t] += 1

    # With repetition penalty, token 7 should appear less than greedy would predict
    print(f"Token 7 selected {counts[7]} / 1000 times with rep_penalty=1.1")
    greedy_sampler = ProductionSampler(temperature=0.8, top_p=0.9, repetition_penalty=1.0)
    counts_nopenalty = Counter()
    for _ in range(1000):
        t = greedy_sampler(logits.clone(), context)
        counts_nopenalty[t] += 1
    print(f"Token 7 selected {counts_nopenalty[7]} / 1000 times without penalty")
```

!!! key "Key Takeaways"

    - Every decoding strategy converts raw logits to a token via a pipeline of *logit processors* (transforms) followed by a *sampler* (multinomial draw or argmax). Processors compose; their order matters.
    - **Greedy decoding** is deterministic but finds the locally optimal token, not the globally most probable sequence. It causes repetitive, low-entropy text.
    - **Temperature** is the master dial: dividing logits by $T < 1$ sharpens the distribution (less diverse, higher confidence); $T > 1$ flattens it (more diverse, potentially incoherent). Models are calibrated for $T \approx 1$.
    - **Top-k** cuts the long tail with a fixed count; **top-p** (nucleus) adapts to the distribution shape by cutting by cumulative mass — typically superior. **Min-p** cuts relative to the peak probability.
    - **Repetition penalty** discounts logits for previously seen tokens; **frequency penalty** discounts proportional to count. Both prevent loops; both interact with temperature (apply penalty before temperature scaling).
    - **Beam search** improves quality for structured tasks (translation, summarisation) but produces bland, repetitive text for open-ended generation. Use $B = 4$–$8$ with length normalisation.
    - **Contrastive decoding** and **DoLa** improve factuality by subtracting a less-knowledgeable model's (or layer's) probability distribution, amplifying the expert signal.
    - During **RL fine-tuning**, sampling temperature controls exploration. Compute KL divergences at the *same* temperature as the rollout to keep the objective consistent.
    - In **distillation**, high-temperature soft targets expose "dark knowledge" — the teacher's inter-token similarity structure — and should be scaled by $T^2$ to maintain gradient magnitude.

!!! sota "State of the Art & Resources (2026)"
    Sampling and decoding research has evolved from simple greedy/beam baselines into a rich family of adaptive, factuality-aware, and watermarking-capable methods. The current frontier focuses on dynamic truncation (min-p, typical sampling), contrastive layer-based decoding to reduce hallucinations, and inference-efficient alternatives that maintain or improve output quality.

    **Foundational work**

    - [Holtzman et al., *The Curious Case of Neural Text Degeneration* (2020)](https://arxiv.org/abs/1904.09751) — introduced nucleus (top-p) sampling and the degeneration analysis that defines the diversity–coherence tradeoff.
    - [Meister et al., *Locally Typical Sampling* (2022)](https://arxiv.org/abs/2202.00666) — information-theoretic framing of sampling; keep tokens whose surprisal is close to the conditional entropy.
    - [Li et al., *Contrastive Decoding: Open-ended Text Generation as Optimization* (ACL 2023)](https://arxiv.org/abs/2210.15097) — expert-minus-amateur log-prob scoring eliminates repetition without any additional training.

    **Recent advances (2023–2026)**

    - [Chuang et al., *DoLa: Decoding by Contrasting Layers Improves Factuality in LLMs* (ICLR 2024)](https://arxiv.org/abs/2309.03883) — single-model contrastive decoding using early vs. final layers to suppress hallucinations; +12–17 pp on TruthfulQA.
    - [Nguyen et al., *Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs* (ICLR 2025)](https://arxiv.org/abs/2407.01082) — dynamic threshold scaled to the peak probability; balances quality and creativity at high temperatures.
    - [Kirchenbauer et al., *A Watermark for Large Language Models* (ICML 2023)](https://arxiv.org/abs/2301.10226) — logit-level "green/red list" watermarking detectable from short spans without model access.
    - [Shi et al., *A Thorough Examination of Decoding Methods in the Era of LLMs* (EMNLP 2024)](https://arxiv.org/abs/2402.06925) — broad empirical comparison of greedy, beam, top-k, top-p, typical, and contrastive decoding across tasks and model sizes.

    **Open-source & tools**

    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — production inference engine; implements the full logit-processor pipeline (temperature, top-k, top-p, min-p, penalties) with PagedAttention and continuous batching.
    - [voidism/DoLa](https://github.com/voidism/DoLa) — official reference implementation of layer-contrastive decoding, compatible with LLaMA-family models.

    **Go deeper**

    - [Patrick von Platen, *How to generate text: using different decoding methods for language generation with Transformers* (HuggingFace Blog, 2020/2023)](https://huggingface.co/blog/how-to-generate) — hands-on walkthrough of greedy, beam search, top-k, and nucleus sampling with code.

## Further Reading

- Holtzman et al., "The Curious Case of Neural Text Degeneration" (ICLR 2020) — introduced nucleus (top-p) sampling and the analysis of degenerate modes.
- Meister et al., "Typical Decoding for Natural Language Generation" (ACL 2023) — information-theoretic motivation for typical sampling.
- Keskar et al., "CTRL: A Conditional Transformer Language Model for Controllable Generation" (2019) — introduced the repetition penalty.
- Li et al., "Contrastive Decoding: Open-ended Text Generation as Optimization" (ACL 2023) — expert vs. amateur contrastive scores.
- Chuang et al., "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models" (ICLR 2024) — layer-contrastive decoding for hallucination reduction.
- Vijayakumar et al., "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models" (AAAI 2018) — diversity-encouraging beam search.
- Kirchenbauer et al., "A Watermark for Large Language Models" (ICML 2023) — logit-based watermarking via vocabulary partitioning.
- Hinton et al., "Distilling the Knowledge in a Neural Network" (NeurIPS 2015 Workshop) — foundational work on temperature-scaled soft targets.
- HuggingFace `transformers` — `src/transformers/generation/logits_process.py` is the canonical reference implementation of every processor discussed here.
