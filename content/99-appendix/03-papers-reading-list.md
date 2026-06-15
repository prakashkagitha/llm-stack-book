#  Key Papers: An Annotated Reading List

This appendix is a curated, annotated bibliography organized by topic. Each entry names the paper, the authors (as commonly cited), the year, and two to three sentences that distill what it introduced and why it matters to a practitioner or interview candidate. The list is deliberately selective — it favors papers that changed how we build systems, not papers that merely refined a number. Read the entries that apply to your current work first, then use the cross-links to the relevant textbook chapters for the deeper mechanism treatment.

A short reading curriculum appears at the end of each section.

---

## Architecture Foundations

These papers define the vocabulary of modern LLMs. If you can explain every entry in this section from first principles, you are ready to answer the architecture questions at any ML interview.

### The Transformer

**Vaswani et al., "Attention Is All You Need," 2017.**
Introduced the Transformer: a sequence-to-sequence model built entirely from multi-head self-attention and position-wise feed-forward layers, with no recurrence. Showed that dispensing with RNNs entirely yielded better translation quality at lower training cost — and, crucially, exposed a parallelism structure that maps perfectly onto GPU matrix operations. This is the paper every LLM engineer must have read in full.

**Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," 2018.**
Demonstrated that a large encoder-only Transformer pretrained with masked language modeling (MLM) and next-sentence prediction could be fine-tuned to achieve state-of-the-art results on eleven NLP tasks with minimal task-specific architecture changes. BERT crystallized the pretraining–fine-tuning paradigm that still dominates applied NLP.

**Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2), 2019.**
Showed that a large decoder-only autoregressive Transformer trained on diverse web text (WebText) can perform a wide variety of tasks — translation, question answering, summarization — in a zero-shot manner, simply by formatting the task as text completion. The paper's central claim — that multitask learning emerges from language modeling at scale — is the conceptual foundation of the GPT line.

**Brown et al., "Language Models are Few-Shot Learners" (GPT-3), 2020.**
Scaled the decoder-only Transformer to 175 billion parameters and demonstrated that few-shot prompting (providing examples in the context window) could match fine-tuned smaller models on many benchmarks. Introduced in-context learning (ICL) as a practical paradigm and triggered the modern era of large language models.

See [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html) and [Building a GPT From Scratch (nanoGPT-style)](../02-transformer/07-build-gpt-from-scratch.html) for hands-on implementations of these ideas.

### Positional Encoding

**Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (RoPE), 2021.**
Proposed encoding position as a rotation in the complex plane applied to query and key vectors, rather than adding a fixed bias. Because the rotation acts multiplicatively on the attention dot product, relative position information flows naturally through the attention score. RoPE is now the default positional scheme in LLaMA, Mistral, Gemma, and most open-source models.

**Press, Smith & Lewis, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (ALiBi), 2021.**
Replaced positional embeddings with a linear bias subtracted from attention logits as a function of token distance. This allows a model trained on short sequences to extrapolate to longer ones at inference time. ALiBi is simple to implement (two lines of code) and was used in MPT and BLOOM.

See [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html) for the full mathematical treatment.

### Attention Variants

**Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (MQA), 2019.**
Proposed multi-query attention (MQA): keys and values share a single head while queries remain multi-head. This shrinks the KV cache by a factor of $h$ (number of heads) with minimal quality loss, dramatically accelerating auto-regressive decoding in memory-bound regimes.

**Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," 2023.**
Generalized MQA to grouped-query attention (GQA): $g$ groups of query heads share one KV head each, so the cache shrinks by $h/g$ instead of $h$. GQA provides a smooth quality-vs-memory tradeoff and is used by LLaMA 2/3, Mistral, and Gemma.

See [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html) for the derivation and implementation.

### Mixture-of-Experts

**Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," 2017.**
Introduced conditional computation at scale: a learned top-$k$ router selects which expert sub-networks process each token, letting the total parameter count grow without proportionally increasing FLOPs per token. This idea directly enabled Switch Transformer, Mixtral, and the rumored architecture of GPT-4.

**Fedus, Zoph & Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," 2021.**
Simplified MoE routing to top-1 (a single expert per token) and showed it can scale to trillion-parameter models while using the same compute as a dense baseline. Introduced auxiliary load-balancing losses that prevent router collapse and remain standard today.

See [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html) for the routing algebra and load-balancing derivation.

### Alternative Sequence Models

**Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," 2023.**
Introduced a selective state-space model (SSM) in which the state-transition matrices are input-dependent rather than fixed — allowing the model to selectively remember or forget content. Mamba achieves linear-time inference (no $O(L^2)$ attention) while matching Transformer perplexity at the billion-parameter scale, with significantly faster throughput on long sequences.

See [Beyond Attention: SSMs, Mamba, RWKV & Linear Attention](../02-transformer/11-ssm-and-alternatives.html).

---

## Pretraining & Scaling

### Scaling Laws

**Kaplan et al., "Scaling Laws for Neural Language Models," 2020.**
Empirically established power-law relationships between model performance (test loss) and model size ($N$), dataset size ($D$), and compute ($C$). The key finding: loss scales as $L \propto N^{-\alpha}$ and $L \propto D^{-\beta}$ with roughly constant exponents, suggesting that bigger models are always better if given proportionally more data.

**Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla), 2022.**
Re-ran the Kaplan scaling analysis more carefully and found that the optimal allocation of a fixed compute budget $C = 6ND$ is approximately equal numbers of parameters $N$ and tokens $D$: for every doubling of model size, you should double the training tokens. This overturned the common practice of under-training large models, directly influencing LLaMA's training recipe.

!!! example "Worked example: Chinchilla optimal sizing"
    Suppose you have a compute budget of $C = 10^{23}$ FLOPs (roughly what it costs to train a 70 B parameter model on 1 T tokens using the $6ND$ approximation).

    Under Chinchilla's equal-allocation rule: $N_{\text{opt}} \approx \sqrt{C / 6}$.

    $$
    N_{\text{opt}} = \sqrt{\frac{10^{23}}{6}} \approx \sqrt{1.67 \times 10^{22}} \approx 4.1 \times 10^{11} \approx 410\text{B parameters}
    $$

    Correspondingly $D_{\text{opt}} \approx C / (6 N_{\text{opt}}) \approx 10^{23} / (6 \times 4.1\times10^{11}) \approx 4\times10^{10}$ tokens (40 B tokens).

    In practice, inference cost makes smaller models more economical to deploy, so practitioners often train a smaller model (e.g., 7 B) on far more tokens than Chinchilla-optimal (e.g., 1–2 T tokens), accepting a small loss penalty for major inference savings. LLaMA 2 7B was trained for about 2 T tokens — roughly 30$\times$ more than the compute-optimal point.

See [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html) for the full curve-fitting methodology.

### Pretraining at Scale

**Touvron et al., "LLaMA: Open and Efficient Foundation Language Models," 2023.**
Showed that a 7 B–65 B parameter model trained on 1 T–1.4 T tokens of publicly available data can outperform GPT-3 (175 B) on many benchmarks, demonstrating that data quality and training duration matter more than raw parameter count. LLaMA bootstrapped the open-source LLM ecosystem.

**Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," 2023.**
Extended LLaMA to 70 B parameters with GQA, trained on 2 T tokens, and released with both the base model and a supervised + RLHF chat variant. Llama 2 was the reference open model for most of 2023–2024 and remains a common fine-tuning base.

**Jiang et al., "Mistral 7B," 2023.**
Introduced sliding-window attention (SWA) and GQA in a 7 B model that outperforms LLaMA-2 13 B on most benchmarks at half the parameter count, emphasizing that architecture efficiency improvements compound with scale.

### Distributed Training Methods

**Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," 2019.**
Introduced the ZeRO (Zero Redundancy Optimizer) family of techniques: partitioning optimizer states (ZeRO-1), gradients (ZeRO-2), and parameters (ZeRO-3) across data-parallel ranks rather than replicating them. ZeRO-3 reduces memory per GPU by a factor of $N_{\text{dp}}$ (data-parallel world size), enabling training of models 8$\times$ larger on the same hardware.

**Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," 2019.**
Showed how to partition Transformer layers across multiple GPUs with minimal communication by splitting the MLP and attention projections column/row-wise — so the all-reduce happens only once per layer. Megatron-LM tensor parallelism is still the dominant technique for very large models.

See [Distributed Training I](../03-pretraining/05-distributed-data-parallel.html) and [Distributed Training II](../03-pretraining/06-distributed-model-parallel.html) for implementation details.

---

## Efficiency & Kernels

### FlashAttention

**Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," 2022.**
Reformulated the attention computation to avoid materializing the full $L \times L$ attention matrix in HBM by tiling the softmax over blocks and using the online softmax trick. The result is $O(L)$ HBM reads/writes instead of $O(L^2)$, making attention IO-bound for long sequences rather than compute-bound — and reducing peak memory from $O(L^2)$ to $O(L)$.

**Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," 2023.**
Improved the CUDA kernel by splitting work across thread blocks more efficiently (reducing non-matmul FLOPs by 2$\times$), exploiting warp-level parallelism for the reduction step, and using asynchronous data loading. FlashAttention-2 is the kernel used in most production training and serving stacks.

See [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html) and [FlashAttention 2 & 3](../04-kernels-efficiency/03-flash-attention-2-3.html).

### Quantization

**Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," 2022.**
Applied second-order weight quantization (using the approximate Hessian from calibration data) to compress large language models to INT4 with minimal perplexity loss — enabling 175 B parameter models to be run on a single A100. GPTQ is the most widely used PTQ method for deployment.

**Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," 2023.**
Observed that a small fraction of weight channels (those corresponding to large-activation channels) are disproportionately important. AWQ scales those channels before quantizing, protecting them from rounding error without needing to store mixed-precision weights, and achieves better accuracy than GPTQ at the same bit-width.

See [Quantization I: Post-Training Quantization](../04-kernels-efficiency/07-quantization-ptq.html) for the mathematical framework.

### Inference Systems

**Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," 2023.**
Borrowed the concept of virtual memory paging to manage the KV cache: keys and values are stored in non-contiguous physical "pages" that are looked up through a block table, eliminating internal fragmentation and enabling fine-grained memory sharing across requests. PagedAttention is the core innovation of vLLM and is now standard in all major serving frameworks.

**Leviathan, Kalman & Matias, "Fast Inference from Transformers via Speculative Decoding," 2022.**
Proposed running a small draft model to generate $\gamma$ tokens in parallel, then verifying them all with the target model in a single forward pass. If all tokens are accepted, you get $\gamma$ tokens for the cost of one target-model step; rejected tokens trigger a correction. Speculative decoding typically achieves 2–3$\times$ wall-clock speedup with zero quality degradation.

See [Speculative Decoding](../07-inference-serving/06-speculative-decoding.html) and [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html).

---

## Post-Training, Fine-Tuning & Alignment

### Supervised Fine-Tuning & Instruction Tuning

**Wei et al., "Finetuned Language Models are Zero-Shot Learners" (FLAN), 2021.**
Showed that fine-tuning a pretrained LM on a diverse collection of tasks formatted as natural language instructions dramatically improves zero-shot generalization to held-out tasks. FLAN established instruction tuning as the standard first step of post-training.

**Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT / RLHF), 2022.**
Described the three-stage pipeline — supervised fine-tuning (SFT) on human demonstrations, reward model (RM) training on human preference pairs, and proximal policy optimization (PPO) to optimize the policy against the RM — that turned GPT-3 into InstructGPT. This paper, more than any other, defines the modern alignment stack.

See [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html) and [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html).

### Parameter-Efficient Fine-Tuning (PEFT)

**Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021.**
Proposed freezing the pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and learning a low-rank perturbation $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$. At merge time $W = W_0 + BA$ and inference cost is identical. LoRA reduces trainable parameters by orders of magnitude while preserving most of the fine-tuning quality.

$$
h = W_0 x + \Delta W x = W_0 x + B A x, \quad r \ll \min(d, k)
$$

**Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," 2023.**
Combined 4-bit NormalFloat (NF4) quantization of the frozen base model with LoRA adapters trained in bfloat16, plus double quantization of the quantization constants. This enabled fine-tuning of a 65 B parameter model on a single 48 GB GPU with quality competitive with full fine-tuning.

See [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html).

### Preference Learning & Alignment

**Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," 2023.**
Showed that the standard RLHF objective has a closed-form optimal policy expressible in terms of the reference LM, allowing the reward to be re-parameterized in terms of policy log-probabilities. This yields a single cross-entropy loss on preference pairs that matches PPO-based RLHF without a separately trained reward model or RL loop.

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

**Bai et al., "Constitutional AI: Harmlessness from AI Feedback," 2022.**
Introduced Constitutional AI (CAI): the model critiques and revises its own outputs according to a set of principles ("the constitution"), then RLHF is run with a reward model trained on the AI-generated comparisons rather than human labels (RLAIF). This greatly reduced the human annotation burden while maintaining safety properties.

See [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html) and [Constitutional AI, RLAIF & Self-Improvement](../05-posttraining-alignment/11-constitutional-rlaif.html).

---

## Reasoning & Chain-of-Thought

**Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," 2022.**
Showed that including step-by-step reasoning exemplars in the few-shot prompt (chain-of-thought prompting) dramatically improves performance on multi-step arithmetic, symbolic, and commonsense reasoning tasks — and that this behavior emerges only at large model scales. CoT is now a near-universal component of reasoning-capable prompting strategies.

**Kojima et al., "Large Language Models are Zero-Shot Reasoners," 2022.**
Demonstrated that simply appending "Let's think step by step" to the query (zero-shot CoT) elicits chain-of-thought reasoning without any exemplars, suggesting the reasoning capability is latent in sufficiently large pretrained models.

**Lightman et al., "Let's Verify Step by Step," 2023.**
Compared process reward models (PRMs) — which assign a reward to each intermediate reasoning step — with outcome reward models (ORMs) that reward only the final answer. PRMs trained on human step-level annotations substantially outperform ORMs on the MATH dataset and form the basis for later reinforcement learning approaches to improving reasoning.

**DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," 2025.**
Showed that applying Group Relative Policy Optimization (GRPO) with verifiable rewards (correct final answer) directly to a pretrained model can bootstrap chain-of-thought reasoning from scratch, producing long "thinking" traces that improve accuracy on math and coding benchmarks. DeepSeek-R1 demonstrated that RL with sparse outcome rewards is a viable path to state-of-the-art reasoning at lower cost than distillation from OpenAI o1.

See [Reasoning, Chain-of-Thought & Test-Time Compute](../05-posttraining-alignment/10-reasoning-test-time-compute.html) and [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html).

---

## RL for Language Models

!!! interview "Interview Corner"
    **Q:** What is the key mathematical difference between PPO and DPO for language model alignment, and when would you prefer each?

    **A:** PPO optimizes a surrogate objective using a separately-trained reward model and a KL penalty: it samples new rollouts each iteration, computes the advantage under the current reward model, and clips the importance-weight ratio to prevent large policy updates. DPO sidesteps the explicit reward model by re-parameterizing the Bradley-Terry reward in terms of log-probability ratios between the policy and a reference model, yielding a direct cross-entropy loss on preference pairs. Prefer PPO (or its variants like GRPO) when your reward signal is verifiable or when you need to run many RL iterations with evolving rollouts; prefer DPO when you have a fixed offline preference dataset and want simplicity — DPO is a single-pass fine-tune with no actor-critic interaction and much lower infrastructure cost.

**Schulman et al., "Proximal Policy Optimization Algorithms" (PPO), 2017.**
Introduced the clipped surrogate objective — a simple, stable alternative to TRPO that prevents excessively large policy updates by clipping the importance-weight ratio $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_\text{old}}(a_t|s_t)$:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]
$$

PPO is the default RL algorithm for RLHF pipelines due to its stability and ease of implementation.

**Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," 2024 (GRPO).**
Introduced Group Relative Policy Optimization (GRPO): instead of a critic network, sample $G$ completions per prompt and normalize advantages by the group mean and standard deviation. This eliminates the value-function baseline while keeping the variance-reduction benefit, reducing GPU memory usage by removing the critic entirely.

$$
\hat{A}_{i} = \frac{r_i - \mu_G}{\sigma_G}, \quad \text{where } \mu_G = \frac{1}{G}\sum_{j=1}^G r_j,\; \sigma_G = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \mu_G)^2}
$$

See [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html) and [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html) for full implementation details.

```python
# Minimal GRPO advantage computation (from scratch, educational)
import torch

def grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    Compute group-normalized advantages for a batch of completions.

    Args:
        rewards: Tensor of shape [G] — one scalar reward per completion
                 in the group for a single prompt.
    Returns:
        advantages: Tensor of shape [G] — zero-mean, unit-variance advantages.
    """
    # Group mean and std across the G completions for this prompt
    mu = rewards.mean()          # scalar
    sigma = rewards.std() + 1e-8 # small epsilon for numerical stability

    # Normalized advantage: each completion is compared to the group average
    # Positive means "this completion did better than the group median"
    advantages = (rewards - mu) / sigma  # shape [G]
    return advantages


# Example: 8 completions for one math problem
# Reward = 1.0 if final answer is correct, 0.0 otherwise
rewards = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
advantages = grpo_advantages(rewards)
# rewards: [1, 0, 1, 1, 0, 0, 1, 0]  → 4 correct out of 8
# mu = 0.5, sigma ≈ 0.535
# advantages ≈ [0.94, -0.94, 0.94, 0.94, -0.94, -0.94, 0.94, -0.94]
print(advantages)
# The four correct completions get positive advantage ~+0.94
# The four incorrect completions get negative advantage ~-0.94
# This signal trains the policy to increase probability of correct completions
```

**Stiennon et al., "Learning to Summarize from Human Feedback," 2020.**
The first large-scale demonstration that RLHF on a language task (TL;DR summarization) can substantially outperform supervised learning alone, and one of the first papers to identify reward hacking as a serious failure mode. Many of the techniques in InstructGPT trace directly to this paper.

**Bai et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback," 2022 (Anthropic).**
Described scaling RLHF to a helpful and harmless dialogue assistant and introduced the concept of the "preference model," empirically characterizing how model quality, reward model quality, and KL penalty interact. This is the technical companion to the Constitutional AI paper.

---

## Inference & Agents

### Inference Optimization

**Pope et al., "Efficiently Scaling Transformer Inference," 2022.**
Analyzed the arithmetic intensity of every transformer operation during inference and derived optimal partitioning strategies for multi-chip inference. Introduced the key insight that decoding is almost always memory-bandwidth-bound, not compute-bound, motivating tensor parallelism for the KV cache.

**Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills," 2023.**
Showed that interleaving "chunked prefill" tokens with decode tokens in the same batch eliminates the "stall" that conventional continuous batching causes when a long prefill monopolizes the GPU. Chunked prefill is now standard in vLLM, TensorRT-LLM, and SGLang.

See [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html).

### Retrieval-Augmented Generation

**Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG), 2020.**
Introduced the now-standard architecture: a frozen retriever (dense passage retriever based on BERT bi-encoders) retrieves relevant documents given the query, which are concatenated with the query and passed to a seq2seq generator. Showed that this approach outperforms purely parametric language models on knowledge-intensive tasks while reducing the need to memorize facts in weights.

See [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html).

### Agents & Tool Use

**Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," 2022.**
Proposed interleaving reasoning traces ("thought") with task-relevant actions ("act") in a unified prompt, allowing the model to dynamically plan and adjust its actions based on intermediate observations. ReAct is the foundational paper for tool-using language model agents.

**Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools," 2023.**
Showed that a language model can learn to insert API calls into its own text completions by bootstrapping from a small set of hand-written examples and filtering self-generated training data. Toolformer produced a model that can call calculators, search engines, and calendars in a zero-shot manner.

See [Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html) and [The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html).

---

## Multimodal Models

**Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), 2021.**
Trained a vision encoder and text encoder jointly by contrastive learning on 400 million image-caption pairs: positive pairs (image, matching caption) are pulled together in embedding space, negative pairs are pushed apart. CLIP's image encoder is used as the visual backbone in nearly every open-source VLM.

**Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT), 2020.**
Applied the standard Transformer directly to images by splitting an image into $16 \times 16$ pixel patches, treating each patch as a "token" with a learned linear embedding. ViT demonstrated that with sufficient data, pure attention architectures match or exceed convolutional networks on image classification.

**Liu et al., "Visual Instruction Tuning" (LLaVA), 2023.**
Connected a CLIP vision encoder to a Vicuna (LLaMA-based) language model via a simple linear projection layer and fine-tuned on GPT-4-generated multimodal instruction-following data. LLaVA showed that a lightweight vision-language interface suffices for impressive visual question answering and was the starting point for most open-source VLMs.

See [Vision Transformers & Image Encoders](../10-multimodal-and-arch/01-vision-transformers.html) and [Vision-Language Models](../10-multimodal-and-arch/02-vision-language-models.html).

---

## Safety, Evaluation & Production

**Perez et al., "Red Teaming Language Models with Language Models," 2022.**
Automated red-teaming by using one language model to generate adversarial test cases that elicit harmful behavior from another. Showed that LM-generated red-team prompts find failure modes that manual red-teamers miss and scale more cheaply.

**Liang et al., "Holistic Evaluation of Language Models" (HELM), 2022.**
Proposed a structured evaluation framework covering seven metrics (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency) across 42 scenarios. HELM was one of the first systematic attempts to move beyond single-number benchmark leaderboards.

**Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," 2023.**
Introduced MT-Bench (multi-turn dialogue benchmark scored by GPT-4) and Chatbot Arena (crowdsourced pairwise human preference) as complementary evaluation tools. Established LLM-as-a-judge as a scalable alternative to human annotation, including careful analysis of its biases (position, verbosity, self-preference).

See [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html) and [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html).

---

## A Curated Reading Curriculum

Use this six-week plan if you are starting from scratch or preparing for a technical interview. Papers are ordered by dependency: reading them in sequence lets each paper build on the last.

```text
WEEK 1 — Architecture foundations
  Day 1: Vaswani et al. (2017) — "Attention Is All You Need"
  Day 2: Devlin et al. (2018) — BERT
  Day 3: Brown et al. (2020) — GPT-3
  Day 4: Su et al. (2021) — RoPE
  Day 5: Shazeer (2019) — MQA; Ainslie et al. (2023) — GQA

WEEK 2 — Scaling & pretraining
  Day 1: Kaplan et al. (2020) — Scaling Laws
  Day 2: Hoffmann et al. (2022) — Chinchilla
  Day 3: Touvron et al. (2023) — LLaMA
  Day 4: Rajbhandari et al. (2019) — ZeRO
  Day 5: Shoeybi et al. (2019) — Megatron-LM

WEEK 3 — Efficiency
  Day 1: Dao et al. (2022) — FlashAttention
  Day 2: Dao (2023) — FlashAttention-2
  Day 3: Frantar et al. (2022) — GPTQ
  Day 4: Kwon et al. (2023) — PagedAttention / vLLM
  Day 5: Leviathan et al. (2022) — Speculative Decoding

WEEK 4 — Post-training & alignment
  Day 1: Ouyang et al. (2022) — InstructGPT / RLHF
  Day 2: Hu et al. (2021) — LoRA
  Day 3: Dettmers et al. (2023) — QLoRA
  Day 4: Rafailov et al. (2023) — DPO
  Day 5: Schulman et al. (2017) — PPO

WEEK 5 — Reasoning & RL
  Day 1: Wei et al. (2022) — Chain-of-Thought
  Day 2: Lightman et al. (2023) — Process Reward Models
  Day 3: Shao et al. (2024) — GRPO / DeepSeekMath
  Day 4: DeepSeek-AI (2025) — DeepSeek-R1
  Day 5: Stiennon et al. (2020) — Learning to Summarize from Human Feedback

WEEK 6 — Agents, multimodal, evaluation
  Day 1: Lewis et al. (2020) — RAG
  Day 2: Yao et al. (2022) — ReAct
  Day 3: Radford et al. (2021) — CLIP
  Day 4: Dosovitskiy et al. (2020) — ViT
  Day 5: Zheng et al. (2023) — MT-Bench / Chatbot Arena
```

---

## Summary Reference Table

The table below lists every annotated paper with its section for quick lookup.

| Paper | Year | Topic | Section |
|---|---|---|---|
| Vaswani et al., "Attention Is All You Need" | 2017 | Architecture | Transformer |
| Devlin et al., BERT | 2018 | Architecture | Encoder pretraining |
| Radford et al., GPT-2 | 2019 | Architecture | Decoder-only LM |
| Brown et al., GPT-3 | 2020 | Architecture | Few-shot / ICL |
| Su et al., RoPE | 2021 | Architecture | Positional encoding |
| Press et al., ALiBi | 2021 | Architecture | Positional encoding |
| Shazeer, MQA | 2019 | Architecture | Attention variant |
| Ainslie et al., GQA | 2023 | Architecture | Attention variant |
| Shazeer et al., MoE | 2017 | Architecture | Expert routing |
| Fedus et al., Switch Transformer | 2021 | Architecture | MoE |
| Gu & Dao, Mamba | 2023 | Architecture | SSM |
| Kaplan et al., Scaling Laws | 2020 | Pretraining | Scaling |
| Hoffmann et al., Chinchilla | 2022 | Pretraining | Scaling |
| Touvron et al., LLaMA | 2023 | Pretraining | Open model |
| Rajbhandari et al., ZeRO | 2019 | Pretraining | Distributed training |
| Shoeybi et al., Megatron-LM | 2019 | Pretraining | Distributed training |
| Dao et al., FlashAttention | 2022 | Efficiency | Kernel |
| Dao, FlashAttention-2 | 2023 | Efficiency | Kernel |
| Frantar et al., GPTQ | 2022 | Efficiency | Quantization |
| Lin et al., AWQ | 2023 | Efficiency | Quantization |
| Kwon et al., PagedAttention | 2023 | Inference | Serving |
| Leviathan et al., Speculative Decoding | 2022 | Inference | Serving |
| Wei et al., FLAN | 2021 | Post-training | Instruction tuning |
| Ouyang et al., InstructGPT | 2022 | Post-training | RLHF |
| Hu et al., LoRA | 2021 | Post-training | PEFT |
| Dettmers et al., QLoRA | 2023 | Post-training | PEFT |
| Rafailov et al., DPO | 2023 | Post-training | Preference learning |
| Bai et al., Constitutional AI | 2022 | Post-training | RLAIF |
| Schulman et al., PPO | 2017 | RL | Policy optimization |
| Shao et al., GRPO | 2024 | RL | Policy optimization |
| Stiennon et al., Summarization RLHF | 2020 | RL | Alignment |
| Wei et al., Chain-of-Thought | 2022 | Reasoning | Prompting |
| Lightman et al., Process RM | 2023 | Reasoning | Reward modeling |
| DeepSeek-AI, DeepSeek-R1 | 2025 | Reasoning | RL for reasoning |
| Lewis et al., RAG | 2020 | Retrieval | RAG |
| Yao et al., ReAct | 2022 | Agents | Tool use |
| Schick et al., Toolformer | 2023 | Agents | Tool use |
| Radford et al., CLIP | 2021 | Multimodal | Vision-language |
| Dosovitskiy et al., ViT | 2020 | Multimodal | Vision |
| Liu et al., LLaVA | 2023 | Multimodal | VLM |
| Perez et al., Red Teaming | 2022 | Safety | Evaluation |
| Liang et al., HELM | 2022 | Evaluation | Benchmarks |
| Zheng et al., MT-Bench | 2023 | Evaluation | LLM-as-judge |

---

!!! key "Key Takeaways"
    - The Transformer (Vaswani 2017), GPT-3 (Brown 2020), and InstructGPT (Ouyang 2022) form the three architectural, scaling, and alignment pillars that define modern LLMs; read each in full.
    - Chinchilla overturned the Kaplan scaling law's parameter-heavy recommendation: for a fixed compute budget, match parameter count to training token count; in practice, over-train smaller models for inference efficiency.
    - FlashAttention made long-context training economically viable by cutting attention's HBM traffic from $O(L^2)$ to $O(L)$; it is now a non-optional kernel dependency in every production training stack.
    - LoRA and QLoRA made fine-tuning accessible without full-parameter updates: a rank-$r$ decomposition of $\Delta W$ reduces trainable parameters to $r(d+k)$ from $dk$, often with $r \in \{8, 16, 64\}$.
    - DPO simplified RLHF to a single cross-entropy loss over preference pairs; PPO and GRPO remain preferable when rewards are verifiable and you need to run many RL iterations.
    - PagedAttention (vLLM) solved KV-cache memory fragmentation at serving time, enabling high-throughput continuous batching with minimal wasted memory.
    - Chain-of-thought prompting (Wei 2022) and process reward models (Lightman 2023) are the two enabling technologies for the reasoning wave; DeepSeek-R1 showed that RL with sparse outcome rewards can bootstrap long thinking traces from scratch.
    - RAG (Lewis 2020) and ReAct (Yao 2022) are the canonical papers for retrieval-augmented generation and tool-using agents, respectively; read them as a pair.
    - Use the six-week curriculum as a structured on-ramp: architecture weeks 1–2, efficiency week 3, alignment weeks 4–5, agents and evaluation week 6.

---

## Further Reading

The entries below go one level deeper than the annotated list above. They are either highly-cited follow-ons to canonical papers, or important technical reports that any serious practitioner should know.

- **Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," 2023** — the full technical report with ablations on data mixing, RLHF reward modeling, and context length extension.
- **Jiang et al., "Mixtral of Experts," 2024** — sparse MoE extension of Mistral with detailed routing statistics and throughput analysis.
- **Anthropic, "Claude's Character" and "Model Card" technical reports** — the most transparent public documentation of a frontier model's post-training process.
- **Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality," 2024 (Mamba-2)** — shows that selective SSMs and attention are both special cases of a unified semiseparable-matrix framework, with an efficient chunk-parallel algorithm.
- **Dubey et al., "The Llama 3 Herd of Models," 2024** — the most comprehensive public account of a frontier pretraining + post-training pipeline, including data curation, scaling experiments, and multimodal extension.
- **Clark et al., "Unified Scaling Laws for Routed Language Models," 2022** — extends Kaplan-style scaling to MoE models, characterizing how expert count and granularity trade off against dense-model equivalents.
- **Zhong et al., "Evaluation Harness" (EleutherAI lm-evaluation-harness)** — the de-facto open-source evaluation framework for zero/few-shot benchmarking; understanding its implementation is essential for rigorous model evaluation.
- **Karpathy, "nanoGPT" (GitHub repo)** — the most-studied minimal GPT implementation; reading and re-writing it from scratch is the single highest-value exercise for learning the Transformer.
