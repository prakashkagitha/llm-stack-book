#  Glossary of Terms

This glossary covers every major concept across the LLM stack — from linear algebra primitives to production serving tricks. Each entry gives a concise one-to-three sentence definition and a cross-link to the chapter that covers the topic in depth. Entries are grouped by theme, then alphabetized within each group, so you can browse by domain or use Ctrl-F to jump to a specific term.

---

## A — Architectural Concepts

**Activation Function**
A nonlinear function applied element-wise after a linear layer; without it, the entire network collapses to a single matrix multiply. Common choices include ReLU, GELU, SiLU (Swish), and GLU variants. See [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html).

**Adapter**
A small bottleneck module inserted into a frozen pretrained model; typically two linear layers with a low-rank hidden dimension. Only adapter parameters are trained, leaving the base model weights unchanged. See [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html).

**ALiBi (Attention with Linear Biases)**
A positional encoding scheme that adds a linear penalty to attention logits based on relative token distance, with no learned parameters. It extrapolates more gracefully than sinusoidal encodings to sequence lengths longer than those seen during training. See [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html).

**Attention Mechanism**
The core operation of the Transformer: each token queries all other tokens to produce a weighted sum of their values. The weight is the softmax-normalized dot product of query and key vectors, scaled by $\frac{1}{\sqrt{d_k}}$ to prevent exploding gradients. See [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html).

**Autoregressive Decoding**
A generation strategy where the model produces one token at a time, each conditioned on all previous tokens. The distribution over the next token is $p(x_t \mid x_1, \ldots, x_{t-1})$ and the full sequence probability is their product. See [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html).

**BPE (Byte-Pair Encoding)**
A data-compression algorithm adapted for tokenization: repeatedly merge the most frequent adjacent pair of tokens until a target vocabulary size is reached. GPT-2 and GPT-4 use BPE on raw bytes. See [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html).

**Causal Mask**
A lower-triangular Boolean mask applied to the attention score matrix to prevent each token from attending to future positions. Essential for autoregressive language models during both training and prefill. See [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html).

**Context Length (Context Window)**
The maximum number of tokens a model can process in one forward pass. Determined architecturally by the positional encoding scheme and, in practice, by KV-cache memory. Modern models support 128 K – 1 M tokens. See [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html).

**Cross-Attention**
Attention where queries come from one sequence (e.g., the decoder) and keys/values come from a different sequence (e.g., the encoder output). The core mechanism in encoder-decoder architectures like T5 and the original Transformer. See [Architecture Variants: Encoder-Decoder, Decoder-Only & Prefix-LM](../02-transformer/08-architecture-variants.html).

**Decoder-Only Model**
A Transformer that uses only the masked self-attention stack and generates tokens autoregressively; no encoder. GPT, LLaMA, Gemini, and Claude are all decoder-only. See [Architecture Variants: Encoder-Decoder, Decoder-Only & Prefix-LM](../02-transformer/08-architecture-variants.html).

**Embedding Layer**
A lookup table that maps integer token IDs to dense real-valued vectors of dimension $d_{model}$. For a vocabulary of 32 K tokens and $d_{model} = 4096$, this table has on the order of 130 M parameters. See [Embeddings & The Input Pipeline](../02-transformer/02-embeddings-input.html).

**Encoder-Decoder Model**
A Transformer with a bidirectional encoder (full self-attention) and an autoregressive decoder (masked self-attention + cross-attention). T5, BART, and the original "Attention Is All You Need" architecture follow this design. See [Architecture Variants: Encoder-Decoder, Decoder-Only & Prefix-LM](../02-transformer/08-architecture-variants.html).

**Feed-Forward Network (FFN)**
The two-layer MLP in each Transformer block, applied independently to each token position. Its hidden dimension is typically $4\times d_{model}$; in gated variants (SwiGLU, GeGLU) it may be $\frac{8}{3} d_{model}$ to keep parameter counts constant. See [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html).

**GELU (Gaussian Error Linear Unit)**
An activation function approximated as $x \cdot \Phi(x)$ where $\Phi$ is the Gaussian CDF. Empirically outperforms ReLU for language model pretraining; used in BERT, GPT-2, and many successors. See [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html).

**GQA (Grouped-Query Attention)**
An attention variant where $h$ query heads share $g$ key-value heads ($g < h$), reducing the KV-cache footprint by a factor of $h/g$ relative to multi-head attention while losing little quality. LLaMA 2 70B and Mistral 7B use GQA. See [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).

**KV Cache**
The per-layer, per-token storage of key and value matrices from previously processed tokens. Eliminates recomputing them during autoregressive decoding; grows linearly with both sequence length and batch size. See [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html).

**Layer Normalization (LayerNorm)**
Normalizes activations across the feature dimension for each individual example, computed as $\hat{x} = (x - \mu)/\sigma$ then rescaled by learned $\gamma, \beta$. Unlike BatchNorm it does not depend on batch size, making it stable for variable-length sequences. See [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html).

**Mamba (Selective State Space Model)**
A sequence model based on structured state space models (S4 and its variants) with input-dependent state transitions, achieving linear-time sequence processing without attention. A strong contender for replacing or supplementing Transformers in long-context tasks. See [Beyond Attention: SSMs, Mamba, RWKV & Linear Attention](../02-transformer/11-ssm-and-alternatives.html).

**MHA (Multi-Head Attention)**
The standard attention module: queries, keys, and values are projected into $h$ independent "heads," attention is computed in parallel for each, and outputs are concatenated and re-projected. Allows each head to specialize in different relational patterns. See [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).

**MLA (Multi-head Latent Attention)**
A DeepSeek innovation that compresses keys and values into a low-rank latent vector before storing them, dramatically reducing the KV-cache memory footprint compared with GQA. See [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).

**MoE (Mixture of Experts)**
An architecture where each Transformer block contains $N$ expert FFN sub-networks, and a learned router selects $k$ of them per token ($k \ll N$). Scales total parameters without proportionally increasing compute per token. See [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html).

**MQA (Multi-Query Attention)**
An extreme form of GQA where all query heads share a single key-value head, minimising KV-cache memory at the cost of some representational capacity. Used in PaLM and early Falcon variants. See [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).

**Prefix-LM**
A Transformer architecture that applies full (bidirectional) attention over a prefix (the prompt) and causal attention over the generated suffix, combining the benefits of encoder (rich prompt representation) and decoder (autoregressive generation) styles. See [Architecture Variants: Encoder-Decoder, Decoder-Only & Prefix-LM](../02-transformer/08-architecture-variants.html).

**ReLU (Rectified Linear Unit)**
The activation $\text{ReLU}(x) = \max(0, x)$; cheap to compute and to differentiate. Largely replaced in LLMs by smoother variants (GELU, SiLU), but still common in vision models and early language models. See [Neural Networks From Scratch: MLPs & Backprop](../01-foundations/06-neural-nets-from-scratch.html).

**Residual Connection (Skip Connection)**
An additive bypass around each sub-layer: $\text{output} = f(x) + x$. Enables gradient flow through deep networks and is critical for training transformers beyond a handful of layers. See [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html).

**RMSNorm (Root-Mean-Square Normalization)**
A simplified variant of LayerNorm that normalizes by the RMS of activations and omits the mean subtraction. Marginally cheaper and equally effective in practice; used in LLaMA and Mistral. See [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html).

**RoPE (Rotary Positional Embedding)**
Encodes position by rotating query and key vectors in 2D subspaces; attention scores become a function of relative position only. Extrapolates well to unseen lengths with techniques like YaRN. See [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html).

**SiLU / Swish**
The activation $\text{SiLU}(x) = x \cdot \sigma(x)$; smooth, non-monotonic, and empirically strong. Its gated variant SwiGLU (used in LLaMA's FFN) computes $\text{SiLU}(xW_1) \odot xW_3$. See [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html).

**Softmax**
The function $\text{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}$, which maps a vector of real scores to a probability distribution. The numerically stable version subtracts $\max(z)$ before exponentiating. See [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html).

**Temperature (Sampling)**
A scalar $T$ that divides logits before the softmax during sampling: low $T$ sharpens the distribution (more deterministic), high $T$ flattens it (more random). $T = 1$ recovers the raw model probabilities. See [Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html).

**Tokenizer**
A module that converts raw text to a sequence of integer IDs and back. The tokenizer vocabulary, merge rules, and special tokens must match the pretrained model exactly. See [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html).

**Transformer**
The architecture introduced by Vaswani et al. (2017) consisting of stacked blocks of multi-head self-attention and FFN sub-layers with residual connections and layer normalization. Now the dominant backbone for nearly all large language and multimodal models. See [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html).

---

## B — Training & Optimization

**Adam Optimizer**
An adaptive gradient method that maintains per-parameter first and second moment estimates ($m_t$ and $v_t$) and updates weights by $-\eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$. The de-facto standard for pretraining most LLMs. See [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html).

**Adafactor**
A memory-efficient optimizer that replaces the full $v_t$ second-moment matrix with a low-rank factored approximation, reducing optimizer state from $O(P)$ to $O(\sqrt{P})$ parameters. Essential for training very large models on memory-constrained hardware. See [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html).

**Automatic Differentiation (Autograd)**
A system for computing exact derivatives of arbitrary computational graphs by applying the chain rule symbolically or numerically. PyTorch's `torch.autograd` builds a dynamic computation graph on the forward pass and traverses it in reverse for gradients. See [Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html).

**Backpropagation**
The algorithm that applies the chain rule through a neural network's computation graph to compute gradients of the loss with respect to every parameter. Requires one forward pass and one backward pass. See [Neural Networks From Scratch: MLPs & Backprop](../01-foundations/06-neural-nets-from-scratch.html).

**Batch Size**
The number of training examples processed before one gradient update. Larger batches reduce gradient noise but require more memory and careful learning-rate scaling (often linear or square-root scaling). See [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html).

**bf16 (Brain Float 16)**
A 16-bit floating-point format with the same 8-bit exponent as float32 (dynamic range) but only 7 mantissa bits (less precision). Preferred over fp16 for LLM training because it rarely overflows on gradient magnitudes. See [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html).

**Chinchilla Scaling Law**
Empirical finding by Hoffmann et al. (2022) that, given a compute budget $C$, the optimal number of training tokens is roughly $20\times$ the number of model parameters — much larger than previously assumed. Implies that LLaMA-style models trained on 1 T+ tokens are "over-trained" relative to the compute-optimal point, which is excellent for inference efficiency. See [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html).

**Cosine Annealing**
A learning-rate schedule that decays the LR following a cosine curve from the peak down to a minimum, optionally restarting. Widely used in LLM pretraining; the decay phase should roughly match total training steps. See [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html).

**Cross-Entropy Loss**
The standard language-model training objective: $\mathcal{L} = -\sum_t \log p_\theta(x_t \mid x_{<t})$. Equivalent to maximizing the log-likelihood of the data and minimizing perplexity. See [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html).

**DDP (Distributed Data Parallelism)**
A PyTorch training pattern where each GPU holds a full model replica, processes a shard of each mini-batch, and synchronizes gradients via all-reduce after each backward pass. Efficient when the model fits on one GPU. See [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html).

**DeepSpeed**
Microsoft's open-source training framework implementing ZeRO memory optimization, pipeline parallelism, and numerous efficiency tricks. Often combined with Megatron-LM for multi-dimensional parallelism. See [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html).

**FP8**
An 8-bit floating-point format (either E4M3 or E5M2 variants) used in the latest generation of training and inference hardware (NVIDIA H100). Provides approximately 2× the throughput of bf16 with careful scaling. See [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html).

**FSDP (Fully Sharded Data Parallel)**
PyTorch's native implementation of ZeRO Stage 3: model parameters, gradients, and optimizer states are all sharded across ranks; each rank reconstructs its needed slice just-in-time via all-gather. Scales to model sizes far beyond a single GPU. See [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html).

**Gradient Accumulation**
Running multiple forward-backward passes without updating weights, then accumulating gradients before applying the optimizer step. Simulates a larger effective batch size when GPU memory is limited. See [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html).

**Gradient Checkpointing (Activation Checkpointing)**
A memory-compute trade-off technique that discards intermediate activations during the forward pass and recomputes them during backprop. Reduces activation memory from $O(n \cdot d)$ to $O(\sqrt{n} \cdot d)$ at the cost of roughly 30% extra FLOPs. See [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html).

**Gradient Clipping**
Rescaling the gradient vector when its norm exceeds a threshold, $g \leftarrow g \cdot \min(1, \tau / \|g\|)$. Stabilizes training by preventing parameter updates that are disproportionately large. See [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html).

**Learning Rate Warmup**
A ramp from near-zero to the peak LR over the first few hundred to few thousand steps. Prevents early gradient explosion when Adam's second moment estimates are unreliable. See [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html).

**Lion Optimizer**
A sign-based optimizer (Liu et al., 2023) that updates weights via $-\eta \cdot \text{sign}(m_t)$ where $m_t$ is an EMA of gradients. Requires less memory than Adam (no $v_t$) and often achieves comparable or better loss. See [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html).

**Loss Spike**
A sudden, transient rise in training loss during pretraining, often caused by a bad batch, hardware bit-flip, or numerical instability. The standard fix is to roll back to a checkpoint and skip or re-weight the offending data. See [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html).

**Mixed Precision Training**
Storing weights in fp32 (the "master copy") but performing forward and backward passes in fp16 or bf16 for compute efficiency. A loss scaling factor prevents underflow of small gradients in fp16. See [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html).

**Perplexity (PPL)**
$\text{PPL} = e^{\mathcal{L}}$ where $\mathcal{L}$ is the average per-token cross-entropy loss. A perplexity of 10 means the model is, on average, as surprised as if it had to choose among 10 equally likely tokens. Lower is better. See [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html).

**Pipeline Parallelism**
A distributed training strategy that splits the model's layers across GPUs; micro-batches are pipelined so multiple GPUs can be active simultaneously. Suffers from a "pipeline bubble" at the start and end of each batch. See [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html).

**Pretraining**
The initial large-scale self-supervised training phase on web-scale text where the model learns a general world model by predicting the next token. Consumes the majority of total compute (often 10–100 GPU-years for frontier models). See [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html).

**Scaling Laws**
Empirical power-law relationships between model size $N$, dataset size $D$, compute $C$, and test loss $L$. Kaplan et al. (2020) and Chinchilla (Hoffmann et al., 2022) are the landmark works. See [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html).

**SGD (Stochastic Gradient Descent)**
The simplest optimizer: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$. Rarely used directly for LLM pretraining (Adam is far more stable) but foundational for understanding optimization. See [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html).

**Tensor Parallelism**
Splitting individual weight matrices across multiple GPUs so that each device holds a column or row shard; partial results are all-reduced. Used in Megatron-LM for intra-layer distribution. See [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html).

**Weight Decay**
L2 regularization added to the optimizer update: $\theta \leftarrow \theta(1 - \lambda) - \eta g$. Prevents overfitting and tends to improve generalization; typically set to 0.1 for LLM pretraining. See [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html).

**ZeRO (Zero Redundancy Optimizer)**
A family of memory-optimization techniques (Stages 1–3) that shards optimizer states, gradients, or parameters across data-parallel ranks to eliminate the redundant copies. Stage 3 (full sharding) is equivalent to FSDP. See [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html).

---

## C — Post-Training & Alignment

**Chain-of-Thought (CoT)**
A prompting technique (Wei et al., 2022) that elicits step-by-step intermediate reasoning before the final answer, substantially improving accuracy on multi-step arithmetic and logical tasks. See [Reasoning, Chain-of-Thought & Test-Time Compute](../05-posttraining-alignment/10-reasoning-test-time-compute.html).

**Constitutional AI (CAI)**
Anthropic's alignment approach where a model critiques and revises its own outputs against a set of written principles (the "constitution"), enabling RLHF-like alignment without large-scale human labeling of every response. See [Constitutional AI, RLAIF & Self-Improvement](../05-posttraining-alignment/11-constitutional-rlaif.html).

**DAPO (Direct Alignment from Preference Optimization)**
An extension of DPO that incorporates token-level KL control and a clip-higher training trick to improve stability and exploration. See [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html).

**Direct Preference Optimization (DPO)**
A fine-tuning method (Rafailov et al., 2023) that bypasses explicit reward model training by directly optimizing the policy from preference pairs using a closed-form reparameterization of the RLHF objective. See [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html).

**DoRA (Weight-Decomposed Low-Rank Adaptation)**
A PEFT extension of LoRA that decomposes each weight into a magnitude component and a direction component, fine-tuning the magnitude normally and using LoRA only for direction updates. Often outperforms vanilla LoRA. See [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html).

**GRPO (Group Relative Policy Optimization)**
A critic-free RL algorithm for language models that computes advantages within a group of rollouts for the same prompt by normalizing rewards, eliminating the need for a separate value network. See [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html).

**Instruction Tuning**
Supervised fine-tuning on curated (instruction, response) pairs to teach the model to follow natural language instructions. FLAN, InstructGPT, and Alpaca are landmark examples. See [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html).

**KL Divergence Penalty**
In RLHF and DPO, a regularization term $\beta \cdot \text{KL}(p_\theta \| p_\text{ref})$ that prevents the policy from straying too far from the supervised fine-tuned reference model, avoiding reward hacking. See [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html).

**LoRA (Low-Rank Adaptation)**
A PEFT technique that adds a low-rank decomposition $\Delta W = BA$ (where $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$) to frozen weight matrices, reducing trainable parameters by 10–1000×. See [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html).

**PEFT (Parameter-Efficient Fine-Tuning)**
A family of methods (LoRA, adapters, prefix tuning, prompt tuning, IA3) that fine-tune a small fraction of model parameters while keeping the majority frozen, enabling adaptation without full-model storage or compute. See [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html).

**PPO (Proximal Policy Optimization)**
The RL algorithm at the core of InstructGPT/ChatGPT training; uses a clipped surrogate objective to constrain policy updates, requiring both a policy model and a value (critic) model. See [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html).

**QLoRA (Quantized LoRA)**
Combines 4-bit NF4 quantization of the base model weights with LoRA adapters trained in bf16, enabling fine-tuning of 65B+ parameter models on a single 48 GB GPU. See [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html).

**Reward Hacking**
When a policy learns to achieve high reward from the proxy reward model by exploiting its failure modes rather than genuinely improving on the intended objective. Also called Goodhart's Law. See [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html).

**Reward Model (RM)**
A model trained to predict human preference scores for model outputs, typically a language model with a linear "reward head" replacing the LM head. Provides the reward signal for PPO-based RLHF. See [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html).

**RLHF (Reinforcement Learning from Human Feedback)**
The three-stage pipeline (SFT → RM training → PPO fine-tuning) used in InstructGPT and ChatGPT to align model behavior with human preferences via reward signal from human comparisons. See [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html).

**RLVR (Reinforcement Learning with Verifiable Rewards)**
An alignment approach where rewards come from ground-truth verifiers (e.g., a math checker or code test suite) rather than a learned reward model, eliminating reward hacking on the proxy. See [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html).

**RLOO (REINFORCE Leave-One-Out)**
A variance-reduction trick for the REINFORCE policy gradient that estimates the baseline for each sample using the average reward of the other samples in the same group, without a separate value network. See [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html).

**SFT (Supervised Fine-Tuning)**
Standard next-token prediction fine-tuning on a curated dataset of (prompt, response) pairs with teacher-forcing. The first stage of the RLHF pipeline and often sufficient for instruction following on narrow tasks. See [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html).

---

## D — Inference & Serving

**Beam Search**
A decoding algorithm that maintains $k$ candidate sequences ("beams") at each step, expanding each by all vocabulary tokens and keeping the top-$k$ by cumulative log probability. More thorough than greedy but slower and often over-generates repetitive text. See [Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html).

**Chunked Prefill**
A technique that splits a long prompt into chunks processed sequentially (instead of a single giant prefill), allowing decode requests to interleave with prefill and improving latency fairness. See [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html).

**Continuous Batching (Iteration-Level Scheduling)**
A serving strategy that allows new requests to join an in-progress batch at token boundaries rather than waiting for the entire batch to finish, maximizing GPU utilization. Pioneered by Orca and adopted by vLLM, TGI, TensorRT-LLM. See [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html).

**Decode Phase**
The autoregressive token-generation phase after the prompt has been processed; typically memory-bandwidth-bound because only one token is generated per step and the KV cache must be loaded. See [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html).

**EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)**
A speculative decoding variant that trains a small auto-regressive draft head to predict multiple tokens ahead and verifies them with the target model in parallel. See [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html).

**Medusa**
A speculative decoding approach that adds multiple "medusa heads" to the base model, each predicting a future token independently; a tree-attention mechanism verifies multiple candidate continuations in one forward pass. See [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html).

**PagedAttention**
A vLLM innovation that manages KV-cache memory using virtual pages (blocks), enabling near-zero fragmentation and allowing KV blocks to be shared across requests with the same prefix. See [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html).

**Prefill Phase**
The initial forward pass that processes the full prompt in parallel; typically compute-bound and can be batched efficiently. A 1 K-token prefill at 4096 model dimension on an A100 takes on the order of milliseconds. See [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html).

**Prefix Caching (Prompt Caching)**
Reusing the computed KV cache for shared prefixes (e.g., a long system prompt) across multiple requests, eliminating redundant prefill compute. Implemented in vLLM and SGLang via radix trees. See [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html).

**RadixAttention (SGLang)**
SGLang's KV-cache sharing mechanism using a radix tree to find the longest common prefix among active and cached requests, maximizing prefix-cache hit rates without manual management. See [SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html).

**Speculative Decoding**
A latency-reduction technique that uses a small fast "draft" model to propose $K$ tokens and the large "target" model to verify them in a single parallel forward pass. On average accepts more than 1 token per target-model step. See [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html).

**TTFT (Time to First Token)**
The end-to-end latency from receiving a request to producing its first output token; dominated by prefill compute and queuing time. A critical user-perceived latency metric for interactive applications. See [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html).

**TGI (Text Generation Inference)**
HuggingFace's production serving library for LLMs, implementing continuous batching, tensor parallelism, and quantization, written in Rust with a Python/Torch backend. See [TensorRT-LLM, TGI & Other Serving Stacks](../07-inference-serving/05-trtllm-tgi-stacks.html).

**TensorRT-LLM**
NVIDIA's optimized inference library for LLMs that uses TensorRT graph compilation, custom CUDA kernels, and quantization (FP8/INT4) to achieve near-peak throughput on NVIDIA GPUs. See [TensorRT-LLM, TGI & Other Serving Stacks](../07-inference-serving/05-trtllm-tgi-stacks.html).

**Throughput (Tokens per Second)**
The number of output tokens a serving system generates per second across all requests. A compute-bound metric that increases with batch size up to memory capacity. See [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html).

**vLLM**
A high-throughput LLM serving system built around PagedAttention and continuous batching that achieves near-zero KV-cache fragmentation and very high GPU utilization. See [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html).

---

## E — Efficiency & Quantization

**Arithmetic Intensity**
The ratio of floating-point operations to bytes of memory traffic for a kernel: $I = \text{FLOPs} / \text{bytes}$. Kernels below the roofline's break-even point are memory-bandwidth-bound; above it they are compute-bound. See [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html).

**AWQ (Activation-Aware Weight Quantization)**
A post-training quantization method that identifies weight channels that are important (those corresponding to large input activations) and applies per-channel scaling before low-bit quantization, preserving accuracy. See [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html).

**FlashAttention**
An IO-aware attention algorithm (Dao et al., 2022) that tiles the QKV matrices into SRAM, fuses the softmax and matmul in a single kernel pass, and avoids materializing the full $N \times N$ attention matrix — reducing memory from $O(N^2)$ to $O(N)$. See [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html).

**GGUF**
A portable binary format for quantized LLM weights used by llama.cpp; supports INT2–INT8 and mixed-precision quantization with metadata. Enables CPU and Apple Silicon inference without CUDA. See [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html).

**GPTQ**
A one-shot post-training quantization method that quantizes weights layer by layer using the Hessian of the loss, minimizing reconstruction error. Achieves near-lossless INT4 quantization of billion-scale models. See [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html).

**INT4 Quantization**
Representing model weights with 4-bit integers. Reduces storage by 8× vs. fp32 and 2× vs. INT8, at the cost of some accuracy; requires careful per-group scaling. Nearly standard for edge and consumer-GPU deployment. See [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html).

**Kernel Fusion**
Combining multiple sequential GPU kernel launches into one to reduce memory traffic and launch overhead. Flash attention is the landmark example; `torch.compile` automates many such fusions. See [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html).

**NF4 (NormalFloat 4)**
A 4-bit data type with quantile-equalized levels optimized for normally distributed weights (as pretrained LLM weights approximately are). Introduced in the QLoRA paper for use with bitsandbytes. See [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html).

**QAT (Quantization-Aware Training)**
A technique that simulates quantization noise during training (using "fake quantize" operations) so the model adapts its weights to the quantization grid, typically yielding better accuracy than PTQ. See [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html).

**Roofline Model**
A visual performance model that plots achievable throughput (FLOPs/s) against arithmetic intensity; memory-bandwidth and peak-compute ceilings create a "roofline." Guides optimization by showing whether a kernel is memory- or compute-bound. See [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html).

**SmoothQuant**
A PTQ method that smooths quantization difficulty by migrating scale challenges from activations (hard to quantize) to weights (easy to quantize) via a mathematically equivalent per-channel scaling. See [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html).

**torch.compile**
PyTorch 2.0's JIT compiler (backed by TorchDynamo + Inductor) that captures and optimizes model computation graphs, fusing kernels and eliminating Python overhead with minimal code changes. See [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html).

**Triton**
OpenAI's Python-based GPU programming language that compiles to PTX; provides a higher-level abstraction than CUDA for writing custom fused kernels without hand-coding assembly. See [Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html).

---

## F — Retrieval, RAG & Agents

**ANN (Approximate Nearest Neighbor Search)**
Algorithms (HNSW, IVF, PQ) that find approximately the $k$ closest vectors in a high-dimensional space in sublinear time, trading a small recall penalty for large speedups over exact search. See [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html).

**BM25**
A classical TF-IDF-based ranking function that scores documents by term frequency, inverse document frequency, and document length normalization. A strong sparse retrieval baseline often used in hybrid search alongside dense embeddings. See [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html).

**Dense Retrieval**
Encoding both queries and documents as dense vectors and performing nearest-neighbor search to retrieve relevant passages. Requires embedding models trained with contrastive objectives (DPR, E5, GTE). See [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html).

**FAISS**
Facebook AI Similarity Search: a production-grade library of ANN algorithms (flat, IVF, HNSW, PQ) with GPU support, used as the retrieval backend in many RAG systems. See [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html).

**Function Calling (Tool Use)**
A mechanism where the LLM emits structured JSON specifying a tool name and arguments; the harness executes the tool and returns results in a subsequent user message. Enables external side effects like web search, code execution, and API calls. See [Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html).

**GraphRAG**
A RAG variant that builds a knowledge graph from documents and uses community detection and graph traversal to retrieve structured relational context beyond what vector similarity captures. See [Advanced RAG: GraphRAG, Agentic RAG & Long-Context vs RAG](../09-rag-retrieval/05-advanced-rag.html).

**HNSW (Hierarchical Navigable Small World)**
A graph-based ANN algorithm that builds a multi-layer proximity graph; query time is $O(\log N)$ and recall is typically above 95% at retrieval speeds orders of magnitude faster than brute force. See [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html).

**MCP (Model Context Protocol)**
Anthropic's open protocol for standardizing how LLM hosts expose tools, resources, and prompts to models via a client-server interface, enabling plug-and-play tool ecosystems. See [The Model Context Protocol (MCP)](../08-agents-harness/06-mcp.html).

**RAG (Retrieval-Augmented Generation)**
A pattern that injects retrieved context (from a vector database or search engine) into the LLM prompt before generation, grounding answers in up-to-date or proprietary information without retraining. See [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html).

**ReAct (Reason + Act)**
An agent paradigm where the LLM alternates between generating reasoning traces (Thought) and grounded actions (Act/Observe), integrating the benefits of chain-of-thought and tool use. See [The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html).

**Reranker**
A cross-encoder model that scores query-document pairs jointly (rather than independently), applied as a second stage to a candidate set from dense or sparse retrieval to improve precision at the cost of latency. See [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html).

**Vector Database**
A database system optimized for storing, indexing, and querying high-dimensional embedding vectors; examples include Chroma, Weaviate, Qdrant, Pinecone, and pgvector. See [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html).

---

## G — Systems, Hardware & Production

**All-Reduce**
A collective communication operation that computes a reduction (e.g., sum) across all participating processes and distributes the result back to each. The core synchronization primitive in DDP gradient averaging. See [Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html).

**CUDA (Compute Unified Device Architecture)**
NVIDIA's programming platform and API for general-purpose GPU computing. CUDA C++ allows writing kernels that execute on thousands of GPU threads in parallel. See [CUDA Programming Essentials for ML Engineers](../04-kernels-efficiency/05-cuda-essentials.html).

**CUDA Graphs**
A mechanism to record a sequence of CUDA operations into a graph and replay it with minimal CPU launch overhead, critical for small-batch inference where CPU launch latency dominates. See [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html).

**Data Flywheel**
A self-reinforcing improvement cycle where deployed model interactions generate labeled data (via user feedback, implicit signals, or RLHF) that is used to train the next model version, which attracts more usage. See [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html).

**GPU Memory Hierarchy**
CUDA GPUs have registers (fastest), L1/shared memory (~100 TB/s), L2 cache (~6 TB/s on A100), and HBM DRAM (~2 TB/s on A100) — each level is orders of magnitude smaller but faster. FlashAttention achieves its gains by keeping data in SRAM. See [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html).

**HBM (High Bandwidth Memory)**
The stacked DRAM found in data-center GPUs (A100 has 80 GB HBM2e at ~2 TB/s bandwidth). This is the "slow" memory relative to SRAM/shared memory but still 10× faster than DDR5 CPU RAM. See [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html).

**Jailbreak**
A prompt crafted to circumvent a model's safety training and elicit policy-violating outputs. Common techniques include role-play framing, encoding, and instruction injection. See [Security: Prompt Injection, Jailbreaks & Defenses](../12-production-mlops/06-security-prompt-injection.html).

**LLMOps**
The operational discipline of monitoring, evaluating, versioning, and improving LLM-based systems in production, including prompt management, eval harnesses, and cost tracking. See [Observability, Logging & LLMOps](../12-production-mlops/02-observability-llmops.html).

**Megatron-LM**
NVIDIA's framework for training very large language models with 3D parallelism (tensor × pipeline × data). The backbone of most frontier model training runs. See [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html).

**NVLink**
NVIDIA's high-speed GPU interconnect that provides ~600–900 GB/s bandwidth between GPUs on the same node (e.g., DGX A100/H100), enabling efficient all-reduce and tensor parallelism. See [Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html).

**Prompt Injection**
An adversarial attack where malicious content in model inputs overrides system-level instructions; a critical security concern for agentic systems with tool access. See [Security: Prompt Injection, Jailbreaks & Defenses](../12-production-mlops/06-security-prompt-injection.html).

**RDMA (Remote Direct Memory Access)**
Network hardware that allows one GPU to read or write another GPU's memory without involving the CPU, used in InfiniBand networks for high-throughput distributed training. See [Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html).

**SM (Streaming Multiprocessor)**
The fundamental compute unit on an NVIDIA GPU; each SM contains CUDA cores, tensor cores, shared memory, and registers. An A100 has 108 SMs. Occupancy — the fraction of warps active per SM — is a key performance lever. See [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html).

**Tensor Core**
Specialized NVIDIA hardware (introduced in Volta) that performs a 4×4 (or 8×8/16×16 in Ampere/Hopper) matrix multiply-accumulate in a single clock cycle, providing up to 16× throughput over regular CUDA cores for matmuls. See [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html).

---

## H — Evaluation & Safety

**Benchmark (LLM)**
A standardized test suite for measuring model capabilities — e.g., MMLU for knowledge breadth, HumanEval for code, GSM8K for math, MATH for competition math, and MT-Bench for instruction following. See [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html).

**Benchmark Contamination**
When pretraining data includes examples from evaluation benchmarks, inflating reported scores above true generalization ability. A major reproducibility concern for all large-scale evaluations. See [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html).

**Constitutional AI**
See the definition above under Post-Training & Alignment. Cross-link: [Constitutional AI, RLAIF & Self-Improvement](../05-posttraining-alignment/11-constitutional-rlaif.html).

**Goodhart's Law**
"When a measure becomes a target, it ceases to be a good measure." In LLM alignment, a reward model is a proxy for human preferences; optimizing it too hard leads to reward hacking. See [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html).

**LLM-as-a-Judge**
Using a language model (often GPT-4 or a specialized judge model) to score other models' outputs, enabling scalable automated evaluation without exhaustive human annotation. Calibrated against human ratings to reduce positional and verbosity bias. See [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html).

**MT-Bench**
A multi-turn instruction-following benchmark of 80 carefully curated questions across 8 categories, evaluated by GPT-4 as judge. Widely used to compare instruction-tuned models on conversational quality. See [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html).

**Red-Teaming**
Systematic adversarial probing of a model to find harmful outputs, jailbreaks, or capability boundaries; critical before deployment and for safety research. See [Red-Teaming, Safety & Robustness Evaluation](../11-evaluation/05-redteaming-safety-eval.html).

**RLHF (see also under Post-Training)**
The alignment pipeline based on learning from human comparative preferences. Also see [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html).

---

!!! example "Worked Example: KV-Cache Memory Budget"

    A 7B decoder-only model with 32 layers, 32 heads, head dimension 128 (so $d_{model} = 4096$), using GQA with 8 KV heads and bf16 precision:

    **Memory per token per layer:**
    - K tensor: 8 heads × 128 dim × 2 bytes = **2 048 bytes**
    - V tensor: 8 heads × 128 dim × 2 bytes = **2 048 bytes**
    - Total per layer: **4 096 bytes = 4 KB**

    **Across all layers at sequence length 4096 tokens:**

    $$\text{KV memory} = 4096 \text{ tokens} \times 32 \text{ layers} \times 4096 \text{ bytes/layer} \approx 536 \text{ MB}$$

    **For a batch of 16 concurrent sequences:**

    $$536 \text{ MB} \times 16 = 8.6 \text{ GB}$$

    On a single 80 GB A100, after the model weights (~14 GB in bf16), this leaves about **57 GB** for KV cache — enough for roughly 16 × 16K-token sequences simultaneously. PagedAttention manages this pool with near-zero fragmentation.

    See [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html) for the implementation.

---

!!! interview "Interview Corner"

    **Q:** What is the difference between GQA, MQA, and standard MHA? When would you pick each, and what are the memory implications?

    **A:** In standard Multi-Head Attention (MHA), each of the $h$ query heads has its own key and value head — so the KV cache scales as $O(h \cdot d_k)$ per token per layer. In Multi-Query Attention (MQA), all query heads share a single KV head, reducing KV cache by a factor of $h$. Grouped-Query Attention (GQA) is a middle ground: $h$ query heads share $g$ KV heads ($1 \le g \le h$), reducing KV cache by $h/g$ with a smaller quality cost than MQA.

    Pick MQA or GQA when memory bandwidth and KV-cache size are the bottleneck (i.e., at inference time with large batches and long contexts). Pick standard MHA when training a small model from scratch where memory is ample and you want maximum expressiveness. LLaMA 2 70B uses GQA with $g=8$ heads, providing an 8× KV-cache reduction with negligible perplexity degradation over MHA.

    The math: for a 70B model at GQA with 8 KV heads vs. 64 query heads, and a 4K context, bf16, 80 layers:
    $8 \times 128 \times 2 \times 2 \times 4096 \times 80 \approx 1.3 \text{ GB per sequence}$ — vs. ~10.5 GB with MHA.

---

## I — Mathematics & Information Theory

**Chain Rule**
The calculus identity $\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$, the mathematical foundation of backpropagation when applied recursively through a computation graph. See [Calculus, Optimization & Convexity](../01-foundations/03-calculus-optimization.html).

**Cosine Similarity**
$\text{cos}(u, v) = \frac{u \cdot v}{\|u\|\|v\|}$, measuring the angle between two vectors regardless of magnitude. Standard metric for comparing dense embeddings in retrieval. See [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html).

**Cross-Entropy**
$H(p, q) = -\sum_x p(x) \log q(x)$; the expected log-loss under the data distribution $p$ of using model distribution $q$. Equivalent to KL divergence plus the data entropy: $H(p,q) = H(p) + D_{KL}(p \| q)$. See [Probability, Statistics & Information Theory](../01-foundations/02-probability-information.html).

**Eigendecomposition**
Factoring a square matrix $A = Q \Lambda Q^{-1}$ where $\Lambda$ is diagonal with eigenvalues and $Q$'s columns are eigenvectors. Fundamental in understanding covariance matrices, optimization landscapes, and attention. See [Linear Algebra for Deep Learning](../01-foundations/01-linear-algebra.html).

**Entropy**
$H(X) = -\sum_x p(x) \log p(x)$; a measure of uncertainty or information content. A uniform distribution over $V$ tokens has entropy $\log V$; a peaked distribution near 0. See [Probability, Statistics & Information Theory](../01-foundations/02-probability-information.html).

**Gradient Descent**
Iterative minimization by moving in the direction of the negative gradient: $\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$. The foundation of all neural network optimizers. See [Calculus, Optimization & Convexity](../01-foundations/03-calculus-optimization.html).

**KL Divergence**
$D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$; measures how much $q$ differs from reference $p$. Always non-negative; zero iff $p = q$. Crucial in RLHF as the regularization term preventing policy divergence. See [Probability, Statistics & Information Theory](../01-foundations/02-probability-information.html).

**Matrix Multiplication**
The core arithmetic operation of deep learning: $(AB)_{ij} = \sum_k A_{ik} B_{kj}$. An $m \times k$ times $k \times n$ matmul costs $2mkn$ FLOPs. Tensor cores execute this with 8–16× throughput over scalar operations. See [Linear Algebra for Deep Learning](../01-foundations/01-linear-algebra.html).

**Softmax Temperature**
See Temperature (Sampling) above.

**SVD (Singular Value Decomposition)**
$A = U \Sigma V^\top$; decomposes any matrix into orthogonal $U$, $V$ and diagonal singular values $\Sigma$. LoRA can be interpreted as constraining $\Delta W$ to the top-$r$ singular value subspace. See [Linear Algebra for Deep Learning](../01-foundations/01-linear-algebra.html).

---

## Worked Code: Looking Up Glossary Concepts Programmatically

The following snippet demonstrates how a production RAG system might index and query this very glossary using dense retrieval — a concrete example of the concepts defined above.

```python
"""
glossary_search.py — Index and search the LLM glossary using dense retrieval.
Demonstrates: tokenization, embedding, ANN search (brute-force for demo).
Requirements: pip install sentence-transformers numpy
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. Glossary entries (term -> definition snippet)
# ---------------------------------------------------------
GLOSSARY = {
    "FlashAttention": (
        "IO-aware attention algorithm that tiles QKV into SRAM, "
        "fusing softmax and matmul to avoid materializing the N×N matrix."
    ),
    "LoRA": (
        "Low-Rank Adaptation: adds a trainable low-rank decomposition ΔW=BA "
        "to frozen weight matrices, reducing trainable params by 10-1000×."
    ),
    "GQA": (
        "Grouped-Query Attention: h query heads share g KV heads (g < h), "
        "reducing KV-cache memory by h/g vs standard multi-head attention."
    ),
    "PagedAttention": (
        "Virtual-page KV-cache management in vLLM; near-zero fragmentation "
        "and cross-request prefix sharing."
    ),
    "Chinchilla Scaling Law": (
        "Compute-optimal training requires ~20 tokens per parameter; "
        "derived by Hoffmann et al. 2022."
    ),
    "RLHF": (
        "Reinforcement Learning from Human Feedback: SFT → reward model → PPO "
        "policy optimization pipeline."
    ),
    "ZeRO": (
        "Zero Redundancy Optimizer: shards optimizer states, gradients, and "
        "parameters across data-parallel ranks to eliminate memory redundancy."
    ),
    "BPE": (
        "Byte-Pair Encoding tokenization: iteratively merge the most frequent "
        "adjacent token pair until reaching target vocabulary size."
    ),
}

# ---------------------------------------------------------
# 2. Encode all definitions using a small embedding model
# ---------------------------------------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

terms = list(GLOSSARY.keys())
definitions = [GLOSSARY[t] for t in terms]

# Shape: (n_entries, embedding_dim)  — all-MiniLM-L6-v2 gives dim=384
embeddings = model.encode(definitions, normalize_embeddings=True)
print(f"Indexed {len(terms)} glossary entries, embedding dim={embeddings.shape[1]}")

# ---------------------------------------------------------
# 3. Query function: brute-force cosine similarity (exact ANN)
# ---------------------------------------------------------
def search_glossary(query: str, top_k: int = 3) -> list[tuple[str, float, str]]:
    """Return top_k (term, score, definition) tuples for the query."""
    q_emb = model.encode([query], normalize_embeddings=True)  # (1, d)
    # Dot product of normalized vectors = cosine similarity
    scores = (embeddings @ q_emb.T).squeeze()                  # (n_entries,)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(terms[i], float(scores[i]), definitions[i]) for i in top_indices]

# ---------------------------------------------------------
# 4. Demo query
# ---------------------------------------------------------
results = search_glossary("how does attention reduce memory usage?")
print("\nTop results for 'how does attention reduce memory usage?'")
for term, score, defn in results:
    print(f"  [{score:.3f}] {term}: {defn[:80]}...")

# Expected output (approximate):
# [0.72] FlashAttention: IO-aware attention algorithm that tiles QKV into SRAM...
# [0.61] GQA: Grouped-Query Attention: h query heads share g KV heads...
# [0.58] PagedAttention: Virtual-page KV-cache management in vLLM...
```

This pattern — embedding a corpus offline, then performing ANN search at query time — is exactly how production RAG pipelines work at scale, replacing `@` matmul with FAISS or HNSW and scaling to millions of documents. See [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html).

---

## Quick-Reference: Symbol Table

| Symbol | Meaning | Chapter |
|--------|---------|---------|
| $d_{model}$ | Hidden dimension of the Transformer | [Embeddings & The Input Pipeline](../02-transformer/02-embeddings-input.html) |
| $d_k$ | Per-head key/query dimension ($= d_{model}/h$) | [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html) |
| $h$ | Number of attention heads | [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html) |
| $N$ | Sequence length | [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html) |
| $V$ | Vocabulary size | [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html) |
| $L$ | Number of Transformer layers | [Building a GPT From Scratch (nanoGPT-style)](../02-transformer/07-build-gpt-from-scratch.html) |
| $P$ | Total parameter count | [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html) |
| $\eta$ | Learning rate | [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html) |
| $\beta$ | KL penalty coefficient in RLHF/DPO | [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html) |
| $r$ | LoRA rank | [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html) |
| $T$ | Sampling temperature | [Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html) |
| $I$ | Arithmetic intensity (FLOPs/byte) | [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html) |

---

!!! key "Key Takeaways"
    - The LLM stack spans five distinct engineering domains: architecture (Transformer internals), training (distributed optimization), efficiency (quantization and kernels), alignment (RLHF/DPO/RL), and serving (batching and KV management) — each with its own vocabulary.
    - Knowing the exact definitions of related but distinct terms (MHA vs. GQA vs. MQA, LoRA vs. QLoRA vs. DoRA, PPO vs. DPO vs. GRPO) is table stakes for technical interviews and system design.
    - Memory arithmetic is fundamental: always derive KV-cache sizes, parameter counts, and gradient storage from first principles using bytes = dtype_size × product_of_dims.
    - Performance bottlenecks are determined by arithmetic intensity relative to the hardware roofline; most LLM inference kernels are memory-bandwidth-bound during decode, compute-bound during prefill with large batches.
    - Alignment terminology conflates three different things — the objective (human preferences), the algorithm (PPO, DPO, GRPO), and the infrastructure (RM, verifier, KL controller) — keep them distinct in any design discussion.
    - RAG, agent, and serving terms (PagedAttention, RadixAttention, TTFT, continuous batching) reflect real implementation details that directly affect system SLAs and cost per token.
    - The appendices [The Math Reference Sheet](../99-appendix/02-math-reference.html), [Key Papers: An Annotated Reading List](../99-appendix/03-papers-reading-list.html), and [Tooling & Environment Setup Cheatsheet](../99-appendix/04-tooling-setup.html) complement this glossary with symbolic derivations, landmark references, and environment setup.

---

## Further Reading

- **Vaswani et al., "Attention Is All You Need" (2017)** — The foundational Transformer paper; defines MHA, positional encodings, and the encoder-decoder architecture.
- **Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)** — Establishes compute-optimal scaling laws; basis for the 20-tokens-per-parameter rule.
- **Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)** — Introduces tiled SRAM-resident attention; now the standard attention implementation.
- **Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2022)** — Defines the LoRA PEFT method; foundational for parameter-efficient fine-tuning.
- **Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)** — Derives DPO from the RLHF objective; eliminated the need for separate reward model training in many pipelines.
- **Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)** — Introduces NF4 quantization and QLoRA; democratized large-model fine-tuning on consumer hardware.
- **Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)** — The vLLM paper; defines PagedAttention and continuous batching.
- **Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019)** — Introduces MQA; the precursor to GQA.
- **nanoGPT (Andrej Karpathy, GitHub)** — A minimal, readable GPT implementation; the best starting point for hands-on architecture study.
