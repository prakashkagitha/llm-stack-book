# Preface: How to Read This Book & The Map of the Stack

> *"What I cannot create, I do not understand."* — Richard Feynman, on his blackboard at the time of his death.

This book is built on Feynman's principle. You will not truly understand attention until you have written the `softmax(QKᵀ/√d)V` loop with your own hands; you will not understand RLHF until you have computed a clipped policy-gradient loss token by token; you will not understand why an H100 is fast until you have stared at the gap between its 989 teraFLOP/s of compute and its 3.35 TB/s of memory bandwidth and felt the tension. So this is a book of explanations *and* of code — toy implementations, from-scratch reconstructions, and pointers into the real open-source systems that run today's frontier models.

## Who this book is for

It is for the engineer who wants to hold the **entire** Large Language Model stack in their head at once — not as a list of buzzwords, but as a connected machine. The person who can move from the IEEE-754 bit layout of a `bfloat16` up through a FlashAttention kernel, through a 3D-parallel pretraining run, through a GRPO reinforcement-learning loop, out to a vLLM serving cluster, and finally into the agentic harness that wraps the model in tools — and explain how each layer constrains the ones above and below it.

If you are preparing for a machine-learning interview — and a free [companion guide](../interview/index.html) is written for exactly that — this breadth-with-depth is precisely what distinguishes a strong candidate from a memorizer.

!!! interview "Interview Corner"
    Throughout the book, callouts like this one flag concepts that come up disproportionately often in ML/LLM interviews, with crisp model answers. When you see the 🎯, treat it as "a sharp interviewer will probe here." The interview track in Part XIII then assembles these into mock questions, a system-design framework, and a day-by-day study plan.

## The map of the stack

The single most useful thing you can carry into any LLM discussion is a mental picture of the **layers**, from silicon to agent. Every chapter in this book lives somewhere on this map.


{{fig:preface-stack-map}}


Read the arrows in both directions. The reason decode-phase inference is *memory-bandwidth-bound* (Part VII) is a fact about HBM (Part I). The reason we invented GQA and MLA (Part II) is to shrink the KV cache that bandwidth has to move. The reason FlashAttention (Part IV) exists is that naively materializing the $N \times N$ attention matrix in HBM is the bottleneck. The reason RL infrastructure (Part VI) is uniquely hard is that it must run *inference* (generation) and *training* in the same loop, fighting over the same GPUs. **Nothing in this stack is arbitrary; every design is a response to a physical or statistical constraint.** Learning to see those constraints is the whole game.

## A worked taste: the one equation under everything

Before we begin, here is the equation that an LLM spends essentially all of its training compute minimizing — the average negative log-likelihood of the next token:

$$
\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta\!\left(x_t \mid x_{<t}\right)
$$

That is the *entire* pretraining objective (Part III). Everything else — the transformer that computes $p_\theta$, the kernels that make it fast, the alignment that reshapes it, the infrastructure that scales it — is in service of this one cross-entropy loss and the things you can do with the distribution it produces. We can even write the heart of it in three lines of PyTorch, and we will return to this snippet many times:

```python
import torch
import torch.nn.functional as F

# logits: (batch, seq_len, vocab)   targets: (batch, seq_len) of token ids
def lm_loss(logits, targets):
    # shift so that position t predicts token t+1
    logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    targets = targets[:, 1:].reshape(-1)
    return F.cross_entropy(logits, targets)   # mean negative log-likelihood
```

If that snippet feels both trivial and mysterious — *trivial* because it is three lines, *mysterious* because it somehow produces systems that write code and prove theorems — then you are in exactly the right frame of mind. This book exists to dissolve the mystery without losing the awe.

## How the book is organized

The book has fifteen parts. You can read it front to back as a course, or treat it as a reference and parachute into any chapter; each is written to stand on its own while linking generously to its neighbors.

| Part | Theme | What you will be able to do |
|------|-------|------------------------------|
| I | Math & systems foundations | Reason about gradients, floats, and GPUs |
| II | The transformer | Build a GPT from scratch |
| III | Pretraining at scale | Plan and run a distributed training job |
| IV | Kernels & efficiency | Explain FlashAttention; quantize a model |
| V | Post-training & alignment | Implement SFT, LoRA, DPO, GRPO |
| VI | RL infrastructure | Understand veRL, TRL, async RL systems |
| VII | Inference & serving | Tune vLLM/SGLang; reason about latency |
| VIII | Agents & harness | Build a tool-using coding agent |
| IX | Retrieval & RAG | Design a grounded retrieval system |
| X | Multimodal & frontiers | Connect vision/audio to an LLM |
| XI | Evaluation | Measure models honestly |
| XII | Production & MLOps | Operate LLMs safely at scale |
| XIII | The Google ML interview | Walk in prepared |
| App. | Glossary, math sheet, papers | Look things up fast |

!!! key "Conventions used in this book"
    - **Math** is rendered with KaTeX; inline like $\sigma(x) = \frac{1}{1+e^{-x}}$ and displayed in its own block.
    - **Code** is syntax-highlighted and copyable; most snippets are runnable as written, and from-scratch reconstructions are collected in the [From-Scratch Code Index](../99-appendix/05-from-scratch-index.html).
    - **🎯 Interview Corners** flag interview-critical ideas.
    - **🔑 Key-idea** boxes (like this one) summarize what you must remember.
    - **⚠️ Warning** boxes flag common mistakes and footguns.
    - **🧪 Example** boxes contain worked, concrete numerical examples.

!!! warning "A note on a fast-moving field"
    This book reflects the state of the art as of **2026**, deliberately covering the most recent ideas in RL infrastructure (GRPO and its descendants, async/disaggregated RL, FP8 training), serving (disaggregated prefill/decode, RadixAttention), and agents (MCP, harness engineering). Specific numbers and library APIs will drift; the *concepts and the constraints they answer to* are durable. When in doubt, re-derive from the physics: bytes moved, FLOPs done, bits of precision, samples of data.

## A ten-day on-ramp (if the clock is ticking)

If you are reading this with an interview looming, do not try to read 2,000 pages in ten days. Go straight to the [Ten-Day Study Plan](../interview/08-ten-day-study-plan.html) in the companion guide, which sequences a focused subset of chapters, then circle back to the rest over the months that follow. Depth compounds; the plan front-loads the highest-yield concepts.

Now turn the page, and let us start where everything starts: with the linear algebra that lives inside a single matrix multiply.

!!! note "Where to go next"
    Continue to **[Linear Algebra for Deep Learning](../01-foundations/01-linear-algebra.html)**, or jump to the [full table of contents](../index.html).
