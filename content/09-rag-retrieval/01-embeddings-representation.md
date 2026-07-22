# 9.1 Embeddings & Representation Learning

Dense text embeddings are the connective tissue of modern information retrieval. Without them, matching a user's natural-language question to a relevant document requires either exact keyword overlap or hand-crafted rules — both brittle in practice. With them, a 768-dimensional vector encodes semantic meaning so precisely that "What is the boiling point of water?" and "At what temperature does H₂O vaporize?" land microseconds apart in vector space. This chapter develops every piece of the pipeline: why dense vectors work, how they are trained with contrastive objectives, how the field settled on architectures like bi-encoders with pooled transformer representations, and how to evaluate them rigorously with MTEB. It ends with a direct bridge to the RAG systems that consume these embeddings at query time.

If you are looking for the token embeddings inside a transformer (the learned lookup table that maps token IDs to vectors at the input layer), that is covered in [Embeddings & The Input Pipeline](../02-transformer/02-embeddings-input.html). This chapter is about *sentence*- and *document*-level representation: a single fixed-size vector that summarizes an entire passage.

## Why Dense Representations?

The classical information retrieval (IR) approach is **sparse retrieval**: represent each document as a high-dimensional bag-of-words vector (TF-IDF or BM25), then compute dot products at query time. BM25 remains a strong baseline, but it fails on vocabulary mismatch — if the query says "automobile" and the document says "car," zero overlap means zero score.

**Dense retrieval** projects documents and queries into a low-dimensional (128–1024 dimensions) continuous space trained so that semantically related pairs are close regardless of word choice. The trade-off is computation: sparse retrieval can be done with an inverted index that touches only non-zero entries; dense retrieval requires either an exhaustive scan or an approximate nearest neighbor (ANN) index (see [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html)).

The two paradigms are not mutually exclusive. Hybrid search — combining BM25 scores with dense scores via reciprocal rank fusion or learned weighting — consistently outperforms either alone. That is covered in [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html).

{{fig:emb-dense-vs-sparse-vocab-mismatch}}

### The Bi-Encoder Architecture

Dense embedding models overwhelmingly use the **bi-encoder** design:

{{fig:emb-biencoder-concept}}

Both query and document are encoded independently by the same (or separate) transformer. The final similarity score is computed as a dot product or cosine similarity at retrieval time. This independence is the key property that enables pre-computation: all document embeddings can be computed offline and stored in an ANN index. Only the query needs to be embedded at inference time.

The **cross-encoder** alternative runs both query and document through the same forward pass with joint attention, producing higher-quality relevance scores but at $O(n)$ inference cost per document. Cross-encoders are used as rerankers on a short candidate list; they cannot be the first-stage retriever. See [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html) for cross-encoder reranking.

## From Token Representations to Sentence Embeddings: Pooling

A transformer operating on a sequence of $n$ tokens produces $n$ contextual vectors, each of shape $d$. We need a single vector. The pooling layer performs this reduction.

### CLS Pooling

BERT-style models prepend a special `[CLS]` token. After the final layer, the vector at position 0 is used as the sequence representation. BERT was pretrained with a next-sentence prediction (NSP) objective that trained `[CLS]` to carry sentence-level signal. In practice, raw BERT CLS embeddings are mediocre for semantic similarity — the representation space is anisotropic (vectors cluster near a cone, not spread across the hypersphere). Models like SBERT fine-tune the encoder specifically for sentence embedding, dramatically improving CLS quality.

### Mean Pooling

Average the final-layer token representations, excluding padding tokens:

$$
\mathbf{e} = \frac{\sum_{i=1}^{n} m_i \mathbf{h}_i}{\sum_{i=1}^{n} m_i}
$$

where $m_i \in \{0,1\}$ is the attention mask and $\mathbf{h}_i$ is the hidden state at position $i$. Mean pooling is empirically stronger than CLS pooling for most bi-encoder tasks, and it is the default in models like `sentence-transformers` (Reimers & Gurevych, 2019).

### Max Pooling and Weighted Pooling

**Max pooling** takes the element-wise maximum across token positions — capturing the strongest activation for each latent feature. It is less common in production. **Weighted pooling** assigns position or attention-based weights before averaging; some models learn these weights as part of fine-tuning.

```python
import torch
import torch.nn.functional as F


def mean_pool(token_embeddings: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute attention-mask-weighted mean of token embeddings.

    Args:
        token_embeddings: shape (batch, seq_len, hidden_dim)
        attention_mask:   shape (batch, seq_len), 1 for real tokens

    Returns:
        sentence_embeddings: shape (batch, hidden_dim)
    """
    # Expand mask to hidden_dim so we can multiply element-wise
    mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)

    # Zero out padding positions
    sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)  # (B, H)

    # Count real tokens per sample (clamp to avoid division by zero)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)             # (B, 1)

    return sum_embeddings / sum_mask  # (B, H)


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """L2-normalize so dot product == cosine similarity."""
    return F.normalize(embeddings, p=2, dim=-1)
```

After pooling, embeddings are typically L2-normalized so that dot products equal cosine similarities, which lie in $[-1, 1]$ and are symmetric and bounded.

## Contrastive Learning & the InfoNCE Objective

The core training objective for dense embedding models is **contrastive learning**: pull representations of semantically similar pairs together, push dissimilar ones apart. The dominant loss function is **InfoNCE** (Information Noise-Contrastive Estimation), also known as the NT-Xent loss in the SimCLR literature.

### InfoNCE Derivation

Given a mini-batch of $N$ (query, positive document) pairs $\{(q_i, d_i^+)\}_{i=1}^N$, the model produces normalized embeddings $\mathbf{e}_{q_i}$ and $\mathbf{e}_{d_j}$ for all queries and all documents. The InfoNCE loss treats all $N-1$ other documents in the batch as **in-batch negatives** for query $i$:

$$
\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\mathbf{e}_{q_i} \cdot \mathbf{e}_{d_i^+} / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{e}_{q_i} \cdot \mathbf{e}_{d_j} / \tau)}
$$

where $\tau > 0$ is the **temperature** hyperparameter. The loss is a cross-entropy over $N$ classes: class $i$ is the correct document for query $i$. Minimizing it encourages $\mathbf{e}_{q_i} \cdot \mathbf{e}_{d_i^+}$ to be the maximum among all $N$ dot products in row $i$.

**Why temperature matters.** A small $\tau$ (e.g., 0.01–0.1) sharpens the softmax, creating a steep gradient signal and forcing the model to distinguish very fine-grained differences. A large $\tau$ (near 1.0) makes the loss soft and nearly uniform. Typical values: 0.05–0.1 for image encoders (SimCLR used 0.07), 0.02–0.05 for text retrieval.

**Scaling with batch size.** With batch size $N$, each query sees $N-1$ negatives. Doubling $N$ roughly doubles the number of negatives per query, which empirically improves representation quality — this is why contrastive learning loves large batches. DPR (Karpukhin et al., 2020) used batch size 128; modern models often train with effective batch sizes in the thousands via gradient accumulation.

{{fig:emb-infonce-similarity-matrix}}

### Full In-Batch Negative Loss Implementation

```python
import torch
import torch.nn.functional as F


def infonce_loss(
    query_emb: torch.Tensor,   # (B, D), already L2-normalized
    doc_emb: torch.Tensor,     # (B, D), already L2-normalized
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss with in-batch negatives.

    Each query is matched to the corresponding document (diagonal).
    All other documents in the batch serve as negatives.

    Args:
        query_emb:   Normalized query embeddings.
        doc_emb:     Normalized positive-document embeddings.
        temperature: Softmax temperature (lower = sharper contrast).

    Returns:
        Scalar loss (mean over batch).
    """
    B = query_emb.size(0)

    # Similarity matrix: (B, B)
    # sim[i][j] = cosine_sim(query_i, doc_j)
    sim = torch.matmul(query_emb, doc_emb.T) / temperature  # (B, B)

    # Targets: each query i matches document i (diagonal)
    labels = torch.arange(B, device=query_emb.device)

    # Query-to-document direction
    loss_q2d = F.cross_entropy(sim, labels)

    # Document-to-query direction (symmetric)
    loss_d2q = F.cross_entropy(sim.T, labels)

    return (loss_q2d + loss_d2q) / 2


# ---- Quick sanity check ----
if __name__ == "__main__":
    torch.manual_seed(42)
    B, D = 4, 768

    # Random normalized embeddings
    q = F.normalize(torch.randn(B, D), dim=-1)
    d = F.normalize(torch.randn(B, D), dim=-1)

    loss = infonce_loss(q, d, temperature=0.05)
    print(f"Loss (random init): {loss.item():.4f}")   # ~ log(B) ≈ 1.386

    # Perfect embeddings: q[i] == d[i]
    d_perfect = q.clone()
    loss_perfect = infonce_loss(q, d_perfect, temperature=0.05)
    print(f"Loss (perfect align): {loss_perfect.item():.6f}")  # ≈ 0.0
```

!!! example "Worked Example: InfoNCE Loss Magnitudes"

    Suppose $B = 4$, $\tau = 0.05$, and query $i=0$ has cosine similarity 0.90 with its positive document
    and 0.10 with each of the three negatives.

    The raw logits (before softmax) are:

    $$
    \text{logits} = [0.90/0.05,\ 0.10/0.05,\ 0.10/0.05,\ 0.10/0.05] = [18.0,\ 2.0,\ 2.0,\ 2.0]
    $$

    Softmax probabilities:

    $$
    p_0 = \frac{e^{18}}{e^{18} + 3e^{2}} \approx \frac{65,659,969}{65,659,969 + 22.17} \approx 0.99997
    $$

    Cross-entropy loss for this query: $-\log(0.99997) \approx 0.00003$. Almost zero — the model
    has near-perfectly separated the positive from the negatives.

    Now suppose similarity with the positive drops to 0.30:

    $$
    \text{logits} = [6.0,\ 2.0,\ 2.0,\ 2.0]
    $$

    $$
    p_0 = \frac{e^6}{e^6 + 3e^2} = \frac{403.4}{403.4 + 22.2} \approx 0.948
    $$

    Loss $\approx -\log(0.948) \approx 0.053$. Meaningful gradient — the model must push the positive closer.

    At random initialization with $B=4$, the expected loss is $\log 4 \approx 1.386$ (uniform over 4 classes).

## Hard Negatives: Going Beyond In-Batch Sampling

In-batch negatives are easy to implement but are often *too easy* — randomly sampled passages are unlikely to be genuinely confusable with the query. Hard negatives are passages that are superficially relevant but do not actually answer the query.

### Mining Hard Negatives

**BM25-mined negatives.** Run BM25 retrieval on the query; take high-ranked documents that are not the gold positive. These share vocabulary with the query but differ in meaning.

**Dense-mined negatives.** Use an earlier checkpoint of the embedding model to retrieve top-$k$ passages; use non-positive ones as negatives. This "ANN negative mining" strategy, used in DPR and ANCE (Xiong et al., 2021), iteratively updates the negatives as the model improves. Training with the model's own hard negatives dramatically accelerates convergence.

**Cross-encoder filtered negatives.** Apply a high-quality cross-encoder reranker to a large candidate set, then use passages that the cross-encoder scores as irrelevant as negatives. These are the hardest and most informative.

### Triplet Loss vs. InfoNCE

An alternative to InfoNCE is the **triplet loss** with a margin $m$:

$$
\mathcal{L}_{\text{triplet}} = \max\!\left(0,\ m - s(q, d^+) + s(q, d^-)\right)
$$

where $s$ is cosine similarity. This only considers one negative per query per step. In practice, InfoNCE with large $N$ outperforms triplet loss because it simultaneously considers many negatives, providing a richer gradient signal.

### GNNeg and Denoised Negatives

A subtle failure mode: the mining procedure mislabels genuinely relevant documents as negatives (false negatives). This corrupts training gradients. Denoising strategies include:

- Re-annotating mined negatives with a strong model
- Soft-labeling using a teacher's confidence score
- Filtering via conditional negative sampling (only include a mined negative if its teacher score is below a threshold)

{{fig:emb-contrastive-space-geometry}}

## Architecture: Sentence-Transformers and the BERT-Based Bi-Encoder

The canonical sentence embedding architecture is SBERT (Sentence-BERT, Reimers & Gurevych, 2019), which fine-tuned a BERT-base backbone on sentence pairs using contrastive/siamese networks. Modern models follow the same structure but with stronger backbones and larger training sets.

{{fig:emb-biencoder-detailed}}

Modern choices for the backbone include:

| Model family | Parameters | Embedding dim | Notes |
|---|---|---|---|
| BERT-base | 110M | 768 | Original SBERT backbone |
| RoBERTa-large | 355M | 1024 | Better pretraining, common in early 2020s |
| MPNet | 110M | 768 | Permuted LM, strong on sentence tasks |
| E5 (Wang et al., 2022) | 110M–560M | 768–1024 | Trained on large-scale text pairs |
| GTE (Li et al., 2023) | 110M–7B | 768–3584 | Includes LLM-based variants |
| BGE (Zhang et al., 2023) | 110M–7B | 768–4096 | Strong MTEB performer |
| Nomic Embed | 137M | 768 | Fully open, MTEB competitive |

### Training Data Pipelines

Modern embedding models are trained in two or three stages:

1. **Weakly supervised pre-finetuning** on hundreds of millions of (query, passage) pairs from heterogeneous web sources: QA pairs, title-body pairs from web pages, Reddit question-reply threads, etc.
2. **Supervised fine-tuning** on curated high-quality datasets (Natural Questions, MS MARCO, SNLI, STSb, etc.) with hard negatives.
3. **Task-specific fine-tuning** for specialized domains (legal, biomedical, code).

The `sentence-transformers` library (Reimers & Gurevych, Hugging Face) is the practical starting point for both training and inference.

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load a pre-trained backbone (already fine-tuned for sentence embedding)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Encode a batch of texts -- returns numpy array by default
texts = [
    "What is the capital of France?",
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower is located in Paris.",
]
embeddings = model.encode(texts, normalize_embeddings=True)

print(f"Embedding shape: {embeddings.shape}")  # (3, 384)

# Cosine similarity between query and each document
import numpy as np
query_emb = embeddings[0]
doc_embs  = embeddings[1:]
scores = doc_embs @ query_emb  # dot product of L2-normalized = cosine sim
print(f"Scores: {scores}")     # e.g. [0.71, 0.58] — first doc more relevant
```

## Matryoshka Representation Learning (MRL)

A practical limitation of fixed-dimension embeddings: the 1024-dim model is always 1024-dim, even when you want to store billions of vectors at lower cost or serve low-latency queries that do not need full precision.

**Matryoshka Representation Learning** (Kusupati et al., 2022, *NeurIPS*) trains a single model to produce embeddings that are useful at *multiple* prefix dimensions simultaneously. Like Russian nesting dolls, the first 64 dimensions alone form a useful embedding, as do the first 128, 256, 512, and 1024.

### How MRL Training Works

Add a loss term for each dimensionality granularity in a set $\mathcal{M} = \{32, 64, 128, 256, 512, d\}$:

$$
\mathcal{L}_{\text{MRL}} = \sum_{m \in \mathcal{M}} \lambda_m \cdot \mathcal{L}_{\text{InfoNCE}}\!\left(\mathbf{e}_{:m},\, \mathbf{e}'_{:m}\right)
$$

where $\mathbf{e}_{:m}$ denotes the first $m$ dimensions of the full embedding and $\lambda_m$ are weighting coefficients (often uniform). The total loss is the weighted sum of contrastive losses at each granularity; gradients flow through all prefix slices simultaneously.

The effect: the first few dimensions encode the most discriminative signal; later dimensions refine it. At test time you can truncate to any supported size and re-normalize.

{{fig:emb-matryoshka-nested-dims}}

```python
def matryoshka_infonce_loss(
    query_emb: torch.Tensor,      # (B, D) full dimension
    doc_emb:   torch.Tensor,      # (B, D) full dimension
    dims:      list[int] = None,  # prefix dimensions to train
    temperature: float = 0.05,
    weights:   list[float] = None,
) -> torch.Tensor:
    """
    Matryoshka contrastive loss: InfoNCE at each prefix dimension.
    Gradients flow through all prefix slices simultaneously.
    """
    D = query_emb.size(-1)
    if dims is None:
        dims = [32, 64, 128, 256, D]
    if weights is None:
        weights = [1.0] * len(dims)

    total_loss = torch.tensor(0.0, device=query_emb.device)

    for dim, w in zip(dims, weights):
        # Slice to prefix dimension and re-normalize
        q_slice = F.normalize(query_emb[:, :dim], dim=-1)
        d_slice = F.normalize(doc_emb[:, :dim], dim=-1)

        # Standard InfoNCE at this granularity
        loss_at_dim = infonce_loss(q_slice, d_slice, temperature)
        total_loss = total_loss + w * loss_at_dim

    return total_loss / sum(weights)
```

MRL is now standard in leading open embedding models. OpenAI's text-embedding-3 family supports variable dimensions via MRL-style training. BGE-M3 (Chen et al., 2024) combines MRL with multi-lingual and multi-granularity retrieval.

## Instruction-Following Embeddings

Generic embedding models treat every query identically. But "software developer" is relevant to both a job-listing query and a biography question — the embedding should reflect the task context. **Instruction embeddings** (E5-mistral-7b-instruct, Instructor, GTE-Qwen) prepend a natural-language task instruction to the query at inference time:

```text
Represent the following sentence for searching relevant passages: <query>
Represent this sentence for retrieval:
  Task: Given a question, find the most relevant academic paper.
  Input: <query>
```

The instruction is tokenized and processed by the encoder together with the query; the pool operation then attends over both. Because the transformer is causal or uses a prefix mask, the instruction's context bleeds into the query representation.

This is especially powerful for LLM-based embedders. Models like E5-mistral-7b-instruct (Wang et al., 2023) use a decoder-only LLM backbone and use the last-token representation instead of mean pooling (since left-to-right autoregressive models produce context-rich representations at the final token, not the first).

```python
from sentence_transformers import SentenceTransformer

# Instruction-aware embedding model
model = SentenceTransformer("intfloat/e5-large-v2")

# Prefix 'query:' vs 'passage:' tells the model the role
queries = ["query: How do transformers handle long sequences?"]
docs    = [
    "passage: Transformers scale quadratically with sequence length in attention.",
    "passage: The history of the Transformer architecture dates to 2017.",
]

q_emb = model.encode(queries, normalize_embeddings=True)
d_emb = model.encode(docs,    normalize_embeddings=True)

scores = q_emb @ d_emb.T
print(scores)  # [[0.73, 0.41]] — first document correctly ranked higher
```

The instruction mechanism gives the same frozen backbone very different behaviors for different retrieval tasks without any fine-tuning, a form of lightweight task specification.

## Evaluation: MTEB

The **Massive Text Embedding Benchmark** (MTEB, Muennighoff et al., 2023) is the standard leaderboard for embedding model evaluation. It covers 58 datasets across 8 task types:

| Task type | Example datasets | Metric |
|---|---|---|
| Retrieval | BEIR (MS MARCO, NQ, HotpotQA, …) | nDCG@10 |
| Clustering | ArXiv topic clustering | V-measure |
| Classification | Amazon review sentiment | Accuracy |
| Pair classification | QQP duplicate detection | AP |
| Reranking | AskUbuntu, StackExchange | MAP |
| STS | STS12-16, STSBenchmark | Spearman ρ |
| Summarization | SummEval | Spearman ρ |
| Bitext mining | Tatoeba (multilingual) | F1 |

MTEB's retrieval sub-tasks (the BEIR benchmark, Thakur et al., 2021) are especially important for RAG use cases. Models are evaluated **zero-shot** on held-out domains, measuring generalization rather than in-distribution performance.

### Running MTEB Evaluation

```bash
pip install mteb
```

```python
import mteb
from sentence_transformers import SentenceTransformer

# Load the model as an MTEB-compatible encoder
model_name = "BAAI/bge-small-en-v1.5"
model = mteb.get_model(model_name)

# Run a single retrieval task
tasks = mteb.get_tasks(tasks=["NFCorpus"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")

# results contains nDCG@10 and other metrics per task
```

For a thorough evaluation, run all 58 tasks and report the mean MTEB score. However, for RAG system design, prioritize the BEIR retrieval subset because it directly predicts retrieval quality in your pipeline. Semantic textual similarity (STS) tasks measure something slightly different — fine-grained similarity rather than ranking relevance.

!!! interview "Interview Corner"

    **Q:** A candidate says "I'll just use cosine similarity on BERT embeddings directly for retrieval." What problems do you anticipate, and how would you fix them?

    **A:** Raw BERT embeddings have two major problems for retrieval. First, they are **anisotropic**: without contrastive fine-tuning, embeddings cluster in a narrow cone of the embedding space rather than being uniformly distributed. This means cosine similarity between arbitrary sentences is high (often 0.6–0.9) even for unrelated pairs, making the scores uninformative for ranking. Second, BERT's `[CLS]` token was trained with masked language modeling and next-sentence prediction — neither objective directly encourages semantically similar sentences to be close in vector space.

    The fix is to fine-tune with a contrastive objective (InfoNCE / NTXent) on labeled (query, positive) pairs, which directly trains the similarity space. Models like SBERT, BGE, E5, and GTE do this. After fine-tuning, the embedding space becomes roughly isotropic (embeddings spread across the hypersphere), and similarity scores are meaningful and well-calibrated. In production you would also use a model with MRL support so you can reduce dimensionality to balance cost and quality.

## Practical Training Recipe

Here is a concrete end-to-end training loop for a fine-tuned bi-encoder, illustrating all the moving pieces:

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
from typing import Optional


@dataclass
class BiEncoderConfig:
    model_name: str = "BAAI/bge-base-en-v1.5"
    max_length: int = 512
    embedding_dim: int = 768
    temperature: float = 0.05
    batch_size: int = 64
    learning_rate: float = 2e-5
    warmup_steps: int = 1000
    matryoshka_dims: Optional[list] = None  # e.g. [128, 256, 768]


class EmbeddingModel(torch.nn.Module):
    def __init__(self, config: BiEncoderConfig):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Tokenize, encode, mean-pool, and L2-normalize a list of texts."""
        encoded = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(next(self.parameters()).device)

        # Forward pass through the transformer backbone
        outputs = self.encoder(**encoded)

        # Mean pool over token dimension (excluding padding)
        embeddings = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])

        # L2 normalize
        return F.normalize(embeddings, p=2, dim=-1)

    def forward(self, queries: list[str], docs: list[str]) -> torch.Tensor:
        """Compute InfoNCE loss for a batch of (query, doc) pairs."""
        q_emb = self.encode(queries)
        d_emb = self.encode(docs)

        if self.config.matryoshka_dims is not None:
            return matryoshka_infonce_loss(
                q_emb, d_emb,
                dims=self.config.matryoshka_dims,
                temperature=self.config.temperature,
            )
        else:
            return infonce_loss(q_emb, d_emb, self.config.temperature)


class PairDataset(Dataset):
    """Simple dataset of (query, positive_document) string pairs."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    queries, docs = zip(*batch)
    return list(queries), list(docs)


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for step, (queries, docs) in enumerate(loader):
        optimizer.zero_grad()

        loss = model(queries, docs)
        loss.backward()

        # Gradient clipping prevents spikes, especially important with
        # large batches and small temperatures
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if step % 100 == 0:
            print(f"  Step {step:5d} | loss {loss.item():.4f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")

    return total_loss / len(loader)


# Example usage (pseudo-code, requires actual data and GPU):
# config = BiEncoderConfig(matryoshka_dims=[128, 256, 768])
# model = EmbeddingModel(config).to("cuda")
# dataset = PairDataset(my_training_pairs)
# loader = DataLoader(dataset, batch_size=config.batch_size,
#                     shuffle=True, collate_fn=collate_fn)
# optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
# scheduler = get_linear_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)
```

### Key Training Decisions

**Gradient caching.** With large batch sizes, storing all intermediate activations for both query and document towers can overflow GPU memory. Gradient caching (GradCache, Gao et al.) computes embeddings in smaller sub-batches for the forward pass, then accumulates gradients in a second pass — achieving the statistical effect of a large batch at lower memory cost.

**Multi-GPU synchronization.** With distributed training, in-batch negatives from other GPUs can be gathered via all-gather before computing the loss, effectively multiplying the batch size by the number of GPUs with no per-GPU memory increase. This "all-gather trick" is used in SimCLR, CLIP, and DPR follow-ups.

**Temperature as a learned parameter.** CLIP (Radford et al., 2021) learned the log-temperature as a scalar parameter, allowing the model to adapt the sharpness of its similarity distribution during training. This avoids manual tuning and is increasingly common.

## Bridging to RAG

Dense embeddings are the front door of every Retrieval-Augmented Generation (RAG) system. The full pipeline:

{{fig:emb-rag-pipeline}}

Several design decisions at the embedding layer propagate through the entire system:

**Asymmetric query/document length.** Queries are short (5–20 tokens); documents are long (100–500 tokens). Some models use separate query and document encoders (DPR-style) or a single encoder with different prefix instructions (E5/BGE). Matching the encoder's training distribution to your actual query and document lengths matters.

**Chunking policy.** A document is split into chunks before embedding. The chunk size (128–512 tokens) and overlap determine how much context each embedded unit contains. This is the first point where RAG quality is determined — see [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html).

**Index refresh latency.** When new documents arrive, their embeddings must be computed and inserted into the ANN index. With asynchronous pipelines, there is a window where new content is not yet retrievable. High-throughput embedding inference on GPU batches can reduce this lag to seconds.

**Embedding drift.** If the embedding model is updated (e.g., fine-tuned on domain data), all stored document embeddings become stale and must be recomputed. This "re-indexing" cost — on the order of millions of API calls or hours of GPU time — is a real operational concern in production RAG.

**Token budget vs. retrieved quality.** Retrieving $k=20$ chunks provides more coverage but costs more LLM context tokens. The embedding model's precision (ratio of relevant chunks among top-$k$) directly determines whether context budget is spent on signal or noise. LLMs with long-context windows (see [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html)) relax this tension but do not eliminate it.

For the next stage — indexing these embeddings and querying them efficiently at scale — see [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html). For the full RAG architecture, see [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html).

!!! warning "Common pitfall: using a general embedding model on a specialized domain"

    A general model trained on web text will embed medical or legal text into a space not calibrated for clinical or statutory similarity. The nearest neighbors in the ANN index will be superficially related but factually irrelevant. Always evaluate your embedding model on a small held-out set from your actual domain before committing to it. If performance is poor, fine-tune on domain-specific (query, passage) pairs — even a few thousand pairs of domain examples can significantly improve recall.

!!! tip "Practitioner tip: normalize before storing"

    Always L2-normalize embeddings before inserting them into your vector database. Then use dot-product distance (IP) rather than cosine distance — they are equivalent for normalized vectors, but IP is faster in most ANN libraries (FAISS IVFFlat, Qdrant, Weaviate). Storing un-normalized vectors and using cosine requires an extra normalization step at query time and often uses a slower code path internally.

!!! key "Key Takeaways"

    - Dense bi-encoders embed queries and documents independently; cosine similarity at retrieval time is fast and pre-computable for all documents.
    - The InfoNCE (NT-Xent) loss treats all other documents in the mini-batch as negatives; a temperature $\tau \approx 0.05$ works well for text retrieval.
    - Hard negatives — BM25-mined, ANN-mined, or cross-encoder-filtered — are essential to push past the plateau achieved with random in-batch negatives.
    - Mean pooling over non-padding tokens consistently outperforms CLS pooling for sentence-level tasks; L2-normalize before computing cosine similarity.
    - Matryoshka Representation Learning trains one model to produce useful embeddings at every prefix dimension, enabling adaptive quality/cost trade-offs at inference time.
    - Instruction embeddings prepend task descriptions to queries; LLM-based encoders (E5-mistral, GTE-Qwen) yield strong zero-shot generalization by leveraging decoder-only backbone representations.
    - MTEB is the standard benchmark; for RAG use cases, prioritize the BEIR retrieval sub-tasks (nDCG@10 on MS MARCO, NQ, HotpotQA, etc.).
    - Embedding model drift after fine-tuning requires full re-indexing of stored document embeddings — plan for this operational cost.
    - Hybrid search (dense + BM25) consistently outperforms either method alone; use embeddings as one signal, not the only signal.

!!! sota "State of the Art & Resources (2026)"
    Dense text embeddings have matured from BERT fine-tuning into multi-billion-parameter LLM-based encoders that top the MTEB leaderboard; Matryoshka training and instruction-aware prefixes are now industry standard, and multilingual 100-language models are the default choice for new RAG systems.

    **Foundational work**

    - [Reimers & Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* (2019)](https://arxiv.org/abs/1908.10084) — established the bi-encoder fine-tuning paradigm that all modern embedding models follow.
    - [Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5, 2022)](https://arxiv.org/abs/2212.03533) — showed that large-scale weakly-supervised CCPairs training yields the first dense model to beat BM25 on BEIR without labeled data.
    - [Kusupati et al., *Matryoshka Representation Learning* (NeurIPS 2022)](https://arxiv.org/abs/2205.13147) — introduced nested prefix-dimension training, now the standard technique for adaptive-cost embeddings.

    **Recent advances (2023–2026)**

    - [Chen et al., *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation* (2024)](https://arxiv.org/abs/2402.03216) — single model supporting dense, sparse, and multi-vector retrieval in 100+ languages with up to 8 192-token inputs.
    - [BehnamGhader et al., *LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders* (COLM 2024)](https://arxiv.org/abs/2404.05961) — converts decoder-only LLMs into strong text encoders via bidirectional attention + masked next-token prediction, reaching SOTA on MTEB using only public data.
    - [Lee et al., *NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models* (ICLR 2025 Spotlight)](https://arxiv.org/abs/2405.17428) — latent-attention pooling and two-stage training bring Llama-3-based embeddings to #1 on MTEB retrieval.
    - [Muennighoff et al., *MTEB: Massive Text Embedding Benchmark* (EACL 2023)](https://arxiv.org/abs/2210.07316) — the 58-dataset, 8-task benchmark that is now the standard evaluation suite for any embedding model.
    - [Thakur et al., *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models* (NeurIPS 2021)](https://arxiv.org/abs/2104.08663) — 18-domain zero-shot retrieval benchmark that exposed brittle generalization in dense models and remains the primary RAG-relevant sub-benchmark inside MTEB.

    **Open-source & tools**

    - [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) — the canonical Python library for training and serving bi-encoder embedding models, maintained by Hugging Face.
    - [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) — BAAI's one-stop toolkit for BGE models, covering fine-tuning, hard-negative mining, reranking, and RAG integration.
    - [embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb) — the official MTEB evaluation library; run any model against all 58 tasks with a single Python call.

    **Go deeper**

    - [MTEB Leaderboard (Hugging Face)](https://huggingface.co/spaces/mteb/leaderboard) — live rankings of 1 000+ embedding models across MTEB tasks; filter by language, task type, and model size.

## Further Reading

- Reimers & Gurevych, **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**, EMNLP 2019 — the foundational bi-encoder fine-tuning paper.
- Karpukhin et al., **Dense Passage Retrieval for Open-Domain Question Answering** (DPR), EMNLP 2020 — established in-batch negative training for retrieval.
- Xiong et al., **Approximate Nearest Neighbor Negative Contrastive Estimation for Dense Text Retrieval** (ANCE), ICLR 2021 — ANN-mined hard negatives.
- Gao et al., **Simcse: Simple Contrastive Learning of Sentence Embeddings**, EMNLP 2021 — dropout as a data augmentation for contrastive pre-training.
- Thakur et al., **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**, NeurIPS 2021 — the standard retrieval evaluation suite.
- Muennighoff et al., **MTEB: Massive Text Embedding Benchmark**, EACL 2023 — comprehensive multi-task embedding leaderboard.
- Kusupati et al., **Matryoshka Representation Learning**, NeurIPS 2022 — multi-granularity embedding training.
- Wang et al., **Text Embeddings by Weakly-Supervised Contrastive Pre-training** (E5), 2022 — large-scale weakly supervised bi-encoder pre-training.
- `sentence-transformers` library (Reimers, Hugging Face) — the practical go-to for embedding model training and inference.
- `MTEB` Python package (Hugging Face) — easy reproducible evaluation of any embedding model.

## Exercises

**1.** *(Conceptual.)* A colleague proposes replacing your dense bi-encoder retriever with a cross-encoder, arguing that cross-encoders score relevance more accurately. Explain (a) why the bi-encoder, not the cross-encoder, is the design that makes first-stage retrieval over millions of documents tractable, and (b) what specifically breaks if you try to use a cross-encoder as the first-stage retriever. Then describe the standard way both architectures are used together in a production RAG pipeline.

??? note "Solution"
    **(a) Why the bi-encoder scales.** In a bi-encoder the query and each document are encoded *independently* by the same transformer, and the relevance score is a plain dot product (or cosine similarity) of the two pooled vectors. Independence is the load-bearing property: because a document's embedding does not depend on the query, every document embedding can be computed *offline*, stored once, and placed in an ANN index. At query time you embed only the query (one forward pass) and then do fast vector lookups. The transformer runs exactly once per query, no matter how large the corpus.

    **(b) Why a cross-encoder cannot be the first-stage retriever.** A cross-encoder runs the query and a document *through the same forward pass with joint attention*, producing one score per (query, document) pair. The score cannot be decomposed into a query vector and a document vector, so nothing can be precomputed. To rank a corpus of $n$ documents you would need $n$ full transformer forward passes *per query* — $O(n)$ cost. For $n$ in the millions that is far too slow and cannot be served from an ANN index at all.

    **Used together.** The bi-encoder is the cheap first stage: it retrieves a short candidate list (e.g., top-$k$ = 50-100) from the full corpus using precomputed embeddings and ANN search. The cross-encoder is then applied only to those few candidates as a *reranker*, spending its expensive joint-attention scoring where accuracy matters most. This two-stage retrieve-then-rerank design gets bi-encoder scalability with cross-encoder precision on the final ordering.

**2.** *(Quantitative.)* A transformer produces the following final-layer token representations for a 3-token sequence with hidden dimension $d = 2$: $\mathbf{h}_1 = [1, 3]$, $\mathbf{h}_2 = [3, 1]$, $\mathbf{h}_3 = [10, 0]$. The attention mask is $\mathbf{m} = [1, 1, 0]$ (the third token is padding). (a) Compute the mean-pooled sentence embedding using the masked formula from the chapter, then L2-normalize it. (b) Now recompute the mean *ignoring the mask* (averaging all three tokens) and normalize. (c) Report the cosine similarity between the two resulting unit vectors and state the lesson.

??? note "Solution"
    **(a) Masked mean pool.** Using $\mathbf{e} = \frac{\sum_i m_i \mathbf{h}_i}{\sum_i m_i}$, only tokens 1 and 2 count:

    $$
    \sum_i m_i \mathbf{h}_i = [1,3] + [3,1] = [4, 4], \qquad \sum_i m_i = 2
    $$

    $$
    \mathbf{e} = [4,4]/2 = [2, 2]
    $$

    Norm $= \sqrt{2^2 + 2^2} = \sqrt{8} = 2.8284$, so the normalized embedding is $[0.7071,\ 0.7071]$.

    **(b) Unmasked mean.** Averaging all three tokens including the padding vector $[10, 0]$:

    $$
    \mathbf{e}' = \frac{[1,3] + [3,1] + [10,0]}{3} = \frac{[14, 4]}{3} = [4.667,\ 1.333]
    $$

    Norm $= \sqrt{4.667^2 + 1.333^2} = \sqrt{21.78 + 1.78} = \sqrt{23.56} = 4.8535$, so the normalized vector is $[0.9615,\ 0.2747]$.

    **(c) Cosine similarity between them.** Both are unit vectors, so cosine similarity is just their dot product:

    $$
    0.7071 \cdot 0.9615 + 0.7071 \cdot 0.2747 = 0.6799 + 0.1942 = 0.874
    $$

    The two embeddings point in noticeably different directions (cosine $0.874$, not $1.0$) purely because one padding token leaked in. **Lesson:** padding tokens carry no meaning but do carry activations; if you forget the mask they corrupt the pooled vector. This is why `mean_pool` multiplies by the attention mask and divides by the count of *real* tokens.

**3.** *(Quantitative.)* You are training with the symmetric in-batch InfoNCE loss from the chapter at temperature $\tau = 0.1$ and batch size $B = 2$. The similarity matrix of L2-normalized embeddings is

$$
\text{sim} = \begin{bmatrix} \mathbf{e}_{q_0}\cdot\mathbf{e}_{d_0} & \mathbf{e}_{q_0}\cdot\mathbf{e}_{d_1} \\ \mathbf{e}_{q_1}\cdot\mathbf{e}_{d_0} & \mathbf{e}_{q_1}\cdot\mathbf{e}_{d_1} \end{bmatrix} = \begin{bmatrix} 0.8 & 0.2 \\ 0.3 & 0.9 \end{bmatrix}
$$

Compute the symmetric loss $(\mathcal{L}_{q\to d} + \mathcal{L}_{d\to q})/2$ that `infonce_loss` returns. (Useful values: $e^2 = 7.389$, $e^3 = 20.086$, $e^8 = 2980.96$, $e^9 = 8103.08$.)

??? note "Solution"
    Divide every entry by $\tau = 0.1$ to get logits, then take a cross-entropy where the diagonal is the correct class.

    **Query-to-document direction** (softmax over each *row*, target = row index):

    Row 0, logits $[8, 2]$, target class 0:
    $$
    p = \frac{e^8}{e^8 + e^2} = \frac{2980.96}{2980.96 + 7.389} = 0.99753, \quad \ell_0 = -\ln 0.99753 = 0.00247
    $$

    Row 1, logits $[3, 9]$, target class 1:
    $$
    p = \frac{e^9}{e^3 + e^9} = \frac{8103.08}{20.086 + 8103.08} = 0.99753, \quad \ell_1 = 0.00248
    $$

    $\mathcal{L}_{q\to d} = (0.00247 + 0.00248)/2 = 0.00248$.

    **Document-to-query direction** (softmax over each *column* of sim, i.e. rows of sim$^\top$):

    Column 0, logits $[8, 3]$, target class 0:
    $$
    p = \frac{e^8}{e^8 + e^3} = \frac{2980.96}{2980.96 + 20.086} = 0.99331, \quad \ell_0 = 0.00672
    $$

    Column 1, logits $[2, 9]$, target class 1:
    $$
    p = \frac{e^9}{e^2 + e^9} = \frac{8103.08}{7.389 + 8103.08} = 0.99909, \quad \ell_1 = 0.00091
    $$

    $\mathcal{L}_{d\to q} = (0.00672 + 0.00091)/2 = 0.00381$.

    **Symmetric loss:**
    $$
    \frac{\mathcal{L}_{q\to d} + \mathcal{L}_{d\to q}}{2} = \frac{0.00248 + 0.00381}{2} \approx 0.0031
    $$

    The loss is tiny because both queries already have their positive as the clear argmax. Note the two directions are *not* equal: query 0 competes against a weak off-diagonal ($0.2$) in its row but document 0 competes against a stronger off-diagonal ($0.3$) in its column, so the $d\to q$ term is larger. That asymmetry is exactly why the chapter's loss averages both directions.

**4.** *(Implementation.)* The chapter implements `mean_pool` but only mentions max pooling in prose. Implement a masked `max_pool(token_embeddings, attention_mask)` with the same signature and shape conventions as `mean_pool` (input `(B, L, H)` and `(B, L)`, output `(B, H)`). It must take the element-wise maximum over *real* token positions only. Explain why simply calling `.max(dim=1)` on the raw tensor is wrong, and how you handle padding correctly.

??? note "Solution"
    A naive `token_embeddings.max(dim=1)` includes padding positions. If a padding token happens to hold a large activation in some dimension, it wins the max and pollutes the pooled vector — the same failure mode as unmasked mean pooling in Exercise 2. The fix is to set padding positions to $-\infty$ *before* taking the max, so they can never be selected.

    ```python
    import torch


    def max_pool(token_embeddings: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Element-wise max over real (non-padding) token positions.

        Args:
            token_embeddings: shape (batch, seq_len, hidden_dim)
            attention_mask:   shape (batch, seq_len), 1 for real tokens

        Returns:
            sentence_embeddings: shape (batch, hidden_dim)
        """
        # Expand mask to hidden_dim so it broadcasts over features
        mask_expanded = attention_mask.unsqueeze(-1).bool()  # (B, L, 1)

        # Send padding positions to -inf so they never win the max
        masked = token_embeddings.masked_fill(~mask_expanded, float("-inf"))

        # Max over the sequence (token) dimension
        return masked.max(dim=1).values  # (B, H)
    ```

    Using `float("-inf")` (rather than a large negative constant) guarantees correctness even when real activations are themselves very negative. As with `mean_pool`, you would L2-normalize the result before computing cosine similarity. (Edge case: a row that is *all* padding would return $-\infty$; in practice every sequence has at least one real token, matching the `clamp(min=1e-9)` guard in `mean_pool`.)

**5.** *(Quantitative + conceptual.)* You store embeddings for a 100-million-chunk corpus in an ANN index using `float32` (4 bytes per dimension). Your model was trained with Matryoshka Representation Learning and produces full 768-dimensional embeddings. (a) Compute the raw vector storage for the full 768-dim embeddings, and for MRL-truncated 128-dim embeddings. (b) What is the memory-savings ratio? (c) After truncating each stored vector to its first 128 dimensions, why must you re-normalize before doing cosine/dot-product search, and why does MRL make the 128-dim prefix usable at all (unlike truncating an ordinary embedding)?

??? note "Solution"
    **(a) Storage.** Bytes per vector = dimensions $\times$ 4.

    Full 768-dim: $768 \times 4 = 3072$ bytes/vector. For $10^8$ vectors:
    $$
    3072 \times 10^8 = 3.072 \times 10^{11}\ \text{bytes} = 307.2\ \text{GB}
    $$

    Truncated 128-dim: $128 \times 4 = 512$ bytes/vector. For $10^8$ vectors:
    $$
    512 \times 10^8 = 5.12 \times 10^{10}\ \text{bytes} = 51.2\ \text{GB}
    $$

    **(b) Savings ratio.** $307.2 / 51.2 = 6\times$ (exactly the dimension ratio $768/128 = 6$, since bytes scale linearly with dimension). You cut index memory sixfold.

    **(c) Re-normalization and why MRL works.** Slicing off the last 640 dimensions changes a vector's L2 norm: the first 128 components of a unit 768-vector generally have norm well below 1. The chapter's practitioner tip relies on stored vectors being unit-norm so that a dot product *equals* cosine similarity; if you skip re-normalization after truncation, the "cosine" scores are distorted by each vector's leftover prefix magnitude and rankings degrade. So you re-normalize the 128-dim prefix to unit length, restoring dot-product = cosine.

    MRL makes the prefix *meaningful* in the first place. During training MRL adds an InfoNCE term at each prefix size ($32, 64, 128, \dots, d$), so gradients force the earliest dimensions to already carry the most discriminative signal — the first 128 dims form a standalone useful embedding. Truncating an *ordinary* (non-MRL) embedding has no such guarantee: its 768 dimensions were only ever trained to be discriminative *together*, so an arbitrary 128-dim prefix can be nearly useless.

**6.** *(Implementation, harder.)* The chapter's `infonce_loss` uses only in-batch negatives. Extend it to `infonce_with_hard_negatives(query_emb, pos_emb, neg_emb, temperature)` where each query has one positive (`pos_emb`, shape `(B, D)`) and its own set of $K$ mined hard negatives (`neg_emb`, shape `(B, K, D)`), all L2-normalized. Build the per-query logit row as `[positive, K hard negatives]`, make the positive the target class, and return the mean cross-entropy. Explain how this connects to the "BM25-mined / ANN-mined negatives" discussion in the chapter, and why hard negatives give a stronger gradient than random in-batch negatives.

??? note "Solution"
    ```python
    import torch
    import torch.nn.functional as F


    def infonce_with_hard_negatives(
        query_emb: torch.Tensor,   # (B, D), L2-normalized
        pos_emb:   torch.Tensor,   # (B, D), L2-normalized
        neg_emb:   torch.Tensor,   # (B, K, D), L2-normalized
        temperature: float = 0.05,
    ) -> torch.Tensor:
        """
        InfoNCE where each query has one positive and K per-query hard negatives.
        The positive is class 0 in each row of logits.
        """
        B, D = query_emb.shape

        # Positive similarity: elementwise dot -> (B, 1)
        pos_logits = (query_emb * pos_emb).sum(dim=-1, keepdim=True)

        # Hard-negative similarities: batched matmul (B,K,D) x (B,D,1) -> (B,K)
        neg_logits = torch.bmm(neg_emb, query_emb.unsqueeze(-1)).squeeze(-1)

        # Concatenate: (B, 1 + K); column 0 is the positive
        logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature

        # Target class is 0 for every query
        labels = torch.zeros(B, dtype=torch.long, device=query_emb.device)

        return F.cross_entropy(logits, labels)


    # ---- Sanity check ----
    if __name__ == "__main__":
        torch.manual_seed(0)
        B, K, D = 4, 3, 768
        q = F.normalize(torch.randn(B, D), dim=-1)
        pos = F.normalize(torch.randn(B, D), dim=-1)
        neg = F.normalize(torch.randn(B, K, D), dim=-1)
        print(infonce_with_hard_negatives(q, pos, neg, 0.05).item())  # ~ log(1+K)
    ```

    At random initialization the loss is about $\log(1 + K)$ (uniform over the $1 + K$ candidates), just as plain in-batch InfoNCE starts near $\log B$.

    **Connection to the chapter.** `neg_emb` is exactly where you feed the mined negatives the chapter describes: BM25-mined passages (high lexical overlap, wrong meaning), ANN/dense-mined passages from an earlier checkpoint (ANCE-style), or cross-encoder-filtered hardest negatives. You would embed those mined passages and pass them as the `(B, K, D)` tensor.

    **Why hard negatives help.** Random in-batch negatives are usually topically unrelated, so their similarity to the query is already low and $\partial \mathcal{L}/\partial \theta$ is nearly zero — little to learn. Hard negatives sit close to the positive in embedding space, so they produce large logits, a non-trivial softmax mass on the wrong class, and therefore a large gradient that forces the model to carve out fine-grained distinctions. That is precisely the plateau-breaking effect the chapter attributes to hard-negative mining. (In practice you often combine both: concatenate these per-query hard negatives with the in-batch negatives for an even richer denominator.)
