# 13.4 ML System Design: Worked Cases

The previous chapter, [ML System Design: A Framework](../13-interview-prep/03-ml-system-design-framework.html), gave you a repeatable seven-step skeleton: **clarify the problem, frame it as ML, define the data, choose the model, design the system, pick the metrics, then stress-test for failure.** A framework is necessary but not sufficient. In the room, the interviewer wants to see you *use* it — to watch you turn a vague prompt ("design YouTube recommendations") into candidate generation, ranking, feature stores, training cadence, and a serving topology with latency budgets, all while narrating tradeoffs out loud.

This chapter is the centerpiece. We work six designs end-to-end:

1. A **recommendation / ranking** system (the classic two-tower + ranker).
2. A **search / semantic retrieval** system.
3. An **LLM serving platform**.
4. A **RAG assistant**.
5. A **content-moderation** system.
6. A **fine-tuning / RLHF pipeline**.

Each one applies the same framework, so you internalize the *motion*, not a memorized script. For every case we plug in real magnitudes — QPS, embedding dimensions, GPU counts, memory budgets — because an answer that says "we'd use a vector DB" without a sense of scale reads as hand-waving. We will lean on sibling chapters for the deep mechanism (attention, PagedAttention, RLHF math) and focus here on *composition*: how the pieces snap together into a system that hits a latency target and a quality bar.

A note on pacing for the interview itself: you have roughly 35–45 minutes. Spend the first 5 clarifying, sketch the high-level boxes by minute 12, then go deep on the one or two components the interviewer steers you toward. Do not try to draw all six of these in one sitting — pick the spine and earn the right to detail by hitting the milestones.

## The Anatomy Every Design Shares

Before the cases, internalize the shape almost all of these systems take. Large-scale ML systems are **funnels**: cheap, high-recall stages at the top narrow billions of candidates down to a handful that an expensive, high-precision model scores at the bottom.

```text
                billions                thousands              hundreds            ~10
   corpus  ──►  CANDIDATE   ──────────►  PRE-RANK / ──────────► RANK  ──────────►  RE-RANK
               GENERATION               FILTERING             (heavy model)       (business
               (ANN, rules)             (light model)                              rules,
                                                                                   diversity)
   cost/item:   ~microseconds            ~10s of µs            ~milliseconds       ~negligible
```

The same funnel describes recommendation (candidate gen → ranking), search (retrieval → reranking), and RAG (retrieval → LLM). The reason is economic: a heavy model that costs 5 ms per item cannot be run over a 10-million-item corpus inside a 200 ms budget ($10^7 \times 5\text{ ms} = 14$ hours), but it can run over the 500 survivors a cheap retriever hands it ($500 \times 5\text{ ms} = 2.5$ s — still too slow serially, so we batch on a GPU and get it to ~10 ms). Whenever you start a design, ask: *what is the corpus size, what is my latency budget, and where do I cut the funnel?*

The second universal pattern is the **offline/online split**. Anything that can be precomputed — item embeddings, ANN indexes, user history features — is computed in batch and stored; the online path does as little as possible. Hold these two patterns in your head and every case below becomes a variation on a theme.

## Case 1 — Recommendation / Ranking System

**Prompt:** "Design the home-feed recommender for a video platform with 500 M daily active users (DAU) and a catalog of 1 B videos."

### Clarify and frame

Ask the cheap questions first. *What are we optimizing — watch time, click-through, long-term retention? Is this the home feed (no query) or search? What's the latency budget? Cold-start for new users and new videos?* Suppose the interviewer says: home feed, optimize for **long-term engagement** (a proxy: expected watch time with a satisfaction penalty for clickbait), p99 latency budget **200 ms**, must handle new videos within minutes.

Frame it as ML: we are estimating, for a (user, video, context) triple, a utility $U$ that combines several heads — $P(\text{click})$, $E[\text{watch time} \mid \text{click}]$, $P(\text{like})$, $P(\text{not dissatisfied})$ — into a single rankable score. This **multi-task** framing is the industry standard (it mirrors the YouTube ranking paper, Covington et al., and the multi-gate MoE work, Zhao et al.).

### The two-stage funnel

**Candidate generation** reduces 1 B videos to ~500. We use several retrievers in parallel and union the results:

- A **two-tower** model: a user tower and an item tower each map to a $d=128$ embedding; relevance is the dot product. Item embeddings are precomputed nightly and indexed in an ANN structure (HNSW or IVF-PQ — see [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html)). At request time we embed the user once and do an approximate top-$k$ search.
- **Co-visitation / graph** retrievers ("users who watched X watched Y").
- **Fresh / trending** retrievers for cold-start items that have no learned embedding yet.

The two-tower model is the heart of candidate generation. Its critical property is that the **user and item never interact until the final dot product**, so item embeddings can be precomputed and the user side is a single forward pass. Here is the model and the in-batch softmax loss that makes it work:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tower(nn.Module):
    """A generic tower: features -> L2-normalized d-dim embedding."""
    def __init__(self, in_dim, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, d),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)   # unit vectors => dot product = cosine

class TwoTower(nn.Module):
    def __init__(self, user_dim, item_dim, d=128):
        super().__init__()
        self.user_tower = Tower(user_dim, d)
        self.item_tower = Tower(item_dim, d)
        # temperature sharpens/softens the softmax over candidates
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, user_feats, item_feats):
        u = self.user_tower(user_feats)     # (B, d)
        v = self.item_tower(item_feats)     # (B, d)
        return u, v

def in_batch_softmax_loss(u, v, log_temp):
    """Sampled-softmax with in-batch negatives.
    Row i's positive is column i; every other item in the batch is a negative.
    This is why large batches matter: more negatives => better retrieval.
    """
    logits = (u @ v.t()) / log_temp.exp()        # (B, B) similarity matrix
    labels = torch.arange(u.size(0), device=u.device)
    return F.cross_entropy(logits, labels)
```

One subtlety worth raising unprompted: **in-batch negatives are biased toward popular items** (popular items appear as positives more often, hence as negatives more often), so we apply a **logQ correction** — subtract $\log Q(\text{item})$, the sampling probability, from each logit. Mentioning this signals you've actually built one of these.

**Ranking** scores the ~500 survivors with a heavy model that *can* let user and item features interact (cross features, attention over the user's recent history). This is where multi-task heads live:

```python
class MultiTaskRanker(nn.Module):
    """Shared bottom -> per-task heads. A real system uses MMoE
    (multi-gate mixture-of-experts) so tasks can share or specialize."""
    def __init__(self, feat_dim, hidden=1024):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.click_head   = nn.Linear(hidden, 1)   # P(click)
        self.watch_head   = nn.Linear(hidden, 1)   # E[watch | click], log-seconds
        self.satisfy_head = nn.Linear(hidden, 1)   # P(not dissatisfied)

    def forward(self, x):
        h = self.shared(x)
        return {
            "click":   torch.sigmoid(self.click_head(h)),
            "watch":   self.watch_head(h),            # regression in log space
            "satisfy": torch.sigmoid(self.satisfy_head(h)),
        }

def combined_utility(out, w=(1.0, 1.0, 0.5)):
    """Weighted product / sum that ops teams tune via A/B tests."""
    wc, ww, ws = w
    # P(click) * exp(E[log watch]) * P(satisfy), in log space for stability
    return (wc * torch.log(out["click"] + 1e-6)
            + ww * out["watch"]
            + ws * torch.log(out["satisfy"] + 1e-6))
```

### Features, data, and the training/serving skew trap

Features fall into user (history embeddings, demographics), item (creator, topic, age, historical CTR), and context (time of day, device). The **feature store** serves these online with low latency and computes them offline for training from the *same* logic — otherwise you get **training/serving skew**, the single most common production bug in recommenders. The canonical fix is **log-and-train**: log the exact features served at inference time, then train on those logs, guaranteeing the distributions match.

Labels come from logged user interactions, which introduces **position bias** (higher-ranked items get more clicks regardless of quality) and **feedback loops** (the model only sees what it already showed). Counter position bias by training a small "position model" and serving with position fixed; counter the feedback loop by injecting exploration (epsilon-random candidates or Thompson sampling on the ranker).

### Serving topology and budget

```text
request ─► feature fetch (10 ms) ─► candidate gen (ANN, 20 ms) ─►
           ranker batch inference on GPU (40 ms) ─► re-rank + diversity (5 ms) ─► response
```

!!! example "Worked example: does the budget close?"

    Catalog 1 B videos, two-tower $d=128$, float32. Item index size: $10^9 \times 128 \times 4\text{ bytes} = 512$ GB. That will not fit one machine's RAM comfortably, so we **shard the ANN index** across, say, 16 nodes (32 GB each) or quantize to int8 ($128$ GB, fits a big node) — a real PQ index compresses far more. Candidate gen does an ANN search returning 500 in ~20 ms.

    Ranking: 500 candidates, batched into one GPU forward pass. A ranker MLP of ~5 M params at 500 rows is a few hundred MFLOPs — trivially under 10 ms on a modern accelerator; budget 40 ms with feature gathering and serialization. Feature fetch 10 ms, re-rank 5 ms. **Total ≈ 75 ms**, comfortably inside the 200 ms p99 even with network and tail variance. The bottleneck is not compute; it is the **feature fetch fan-out** and ANN tail latency, so we cache hot user features and replicate the index for QPS headroom.

    QPS: 500 M DAU, suppose each opens the app 5×/day with feed refreshes ≈ 10 ranking calls per visit ⇒ $5 \times 10^8 \times 5 \times 10 / 86400 \approx 2.9 \times 10^5$ QPS average, with peaks 3–5× higher. At ~1k QPS per ranker replica we need on the order of a few thousand replicas at peak — a real number to state.

### Metrics

Offline: AUC / log-loss per head, and **recall@k** for candidate generation. Online (the ones that decide launches): the north-star (watch time per user, 7-day retention) plus guardrails (dissatisfaction rate, diversity, creator fairness). Always evaluate via **A/B test** — offline AUC gains routinely fail to translate, which is itself a great thing to say.

## Case 2 — Search / Semantic Retrieval

**Prompt:** "Design a search system over 100 M documents that understands meaning, not just keywords."

### Frame and the hybrid insight

Pure keyword search (BM25) is unbeatable for exact matches, rare terms, and proper nouns; pure semantic (dense) search wins on paraphrase and intent. The right answer is almost always **hybrid**: run both and fuse. Framing the task: given a query, return a ranked list maximizing relevance (graded, judged by humans or click models), under a latency budget of ~150 ms.

We reuse the funnel: **retrieve** (BM25 + dense, fuse) → **rerank** (cross-encoder) → return.

### Dense retrieval and the bi-encoder vs cross-encoder distinction

This distinction is the crux of the whole design and a frequent interview probe (see [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html)):

- A **bi-encoder** encodes query and document *separately* into vectors; similarity is a dot product. Documents are precomputed and ANN-indexed ⇒ cheap, scalable, but it can't model fine-grained query–document interaction.
- A **cross-encoder** concatenates query and document and runs them through a transformer *together*, so every query token attends to every document token ⇒ far more accurate, but it cannot be precomputed (the query is part of the input) and costs a full forward pass *per document*. Use it only on the top ~100 the retriever returns.

```python
# Reciprocal Rank Fusion (RRF): a parameter-light, robust way to fuse
# rankings from heterogeneous retrievers (BM25 ranks + dense ranks).
def reciprocal_rank_fusion(rank_lists, k=60):
    """rank_lists: list of lists of doc_ids, each ordered best-first."""
    scores = {}
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking):        # rank starts at 0
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

# Usage: fuse keyword and dense candidate sets, then rerank the top of the fused list.
fused = reciprocal_rank_fusion([bm25_results, dense_results])
top100 = fused[:100]
# reranked = cross_encoder.rank(query, top100)  # heavy model, top-100 only
```

RRF is worth knowing because it needs no score normalization across systems whose scores live on incomparable scales — it only uses ranks. State it; interviewers like it.

### Indexing pipeline and freshness

Offline: chunk documents (see [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html)), embed each chunk with the bi-encoder, build an HNSW index, and build an inverted index for BM25. Freshness matters: a news search must index new documents in seconds. Architect a **two-tier index** — a large, immutable, optimized base index rebuilt nightly plus a small, mutable "fresh" index that absorbs new documents in near real time and is merged periodically. Queries hit both and merge results. This LSM-tree-like pattern recurs in every real search system.

!!! tip "Practitioner tip"

    Embedding model versioning is a silent killer. When you upgrade the bi-encoder, *every* document vector must be re-embedded, and you cannot mix vectors from two model versions in one index (their geometries differ). Plan for a full backfill and a dual-index cutover. Store the embedding model version alongside each vector so you can detect mismatches.

### Evaluation

Use graded relevance and rank-aware metrics: **NDCG@10** (rewards putting the most relevant docs highest), **MRR** (first relevant result), and recall@k for the retrieval stage in isolation. Build a judged eval set of (query, doc, grade) triples; supplement with online interleaving experiments, which compare two rankers within a single result list and are far more sensitive than A/B for ranking changes.

!!! interview "Interview Corner"

    **Q:** Your dense retriever has great recall but users complain that searches for exact error codes like "ORA-00942" return semantically-similar-but-wrong results. What's happening and how do you fix it?

    **A:** Dense embeddings smear rare tokens and exact identifiers into a fuzzy neighborhood — they're built for semantic similarity, not lexical exactness, so "ORA-00942" lands near other Oracle errors. The fix is hybrid retrieval: keep the dense path for intent but add a BM25 / exact-match path that nails rare literal tokens, and fuse with RRF. Optionally boost documents containing the exact query string, and route queries that look like identifiers (regex on code-like tokens) more heavily to the lexical retriever. This is the canonical argument for *why* hybrid beats pure-dense in production.

## Case 3 — LLM Serving Platform

**Prompt:** "Design a platform to serve a 70 B-parameter chat model to external developers via an API, with strict latency SLAs."

This case leans heavily on Part VII; here we focus on the *system* around the engine. See [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html) and [Designing an LLM Serving System](../12-production-mlops/01-serving-system-design.html) for mechanism.

### Clarify the SLAs and the two-phase reality

The key clarification: LLM latency is **two numbers**, not one. **Time-to-first-token (TTFT)** is dominated by *prefill* (compute-bound, processes the whole prompt at once). **Inter-token latency (ITL)** is dominated by *decode* (memory-bandwidth-bound, one token at a time, reading the entire KV cache every step). Suppose the SLA is TTFT < 500 ms and ITL < 50 ms (≈ 20 tokens/s, comfortably faster than reading speed).

### Memory budget — the gate on everything

A 70 B model in bf16 needs $70 \times 10^9 \times 2 = 140$ GB just for weights — already more than a single 80 GB GPU. So we need **tensor parallelism** across at least 2 GPUs for weights, and more for the KV cache and activation headroom (see [Multi-GPU & Multi-Node Inference](../07-inference-serving/11-multi-gpu-inference.html)).

!!! example "Worked example: KV-cache budget sets the batch size"

    Take a 70 B model with 80 layers, 64 attention heads (using GQA with 8 KV heads), head dim 128. Per token, the KV cache stores K and V for each KV head across all layers:

    $$
    \text{bytes/token} = 2 \times n_\text{layers} \times n_\text{kv heads} \times d_\text{head} \times \text{bytes}
    = 2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \text{ bytes} \approx 0.31 \text{ MB}.
    $$

    For a context of 8k tokens, one sequence's KV cache is $8192 \times 0.31 \approx 2.5$ GB. After 140 GB of weights split over (say) 4× 80 GB GPUs (320 GB total, 35 GB/GPU for weights), we have on the order of $320 - 140 = 180$ GB left for KV cache and activations. That allows roughly $180 / 2.5 \approx 70$ concurrent 8k-context sequences — *that* number is the engine's max batch size, and it directly sets throughput. Without GQA (64 KV heads instead of 8) the per-token cost is 8× larger and we'd fit ~9 sequences. **This is why GQA exists**, and stating the arithmetic is gold in an interview.

### The serving engine and platform

Use a continuous-batching engine (vLLM / SGLang) with **PagedAttention** so the KV cache is allocated in non-contiguous blocks, eliminating fragmentation (see [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html) and [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)). Around the engine:

```text
                       ┌─────────── control plane ───────────┐
client ─► API gateway ─► auth / rate-limit / quota ─► router ─► [ engine replicas ]
          (TLS, keys)    (token-bucket per key)      (load-     vLLM, TP=4, paged KV
                                                       aware)    continuous batching
                                                                       │
                              metrics / billing / abuse  ◄─────────────┘
```

Design decisions to narrate:

- **Routing.** Route by *load*, not round-robin: a replica's queue depth and free KV blocks predict latency better than request count. Use **prefix-cache-aware routing** (SGLang's RadixAttention, see [SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html)) so requests sharing a long system prompt land on the replica that already has those KV blocks cached — a large win for chat with shared system prompts.
- **Disaggregation.** Prefill (compute-bound) and decode (bandwidth-bound) interfere when colocated: a long prefill stalls everyone's decode. **Disaggregate** them onto separate pools and stream the KV cache between them (see [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html)). Or use **chunked prefill** to interleave.
- **Speculative decoding** to cut ITL: a small draft model proposes several tokens, the big model verifies them in one pass (see [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html)).
- **Quantization** (FP8/INT8 weights and KV cache) to fit more on each GPU and raise the batch ceiling (see [Quantization II](../04-kernels-efficiency/08-quantization-formats-qat.html)).

### Capacity and cost

```python
def replicas_needed(qps, avg_output_tokens, tokens_per_sec_per_replica):
    """Decode throughput is the usual bottleneck for chat workloads.
    A replica producing T tokens/s (summed over its batch) serves
    QPS = T / avg_output_tokens requests per second."""
    per_replica_qps = tokens_per_sec_per_replica / avg_output_tokens
    return qps / per_replica_qps

# e.g. 2000 QPS, 300 output tokens each, replica does 3000 tok/s aggregate
print(replicas_needed(2000, 300, 3000))   # -> ~200 replicas
```

The lever that matters most for cost is **aggregate tokens/sec per GPU**, which continuous batching maximizes by keeping the GPU saturated. State the autoscaling signal: scale on **queue wait time / KV-cache utilization**, not CPU — GPU servers look idle on CPU while the GPU is pinned. See [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html).

## Case 4 — RAG Assistant

**Prompt:** "Design an enterprise assistant that answers questions over a company's 10 M internal documents, with citations and no hallucinated facts."

RAG composes Case 2 (retrieval) with Case 3 (serving). The design discipline is in the *glue* and the *failure modes*. See [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html).

### Pipeline

```text
query ─► query rewrite/expand ─► hybrid retrieve (BM25 + dense) ─► rerank (cross-enc)
      ─► select top-k chunks under token budget ─► build prompt (with citations)
      ─► LLM generate ─► post-check (grounding/citation verify) ─► answer
```

Each stage earns its place:

- **Query rewriting.** Conversational queries ("what about its pricing?") are under-specified; rewrite using chat history into a standalone query before retrieval, or generate multiple sub-queries for complex questions.
- **Retrieval + rerank.** Exactly Case 2. Chunking strategy (see [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html)) is the highest-leverage knob: too small and you lose context, too large and you dilute relevance and waste the LLM's token budget.
- **Context assembly under budget.** You cannot stuff all retrieved chunks in. Pack the highest-reranked chunks until you hit a budget (e.g. 4k tokens of context), deduplicate near-identical passages, and place the most relevant chunk near the *end* of the context (the "lost-in-the-middle" effect: models attend most to the start and end).

```python
def assemble_context(chunks, token_budget, count_tokens):
    """chunks: reranked best-first. Greedily pack under budget,
    then reorder so the single best chunk sits last (recency bias)."""
    selected, used = [], 0
    for c in chunks:
        n = count_tokens(c.text)
        if used + n > token_budget:
            break
        selected.append(c); used += n
    if len(selected) >= 2:
        # move best chunk (index 0) to the end of the prompt
        selected = selected[1:] + selected[:1]
    return selected
```

### The anti-hallucination contract

The hard requirement — "no hallucinated facts, with citations" — drives the design more than retrieval quality does. Tactics, in layers:

1. **Constrain the prompt:** "Answer *only* from the provided context. If the answer is not present, say you don't know. Cite the chunk id for every claim."
2. **Force citations structurally** by having the model emit claims tagged with chunk ids, then **verify** post-hoc: check that each cited chunk actually entails the claim (a small NLI model or an LLM-as-judge grounding check — see [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html)). Drop or flag unsupported claims.
3. **Abstain** when retrieval confidence is low (top reranker score below a threshold) rather than letting the LLM improvise.

### Evaluation — the RAG triad

Evaluate three relationships independently, because a failure in any one sinks the answer:

- **Context relevance:** did retrieval fetch the right chunks? (retrieval recall / NDCG)
- **Faithfulness / groundedness:** is every claim supported by the retrieved context? (NLI or judge)
- **Answer relevance:** does the answer address the question?

Decomposing the metric this way lets you localize failures: low context relevance ⇒ fix retrieval; low faithfulness with good context ⇒ fix the prompt or model. This decomposition is exactly what a strong candidate volunteers.

!!! warning "Common pitfall"

    Teams obsess over the generator and starve retrieval. In practice **most RAG failures are retrieval failures** — the answer simply wasn't in the fetched chunks, so even a perfect LLM cannot produce it. Before touching prompts, measure retrieval recall on a labeled set. Also beware **stale indexes**: if a document changes and the index lags, the assistant confidently cites outdated policy. Wire document updates to incremental re-indexing and stamp each chunk with a freshness date.

For when to reach for long-context instead of (or alongside) retrieval, see [Advanced RAG: GraphRAG, Agentic RAG & Long-Context vs RAG](../09-rag-retrieval/05-advanced-rag.html).

## Case 5 — Content-Moderation System

**Prompt:** "Design a system to detect policy-violating content (hate, harassment, CSAM, spam, violence) across 1 M posts per minute, in 50 languages."

Moderation is a *high-throughput, high-stakes classification* problem where the cost of errors is wildly asymmetric and adversaries adapt. See [Safety, Guardrails & Content Moderation](../12-production-mlops/04-safety-guardrails.html) and [Red-Teaming, Safety & Robustness Evaluation](../11-evaluation/05-redteaming-safety-eval.html).

### Frame the asymmetry first

The most important clarification: **what does a false negative cost vs a false positive?** For CSAM, a false negative is catastrophic and legally mandated to be caught — push the threshold toward near-zero misses even at the cost of precision, then human-review the flags. For borderline hate speech, a false positive (silencing legitimate speech) is itself harmful — calibrate carefully. **Per-category thresholds**, not one global threshold, are the design consequence.

### Tiered architecture

The funnel again, sorted by cost and confidence:

```text
post ─► T0: hash & rule match (sub-ms)  ──flag──► action
      ─► T1: fast multilingual classifier (5 ms) ──high-conf violate──► auto-remove
                                                  ──high-conf clean────► allow
      ─► T2: heavy multimodal LLM/model (100 ms+) on the uncertain middle band
                                                  ──► allow / remove / ESCALATE
      ─► T3: human review queue (minutes) for escalations & appeals
```

- **T0 — exact/near-exact match:** hash databases (e.g. PhotoDNA-style perceptual hashes for known CSAM, URL blocklists, known spam fingerprints). Catches the bulk cheaply and deterministically.
- **T1 — lightweight model:** a distilled multilingual transformer (one shared encoder over 50 languages — see [Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html)) outputs per-category probabilities. Auto-action only on **high-confidence** decisions in both directions; pass the uncertain middle to T2.
- **T2 — heavy model:** a multimodal model that reads text + image + maybe video frames, with enough capacity to catch subtle, contextual, or multimodal violations (a benign caption on a violating image). See [Vision-Language Models](../10-multimodal-and-arch/02-vision-language-models.html).
- **T3 — humans:** the ground-truth source and the appeals path. Human labels feed back into training (the data flywheel — see [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html)).

### Throughput math

!!! example "Worked example: how many T1 GPUs?"

    1 M posts/minute = $\approx 16{,}700$ posts/sec. Say T0 rules dispose of 30% deterministically, leaving ~11,700/sec for T1. A distilled classifier on a GPU at batch 64 might do ~5,000 inferences/sec/GPU. We need $\lceil 11{,}700 / 5{,}000 \rceil = 3$ GPUs for the steady state, but provision for spikes (a viral event, a coordinated attack) at 3–5× ⇒ ~12 GPUs with autoscaling. If T1 sends, say, 5% to T2 (≈ 585/sec) and T2 runs at 200/sec/GPU, that's ~3 GPUs for T2. The point: **the cheap tier must dispose of the vast majority**, or the heavy tiers explode in cost. State this funnel economics explicitly.

### The adversarial and feedback dimensions

Moderation is unique among these cases in that **adversaries actively evade** you: leetspeak ("h@te"), text-in-image to dodge text classifiers, coded language that shifts weekly. Consequences for the design:

- **Continuous retraining** on fresh human-labeled adversarial examples — a tight data flywheel, not a static model.
- **Robustness to perturbation:** normalize text (unicode confusables, spacing), OCR images, and red-team your own classifiers (see [Red-Teaming, Safety & Robustness Evaluation](../11-evaluation/05-redteaming-safety-eval.html)).
- **Calibration and review budget:** the threshold that routes to humans must respect a finite human-review capacity; if you flag too much, the queue overflows and review latency violates legal deadlines. This couples model precision to staffing — a great real-world tradeoff to surface.

### Metrics

Per category: precision/recall at the operating threshold, but the headline metrics are **prevalence** (fraction of violating content that slips through, measured by sampling and human-labeling live traffic — this is the true north star) and **appeal-overturn rate** (a proxy for false-positive harm). Note that you can't measure recall directly without knowing all violations, so prevalence-via-sampling is the rigorous substitute — saying this shows measurement maturity.

## Case 6 — Fine-Tuning / RLHF Pipeline

**Prompt:** "Design the pipeline that takes a pretrained base model and produces an aligned, instruction-following chat model, run continuously as new data arrives."

This is a *training-infrastructure* design, not a serving one. It composes Part V and Part VI. See [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html) and [The Anatomy of an RL-for-LLM System](../06-rl-infra/01-anatomy-rl-system.html).

### The three stages

The standard recipe (InstructGPT, Ouyang et al.) is **SFT → reward model → RL**:

1. **Supervised fine-tuning (SFT):** train the base model on high-quality (prompt, ideal-response) pairs to teach the chat format and instruction-following (see [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html)). Data formatting and chat templates matter enormously (see [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html)).
2. **Reward model (RM):** collect human **preference pairs** (response A vs B for the same prompt), train a model to score responses so the preferred one scores higher. The loss is the Bradley–Terry pairwise objective:

    $$
    \mathcal{L}_\text{RM} = -\log \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big)
    $$

    where $y_w$ is the human-preferred ("winning") response and $y_l$ the rejected one.

3. **RL optimization:** optimize the policy to maximize reward while staying close to the SFT model via a KL penalty (PPO — see [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html)), or skip the explicit RM with **DPO** (see [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html)), or use **GRPO** for critic-free RL (see [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html)).

```python
import torch
import torch.nn.functional as F

def reward_model_loss(reward_chosen, reward_rejected):
    """Bradley-Terry: maximize the margin between preferred and rejected.
    reward_* are scalar scores from the RM head."""
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()

def dpo_loss(pi_logp_chosen, pi_logp_rejected,
             ref_logp_chosen, ref_logp_rejected, beta=0.1):
    """DPO turns the RLHF objective into a classification loss on pairs,
    eliminating the separate reward model AND the RL loop. The policy's
    log-prob ratio against a frozen reference IS the implicit reward."""
    pi_logratios  = pi_logp_chosen  - pi_logp_rejected
    ref_logratios = ref_logp_chosen - ref_logp_rejected
    return -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
```

A strong candidate states the **DPO-vs-PPO tradeoff** unprompted: DPO is dramatically simpler (no reward model to train, no rollout engine, no RL instability) and is the right default for most teams; PPO/GRPO can reach higher ceilings and are necessary for **online** RL with *verifiable* rewards (math, code — see [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html)) where the signal is a programmatic checker rather than a learned preference model.

### The system: why RL is an infrastructure problem

PPO/GRPO is hard *as a system* because each step alternates between **generation** (autoregressive rollout — an inference workload) and **training** (a backward pass), and these have opposite hardware profiles. The dominant architecture co-locates or disaggregates an inference engine (vLLM) for rollouts and a training engine (FSDP/Megatron) for updates, synchronizing weights between them each step (see [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html) and [Colocated vs Disaggregated RL & Weight Synchronization](../06-rl-infra/07-colocated-vs-disaggregated.html)).

```text
   ┌──────────────────────── one RL step ───────────────────────┐
   │  1. sample prompts from dataset                             │
   │  2. ROLLOUT: policy (vLLM) generates responses   ◄── inference, GPU-bound on decode
   │  3. SCORE: reward model / verifier / sandbox scores them    │
   │  4. compute advantages (GAE for PPO; group-relative for GRPO)│
   │  5. TRAIN: policy gradient update (FSDP)          ◄── training, gradient/optimizer mem
   │  6. SYNC: copy updated weights back to the rollout engine    │
   └─────────────────────────────────────────────────────────────┘
```

Memory is the binding constraint. PPO keeps **four** models resident — policy, reference (for KL), reward model, and a value/critic — which is why critic-free methods like GRPO and RLOO (which drop the value network and estimate advantage from a group of sampled completions) are attractive at scale (see [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html)). For parameter-efficient adaptation that slashes memory, use LoRA/QLoRA (see [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html)) and the LoRA math in [Memory-Efficient Training](../04-kernels-efficiency/10-memory-efficient-training.html).

### Data engine, evaluation, and the "run continuously" requirement

The prompt says *continuously as new data arrives* — that means a **data flywheel** (see [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html)): production traffic and thumbs-up/down feed a labeling pipeline, which yields fresh preference data, which triggers periodic retraining. Guard against:

- **Reward hacking / over-optimization:** the policy exploits quirks in the RM (verbosity, sycophancy, formatting tricks) rather than genuinely improving. Monitor the **KL divergence** from the reference — runaway KL is the canonical alarm — and cap it (see [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html) and [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html)).
- **Evaluation:** offline benchmarks plus a held-out preference test set plus, crucially, **A/B on real users** and a red-team pass before any launch (see [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html) and [Red-Teaming, Safety & Robustness Evaluation](../11-evaluation/05-redteaming-safety-eval.html)).

!!! example "Worked example: RM data and SFT compute"

    Suppose we fine-tune a 8 B model. SFT on 100k high-quality examples averaging 1k tokens = $10^8$ tokens. With a rough training cost of $6N$ FLOPs/token ($N$ = params), that's $6 \times 8\times10^9 \times 10^8 = 4.8\times10^{18}$ FLOPs ≈ 4.8 ExaFLOPs — a few hours on 8 high-end GPUs delivering, say, ~$10^{15}$ effective FLOPs/s combined. For the RM, preference data is the bottleneck, not compute: getting 50k well-labeled human preference pairs at, say, 2 minutes of annotator time per pair is $\approx 1{,}700$ annotator-hours — *that* line item, not GPU cost, gates the schedule. Recognizing that **human preference labeling, not compute, is the critical-path resource for RLHF** is a senior-level observation.

## Bringing It Together: A Reusable Checklist

Across all six cases the same questions recur. When you sit down in the interview, run this loop out loud:

1. **Clarify** objective, scale (corpus size, QPS), latency budget, and the cost asymmetry of errors.
2. **Frame** as ML: inputs, outputs, label source, loss.
3. **Funnel** the work: cheap high-recall retrieval → expensive high-precision scoring, cutting the funnel where the latency budget forces it.
4. **Split** offline (precompute embeddings/indexes/features) from online (do the minimum).
5. **Plug in numbers:** memory (weights + KV cache + index), throughput (tokens/s, QPS per replica), and confirm the budget closes.
6. **Metrics:** an offline proxy *and* an online north-star, always validated by A/B.
7. **Stress-test:** training/serving skew, feedback loops, staleness, adversaries, cold-start, and the dominant failure mode for *this* system.

!!! interview "Interview Corner"

    **Q:** You've designed a recommender and a RAG system in two different interviews. The interviewer asks: "These feel like the same architecture. Are they?"

    **A:** Structurally, yes — both are two-stage funnels: a cheap retriever (two-tower ANN for recs, hybrid BM25+dense for RAG) narrows a huge corpus to a small candidate set, then an expensive model (a multi-task ranker for recs, an LLM for RAG) produces the final output. The differences are in the *output* and the *failure modes*: a recommender outputs a ranked list optimized for engagement and fights feedback loops and position bias; a RAG system outputs grounded text and fights hallucination and staleness. Recognizing the shared skeleton lets you reuse the retrieval/serving machinery; recognizing the divergent failure modes is what makes each design correct rather than generic.

!!! key "Key Takeaways"

    - Almost every large-scale ML system is a **funnel**: cheap high-recall candidate generation feeds expensive high-precision scoring; you cut the funnel where the latency budget forces it.
    - The **offline/online split** is universal — precompute embeddings, indexes, and features in batch; keep the online path minimal.
    - **Recommendation** = two-tower retrieval + multi-task ranker; the silent killers are training/serving skew, position bias, and feedback loops.
    - **Search** = hybrid (BM25 + dense) retrieval fused with RRF, then a cross-encoder reranker; bi-encoders scale, cross-encoders are accurate — use each where it belongs.
    - **LLM serving** is gated by the **KV-cache memory budget**, which sets the batch size and thus throughput; GQA, PagedAttention, continuous batching, and disaggregated prefill/decode are the levers, and TTFT/ITL are *two* SLAs.
    - **RAG** composes retrieval + serving; most failures are *retrieval* failures, and the anti-hallucination contract (ground, cite, verify, abstain) drives the design more than the generator does.
    - **Moderation** is a tiered, adversarial, asymmetric-cost classifier where the cheap tier must dispose of the majority and per-category thresholds route the uncertain middle to humans.
    - **RLHF** is SFT → reward model → RL (or DPO to collapse the last two); it is an *infrastructure* problem of alternating generation and training, and human preference labeling — not compute — is the critical-path resource.
    - In the room: **clarify, frame, funnel, split, quantify, measure, stress-test** — narrate the tradeoffs and always validate with an A/B test.

## Further reading

- Covington, Adams & Sargin, *Deep Neural Networks for YouTube Recommendations* (2016) — the canonical two-stage candidate-generation + ranking design.
- Zhao et al., *Recommending What Video to Watch Next: A Multitask Ranking System* (2019) — multi-gate mixture-of-experts and multi-task ranking heads.
- Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering* (2020) — bi-encoder dense retrieval with in-batch negatives.
- Nogueira & Cho, *Passage Re-ranking with BERT* (2019) — the cross-encoder reranker.
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020) — the original RAG formulation.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, 2023) — the serving-engine memory model.
- Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT, 2022) — the SFT → RM → PPO pipeline.
- Rafailov et al., *Direct Preference Optimization* (2023) — collapsing reward modeling and RL into a single classification loss.
- Shao et al., *DeepSeekMath / GRPO* (2024) — critic-free group-relative policy optimization at scale.
