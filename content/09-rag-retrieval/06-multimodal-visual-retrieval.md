# 9.6 Multimodal & Visual-Document Retrieval: ColPali & Late Interaction

Most of Part IX has quietly assumed that a "document" is text — a paragraph you can tokenize, embed, and stuff into a context window. But an enormous fraction of the world's high-value knowledge does not live as clean text. It lives in PDFs full of multi-column layouts, financial reports dense with tables, slide decks where the meaning is in the figure, scanned contracts, scientific papers with equations, invoices, engineering diagrams, and screenshots. The moment you try to retrieve over these with the standard pipeline, you hit a wall that has nothing to do with embeddings and everything to do with **parsing**.

The traditional answer is an OCR-and-layout pipeline: run optical character recognition, detect tables and reading order, reconstruct a linearized text stream, chunk it, embed it, and proceed as in [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html). This works, but it is brittle and lossy. OCR mangles multi-column reading order, drops table structure, and silently discards the very thing that made a chart informative — its visual form. Every error compounds downstream: a misread row in a table becomes a wrong retrieval becomes a wrong answer.

This chapter is about the alternative that emerged in 2024–2025 and rapidly became a default for visually rich corpora: treat **each page as an image** and retrieve directly over the pixels, skipping OCR entirely. We will build up from cross-modal dense retrieval (CLIP/SigLIP), then develop the key idea — **late interaction**, borrowed from ColBERT and extended to vision in **ColPali** and **ColQwen2** — and finally engineer the full system: how to index thousands of multi-vector page embeddings (PLAID, HNSW-over-patches), how to rerank, what it costs, and how to evaluate it on ViDoRe. The generation half — feeding retrieved page images to a vision-language model (VLM) — connects directly to [Vision-Language Models](../10-multimodal-and-arch/02-vision-language-models.html).

## Cross-Modal Dense Retrieval: CLIP and SigLIP

Before late interaction, the first question is simpler: can we put an image and a text query into the **same vector space** so that nearest-neighbor search retrieves the right image for a textual query? This is **cross-modal dense retrieval**, and the canonical answer is CLIP.

### The Dual-Encoder Contrastive Recipe

CLIP (Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021) trains two encoders — an image encoder $f_\text{img}$ (a Vision Transformer; see [Vision Transformers & Image Encoders](../10-multimodal-and-arch/01-vision-transformers.html)) and a text encoder $f_\text{txt}$ — to map their inputs into a shared $d$-dimensional space. Each image and each text is reduced to a **single normalized vector**. Given a batch of $N$ matched (image, caption) pairs, we compute all $N \times N$ cosine similarities and apply a symmetric contrastive loss (InfoNCE) that pulls matched pairs together and pushes mismatched pairs apart.

Let $u_i = f_\text{img}(I_i) / \lVert f_\text{img}(I_i)\rVert$ and $v_j = f_\text{txt}(T_j) / \lVert f_\text{txt}(T_j)\rVert$. With a learned temperature $\tau$, the image-to-text loss is

$$
\mathcal{L}_{\text{i}\to\text{t}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(u_i^\top v_i / \tau)}{\sum_{j=1}^{N}\exp(u_i^\top v_j / \tau)}
$$

and the full loss symmetrizes over text-to-image as well. The mechanics are exactly the contrastive learning from [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html), just with two modalities sharing one space.

**SigLIP** (Zhai et al., *Sigmoid Loss for Language Image Pre-Training*, 2023) replaces the softmax-over-the-batch with an independent **sigmoid** loss per pair, treating every (image, text) pair as a binary classification:

$$
\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N} \log \sigma\!\big(z_{ij}\,(t\, u_i^\top v_j + b)\big), \quad z_{ij} = \begin{cases} +1 & i = j \\ -1 & i \ne j\end{cases}
$$

where $\sigma$ is the logistic sigmoid, $t$ is a learned scale, and $b$ a learned bias initialized negative (since most pairs are negatives). The practical win is that the sigmoid loss does not require a global softmax normalization across the batch, so it scales to very large batches without the all-gather coupling that softmax needs — and it tends to give better small-batch behavior. SigLIP's vision tower later became the backbone of several VLMs and, crucially for us, of ColPali.

### Using CLIP/SigLIP for Visual-Document Retrieval

```python
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

# Load a SigLIP checkpoint (image + text towers sharing one space).
model = AutoModel.from_pretrained("google/siglip-base-patch16-224").eval()
proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

@torch.no_grad()
def embed_images(paths):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    inp = proc(images=imgs, return_tensors="pt")
    emb = model.get_image_features(**inp)        # [N, d]
    return F.normalize(emb, dim=-1)              # unit vectors -> cosine = dot

@torch.no_grad()
def embed_text(queries):
    inp = proc(text=queries, padding="max_length", return_tensors="pt")
    emb = model.get_text_features(**inp)         # [M, d]
    return F.normalize(emb, dim=-1)

# Index: one vector per page image.
page_paths = ["page_001.png", "page_002.png", "page_003.png"]
page_vecs  = embed_images(page_paths)            # [3, d]

q = embed_text(["What was Q3 revenue growth?"]) # [1, d]
scores = q @ page_vecs.T                         # [1, 3] cosine similarities
best = scores.argmax(dim=-1).item()
print("Best page:", page_paths[best], "score:", scores[0, best].item())
```

This is genuinely useful for **natural-image** retrieval ("find the photo of a dog on a beach"). But it is weak for **dense documents**, and the reason is structural, not a tuning problem. CLIP/SigLIP compress an entire page into one vector. A page of a 10-K filing contains dozens of distinct facts: a revenue figure, a footnote about currency hedging, a segment table, a risk paragraph. A single 768-dimensional vector cannot simultaneously preserve all of them in a way that a *specific* keyword-like query can latch onto. This is the **information-bottleneck** problem of single-vector ("bi-encoder") retrieval, and it is exactly the problem that late interaction was invented to solve in the text world.

!!! note "Aside: contrastive captions vs. document queries"

    CLIP and SigLIP are trained on web image–caption pairs ("a golden retriever running on a beach"). The query distribution for documents is utterly different: "depreciation schedule for fiscal 2022," "the bar chart comparing latency across GPUs." A model trained on captions has never learned to align fine-grained query terms to the small region of a page that answers them. This domain mismatch is why off-the-shelf CLIP underperforms on ViDoRe, and why ColPali fine-tunes specifically on document-query pairs.

## Late Interaction: From ColBERT to ColPali

### The ColBERT Idea

ColBERT (Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*, 2020) sits between two extremes. A **cross-encoder** concatenates query and document, runs full attention across both, and scores them jointly — maximally expressive but $O(\text{queries} \times \text{docs})$ forward passes, far too slow to scan a corpus. A **bi-encoder** (single vector each) is fast — you precompute document vectors and do a dot product — but throws away token-level detail.

Late interaction is the middle path. Encode the query into **one vector per query token** $\{q_1, \dots, q_n\}$ and the document into **one vector per document token** $\{d_1, \dots, d_m\}$. Precompute and store all document token vectors offline. At query time, score with the **MaxSim** operator: for each query token, find its best-matching document token, then sum:

$$
S(Q, D) = \sum_{i=1}^{n} \max_{j=1}^{m} \; q_i^\top d_j
$$

Intuitively, each query term gets to "go shopping" across the whole document for its single best evidence, and the score adds up that evidence. There is no cross-attention between query and document — the only interaction is the cheap MaxSim dot products at the very end, hence *late* interaction. This preserves token granularity (a rare query term can find its exact match) while keeping documents fully precomputable.

```python
import torch

def maxsim(Q, D):
    """
    Q: [n, d] query token embeddings (L2-normalized)
    D: [m, d] document token embeddings (L2-normalized)
    Returns the ColBERT late-interaction score (a scalar).
    """
    sim = Q @ D.T                  # [n, m] all query-token x doc-token sims
    per_query = sim.max(dim=1).values   # [n] best doc token per query token
    return per_query.sum()         # sum over query tokens

# Toy: 4 query tokens, 6 doc tokens, d=8
Q = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
D = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
print("MaxSim score:", maxsim(Q, D).item())
```

The cost is storage: instead of one vector per document you store $m$ (often 100–300 for a passage). This is the central trade-off of all multi-vector retrieval, and most of the systems engineering later in this chapter exists to make it affordable.

### ColPali: Late Interaction Over Page Patches

ColPali (Faysse et al., *ColPali: Efficient Document Retrieval with Vision Language Models*, 2024) makes one conceptually clean move: **replace ColBERT's BERT-token document encoder with a vision-language model that turns a page image into a grid of patch embeddings.** The query is still text; the document is now an image.

The pipeline:

1. Render each PDF page to an image (e.g., at ~150 DPI). **No OCR.**
2. Feed the image to a VLM's vision encoder + projection. The original ColPali used PaliGemma (a SigLIP vision tower feeding a Gemma language model). The image becomes a sequence of patch tokens — for PaliGemma, a $32\times 32$ grid yields **1024 patch embeddings** per page.
3. Project each patch embedding down to a low dimension $d$ (ColPali uses $d = 128$, matching ColBERT) with a linear layer, and L2-normalize. Store these 1024 vectors as the page's representation.
4. The text query is tokenized and run through the **same model's** language tower to produce one $d$-vector per query token.
5. Score with MaxSim, identical to ColBERT.

{{fig:colpali-late-interaction-maxsim}}

The magic is in step 2. Because the patch embeddings come from a VLM that was pretrained to *read*, each patch vector is contextualized: a patch sitting on the cell of a table that says "Revenue: USD 4.2B" carries a representation that a query token for "revenue" can match against — **without anyone ever running OCR**. The model learned to associate visual glyphs and layout with textual meaning during pretraining, and ColPali's fine-tuning sharpens that for the retrieval objective.

### ColQwen2 and the Family

**ColQwen2** swaps the PaliGemma backbone for Qwen2-VL. Qwen2-VL's vision encoder supports **dynamic resolution** — it does not force every page into a fixed $448\times 448$ box but processes the native aspect ratio, producing a variable number of patch tokens. For a tall, dense page this means *more* patches (finer evidence) and for a sparse slide *fewer* (cheaper). This typically lifts retrieval quality on dense documents at the cost of variable, sometimes larger, per-page storage. The broader family — **ColSmol** (small, for edge/CPU), and later **ColPali/ColQwen** revisions, plus the general "ColVision" recipe — all share the same skeleton: VLM patch encoder + low-dim projection + MaxSim. The contrast with ColBERT-text is only in the document encoder.

!!! warning "Common pitfall: forgetting the query side is also multi-vector"

    A frequent misreading is "ColPali embeds pages into 1024 vectors and queries into one vector." No — the query is **also** multi-vector (one $d$-dim vector per query token, typically a few dozen). MaxSim runs over the full $n \times m$ grid. If you accidentally mean-pool the query into a single vector, you have silently reverted to a worse bi-encoder and thrown away ColPali's entire advantage. Keep both sides token/patch-level until the MaxSim.

### Training ColPali

ColPali is fine-tuned with a contrastive in-batch loss on (query, positive-page) pairs, where the negatives are the other pages in the batch. The score is the MaxSim, and the loss is the same InfoNCE softmax over MaxSim scores. A common refinement adds a margin-aware or in-batch hard-negative term. The training data is the crux: synthetic and curated **document-query pairs** spanning tables, figures, infographics, and full pages, so the model learns the document query distribution that CLIP never saw.

```python
import torch
import torch.nn.functional as F

def colbert_scores(Qb, Db, q_mask, d_mask):
    """
    Vectorized MaxSim for a batch (used both in training and scoring).
    Qb: [B, n, d] query token embeddings (padded)
    Db: [B, m, d] doc patch embeddings (padded)
    q_mask: [B, n] 1 for real query tokens, 0 for padding
    d_mask: [B, m] 1 for real patches
    Returns: [B, B] score matrix S[i, j] = MaxSim(query_i, doc_j)
    """
    # Pairwise sims across the cross product of the batch:
    #   [B, n, d] x [B, m, d] -> [B(query), B(doc), n, m]
    sim = torch.einsum("ind,jmd->ijnm", Qb, Db)
    # Mask out padded doc patches before the max over patches.
    sim = sim.masked_fill(~d_mask[None, :, None, :].bool(), -1e4)
    sim = sim.max(dim=-1).values                 # [B, B, n] max over patches
    # Zero out padded query tokens before summing over query tokens.
    sim = sim * q_mask[:, None, :]
    return sim.sum(dim=-1)                        # [B, B]

def colpali_loss(Qb, Db, q_mask, d_mask):
    S = colbert_scores(Qb, Db, q_mask, d_mask)   # [B, B], diagonal = positives
    labels = torch.arange(S.size(0), device=S.device)
    # Standard in-batch InfoNCE over MaxSim scores (both directions optional).
    return F.cross_entropy(S, labels)
```

## A Worked Example: Storage and Scoring Cost

!!! example "Worked example: indexing 100k pages with ColPali"

    Suppose a corpus of **100,000 pages**. ColPali (PaliGemma backbone) produces **1024 patch vectors per page**, each of dimension **$d = 128$**. Store at fp16 (2 bytes/element).

    **Per page:**

    $$
    1024 \text{ patches} \times 128 \text{ dims} \times 2 \text{ bytes} = 262{,}144 \text{ bytes} \approx 256 \text{ KiB}
    $$

    **Full corpus:**

    $$
    100{,}000 \times 256 \text{ KiB} \approx 25.6 \text{ GiB}
    $$

    Compare a single-vector SigLIP index at the same $d=128$ and fp16: $100{,}000 \times 128 \times 2 = 25.6$ MB — a **1000×** difference, because each page now holds 1024 vectors instead of 1. That factor of 1024 is the price of late interaction and the reason indexing strategy matters enormously.

    **Scoring cost (brute force) for one query** with $n = 20$ query tokens against all pages:

    $$
    100{,}000 \text{ pages} \times 1024 \text{ patches} \times 20 \text{ q-tokens} \times 128 \text{ flops/dot} \approx 2.6 \times 10^{11} \text{ FLOPs}
    $$

    That is ~260 GFLOPs of dot products **per query** if you score every page exhaustively. On a modern GPU this is milliseconds of compute but tens of GiB of memory traffic — memory bandwidth, not arithmetic, is the bottleneck (see the [Roofline Model](../04-kernels-efficiency/01-roofline-performance.html)). At scale you cannot brute-force every query; you need an approximate candidate-generation stage (next section) and only run full MaxSim on a shortlist.

    **Binary quantization** (1 bit/dim instead of 16) shrinks the index from 25.6 GiB to **1.6 GiB** at a small recall cost — often the single highest-leverage optimization for multi-vector indexes.

## Indexing Multi-Vector Embeddings at Scale

The brute-force MaxSim above is fine for a few thousand pages but collapses at corpus scale. The whole field of efficient multi-vector retrieval is about avoiding the full $O(\text{pages} \times \text{patches} \times \text{q-tokens})$ scan. There are two dominant approaches.

### PLAID: ColBERT's Native Engine

PLAID (Santhanam et al., *PLAID: An Efficient Engine for Late Interaction Retrieval*, 2022) is the production indexing system designed for ColBERT, and it carries over to ColPali. Its pipeline:

1. **Residual compression.** Cluster all patch/token vectors in the corpus with k-means into a codebook of centroids. Store each vector as `(centroid_id, residual)`, where the residual (vector minus its centroid) is **quantized to 1–2 bits per dimension**. This is the dominant memory saving.
2. **Centroid-based candidate generation.** Each query token retrieves the nearest centroids. Any document that has patches assigned to those centroids becomes a candidate. This avoids touching documents with no plausible matching patch.
3. **Centroid-pruned scoring.** Approximate MaxSim using centroid similarities first to prune, then refine only promising candidates with the decompressed residuals.
4. **Full re-ranking.** Decompress the top candidates and compute exact MaxSim for the final ordering.

The staged funnel — cheap-and-approximate to expensive-and-exact — is the recurring pattern of all large-scale retrieval, and you saw it for single-vector ANN in [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html).

### HNSW Over Flattened Patch Vectors

A simpler, store-agnostic approach: **flatten** every patch vector of every page into one giant single-vector ANN index, tagging each with its page id. Build an HNSW graph (see the ANN chapter) over all $100{,}000 \times 1024 \approx 10^8$ patch vectors. Then:

1. For each of the $n$ query token vectors, run an ANN search to retrieve the top-$k$ nearest **patches** (across all pages).
2. Collect the **union of page ids** those patches belong to — this is the candidate page set.
3. For each candidate page, fetch its full patch matrix and compute **exact MaxSim**.
4. Sort by exact score; return top pages.

This is the strategy used by libraries that build ColPali on top of generic vector databases (Qdrant's multivector support, Vespa's tensor/MaxSim ranking, Weaviate's MUVERA-style approaches). The candidate-generation step is approximate (you might miss a page whose best patch did not make any query token's top-$k$), but in practice recall is high because a relevant page usually has *several* strongly matching patches.

```python
import numpy as np
import hnswlib

class PatchHNSWIndex:
    """Flatten all page patches into one HNSW index; rerank pages with exact MaxSim."""
    def __init__(self, dim=128):
        self.dim = dim
        self.index = hnswlib.Index(space="ip", dim=dim)  # inner product
        self.page_patches = {}   # page_id -> [m, d] float32 patch matrix
        self.label_to_page = {}  # global patch label -> page_id
        self._next = 0

    def init(self, max_patches):
        self.index.init_index(max_elements=max_patches, ef_construction=200, M=16)

    def add_page(self, page_id, patches):
        patches = patches.astype(np.float32)
        n = patches.shape[0]
        labels = np.arange(self._next, self._next + n)
        self.index.add_items(patches, labels)
        for lab in labels:
            self.label_to_page[int(lab)] = page_id
        self.page_patches[page_id] = patches
        self._next += n

    def search(self, query_patches, k_per_token=50, topn=10):
        query_patches = query_patches.astype(np.float32)
        # 1) candidate generation: nearest patches per query token -> union of pages
        candidates = set()
        for qtok in query_patches:
            labels, _ = self.index.knn_query(qtok, k=k_per_token)
            for lab in labels[0]:
                candidates.add(self.label_to_page[int(lab)])
        # 2) exact MaxSim rerank over candidate pages only
        scored = []
        for pid in candidates:
            D = self.page_patches[pid]               # [m, d]
            sim = query_patches @ D.T                # [n, m]
            scored.append((pid, sim.max(axis=1).sum()))   # MaxSim
        scored.sort(key=lambda x: -x[1])
        return scored[:topn]
```

### Token Pooling and Compression

Because 1024 patches per page is the cost driver, a cheap win is **token pooling**: cluster a page's patch vectors (e.g., hierarchical agglomerative clustering) and keep a smaller set of representative vectors — say 256 instead of 1024 — before indexing. Empirically you can drop a large fraction of patches with little retrieval-quality loss, because many patches (margins, whitespace, repeated background) are redundant. Combined with **binary quantization** of what remains, real ColPali deployments routinely shrink the index by 10–30× from the naive 256 KiB/page figure. **MUVERA** (Multi-Vector Retrieval via Fixed Dimensional Encodings) goes further: it deterministically projects the whole multi-vector set into a *single* fixed-dimensional vector whose dot product approximates MaxSim, letting you reuse a standard single-vector ANN index for candidate generation and only fall back to exact MaxSim for reranking.

!!! tip "Practitioner tip: separate the two storage tiers"

    Keep two representations of each page's patches: a **compressed** form (binary or 2-bit residuals) for fast candidate generation, and the **full fp16** form (possibly on cheaper storage or memory-mapped) for exact MaxSim reranking of the ~50 candidates that survive. You almost never need the full-precision vectors of pages that never enter the shortlist, so they can live on disk. This two-tier split is what keeps RAM bounded while preserving final-ranking accuracy.

## The OCR-Free Page-as-Image RAG Pipeline

Retrieval is half the system. The other half feeds the retrieved **page images** to a vision-language model to generate the answer — never reconstructing text at all. This is the OCR-free RAG loop, and it composes cleanly with everything in [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html), with images replacing text chunks.

{{fig:colpali-ocrfree-rag-pipeline}}

```python
# End-to-end sketch with the `colpali-engine` library + a VLM generator.
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from pdf2image import convert_from_path

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ColPali.from_pretrained("vidore/colpali-v1.3",
                                torch_dtype=torch.bfloat16).to(device).eval()
processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")

# --- OFFLINE: render pages and embed them ---
pages = convert_from_path("annual_report.pdf", dpi=150)  # list[PIL.Image]
page_embeddings = []
with torch.no_grad():
    for batch_start in range(0, len(pages), 4):
        batch = pages[batch_start:batch_start + 4]
        inp = processor.process_images(batch).to(device)
        emb = model(**inp)                  # [b, num_patches, 128]
        page_embeddings.extend(list(emb.to(torch.float16).cpu()))
# (Index page_embeddings into PLAID/HNSW here; see previous section.)

# --- ONLINE: embed the query, score with MaxSim, retrieve top pages ---
query = "How did operating margin change from 2021 to 2022?"
with torch.no_grad():
    q_inp = processor.process_queries([query]).to(device)
    q_emb = model(**q_inp)                  # [1, n_tokens, 128]

# Exact MaxSim against all pages (small corpus; use the index at scale).
def maxsim(q, d):
    return (q @ d.T).max(dim=1).values.sum()
scores = torch.tensor([maxsim(q_emb[0].float(), d.float()) for d in page_embeddings])
top = scores.topk(3).indices.tolist()
retrieved_images = [pages[i] for i in top]

# --- GENERATION: hand the raw page images to a VLM (no OCR text!) ---
from transformers import AutoModelForImageTextToText, AutoProcessor
vlm = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16).to(device).eval()
vproc = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

messages = [{"role": "user", "content":
    [{"type": "image", "image": im} for im in retrieved_images]
    + [{"type": "text", "text": query
        + " Cite the page number you used."}]}]
prompt = vproc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
vin = vproc(text=[prompt], images=retrieved_images, return_tensors="pt").to(device)
with torch.no_grad():
    out = vlm.generate(**vin, max_new_tokens=256)
print(vproc.batch_decode(out, skip_special_tokens=True)[0])
```

### Why "no OCR" Is a Feature, Not a Hack

The instinct of a careful engineer is to distrust skipping OCR — surely text is more reliable than pixels? But consider what survives the round-trip. A bar chart's *meaning* is the relative height of bars; OCR captures only the axis labels and legend, discarding the comparison. A complex table's meaning is its 2-D structure; linearized OCR text frequently scrambles which number belongs to which row/column header. The page-as-image approach keeps the layout, the chart geometry, the equation rendering, and the spatial relationships intact, and a modern VLM can read all of it. You trade OCR's parsing errors for the VLM's vision — and for visually rich documents that is a strongly favorable trade.

The cost is on the **generation** side: page images are expensive in tokens. A single high-resolution page can consume hundreds to over a thousand vision tokens in the VLM's context, so retrieving $k=10$ pages can blow a context budget that $k=10$ text chunks would not. This makes **precision of retrieval** more important here than in text RAG — you want the top 1–3 pages to be right, because you cannot afford to dump 20 page images into the VLM. It also makes reranking valuable.

## Visual Reranking and Hybrid Strategies

Late-interaction retrieval already does fine-grained matching, so do we still need a reranker? Often yes — for the same staged-funnel reason as text RAG (see [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html)). Three flavors of visual reranking exist.

1. **Exact-MaxSim rerank.** If candidate generation was approximate (HNSW union, MUVERA, or PLAID centroid pruning), recomputing exact MaxSim on the shortlist is itself a reranker. Cheap and almost always worth it.

2. **Cross-encoder / VLM-as-judge rerank.** Pass each candidate page *image* and the query jointly into a VLM and ask for a relevance score. This is the visual analogue of a text cross-encoder: maximally expressive (full cross-attention between query and page) but expensive, so apply it only to the top ~20 candidates. A prompt like "On a scale of 0–10, does this page contain the information to answer: `<query>`?" turns any capable VLM into a reranker.

3. **Hybrid with text/OCR signals.** Nothing forbids running OCR *in addition* for a BM25 lexical channel (see hybrid search). For documents with exact identifiers — invoice numbers, part codes, legal citations — a lexical match on OCR'd text is unbeatable, while ColPali handles the semantic/visual layout. Fuse the rankings with Reciprocal Rank Fusion (RRF):

$$
\text{RRF}(d) = \sum_{r \in \{\text{ColPali},\,\text{BM25}\}} \frac{1}{k + \operatorname{rank}_r(d)}, \quad k \approx 60
$$

```python
def reciprocal_rank_fusion(rankings, k=60):
    """rankings: list of ranked lists, each a list of page_ids best-first."""
    scores = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda p: -scores[p])

colpali_order = ["p17", "p4", "p23", "p9"]
bm25_order    = ["p4", "p17", "p88", "p23"]   # from OCR + BM25
print(reciprocal_rank_fusion([colpali_order, bm25_order]))
```

!!! interview "Interview Corner"

    **Q:** Your team retrieves over 2 million scanned pages. ColPali at 1024 patches/page would need a multi-vector index in the tens-to-hundreds of GB and exact MaxSim is too slow to scan. Walk me through a serving design that hits sub-200 ms p95 retrieval latency.

    **A:** I would build a three-stage funnel and aggressively compress. **(1) Compression at index time:** token-pool each page from 1024 to ~128 patches via clustering, then store residuals at 1–2 bits/dim (PLAID) or binary-quantize. That alone shrinks 2M pages from hundreds of GB to a few tens of GB that fit in RAM across a couple of shards. **(2) Approximate candidate generation:** either PLAID's centroid-based retrieval or a MUVERA fixed-dimensional encoding so each query token hits a standard HNSW index; take the union of, say, the top few hundred candidate pages. This stage never touches full-precision vectors. **(3) Exact MaxSim rerank:** decompress the full fp16 patches for only those few hundred candidates (the two-tier storage split, full vectors memory-mapped on NVMe) and compute exact MaxSim, returning the top 3–5 pages. The latency budget goes mostly to stage 2's ANN graph traversal; stage 3 is a few hundred small matmuls. I would shard by document, replicate for QPS, cache query embeddings for repeated queries, and keep generation precision high by returning few but accurate pages. If latency still misses, I trade recall for speed by lowering `ef`/`nprobe` and the per-token top-$k$, and I validate the recall hit on a held-out ViDoRe-style eval before shipping.

## Evaluation: ViDoRe and What to Measure

You cannot tune what you cannot measure, and visual-document retrieval needed its own benchmark because text-retrieval datasets do not exist as page images. **ViDoRe** (the Visual Document Retrieval Benchmark, introduced alongside ColPali) is the standard. It contains query–page tasks spanning academic papers, figures, infographics, tables, and multi-domain documents (medical, energy, government, AI), in multiple languages, with both synthetic and human-curated queries. Crucially, each task is *page-level*: given a query, retrieve the correct page(s) from a corpus of page images.

The primary metric is **nDCG@k** (Normalized Discounted Cumulative Gain), with **Recall@k** and **MRR** as companions. nDCG rewards placing relevant pages near the top and discounts gains logarithmically by rank:

$$
\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \qquad \text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}
$$

where $\text{rel}_i$ is the relevance of the page at rank $i$ and IDCG is the DCG of the ideal ordering (so nDCG $\in [0, 1]$).

```python
import numpy as np

def dcg(relevances):
    relevances = np.asarray(relevances, dtype=float)
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum((2**relevances - 1) / discounts)

def ndcg_at_k(retrieved_rels, ideal_rels, k=5):
    """retrieved_rels: relevance of each retrieved page, in retrieved order.
       ideal_rels: all true relevances sorted descending (for the ideal ranking)."""
    actual = dcg(retrieved_rels[:k])
    ideal  = dcg(sorted(ideal_rels, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0

# Retrieved pages had relevances [1, 0, 1, 0, 0]; one other relevant page (rel 1)
# existed but was missed, so the ideal top-5 is [1, 1, 1, 0, 0].
print(round(ndcg_at_k([1, 0, 1, 0, 0], [1, 1, 1, 1, 0], k=5), 4))
```

A few measurement subtleties specific to this setting:

- **Report retrieval and end-to-end separately.** A page can be retrieved correctly yet the VLM still answers wrong, or vice versa. Measure nDCG@k for retrieval and an answer-correctness metric (exact match, LLM-as-judge; see [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html)) for the full pipeline.
- **Watch DPI and resolution.** Rendering pages too low-res hurts ColPali's patch evidence and the VLM's reading; too high-res inflates cost. ViDoRe-style ablations across DPI catch this.
- **Beware synthetic-query leakage.** Synthetic queries generated by the *same* family of models used in retrieval can inflate scores; trust human-curated splits more. ViDoRe-v2 specifically broadened domains and tightened this.
- **Language and OCR-hardness.** ColPali's biggest wins over OCR pipelines are on non-Latin scripts, handwriting, and heavy-layout pages where OCR degrades; stratify your eval by document difficulty to see where the visual approach actually pays off.

!!! warning "Common pitfall: comparing ColPali to OCR on the wrong corpus"

    If your corpus is clean, single-column, born-digital text (e.g., Markdown docs or simple PDFs), a good OCR/text pipeline will match or beat ColPali at a fraction of the index size — the page images buy you nothing because there is no visual structure to lose. ColPali earns its 100–1000× storage premium specifically on **visually rich, layout-heavy, OCR-hostile** documents. Always benchmark both on *your* corpus before committing; do not assume the fancier method wins universally.

## Putting It Together: When to Reach for This

The decision is a corpus question, not a fashion question. Reach for ColPali-style visual-document retrieval when:

- Your corpus is **PDFs, slides, scans, or screenshots** with meaningful **layout, tables, figures, or charts**.
- OCR is **lossy or failing** on your documents (multi-column, non-Latin scripts, handwriting, complex tables).
- You can afford a **multi-vector index** (or the compression to make it affordable) and you want to skip the brittle OCR-and-layout engineering entirely.

Stay with text RAG (and OCR if needed) when the corpus is **born-digital clean text**, when **index size is tightly constrained**, or when you need **exact lexical matching** on identifiers as the dominant signal (though hybrid fusion lets you have both). And remember the cost asymmetry: visual retrieval shifts expense from a parsing pipeline (offline, one-time) to **storage** (multi-vector index) and **generation tokens** (page images in the VLM). For high-value, layout-heavy corpora that trade is usually worth it; for commodity text it usually is not.

This chapter closes Part IX. The retrieval mechanisms here — dual encoders, late interaction, multi-vector indexing, staged reranking — are the same primitives you have seen throughout the part, recombined for pixels instead of tokens. The generation side hands off directly to Part X: see [Vision-Language Models](../10-multimodal-and-arch/02-vision-language-models.html) for how the VLM actually reads those retrieved pages, and [Vision Transformers & Image Encoders](../10-multimodal-and-arch/01-vision-transformers.html) for the patch encoders that make any of this possible.

!!! key "Key Takeaways"

    - **Single-vector cross-modal retrieval (CLIP/SigLIP)** puts images and text in one space but compresses each page to one vector — an information bottleneck that fails on dense, multi-fact documents.
    - **Late interaction (ColBERT's MaxSim)** keeps one vector per token/patch and scores by summing each query token's best match, recovering fine-grained matching while keeping documents precomputable.
    - **ColPali / ColQwen2** extend late interaction to vision: a VLM turns a page *image* into ~1024 patch vectors (128-dim), scored against multi-vector text queries via MaxSim — **no OCR required**.
    - **Multi-vector indexing is the engineering crux:** ~1000× more vectors than single-vector means PLAID (residual/centroid compression), HNSW-over-patches, token pooling, MUVERA, and binary quantization are essential, not optional.
    - **The OCR-free pipeline** feeds retrieved **page images** straight to a VLM generator, preserving layout, tables, and charts that OCR destroys — but page images are token-expensive, so retrieval **precision** matters more than in text RAG.
    - **Reranking still helps:** exact-MaxSim rerank after approximate candidate generation, VLM-as-judge cross-encoding, and RRF fusion with OCR/BM25 for exact-identifier queries.
    - **Evaluate on ViDoRe** with nDCG@k/Recall@k, separating retrieval quality from end-to-end answer correctness, and stratify by document difficulty and language.
    - **Reach for it on layout-heavy, OCR-hostile corpora**; stay with text RAG on clean born-digital text where the storage premium buys nothing.

!!! sota "State of the Art & Resources (2026)"

    Visual-document retrieval went from a niche idea to a default for layout-heavy corpora in under two years, driven by the ColPali line of work and a fast-maturing indexing ecosystem.

    **Foundational work**

    - [Radford et al., *Learning Transferable Visual Models From Natural Language Supervision (CLIP)* (2021)](https://arxiv.org/abs/2103.00020) — the dual-encoder contrastive recipe that put images and text in a shared space.
    - [Zhai et al., *Sigmoid Loss for Language Image Pre-Training (SigLIP)* (2023)](https://arxiv.org/abs/2303.15343) — sigmoid contrastive loss; the vision backbone underpinning PaliGemma and ColPali.
    - [Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT* (2020)](https://arxiv.org/abs/2004.12832) — introduced late interaction and the MaxSim operator.

    **Visual late interaction**

    - [Faysse et al., *ColPali: Efficient Document Retrieval with Vision Language Models* (2024)](https://arxiv.org/abs/2407.01449) — the central paper of this chapter; VLM patch embeddings + MaxSim + the ViDoRe benchmark.
    - [Santhanam et al., *PLAID: An Efficient Engine for Late Interaction Retrieval* (2022)](https://arxiv.org/abs/2205.09707) — residual compression and centroid pruning for production multi-vector indexing.
    - [Jayaram et al., *MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings* (2024)](https://arxiv.org/abs/2405.19504) — collapses multi-vector sets into single vectors so standard ANN indexes can do candidate generation.

    **Open-source & tools**

    - [illuin-tech/colpali (`colpali-engine`)](https://github.com/illuin-tech/colpali) — reference ColPali/ColQwen2 training and inference.
    - [stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT) — the original ColBERT + PLAID engine.
    - Multi-vector support in Qdrant, Vespa (tensor MaxSim ranking), and Weaviate for building ColPali indexes on production vector databases.

## Further Reading

- Faysse et al., *ColPali: Efficient Document Retrieval with Vision Language Models*, 2024 — the foundational paper and source of the ViDoRe benchmark.
- Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*, SIGIR 2020.
- Santhanam et al., *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*, NAACL 2022.
- Santhanam et al., *PLAID: An Efficient Engine for Late Interaction Retrieval*, CIKM 2022.
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, ICML 2021.
- Zhai et al., *Sigmoid Loss for Language Image Pre-Training (SigLIP)*, ICCV 2023.
- Beyer et al., *PaliGemma: A versatile 3B VLM for transfer*, 2024 — the backbone of the original ColPali.
- Jayaram et al., *MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings*, 2024.
- The `colpali-engine` and `ColBERT` open-source repositories for reference implementations.
