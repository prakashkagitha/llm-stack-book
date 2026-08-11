# 15.2 Data at Scale: datatrove and Hugging Face datasets

[Data: Sourcing, Filtering, Dedup, Tokenize & Pack ~20B Tokens](../14-capstone/02-data-pipeline.html) built `Stack-100M`'s corpus by hand: a source registry, a lean domain-routed quality filter, a MinHash-and-LSH deduplicator written against nothing but `numpy`, a greedy packer, and a `uint16` memmap shard writer. Every line of that code was chosen so the *mechanism* would be visible — you could set a breakpoint inside the MinHash permutation sweep and watch a Jaccard estimate materialize from raw hashes. That chapter also told you, honestly, where the from-scratch version stops working: past a few hundred thousand documents its in-RAM LSH index silently stops catching duplicates, and its single-threaded MinHash pass costs on the order of 44 core-hours for 20 million documents — a controllable but real tax that does not shrink further without more machines.

This chapter is the answer to "fine, now what do I actually run." Two libraries do this job at production scale in 2026: **Hugging Face `datasets`**, for sourcing, streaming, and mixing already-curated corpora such as FineWeb-Edu and Cosmopedia v2 straight off the Hub, and **`datatrove`** (HuggingFace's own pipeline library — the tool FineWeb itself was built with), for the heavier stages: extracting text from raw Common Crawl WARC files, running the quality-filter battery, deduplicating at cluster scale, and dispatching all of it across a Slurm cluster with one line changed. We will build the same FineWeb-style pipeline `datatrove`'s own examples ship, and then close the loop by writing its output into the *exact* `uint16` memmap shard format `PackedMemmapDataset` (Ch. 14.2) reads — so `stacklm`'s training loop cannot tell whether a shard came from the toy synthetic corpus or a real multi-terabyte Common Crawl dump. Along the way we will look at **NeMo Curator** and **Dolma**, the two other toolkits a working engineer is likely to meet, and be explicit about what each buys you that the others don't.

Throughout, treat every API call as a snapshot: `datatrove` and `datasets` are both under active development, block names move between minor releases, and executor keyword arguments occasionally get renamed. Pin your versions (`pip install "datatrove[all]==<version>" "datasets==<version>"` in a lockfile, not a bare `pip install datatrove`), and when a call in this chapter does not match what you see installed, trust the installed package's docstring and the upstream `examples/` directory over this page — the *mechanism* below (readers feed filters feed dedup feed writers, executed by a task-parallel runner) is what will still be true.

## From Hand-Rolled to Production: What Changes

Before diving into code, it is worth being precise about which of Ch. 14.2's problems each library actually solves, because "use the library" is not automatically an upgrade on every axis — it trades one set of costs (your engineering time, your visibility into the mechanism) for another (a dependency, a less legible stack trace, an API that moves under you).

| Stage | `stacklm` (Ch. 14.2, hand-rolled) | Production library | What you get that you didn't have |
|---|---|---|---|
| Sourcing curated HF datasets | `DataMixEntry` registry + `load_dataset(streaming=True)` wrapped in a hand-checked column probe | `datasets` (same call, no wrapper) | Nothing new — Ch. 14.2 *is* already the `datasets` streaming API; the registry exists to catch config/column mistakes before they cost GPU-hours |
| Raw WARC → clean text | not implemented (Ch. 14.2 starts from already-curated HF datasets) | `datatrove` `WarcReader` + `Trafilatura` extractor | The entire "build your own FineWeb from Common Crawl" capability |
| Quality filtering | `filters.py`, ~60 lines, 3 domain-routed heuristics | `datatrove.pipeline.filters` (Gopher, C4, FineWeb, language, URL) | The published, ablated filter battery behind FineWeb-Edu itself, not a reimplementation of its *spirit* |
| Near-duplicate dedup | `dedup.py`, in-RAM `SignatureStore` + `LSHIndex`, hard ceiling ≈ 500k–2M docs | `datatrove.pipeline.dedup` 4-stage disk-backed MinHash | No RAM ceiling; crash-safe per-stage checkpoints; runs on a cluster |
| Distributing the work | none — single process, single core | `LocalPipelineExecutor` → `SlurmPipelineExecutor` | Task parallelism and cluster dispatch from a one-line executor swap |
| Packing + shard format | `pack.py` + `shard.py`, bespoke `<bos>`-marked `uint16` memmap | *(no drop-in equivalent — see §5)* | Nothing; this stays yours, and §5 explains why |

The last row matters enough to say up front: there is no library call that produces `stacklm`'s exact on-disk shard format, because that format is a *design decision* specific to this book's training loop (position ids derived from `<bos>` markers, no stored `.pos.bin`, a `manifest.json` carrying `bos_id`). `datatrove` ships its own `DocumentTokenizer` writer, which packs and tokenizes into a binary stream tuned for frameworks like `nanotron`; it is excellent if you are adopting that whole stack, but it does not write `stacklm`'s shards. Section 5 shows the bridge: use `datatrove` (or `datasets`) for everything upstream of packing, and keep Ch. 14.2's `pack_documents`/`ShardWriter` as the final, bespoke step. This is the realistic shape of most production data pipelines — a general-purpose library for the expensive, generic stages, and a thin, hand-written adapter at the boundary where your training code has opinions.

## Streaming the Mix with Hugging Face `datasets`

`Stack-100M`'s four-source mix is fixed in `capstone/PLAN.md` §2 and reproduced exactly in Ch. 14.2's `STACK100M_MIX` registry: 70% FineWeb-Edu, 15% Cosmopedia v2, 10% StarCoder (Python), 5% FineMath. Every one of those sources already lives on the Hugging Face Hub as a `datasets`-compatible corpus, stored as sharded Parquet under the hood — which is exactly the case `datasets` streaming was built for. There is no need to hand-write a source registry to reproduce Ch. 14.2's caution about multi-config repos and wrong text columns; `datasets` will tell you about the first one loudly (`ValueError: Config name is missing`) and stay silent about the second, which is exactly the failure mode Ch. 14.2's `stream_hf` probe guards against. That guard is still worth keeping in production code — `datasets` gives you the primitive, not the safety net.

```python
"""
Stream the Stack-100M mix directly from the Hub, in production.

pip install "datasets>=2.19,<4" huggingface_hub
(the streaming/interleave API here has been stable since datasets 2.x;
pin a version in your lockfile rather than trusting "latest")
"""
from datasets import load_dataset, interleave_datasets

# Each load_dataset(..., streaming=True) call returns an IterableDataset:
# no download, no disk write -- rows are pulled from the Hub's Parquet
# shards lazily, exactly like Ch. 14.2's stream_hf but without the
# hand-rolled generator plumbing.
fineweb_edu = load_dataset(
    "HuggingFaceFW/fineweb-edu", name="sample-100BT",
    split="train", streaming=True, revision="main",
)
cosmopedia_v2 = load_dataset(
    "HuggingFaceTB/smollm-corpus", name="cosmopedia-v2",
    split="train", streaming=True, revision="main",
)
# starcoderdata is gated (accept terms on the Hub + huggingface_hub.login()
# first) and sharded by language via data_dir=, exactly as Ch. 14.2 notes.
starcoder_py = load_dataset(
    "bigcode/starcoderdata", data_dir="python",
    split="train", streaming=True, revision="main",
)
finemath = load_dataset(
    "HuggingFaceTB/finemath", name="finemath-4plus",
    split="train", streaming=True, revision="main",
)

# --- normalize the text column BEFORE interleaving --------------------
# starcoderdata's text lives in "content", not "text" -- the same pitfall
# Ch. 14.2 asserts against. .map() on a streaming dataset is lazy: nothing
# runs until you iterate.
starcoder_py = starcoder_py.map(
    lambda row: {"text": row["content"], "source": "starcoder", "domain": "code"}
)
fineweb_edu = fineweb_edu.map(
    lambda row: {"text": row["text"], "source": "fineweb_edu", "domain": "web"}
)
cosmopedia_v2 = cosmopedia_v2.map(
    lambda row: {"text": row["text"], "source": "cosmopedia_v2", "domain": "synthetic"}
)
finemath = finemath.map(
    lambda row: {"text": row["text"], "source": "finemath", "domain": "math"}
)

# interleave_datasets is the library equivalent of Ch. 14.2's
# interleave_budgeted -- weighted round-robin sampling over several
# IterableDatasets. Note the stopping_strategy tradeoff called out there:
mix = interleave_datasets(
    [fineweb_edu, cosmopedia_v2, starcoder_py, finemath],
    probabilities=[0.70, 0.15, 0.10, 0.05],
    seed=1337,
    stopping_strategy="all_exhausted",  # see warning below
)

mix = mix.shuffle(seed=1337, buffer_size=100_000)  # reservoir shuffle, streaming
```

!!! warning "`stopping_strategy` changes what "70/15/10/5" actually means"

    `interleave_datasets` supports two stopping strategies, and the choice silently changes the realized mix over a long stream. `"first_exhausted"` (the default) stops the *entire* interleaved stream the moment any one source runs dry — with FineMath (the smallest source at 5% weight and the fewest raw rows) this can truncate the whole 20B-token run early, well before FineWeb-Edu's much larger pool is touched. `"all_exhausted"` instead keeps going until every source is exhausted, **oversampling** — repeating — whichever sources run out first, so a source can appear more than once per "epoch" over the mix. Neither behavior matches Ch. 14.2's `interleave_budgeted`, which enforces an explicit token budget per source and *drops* a source once its budget is met rather than truncating the whole stream or oversampling. For a fixed, exact 20B-token target, wrap `interleave_datasets` output in the same kind of budget-tracking loop Ch. 14.2 uses, or accept `"all_exhausted"`'s oversampling as an acceptable approximation and monitor the realized per-source token counts in your manifest, the same way `build_corpus.py` does.

For multi-GPU pretraining, one more `datasets` primitive matters: **`split_dataset_by_node`**. Each data-parallel rank must see a disjoint slice of the stream — not the same shuffled stream re-read from rank 0, which would mean every GPU trains on identical data.

```python
from datasets.distributed import split_dataset_by_node

# Call this AFTER shuffling, once per rank, inside the training script.
# It shards the underlying Parquet files across ranks where possible and
# falls back to a strided skip otherwise -- either way, rank r never sees
# a document rank r' != r also sees in the same epoch.
rank, world_size = 2, 8  # from torch.distributed.get_rank()/get_world_size()
mix_for_this_rank = split_dataset_by_node(mix, rank=rank, world_size=world_size)
```

This is the piece that has no analogue in Ch. 14.2 at all, because the from-scratch pipeline pre-shards to disk once (`ShardWriter`) and lets `PackedMemmapDataset` + `DistributedSampler` handle rank assignment at read time — a materialize-then-shard design. `datasets` streaming supports the opposite design, shard-at-read-time, which is attractive when the corpus is too large to ever fully materialize or when you want to change the mix between runs without re-writing shards. Both are legitimate; `Stack-100M`'s 20B tokens (about 40 GB tokenized) comfortably fits the materialize-once design, which is why Ch. 14.2 chose it. At the scale of FineWeb's full multi-trillion-token release, materializing every mix is not an option, and streaming-with-rank-sharding is closer to how frontier labs actually feed a training job.

`datasets` gives you sourcing, streaming, mixing, and rank-sharding — the input side of the pipeline. It does **not** give you quality filtering beyond whatever `.filter()` predicate you write by hand, and it has no fuzzy deduplication at all; those are exactly the stages `datatrove` exists for, and exactly the stages where the from-scratch code in Ch. 14.2 hits its real ceiling.

## The FineWeb Recipe With `datatrove`: Readers, Extractors, and Filters

`Stack-100M`'s mix starts from already-curated Hub datasets, which sidesteps the hardest stage of building a corpus like FineWeb-Edu in the first place: turning raw Common Crawl WARC (Web ARChive) dumps into clean, deduplicated, quality-filtered text. That is the job `datatrove` was built for, and walking through it — even though `Stack-100M`'s flagship run does not need to redo it — is the fastest way to understand what "quality-filtered web text" actually costs to produce, and it is the pipeline you would run if you wanted your *own* FineWeb-Edu variant over a different crawl, a different language, or a different time slice of the web.

A `datatrove` pipeline is a plain Python list of `PipelineStep` objects, executed in order by an `Executor`. Each step consumes a stream of `Document` objects and yields a (possibly filtered, possibly transformed) stream of `Document` objects — the same generator-of-documents shape as every function in Ch. 14.2's `dedup.py` and `filters.py`, which is not a coincidence; `datatrove`'s block interface is close to the smallest abstraction that supports both streaming memory and independent, restartable stages.

```python
"""
capstone/scripts/datatrove_fineweb_pipeline.py  (production path -- needs
`pip install "datatrove[all]"`; not run in CI -- needs real WARC data and
several CPU-hours minimum)

Reproduce the shape of the FineWeb / FineWeb-Edu extraction-and-filter
pipeline over a slice of Common Crawl, ending in filtered JSONL that Section 5
below tokenizes and packs into stacklm's shard format.
"""
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    URLFilter,               # blocklists: adult content, spam domains, etc.
    LanguageFilter,          # fastText language ID; keep English only
    GopherRepetitionFilter,  # reject documents dominated by repeated n-grams
    GopherQualityFilter,     # Rae et al. 2021 heuristic quality gate
    C4QualityFilter,         # Raffel et al. 2020's C4 heuristics
    FineWebQualityFilter,    # the additional filters FineWeb itself adds
)
from datatrove.pipeline.writers.jsonl import JsonlWriter

# A Common Crawl "segment" path -- one dump's worth of WARC file listings.
# In production this comes from CC's published warc.paths.gz index; a real
# run processes many segments across many crawl dates.
CC_SEGMENT = "s3://commoncrawl/crawl-data/CC-MAIN-2025-XX/segments/.../warc.paths.gz"
OUT_DIR = "/scratch/stack100m/fineweb_style_filtered"

pipeline = [
    # 1. Read raw WARC records: HTTP response bodies as they were crawled,
    #    including headers, encoding quirks, and non-HTML content we'll
    #    never keep.
    WarcReader(CC_SEGMENT, glob_pattern="*.warc.gz"),

    # 2. Extract the main article text out of the raw HTML, discarding
    #    navigation bars, ads, and boilerplate. Trafilatura is the
    #    extractor FineWeb itself uses; it is meaningfully better at this
    #    than a naive readability heuristic, at a real CPU cost per page.
    Trafilatura(favour_precision=True),

    # 3. URL-level filtering: drop known-bad domains before spending any
    #    more compute on a document that will be rejected anyway. Order
    #    matters -- URLFilter is cheap and goes first.
    URLFilter(),

    # 4. Language ID: keep only documents fastText scores as English above
    #    threshold. Running this before the heavier quality filters avoids
    #    scoring "quality" on text in the wrong language at all.
    LanguageFilter(languages=["en"]),

    # 5. The Gopher (Rae et al., 2021) and C4 (Raffel et al., 2020) heuristic
    #    batteries: word-count bounds, mean word length, symbol-to-word
    #    ratio, stop-word presence, repeated-line/paragraph fractions --
    #    the published, ablated version of what Ch. 14.2's filters.py
    #    approximates by hand in ~15 lines per domain.
    GopherRepetitionFilter(),
    GopherQualityFilter(),
    C4QualityFilter(),

    # 6. FineWeb's own additional heuristics on top of Gopher+C4 -- the
    #    filters Penedo et al. 2024 found made a measurable difference
    #    over the C4/Gopher baseline alone.
    FineWebQualityFilter(),

    # 7. Write what survives as gzip-compressed JSONL, sharded by task.
    JsonlWriter(OUT_DIR),
]

executor = LocalPipelineExecutor(
    pipeline=pipeline,
    tasks=32,                      # one task per available CPU core
    logging_dir="/scratch/stack100m/logs/fineweb_filter",
)

if __name__ == "__main__":
    executor.run()
```

Every filter in step 5–6 is a *published, citable* heuristic battery — Gopher's filters come from Rae et al.'s *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* (2021), C4's from Raffel et al.'s *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (2020), and `FineWebQualityFilter` from Penedo et al.'s FineWeb paper (2024). Ch. 14.2's hand-written `passes_web_filter` (min/max word count, mean word length bounds, alpha fraction, digit fraction, duplicate-line fraction) is a compressed, from-first-principles reconstruction of the *spirit* of exactly these filters — same intuitions (reject boilerplate, reject junk, reject repetition), a fraction of the coverage. Reading `datatrove`'s filter source after having implemented the toy version is one of the most efficient ways to see what a decade of "we tried removing this heuristic and perplexity got worse" institutional knowledge looks like in code.

!!! tip "Practitioner tip: order your filters cheapest-first"

    `URLFilter` costs a dictionary lookup; `GopherQualityFilter` tokenizes the document and computes half a dozen statistics; `Trafilatura` extraction is the most expensive step of all, often tens of milliseconds per page. `datatrove`'s pipeline list runs top to bottom per document, so ordering matters for wall-clock even though it does not change the final filtered set: reject on URL before you pay for language ID, reject on language before you pay for the Gopher battery. The pipeline above already follows this rule; if you add a custom filter, insert it by cost, not by conceptual tidiness.

This chapter builds directly on [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html), which covers the theory behind every one of these filters — what repetition detection is actually catching, why C4's heuristics were chosen, how language ID models are trained — in far more depth than a pipeline listing can. Treat this section as that chapter's applied, at-scale instantiation, the same relationship Ch. 14.2 has to [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html).

## Deduplication and Tokenization at Cluster Scale

Ch. 14.2 was explicit about where its from-scratch `near_dedup_stream` breaks: an in-RAM `SignatureStore` plus LSH buckets that measure roughly 3.8 KB per indexed document, giving a hard ceiling around 500k–2M documents on a typical box, past which recall for new documents silently drops to zero. `datatrove`'s `MinhashDedup*` blocks fix this the same way any large distributed system fixes an unbounded-memory problem: put the state on disk, keyed so each worker only ever needs its own slice.

```python
"""
capstone/scripts/datatrove_minhash_dedup.py  (production path)

MinHash near-duplicate detection over the filtered corpus from the previous
section, using the same 4-stage disk-backed pipeline HuggingFace used for
FineWeb. This is the library version of Ch. 14.2's dedup.py MinHasher/
LSHIndex/SignatureStore -- same algorithm (Broder 1997 MinHash + LSH banding),
no in-RAM ceiling.
"""
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import (
    MinhashDedupSignature, MinhashDedupBuckets,
    MinhashDedupCluster, MinhashDedupFilter,
)
from datatrove.pipeline.dedup.minhash import MinhashConfig
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig

# FineWeb's own settings: 5-grams, 112 permutations as 14 buckets x 8 hashes
# per bucket. The (bands, rows) = (14, 8) choice targets the same "~75%+
# Jaccard similarity" region Ch. 14.2's warning box derives from first
# principles for its own (16, 8) default -- (1/14)^(1/8) ~ 0.72 vs
# (1/16)^(1/8) ~ 0.71, essentially the same S-curve at 112 vs 128 hashes.
cfg = MinhashConfig(hash_config=HashConfig(precision=64),
                    num_buckets=14, hashes_per_bucket=8, n_grams=5)

IN = "/scratch/stack100m/fineweb_style_filtered"
WORK = "/scratch/stack100m/minhash"
OUT = "/scratch/stack100m/deduped"
TASKS = 64  # one task per core; SlurmPipelineExecutor scales this to a cluster

# Stage 1: compute a MinHash signature per document, sharded across tasks.
# Each task only ever holds its own slice of documents -- no global index.
stage1 = LocalPipelineExecutor(
    pipeline=[JsonlReader(IN),
              MinhashDedupSignature(output_folder=f"{WORK}/signatures", config=cfg)],
    tasks=TASKS, logging_dir=f"{WORK}/logs/signatures")

# Stage 2: one task PER BUCKET. Because signatures are sharded by bucket,
# each bucket's candidate-pair search is embarrassingly parallel and never
# needs another bucket's data -- the disk-backed analogue of Ch. 14.2's
# LSHIndex.buckets, but with no shared-memory ceiling.
stage2 = LocalPipelineExecutor(
    pipeline=[MinhashDedupBuckets(input_folder=f"{WORK}/signatures",
                                  output_folder=f"{WORK}/buckets", config=cfg)],
    tasks=cfg.num_buckets, logging_dir=f"{WORK}/logs/buckets", depends=stage1)

# Stage 3: a single task performs union-find clustering over every candidate
# pair found in stage 2, producing a canonical "keep" id per duplicate
# cluster -- global, deterministic, and order-independent, unlike Ch. 14.2's
# greedy streaming "keep whichever arrives first" rule.
stage3 = LocalPipelineExecutor(
    pipeline=[MinhashDedupCluster(input_folder=f"{WORK}/buckets",
                                  output_folder=f"{WORK}/remove_ids", config=cfg)],
    tasks=1, logging_dir=f"{WORK}/logs/cluster", depends=stage2)

# Stage 4: re-read the filtered corpus, drop every document flagged for
# removal, write the final deduplicated JSONL.
stage4 = LocalPipelineExecutor(
    pipeline=[JsonlReader(IN),
              MinhashDedupFilter(input_folder=f"{WORK}/remove_ids"),
              JsonlWriter(OUT)],
    tasks=TASKS, logging_dir=f"{WORK}/logs/filter", depends=stage3)

if __name__ == "__main__":
    stage4.run()  # `depends` chains the DAG; running the last stage runs all four
```

Three properties of this design are worth naming, because each is the direct fix for a specific limitation of Ch. 14.2's hand-rolled version. **Disk-backed hand-off between stages** means a crash at hour 30 of stage 2 loses stage 2's partial work, not stages 1's signatures or the whole run — the same instinct behind sharded checkpoints in [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html), applied to data engineering rather than training. `logging_dir` also doubles as a completion marker: re-running `stage4.run()` after a partial failure skips any task whose output already exists, so a crashed 64-task job resumes from task 41 rather than task 0. **Per-bucket task parallelism** in stage 2 is what removes the in-RAM ceiling entirely — there is no structure in this design analogous to `SignatureStore`'s fixed-capacity buffer, because no single process ever holds more than one bucket's signatures at once. **Global union-find clustering** in stage 3 fixes an accuracy issue, not just a scaling one: Ch. 14.2's `near_dedup_stream` decides duplicates greedily as it streams, so which member of a 5-way duplicate cluster survives depends on stream order — re-running with a different shuffle seed can keep a different copy. Union-find over the full candidate graph picks a canonical representative deterministically, independent of processing order.

!!! example "How long does this actually take?"

    Ch. 14.2 measured its from-scratch MinHash signing at roughly 8 ms/document on one core (vectorized `numpy`, `num_perm=128`), and estimated ≈ 20 million kept documents for a 20B-token corpus, giving

    $$
    20\times10^{6}\ \text{docs} \times 8\ \text{ms} \approx 1.6\times10^{5}\ \text{s} \approx 44\ \text{single-core-hours}
    $$

    for signing alone (bucketing, clustering, and re-filtering add more; this isolates the piece both implementations share). That number does not change just because the work moved to `datatrove` — the *algorithm* is identical, and this arithmetic is a lower bound on stage 1's compute either way. What changes is how many cores you can throw at it at once, and how gracefully a failure is absorbed.

    - **Single core (Ch. 14.2's script, run to completion):** ≈ 44 hours, and a crash at hour 40 costs all 40 hours.
    - **`LocalPipelineExecutor(tasks=64)` on one 64-core box:** signing parallelizes almost perfectly across tasks (each task hashes its own document shard independently), so wall-clock drops to roughly $44 / 64 \approx 0.7$ hours ≈ 41 minutes for stage 1 — plus bucketing (stage 2, `tasks=14`, comparably fast since each bucket's file is far smaller than the full corpus), clustering (stage 3, single task, operates only on the much smaller candidate-pair graph, not raw text), and the final filter pass (stage 4, another full corpus read, ≈ same order as stage 1). Budget on the order of two to three hours wall-clock for the whole four-stage job on one large box, dominated by I/O and the two full-corpus passes rather than by MinHash arithmetic itself.
    - **`SlurmPipelineExecutor` across 4 nodes × 64 cores (256 workers):** stage 1 alone drops toward $44/256 \approx 10$ minutes; the ceiling shifts to whichever stage has the least parallelism (stage 3's single-task clustering) and to shared filesystem I/O bandwidth, which a 4-node job stresses far more than a single box does.

    The honest caveat: this arithmetic isolates signing cost and ignores extraction (`Trafilatura`, the most expensive stage in the *whole* pipeline, often tens of milliseconds per page), filter evaluation, and network/disk I/O for a real multi-terabyte WARC input — all of which typically dominate total wall-clock at true FineWeb scale. Treat these numbers as "how the MinHash stage specifically scales with task count," not as a full-pipeline benchmark; measure your own throughput on your own hardware and corpus before committing a cluster budget.

`datatrove` also exposes the alternatives worth knowing when its defaults don't fit: **[`text-dedup`](https://github.com/ChenghaoMou/text-dedup)** packages the same MinHash/SimHash/suffix-array algorithms as focused, dependency-light implementations outside a full pipeline framework — a reasonable middle ground between Ch. 14.2's from-scratch code and `datatrove`'s cluster-oriented design, if all you need is dedup and not the whole readers-filters-writers stack.

## From Filtered Documents to Stack-100M's Packed Shards

Here is where the production pipeline has to become Ch. 14.2's again. `datatrove`'s own `DocumentTokenizer` writer (`datatrove.pipeline.tokens`) tokenizes and packs documents into a binary token stream tuned for consumption by frameworks such as `nanotron` — genuinely useful if that is your downstream trainer, but it is not `stacklm`'s shard format: no `<bos>`-derived position/segment reconstruction, a different on-disk layout, a different manifest schema. Rather than fight the writer to match a format it was never designed to produce, we treat everything upstream of packing — sourcing, filtering, deduplication — as `datatrove`'s job, and keep Ch. 14.2's `pack_documents` + `ShardWriter` as the final, bespoke bridge. This is not a compromise; it is the realistic shape of most real data pipelines, where a general-purpose library handles the expensive generic stages and a thin adapter handles the one stage where your training code has strong opinions.

The bridge runs as its own `datatrove` pipeline stage, which means it inherits task parallelism, resumability, and (if you need it) Slurm dispatch for free — each worker tokenizes and packs its own slice of the deduplicated corpus and writes its own shard set, then a small merge step combines the per-worker manifests.

```python
"""
capstone/scripts/datatrove_to_shards.py  (production path)

Final stage: read datatrove's filtered + deduplicated JSONL (Sections 3-4),
tokenize with Stack-100M's trained BPE (Ch. 14.3), and write the EXACT SAME
uint16 memmap shard format capstone/stacklm/data/shard.py's ShardWriter
produces by hand -- PackedMemmapDataset cannot tell the difference.
"""
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader

from stacklm.data.pack import pack_documents
from stacklm.data.shard import ShardWriter
from stacklm.tokenizer import load_tokenizer  # Ch. 14.3's trained BPE, wrapped

class StackShardWriterStep(PipelineStep):
    """A datatrove pipeline block: consume this task's slice of documents,
    tokenize + pack to SEQ_LEN=2048 windows, write one uint16 shard set.

    Reuses Ch. 14.2's pack_documents/ShardWriter unmodified -- this class is
    only glue between datatrove's Document stream and that generator-of-dicts
    interface.
    """
    name = "Stack-100M shard writer"
    type = "writer"

    def __init__(self, out_dir: str, tokenizer_path: str, seq_len: int = 2048,
                 tokens_per_shard: int = 100_000_000):
        super().__init__()
        self.out_dir = out_dir
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.tokens_per_shard = tokens_per_shard

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        tok = load_tokenizer(self.tokenizer_path)  # exposes bos_id/eos_id/pad_id/encode
        worker_dir = f"{self.out_dir}/worker{rank:02d}"
        writer = ShardWriter(worker_dir, seq_len=self.seq_len,
                             tokens_per_shard=self.tokens_per_shard)
        # datatrove's Document has .text plus .metadata; we only need .text --
        # pack_documents wants the same {"text": ...} dict shape Ch. 14.2 uses
        # everywhere, so this generator is the entire adapter.
        docs = ({"text": d.text} for d in data)
        for input_ids, position_ids in pack_documents(docs, tok, seq_len=self.seq_len):
            writer.add(input_ids, position_ids)
        writer.close()
        writer.write_manifest(tokenizer=tok, extra={"worker_rank": rank})
        return
        yield  # PipelineStep.run must be a generator; this step is terminal

IN = "/scratch/stack100m/deduped"
OUT = "/scratch/stack100m/shards"

executor = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(IN),
        StackShardWriterStep(out_dir=OUT, tokenizer_path="capstone/artifacts/tokenizer"),
    ],
    tasks=32,
    logging_dir="/scratch/stack100m/logs/shard_write",
)

if __name__ == "__main__":
    executor.run()
```

Each of the 32 workers now owns a `worker{rank:02d}/` directory with its own `shard_00000.tokens.bin`, `.meta.bin`, and `manifest.json` — correct, self-contained, and immediately readable by `PackedMemmapDataset(f"{OUT}/worker07")` on its own. To hand the whole corpus to the training loop as one directory, flatten and merge:

```python
"""
capstone/scripts/merge_shard_workers.py

Combine N workers' shard_*.{tokens,meta}.bin + manifest.json into one flat
directory with globally renumbered shards, so PackedMemmapDataset(OUT) (which
globs shard_*.meta.bin non-recursively) sees the whole corpus.
"""
import json
import shutil
from pathlib import Path

def merge_shard_workers(worker_root: str, out_dir: str) -> dict:
    worker_root, out = Path(worker_root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    global_idx = 0
    totals = {"n_shards": 0, "n_sequences": 0, "n_tokens": 0}
    per_worker_manifests = sorted(worker_root.glob("worker*/manifest.json"))
    assert per_worker_manifests, f"no worker manifests under {worker_root}"

    base_manifest = json.loads(per_worker_manifests[0].read_text())
    for man_path in per_worker_manifests:
        man = json.loads(man_path.read_text())
        # bos_id/eos_id/pad_id/seq_len must agree across every worker -- they
        # all loaded the same tokenizer, but a stale worker from a previous
        # tokenizer version would silently corrupt position-id reconstruction.
        for key in ("bos_id", "eos_id", "pad_id", "seq_len"):
            assert man[key] == base_manifest[key], f"{man_path}: {key} mismatch"
        worker_dir = man_path.parent
        for meta_path in sorted(worker_dir.glob("shard_*.meta.bin")):
            stem = str(meta_path)[: -len(".meta.bin")]
            new_stem = out / f"shard_{global_idx:05d}"
            shutil.copy(stem + ".tokens.bin", str(new_stem) + ".tokens.bin")
            shutil.copy(stem + ".meta.bin", str(new_stem) + ".meta.bin")
            global_idx += 1
        totals["n_shards"] += man["n_shards"]
        totals["n_sequences"] += man["n_sequences"]
        totals["n_tokens"] += man["n_tokens"]

    merged = {**base_manifest, **totals}
    (out / "manifest.json").write_text(json.dumps(merged, indent=2))
    return merged

if __name__ == "__main__":
    stats = merge_shard_workers("/scratch/stack100m/shards", "/scratch/stack100m/shards_merged")
    print(f"merged {stats['n_shards']} shards, {stats['n_tokens']:,} tokens")
```

`shutil.copy` is deliberate rather than `os.rename`: worker directories and the merged output frequently live on different filesystems in a real cluster job (local scratch per node vs. shared storage), where a rename across filesystems fails. If `worker_root` and `out_dir` are guaranteed to share a filesystem, swap in `Path.rename` for a near-instant move instead of a full copy — at 40 GB the difference between the two is minutes, not seconds.

The same bridge pattern applies verbatim to the `datasets`-streaming path from Section 2: pipe `mix` (the interleaved `IterableDataset`) through `quality_filter` (Ch. 14.2's `filters.py`, unchanged) and directly into `pack_documents`/`ShardWriter`, skipping `datatrove` entirely for the smaller, curated-source case where you don't need cluster-scale WARC extraction or fresh MinHash dedup.

```python
"""
capstone/scripts/hf_streaming_to_shards.py

The lighter-weight bridge: datasets streaming -> quality filter (reused from
Ch. 14.2, unchanged) -> pack -> shard. No datatrove needed when the sources
are already-curated HF datasets and you trust their existing dedup.
"""
from stacklm.data.filters import quality_filter
from stacklm.data.pack import pack_documents
from stacklm.data.shard import ShardWriter
from stacklm.tokenizer import load_tokenizer

# `mix` is the interleave_datasets() IterableDataset built in Section 2.
def build_shards_from_hf_stream(mix, tokenizer_path: str, out_dir: str,
                                token_budget: int = 20_000_000_000) -> None:
    tok = load_tokenizer(tokenizer_path)
    writer = ShardWriter(out_dir, seq_len=2048, tokens_per_shard=100_000_000)
    filtered = (row for row in mix if quality_filter(row))
    n_tokens = 0
    for input_ids, position_ids in pack_documents(filtered, tok, seq_len=2048):
        writer.add(input_ids, position_ids)
        n_tokens += len(input_ids)
        if n_tokens >= token_budget:
            break
    writer.close()
    writer.write_manifest(tokenizer=tok, extra={"source": "hf_datasets_streaming"})
```

Because a real FineWeb-Edu/Cosmopedia mix has already been through its own filtering and dedup upstream on the Hub, this path is legitimately simpler than the full `datatrove` pipeline — `Stack-100M`'s flagship recipe uses exactly this shape, which is why Ch. 14.2's registry and this section's code share so much surface area. Reach for the full `datatrove` WARC pipeline in Sections 3–4 only when you need to build a corpus that does not already exist on the Hub in curated form.

## Scaling Out: `LocalPipelineExecutor` → `SlurmPipelineExecutor`

Every pipeline in this chapter has used `LocalPipelineExecutor`. Moving to a cluster is deliberately a one-line change — `datatrove`'s executor abstraction is built around exactly this promise, and it is worth seeing the Slurm version to know what it actually does under the hood rather than treating it as magic.

```python
"""
Same MinHash-signing pipeline as Section 4, dispatched to a Slurm cluster
instead of one local box. Everything upstream of the executor is unchanged.
"""
from datatrove.executor.slurm import SlurmPipelineExecutor
# ... same `pipeline=[JsonlReader(IN), MinhashDedupSignature(...)]` as before

stage1 = SlurmPipelineExecutor(
    pipeline=[...],                      # identical pipeline list
    tasks=2000,                          # total parallel tasks across the job
    time="06:00:00",                     # per-task Slurm walltime limit
    partition="cpu",                     # your cluster's partition/queue name
    cpus_per_task=1,
    mem_per_cpu_gb=4,
    logging_dir="/shared/stack100m/logs/signatures",  # MUST be shared storage
    job_name="fineweb-minhash-sign",
    max_array_size=1000,                 # Slurm job-array size cap; datatrove
                                          # chunks 2000 tasks into batches of it
)

if __name__ == "__main__":
    stage1.run()  # generates an sbatch script, submits it, blocks until done
```

Under the hood, `SlurmPipelineExecutor.run()` writes an `sbatch` job-array script that calls back into the same Python pipeline definition once per array task, with `rank`/`world_size` set from `SLURM_ARRAY_TASK_ID`; each task processes its own slice of the input and writes to its own output file, exactly as `LocalPipelineExecutor`'s worker processes do — the only thing that changed is *which scheduler* launches those workers. Two consequences follow directly from that design, and both matter operationally:

- **`logging_dir` must be on shared storage** (NFS, Lustre, or equivalent) that every compute node can read and write, because that is how the executor knows which tasks already completed on a previous, partially-failed run — re-submitting the same job skips finished tasks rather than redoing them, the cluster-scale version of the resumability property noted in Section 4.
- **`depends=stageN`** chains stages into a DAG the same way on Slurm as locally, but on Slurm this becomes a real Slurm job dependency (`--dependency=afterok:<jobid>`) rather than an in-process wait — stage 2 does not occupy allocation while it waits for stage 1's array to finish.

!!! warning "Version drift: executor kwargs move between releases"

    The keyword arguments shown above (`time`, `partition`, `cpus_per_task`, `mem_per_cpu_gb`, `max_array_size`) reflect the shape of `SlurmPipelineExecutor`'s configuration as of the versions this book was written against, and are exactly the kind of surface that changes between `datatrove` releases — a new Slurm-specific feature (QOS flags, GPU partitions, array-size defaults) tends to arrive as a renamed or added kwarg rather than a breaking rewrite. Before running this in production, `python -c "from datatrove.executor.slurm import SlurmPipelineExecutor; help(SlurmPipelineExecutor)"` against your installed version and diff it against what's above. The *shape* — pipeline list, `tasks`, `depends`, `logging_dir` on shared storage — has been stable across the versions we've used; the cluster-specific knobs are what drift.

The practical upshot for `Stack-100M`'s own scale — a single machine, a single afternoon, 20B tokens — is that you will likely never need `SlurmPipelineExecutor` at all; `LocalPipelineExecutor(tasks=<your core count>)` is the whole story. It becomes load-bearing the moment you rebuild a *fresh* FineWeb-Edu-scale corpus (trillions of tokens) rather than sampling an existing one, which is a different, much larger project than this capstone undertakes — but it is the exact code you would reach for if you did.

## Alternatives: NVIDIA NeMo Curator and AI2's Dolma Toolkit

`datatrove` is HuggingFace's tool and the one FineWeb itself was built with, but it is not the only production data-curation library a working engineer will meet, and it is worth knowing the other two well enough to pick correctly.

**NVIDIA NeMo Curator** takes the same readers-filters-dedup-writers shape as `datatrove` but targets GPU acceleration end to end, built on RAPIDS (`cuDF`/`Dask-cuDF`) rather than plain CPU Python/`numpy`. Its fuzzy deduplication runs MinHash-LSH on GPU, and it ships GPU-accelerated quality classifiers (including a "quality classifier" and domain/toxicity classifiers trained by NVIDIA) alongside the same heuristic filter families `datatrove` has. The case for reaching for it: once a CPU-bound `datatrove` pipeline's bottleneck genuinely is dedup or classifier-based filtering throughput — not I/O, not extraction — and you have GPUs sitting idle between training runs, NeMo Curator can turn that CPU-hour bill into a GPU-hour bill at a very different price point. It integrates naturally if the rest of your stack is already NVIDIA-centric (NeMo for training, Megatron-derived parallelism), the same way `datatrove` integrates naturally if the rest of your stack is HuggingFace-centric.

**AI2's Dolma toolkit** is the curation pipeline behind the Dolma corpus (Soldaini et al., 2024), and it takes a different shape entirely: rather than a Python pipeline-of-objects, Dolma is CLI-first and config-driven — you write a YAML spec describing taggers (heuristic and model-based document annotators, applied without discarding anything yet), a mixer (which combines tagger outputs into keep/drop decisions and produces the filtered corpus), and a deduper, then invoke the `dolma` CLI against that config. The taggers-then-mixer split is a genuine design difference worth understanding on its own terms: it separates "annotate every document with every signal you might ever want" from "decide what to keep," which means you can change your filtering *decision* — tighten a quality threshold, add a new exclusion rule — by re-running the (cheap) mixer over already-computed tags, without re-running the (expensive) taggers. `datatrove`'s filter blocks, by contrast, discard documents as they go, so revisiting a filtering decision means re-running extraction and filtering together, unless you have separately checkpointed an unfiltered intermediate output. If your team's workflow is "curate a corpus once, then iterate on the exact filtering thresholds for months," Dolma's separation of annotation from decision is a real advantage; if your workflow is closer to "one filtered corpus, move on," it is extra machinery.

| Tool | Language / execution model | GPU support | Best fit |
|---|---|---|---|
| `datatrove` | Python pipeline objects; `Local`/`Slurm` executors | CPU-primary | The HF-ecosystem default; what FineWeb was built with; this chapter's main path |
| `datasets` | Python `IterableDataset`; streaming/interleave | CPU-primary | Sourcing and mixing already-curated Hub corpora; no filtering/dedup of its own |
| NeMo Curator | Python, Dask/RAPIDS (`cuDF`) | GPU-accelerated | CPU-bound dedup/classifier bottlenecks; NVIDIA-centric stacks |
| Dolma toolkit | CLI + YAML config; taggers → mixer → deduper | CPU-primary | Re-iterable filtering decisions without re-running extraction; AI2/Dolma-corpus-compatible pipelines |

None of these tools disagree on the underlying algorithms — Gopher/C4 heuristics, fastText language ID, MinHash-LSH near-dedup are the shared vocabulary across all four, which is exactly why Ch. 14.2's from-scratch implementations transfer as *understanding* regardless of which production tool you end up running. What differs is execution model, GPU support, and how easy it is to revisit a decision after the fact — pick based on your cluster, your team's existing stack, and how often you expect to iterate on the filtering thresholds, not based on which paper you read most recently.

!!! interview "Interview Corner"

    **Q:** You have a from-scratch MinHash deduplicator that works correctly on a million-document corpus but falls over past a few million. Your production pipeline uses `datatrove`'s 4-stage `MinhashDedup*` blocks instead. What specifically does that redesign fix, and why can't you fix the from-scratch version by just increasing a buffer size?

    **A:** The from-scratch version keeps a single in-process index — signatures plus LSH buckets — sized by a fixed `index_capacity`. Every document's candidate lookup and insertion touches that one shared structure, so its memory is bounded by design, but the bound is a *hard ceiling*: past it, the code keeps running and keeps yielding documents, but silently stops detecting duplicates for everything after the ceiling — a correctness failure with no error, discoverable only by noticing an unexpectedly high duplicate rate later. Raising `index_capacity` only moves the ceiling; at 20 million documents no single machine's RAM moves it far enough, because the index cost is linear in corpus size by construction.

    `datatrove`'s design removes the single shared structure entirely. Stage 1 signs documents independently per task — no cross-task communication needed. Stage 2 partitions candidate search by LSH *bucket*, and because a document's signature is split into fixed bucket slices deterministically, every bucket's candidate set can be computed from only that bucket's data, in its own task, without ever seeing another bucket's documents. That is what makes it disk-backed and horizontally scalable: no process ever needs the whole index in memory, because no single index exists — only per-bucket shards that get combined once, in stage 3's union-find, over a much smaller graph of candidate *pairs* rather than raw documents. The fix isn't a bigger buffer; it's replacing an architecture with a fundamental memory bound by one whose per-worker memory doesn't grow with total corpus size at all.

## Worked Example: Sizing a Corpus Rebuild

!!! example "From a filtered-corpus target to a cluster request"

    Suppose you want to rebuild `Stack-100M`'s FineWeb-Edu slice yourself from a fresh Common Crawl segment rather than sampling the pre-built `sample-100BT` config, because you want a more recent crawl date. You need roughly 14B kept tokens (70% of the 20B budget), and FineWeb-Edu's own published pipeline keeps only a modest fraction of raw crawled pages after its educational-quality classifier — call it on the order of a few percent, since the vast majority of the raw web is not educational content by that classifier's standard (treat this as illustrative, not a number to build a cluster budget on without checking your own classifier's actual keep rate on a sample first).

    If your keep rate after extraction + language filter + Gopher/C4 + the educational classifier is $r$, you need to *process* roughly $14\text{B} / r$ raw tokens to end up with 14B kept ones. At an illustrative $r \approx 0.03$ (3% — in the right ballpark for aggressive educational-content filtering, not a verified FineWeb figure), that is:

    $$
    \text{tokens to process} \approx \frac{14\times10^{9}}{0.03} \approx 4.7\times10^{11} \approx 470\text{B raw tokens.}
    $$

    At a typical web document averaging on the order of 500–1000 tokens, that is roughly 500–950 million documents to run through extraction and filtering — two to three orders of magnitude more than the 20 million *kept* documents Section 4's dedup arithmetic assumed. This is the number that should set your cluster request, not the 20M-document post-filter estimate: `Trafilatura` extraction alone, at even an optimistic ~50 ms/page on one core, is

    $$
    7\times10^{8}\ \text{docs} \times 0.05\ \text{s} \approx 3.5\times10^{7}\ \text{s} \approx 9{,}700\ \text{core-hours}
    $$

    for extraction before any filtering or dedup runs at all — roughly 40 core-*days* on a single 256-core allocation (`SlurmPipelineExecutor(tasks=2000)` spread across several nodes gets you well under a day). This is precisely why Section 1's table lists "raw WARC → clean text" as a capability Ch. 14.2 never needed and `datatrove` exists to provide: `Stack-100M`'s actual flagship recipe sidesteps this entire calculation by sampling the Hub's *already-extracted, already-filtered* `sample-100BT` config, which is exactly the corner NeMo Curator's GPU acceleration and a real production cluster both exist to cut through, and exactly why "just sample an existing curated dataset" is the right default unless you have a specific reason (a fresher crawl date, a different language, a bespoke filter) to redo this work yourself.

## Key Takeaways

!!! key "Key Takeaways"

    - `datasets` streaming (`load_dataset(..., streaming=True)`, `interleave_datasets`, `.shuffle(buffer_size=...)`, `split_dataset_by_node`) is the production version of Ch. 14.2's source registry and `interleave_budgeted` — it handles sourcing, mixing, and rank-sharding curated Hub corpora, but does no fuzzy deduplication and only whatever filtering you write yourself.
    - `datatrove` — HuggingFace's own pipeline library, and the tool FineWeb was actually built with — is what fills that gap: `WarcReader` + `Trafilatura` for raw-crawl extraction, the Gopher/C4/FineWeb filter families for quality gating, and a 4-stage disk-backed `MinhashDedup*` pipeline for near-duplicate detection at a scale Ch. 14.2's in-RAM `SignatureStore` structurally cannot reach.
    - The redesign that removes MinHash's memory ceiling is architectural, not a bigger buffer: signing is per-task, bucketing is per-LSH-bucket, and only the much smaller candidate-pair graph (not raw documents) is ever globally clustered.
    - `LocalPipelineExecutor` → `SlurmPipelineExecutor` is deliberately a near-one-line change; both dispatch the identical pipeline list, and `logging_dir` on shared storage is what makes a partially-failed cluster job resumable rather than a restart-from-zero.
    - There is no library call that reproduces `stacklm`'s bespoke `<bos>`-marked `uint16` shard format — `datatrove`'s own `DocumentTokenizer` targets a different downstream trainer. The realistic pattern is a library for the expensive generic stages and a thin hand-written bridge (reusing Ch. 14.2's `pack_documents`/`ShardWriter` unmodified) at the one boundary where your training code has opinions.
    - NeMo Curator (GPU-accelerated, RAPIDS-based) and the Dolma toolkit (CLI-first, taggers-then-mixer) are the two other production tools worth knowing; pick based on GPU availability and how often you expect to revisit filtering thresholds, not algorithmic differences — Gopher/C4 heuristics and MinHash-LSH are the shared vocabulary across all of them.
    - Pin every version in a lockfile. Block names, executor kwargs, and even config schemas move between `datatrove` and `datasets` releases; treat this chapter's exact API calls as a snapshot of the mechanism, and check the installed package's own examples before trusting a signature verbatim.

## Further Reading

- Penedo, Kydlíček, et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* (2024) — the paper behind FineWeb-Edu and the filter battery this chapter's `datatrove` pipeline reproduces, including the published MinHash configuration.
- Rae et al., *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* (2021) — source of the Gopher quality and repetition heuristics used by `GopherQualityFilter`/`GopherRepetitionFilter`.
- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (2020) — introduces the C4 corpus and its heuristic filters (`C4QualityFilter`).
- Soldaini et al., *Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research* (2024) — the corpus and the taggers-then-mixer curation toolkit discussed in Section 6.
- Lee et al., *Deduplicating Training Data Makes Language Models Better* (2022) — the empirical case for near-duplicate removal that both Ch. 14.2's from-scratch MinHash and `datatrove`'s `MinhashDedup*` pipeline implement.
- [huggingface/datatrove](https://github.com/huggingface/datatrove) — source and `examples/` directory; check the FineWeb reproduction script there for the current, exact API against whichever version you install.
- [huggingface/datasets](https://github.com/huggingface/datasets) — streaming, `interleave_datasets`, and `datasets.distributed.split_dataset_by_node` documentation.
- [NVIDIA/NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator) — GPU-accelerated fuzzy dedup and classifier-based filtering.
- [allenai/dolma](https://github.com/allenai/dolma) — AI2's CLI-first curation toolkit (taggers, mixer, deduper) behind the Dolma corpus.
- [ChenghaoMou/text-dedup](https://github.com/ChenghaoMou/text-dedup) — focused MinHash/SimHash/suffix-array dedup implementations outside a full pipeline framework.
- [Data: Sourcing, Filtering, Dedup, Tokenize & Pack ~20B Tokens](../14-capstone/02-data-pipeline.html) — the from-scratch version this chapter builds on throughout.
- [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) and [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html) — the general theory behind every filter and dedup stage in this chapter.
- [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html) — the fault-tolerance principles behind `datatrove`'s per-stage, resumable disk hand-offs.
