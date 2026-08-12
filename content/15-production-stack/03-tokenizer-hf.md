# 15.3 Tokenizer Training: Hugging Face tokenizers and SentencePiece

[Chapter 14.3](../14-capstone/03-tokenizer.html) built a byte-level BPE tokenizer from raw Python: an incremental heap-based trainer, a hand-rolled encoder with a per-chunk cache, and — at the very end — an exporter into `tiktoken` and `transformers.PreTrainedTokenizerFast` so the rest of the ecosystem could actually load the thing. That chapter earned its keep: you now know exactly what a `tokenizer.json` *means*, because you built one. This chapter answers the question that follows naturally from that: if I'm not doing this to learn the algorithm, what do I actually run at work?

The answer, in 2026, is almost always Hugging Face's [`tokenizers`](https://github.com/huggingface/tokenizers) library — a Rust implementation with Python bindings that trains and encodes orders of magnitude faster than pure Python, and produces the exact `tokenizer.json` artifact that `transformers`, `TRL`, `vLLM`, `SGLang`, and `llama.cpp`'s converter all already know how to read. For a narrower set of cases — mostly language modeling on non-English-dominant corpora, or when you need a `Unigram` language model instead of BPE — you reach for Google's [`sentencepiece`](https://github.com/google/sentencepiece) instead. This chapter trains Stack-100M's real production tokenizer with both, shows exactly which knobs matter, and pins down the two or three places where the library's defaults quietly disagree with the from-scratch chapter's choices — because those disagreements are the kind of thing that costs you a day of debugging if you don't know to look for them.

We will not re-derive the BPE algorithm — that's [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html) and [Chapter 14.3](../14-capstone/03-tokenizer.html). This chapter is about the tool.

## Why Reach for a Library At All

The from-scratch trainer in Chapter 14.3 is not slow because Python is slow in some vague sense — it is slow because Python's interpreter overhead dominates an algorithm that touches millions of small integer tuples in a tight loop, and because it runs on a single core. `tokenizers` fixes both: the trainer and the encoder are written in Rust, and the encoder additionally parallelizes over documents with Rust-native threads (no GIL to fight). Chapter 14.3 measured this directly on the same corpus: HuggingFace's `trainers.BpeTrainer` finished a 32,768-entry vocabulary on an 8.34 MB sample in about the same wall-clock time as the from-scratch Python trainer at that *small* scale (the library's thread pool doesn't pay for its own setup cost until the corpus is much bigger) — but its **encoding** throughput, batched over 16 threads, is the fastest row in that chapter's table. At the corpus sizes a real pretraining run actually uses — hundreds of megabytes for training the tokenizer, tens to hundreds of gigabytes for encoding the full mix — the constant-factor gap between a Rust `encode_batch` and a pure-Python loop is the difference between a job that finishes on your laptop while you get coffee and one that needs a cluster.

There is a second, less obvious reason to prefer the library: **you get the ecosystem for free.** A `tokenizer.json` written by `tokenizers` is the file format `transformers.PreTrainedTokenizerFast`, `vllm serve --tokenizer`, `SGLang`, TRL's `SFTTrainer`, and (with the caveats in the next chapter's serving discussion) `llama.cpp`'s GGUF converter all expect. Chapter 14.3 had to build a 60-line exporter and a byte-identity test to bridge the gap between its bespoke JSON and that ecosystem. If you train with `tokenizers` in the first place, there is no gap to bridge — `save()` writes the ecosystem's native format directly.

!!! note "What the library does *not* replace"
    Reaching for `tokenizers` does not make Chapter 14.3 optional reading. The library trains BPE merges and encodes text; it does not decide *what* your pre-tokenizer regex should do with digit runs, *how many* special tokens to reserve, or *why* 32,768 rather than 50,257 is the right vocabulary size for a 100M-parameter model. Those are modeling decisions this chapter inherits from Chapter 14.3 verbatim — we are changing the implementation, not the design.

The vocabulary size is a modeling decision precisely because, with tied embeddings, the token table costs

$$
P_{\text{embed}} = V \cdot d_{\text{model}}
$$

parameters, counted once. For Stack-100M ($d_{\text{model}}=512$) a $V=32{,}768$ table is $32768 \times 512 \approx 16.8\text{M}$ parameters — about 16.6% of the 101M budget — whereas GPT-2's $V=50{,}257$ would spend $\approx 25.7\text{M}$, roughly a quarter of the model, on the table alone. The other side of the ledger is the **compression ratio** $r = B / T$ (source bytes $B$ per emitted token $T$): a larger vocabulary raises $r$, so the same document trains on fewer steps, but past a point the extra rows buy diminishing $r$ while still costing $d_{\text{model}}$ parameters each. At 100M scale that tension is why 32,768 wins; at 7B scale, where $P_{\text{embed}}$ is a rounding error, the balance tips toward the larger vocabulary and its higher $r$.

## Setup and Version Pinning

Pin your versions. Tokenizer APIs are unusually stable release-to-release compared to, say, a training framework, but flags do move, and a `tokenizer.json` trained with one major version is not guaranteed to load identically in another five years from now.

```bash
# As of writing (2026), the current stable line. Pin exact versions in your
# lockfile / requirements.txt -- do not float `tokenizers` and `transformers`
# independently. transformers is pure Python, but it declares a NARROW version
# range on `tokenizers` (both the Python API it calls and the tokenizer.json
# schema move across tokenizers minors), so let transformers' own pin drive the
# tokenizers version rather than picking the two separately.
pip install "tokenizers>=0.20,<0.22" "transformers>=4.46,<4.56"
pip install sentencepiece==0.2.0    # separate C++ library, separate release cadence

python -c "import tokenizers, transformers, sentencepiece as sp; \
print(tokenizers.__version__, transformers.__version__, sp.__version__)"
```

If you hit a `tokenizer.json` that fails to load ("unknown field", "version mismatch"), the fix is almost always a version pin problem, not a corruption problem — check the `tokenizers` changelog on GitHub before you start debugging the file itself.

## Training a Byte-Level BPE Tokenizer With `tokenizers`

Chapter 14.3 fixed Stack-100M's tokenizer design: `vocab_size = 32768`, a `cl100k`-style pre-tokenizer regex with digit runs capped at three characters, and nine reserved special tokens occupying the top of the id space. We reproduce that design exactly, but with the library doing the heavy lifting.

### The building blocks

A `tokenizers.Tokenizer` is assembled from four independently swappable pieces — this is the library's central design idea, and it maps cleanly onto the pipeline Chapter 14.3 built by hand:

```text
raw text
   │
   ▼
┌─────────────┐   normalizer   (NFC/NFKC, lowercasing, etc. -- NONE for us;
│ normalizers │                 byte-level BPE handles case via the vocabulary,
└──────┬──────┘                 not normalization)
       ▼
┌─────────────┐   pre_tokenizer  (OUR regex, exactly as in Ch. 14.3, wrapped
│pre_tokenizers│                  in a Split step, then ByteLevel remaps bytes
└──────┬──────┘                  to printable codepoints)
       ▼
┌─────────────┐   model          (models.BPE -- the trained merge table)
│   models    │
└──────┬──────┘
       ▼
┌─────────────┐   post_processor (adds <|bos|>/<|eos|> around encoded output --
│post_processor│                  we leave this OFF and let the chat template
└──────┬──────┘                  add specials explicitly, exactly like 14.3)
       ▼
┌─────────────┐   decoder        (ByteLevel -- inverts the printable-codepoint
│  decoders   │                  remap back to raw bytes)
└─────────────┘
```

This is the same pipeline Chapter 14.3's exporter built by hand in `to_hf_tokenizer()` — a `Sequence` of `Split(Regex(...))` then `ByteLevel(use_regex=False)`. The difference here is that we *train* directly inside this object instead of training our own structure and converting it afterward.

### The trainer

```python
# scripts/train_hf_tokenizer.py
"""
Train Stack-100M's production tokenizer with HuggingFace `tokenizers`.

This reproduces the DESIGN fixed in ../14-capstone/03-tokenizer.html exactly:
  - vocab_size = 32768             (capstone/PLAN.md Sec. 1 / Sec. 3)
  - the cl100k-style pre-tokenizer regex, digits capped at 3 characters
  - 9 reserved special tokens, occupying the FINAL 9 ids

What changes vs. the from-scratch trainer: the merge-counting loop, the
incremental pair-count bookkeeping, and the encode loop are now Rust, not
Python -- see the throughput comparison at the end of this chapter.
"""
from tokenizers import (
    Tokenizer, Regex, AddedToken, decoders, pre_tokenizers, trainers, models,
)

# --- 1. The SAME regex as Chapter 14.3, unchanged. ---------------------------
# `\p{N}{1,3}` caps digit runs so that '2026' and '2031' are segmented
# identically -- see the worked comparison table in Ch. 14.3. The `tokenizers`
# crate embeds Rust's `onig`/`fancy-regex` engine, which DOES support \p{L}/\p{N}
# Unicode-property escapes natively (unlike Python's stdlib `re`), so there is
# no stdlib-fallback branch needed here -- one less thing to keep in sync.
SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# --- 2. The SAME 9 special tokens, SAME order, so ids land at the SAME
# positions (32759..32767) as the from-scratch tokenizer in Ch. 14.3. ---------
# IMPORTANT: we do NOT pass these to BpeTrainer(special_tokens=...). That
# argument allocates them FIRST, at ids 0..8, which is the opposite of the
# layout Ch. 14.3 fixed. To put them on top we train a vocabulary of
# VOCAB_SIZE - 9 and then append them with `add_special_tokens`, which assigns
# ids at the current end of the vocabulary. See the "Reserving Special Tokens"
# section below.
SPECIAL_TOKENS = [
    "<|bos|>", "<|eos|>", "<|pad|>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
    "<|tool_call|>", "<|tool_result|>",
]
VOCAB_SIZE = 32768
BASE_VOCAB_SIZE = VOCAB_SIZE - len(SPECIAL_TOKENS)   # 32759 = 256 bytes + 32503 merges

# --- 3. Assemble the pipeline. ------------------------------------------------
tokenizer = Tokenizer(models.BPE(unk_token=None, fuse_unk=False))

# Split on our regex FIRST (never let a merge cross a pre-token boundary),
# THEN ByteLevel to remap the 256 raw byte values to printable codepoints.
# use_regex=False is load-bearing: ByteLevel ships its OWN GPT-2 splitting
# regex, and if you leave use_regex=True it re-splits AFTER ours, silently
# overriding the digit cap you just wrote. add_prefix_space=False matches
# Ch. 14.3 -- we never inject a synthetic leading space.
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated", invert=False),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
])
tokenizer.decoder = decoders.ByteLevel()

# --- 4. The trainer. ----------------------------------------------------------
trainer = trainers.BpeTrainer(
    vocab_size=BASE_VOCAB_SIZE,              # leave the top 9 ids free for the specials
    min_frequency=2,                         # a pair must repeat to be worth a merge id
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 byte-codepoints,
                                              # so every raw byte is representable even
                                              # if it never appears in the training sample
    show_progress=True,
)


def corpus_iterator(paths, byte_budget=None):
    """Stream training documents -- never materialize the corpus as one string.
    Mirrors `train_from_iterable` in Ch. 14.3: memory is O(distinct chunks),
    not O(corpus)."""
    seen = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if byte_budget is not None and seen >= byte_budget:
                    return
                seen += len(line.encode("utf-8"))
                yield line


if __name__ == "__main__":
    import glob
    sample_files = sorted(glob.glob("data/tokenizer_sample/*.txt"))
    # train_from_iterator is the streaming entry point -- the Rust trainer
    # pulls batches from the Python generator, so this scales the same way
    # Ch. 14.3's train_from_iterable does, just with the hot loop in Rust.
    tokenizer.train_from_iterator(
        corpus_iterator(sample_files, byte_budget=500_000_000),  # ~500 MB sample
        trainer=trainer,
    )
    assert tokenizer.get_vocab_size() == BASE_VOCAB_SIZE, tokenizer.get_vocab_size()

    # Append the specials AFTER training -> they take ids 32759..32767.
    # `special=True` marks them as control tokens; `normalized=False` means they
    # must match the exact bytes, immune to any normalizer added later.
    tokenizer.add_special_tokens(
        [AddedToken(t, special=True, normalized=False) for t in SPECIAL_TOKENS]
    )

    tokenizer.save("tokenizer/stack100m-32768-raw.json")
    print("vocab_size:", tokenizer.get_vocab_size())
```

Run it:

```bash
python scripts/train_hf_tokenizer.py
# Training  BPE ██████████████████████████████████████████ 100%
# vocab_size: 32768
```

!!! tip "Practitioner tip: how big a sample do you actually need"
    Chapter 14.3 makes this point with real numbers and it holds here too: pair-frequency statistics converge long before you've shown the trainer the full ~20B-token corpus. A few hundred megabytes to a couple of gigabytes, drawn proportionally from the same FineWeb-Edu / Cosmopedia / code / math mix ([Chapter 14.2](../14-capstone/02-data-pipeline.html)), is enough to learn merges that generalize to the full run. Training on more than that mostly burns wall-clock without moving the merge table much — spend the extra budget on a bigger *encoding* pass instead, since that's where the real corpus size shows up.

### `train_from_iterator` versus `train`

`BpeTrainer` (and every `tokenizers` trainer) exposes two entry points on the `Tokenizer` object:

- `tokenizer.train(files=[...])` — takes a list of **file paths** directly; the Rust side handles the I/O and streaming internally. This is the simplest option when your corpus already lives as a handful of large text files.
- `tokenizer.train_from_iterator(iterator, trainer=trainer)` — takes any Python iterable of strings. This is the one to reach for when your data comes from a `datasets` streaming split, a `datatrove` pipeline output, or anything that isn't already flat files on disk — which, per [Data at Scale: datatrove and Hugging Face datasets](../15-production-stack/02-data-datatrove.html), is exactly the shape the rest of the Part XV pipeline produces.

```python
# Training directly from an HF `datasets` streaming split -- no local files at all.
from datasets import load_dataset

def hf_streaming_iterator(ds, text_field="text", n_docs=2_000_000):
    for i, example in enumerate(ds):
        if i >= n_docs:
            return
        yield example[text_field]

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                   split="train", streaming=True)
tokenizer.train_from_iterator(hf_streaming_iterator(ds), trainer=trainer)
```

### Wrapping in `PreTrainedTokenizerFast`

A raw `tokenizers.Tokenizer` object is not yet the artifact TRL and vLLM expect. `PreTrainedTokenizerFast` is the `transformers`-side wrapper that adds the special-token *attributes* (`.bos_token_id`, `.pad_token`, …), `save_pretrained` / `from_pretrained`, and — critically — the Jinja `chat_template` that `apply_chat_template` executes. This is the exact same wrapping step Chapter 14.3's `save_pretrained()` exporter performs; here it's the primary path, not a bolt-on.

```python
# scripts/wrap_pretrained_tokenizer.py
from transformers import PreTrainedTokenizerFast

# Reuse the SAME ChatML-style template as Chapter 14.3's exporter, so a model
# fine-tuned with THIS tokenizer sees byte-identical prompts to one trained
# against the from-scratch artifact -- one source of truth for the chat format,
# not two that can silently drift.
CHAT_TEMPLATE = (
    "{{ '<|bos|>' }}"
    "{% for m in messages %}"
    "{{ '<|' + m['role'] + '|>' + m['content'] + '<|end|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}"
    "{% else %}{{ '<|eos|>' }}{% endif %}"
)

fast = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,              # the trained Tokenizer from above
    bos_token="<|bos|>",
    eos_token="<|eos|>",
    pad_token="<|pad|>",
    additional_special_tokens=[
        "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
        "<|tool_call|>", "<|tool_result|>",
    ],
    chat_template=CHAT_TEMPLATE,
    model_max_length=8192,   # the mid-training context length (capstone/PLAN.md Sec. 7);
                             # NOT the 2048 pretrain length -- see Ch. 14.3's note on this
)
fast.save_pretrained("tokenizer/stack100m-32768-hf")
# -> tokenizer.json, tokenizer_config.json, special_tokens_map.json
```

```bash
ls tokenizer/stack100m-32768-hf/
# tokenizer.json  tokenizer_config.json  special_tokens_map.json
```

Load it back exactly the way TRL's `SFTTrainer`, vLLM, and every other downstream consumer will:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("tokenizer/stack100m-32768-hf")
print(len(tok))                                   # 32768
print(tok.bos_token_id, tok.pad_token_id)          # 32759 32761
print(tok.convert_tokens_to_ids("<|tool_result|>"))  # 32767

msgs = [{"role": "user", "content": "What is 17 * 23?"}]
print(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
# <|bos|><|user|>What is 17 * 23?<|end|><|assistant|>
```

!!! warning "Common pitfall: `add_special_tokens=False` does not stop special-string injection"
    Chapter 14.3 spends a whole section on this and it is worth restating here because it is *the same trap*, in the same library, and it is easy to assume the flag means what its name suggests. `AddedVocabulary` extraction in `tokenizers` runs unconditionally on the raw string, **before** the pre-tokenizer ever sees it — `add_special_tokens=False` only suppresses the automatic BOS/EOS *post-processor*, not this extraction step. If a user's message literally contains the text `<|assistant|>`, it tokenizes to the real role-boundary id 32764 unless you pass `split_special_tokens=True`:

    ```python
    tok("hello <|assistant|> world", add_special_tokens=False).input_ids
    # id 32764 for <|assistant|> appears -- WRONG for untrusted text

    tok("hello <|assistant|> world", add_special_tokens=False,
        split_special_tokens=True).input_ids
    # <|assistant|> tokenized as ordinary bytes -- CORRECT for untrusted text
    ```

    The rule, unchanged from Chapter 14.3: **untrusted content is always encoded with `split_special_tokens=True`**; only your own chat-template code, rendering a validated conversation, tokenizes with it off. This is the same class of bug as prompt injection at the text layer — see [Security: Prompt Injection, Jailbreaks & Defenses](../12-production-mlops/06-security-prompt-injection.html) — and it is worth adding the identical equivalence test Chapter 14.3 wrote, now pointed at this artifact instead of the bespoke one.

### What the library buys you beyond speed

Beyond raw throughput, three things `tokenizers` gives you that the from-scratch trainer in Chapter 14.3 had to build by hand or simply didn't have:

- **`encode_batch` with real thread parallelism.** `tokenizer.encode_batch(list_of_texts)` fans the batch across a Rust (rayon) thread pool with the GIL released — no `multiprocessing.Pool` process-spawn overhead, no pickling documents across process boundaries. There is no `num_threads` argument; the pool is configured out-of-band by the `TOKENIZERS_PARALLELISM=true` and `RAYON_NUM_THREADS=16` environment variables, which must be set *before* the library is imported.
- **Offset tracking for free.** Every `Encoding` object carries `.offsets`, the exact `(start, end)` character span each token came from — essential for building token-level loss masks, highlighting spans in a UI (see the Tokenizer Playground below), or aligning entity labels to sub-word tokens in an NER pipeline. The from-scratch encoder in Chapter 14.3 does not track this at all.
- **Truncation and padding built in.** `tokenizer.enable_truncation(max_length=2048)` and `tokenizer.enable_padding(pad_id=..., pad_token="<|pad|>")` are one-line configuration instead of code you write and test yourself.

```python
enc = tok("The year 2026 arrives.", return_offsets_mapping=True)
for tid, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
    print(tid, repr("The year 2026 arrives."[s:e]))
# 851  'The'
# 954  ' year'
# 220  ' '
# 5896 '202'
# 21   '6'
# 8442 ' arrives'
# 13   '.'
```

## The SentencePiece Path, and When to Use It

`sentencepiece` (Kudo & Richardson, 2018) predates `tokenizers` and takes a different stance on one core question: **should the tokenizer see raw text, or normalized text?** HuggingFace's `tokenizers` assumes you pre-tokenize with a regex (word/punctuation boundaries are meaningful) and then apply BPE inside each chunk. SentencePiece instead treats the input as a raw Unicode character stream — it does **not** assume whitespace-delimited words, replacing every space with a reserved metasymbol (`▁`, U+2581) *before* any segmentation happens, so word boundaries become ordinary vocabulary content rather than a hard-coded rule. That single design choice is why SentencePiece is the default in every major multilingual model family — Llama's original tokenizer, T5, Gemma, and most models trained on CJK-heavy or agglutinative-morphology data (Japanese, Korean, Thai, Finnish, Turkish) use it, because "split on whitespace first" is a poor prior for languages that don't reliably use whitespace as a word boundary.

SentencePiece supports two segmentation algorithms under one C++ library and one Python API:

- **BPE mode** (`--model_type=bpe`) — the same merge algorithm as `tokenizers`, but operating on the whitespace-as-`▁` character stream instead of byte-level regex chunks.
- **Unigram mode** (`--model_type=unigram`, Kudo, 2018) — a fundamentally different algorithm: start from a large candidate vocabulary, then iteratively prune the least-useful entries under an EM-fit unigram language model, keeping the segmentation that maximizes corpus likelihood. Unigram additionally gives you **stochastic subword regularization** at training time — sampling different valid segmentations of the same string turns the tokenizer into a form of data augmentation, which measurably helps low-resource or morphologically rich settings.

### Training a SentencePiece model

```python
import sentencepiece as spm

# Concatenate a training sample the same way Ch. 14.3 samples the mix.
# SentencePiece trains directly from a flat text file (one document per line
# is fine; it treats each line as a training "sentence").
spm.SentencePieceTrainer.train(
    input="data/tokenizer_sample/sample.txt",
    model_prefix="tokenizer/stack100m-32768-sp",
    vocab_size=32768,
    model_type="bpe",                 # or "unigram" for the EM-pruned variant
    character_coverage=0.9995,        # fraction of characters covered directly;
                                       # the rest fall back to byte pieces (below)
    byte_fallback=True,               # CRITICAL: without this, any character
                                       # outside the training set's coverage
                                       # becomes <unk> and is UNRECOVERABLE --
                                       # byte_fallback keeps SentencePiece's
                                       # guarantee equivalent to Ch. 14.3's
                                       # byte-level "every input is representable"
    user_defined_symbols=[            # SentencePiece's equivalent of our
        "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
        "<|tool_call|>", "<|tool_result|>",
    ],
    bos_id=0, eos_id=1, pad_id=2, unk_id=3,  # SentencePiece's OWN defaults are
                                              # unk_id=0, bos_id=1, eos_id=2 and
                                              # pad_id=-1 (padding disabled), so we
                                              # override all four explicitly -- the
                                              # id layout is ours, not the library's.
                                              # The piece TEXT is set via *_piece below
    bos_piece="<|bos|>", eos_piece="<|eos|>", pad_piece="<|pad|>",
    unk_piece="<unk>",
    num_threads=16,
    train_extremely_large_corpus=False,   # flip on for multi-hundred-GB samples;
                                           # changes internal memory strategy
)

sp = spm.SentencePieceProcessor(model_file="tokenizer/stack100m-32768-sp.model")
print(sp.encode("The year 2026 arrives.", out_type=str))
# ['▁The', '▁year', '▁20', '26', '▁arrives', '.']
```

Two things to notice against the `tokenizers` output above. First, `▁The` carries the leading space *inside* the token as the `▁` metasymbol — decode is `"".join(pieces).replace("▁", " ")`, not a separate ByteLevel remap step. Second, without our custom digit cap, `2026` segments however BPE's frequency statistics happen to land (`▁20` + `26` here) — SentencePiece has no first-class concept of a pre-tokenizer regex the way `tokenizers` does; if you need the digit-cap discipline from Chapter 14.3, you either normalize digits yourself before training, or use `tokenizers` where the pre-tokenizer step is designed for exactly this.

### Loading a SentencePiece model into `transformers`

```python
from transformers import LlamaTokenizerFast

# transformers ships fast-tokenizer wrappers for the common SentencePiece-based
# families; LlamaTokenizerFast (and T5TokenizerFast, GemmaTokenizerFast) know
# how to read a raw .model file and convert on load. For a fully custom
# vocabulary + special-token layout like ours, the more direct path is
# converting the .model into a tokenizers.Tokenizer once, offline. That goes
# through transformers' `convert_slow_tokenizer`, which is exactly what the
# *TokenizerFast constructors invoke under the hood:
fast = LlamaTokenizerFast(vocab_file="tokenizer/stack100m-32768-sp.model")
fast.backend_tokenizer.save("tokenizer/stack100m-32768-sp-converted.json")

# NOTE: `tokenizers` does ship a `from_spm` helper, but ONLY on
# SentencePieceUnigramTokenizer -- it parses the protobuf as a Unigram model, so
# it cannot read the model_type="bpe" file trained above. There is no
# SentencePieceBPETokenizer.from_spm. Use the conversion above for BPE, or train
# with model_type="unigram" if you want the from_spm one-liner.
```

### The decision

| | `tokenizers` (byte-level BPE) | `sentencepiece` |
|---|---|---|
| Input assumption | Regex pre-tokenizer decides word/punctuation boundaries | Raw character stream; whitespace is ordinary content (`▁`) |
| Every byte representable? | Yes, by construction (256 base ids) | Only with `byte_fallback=True` |
| Best fit | English-dominant / code-heavy mixes where a pre-tokenizer regex is a genuine asset (our digit cap, contraction handling) | Multilingual, especially CJK / agglutinative languages with weak whitespace conventions |
| Segmentation algorithm | BPE only | BPE **or** Unigram (EM-pruned, supports subword regularization) |
| Ecosystem artifact | `tokenizer.json` — native `transformers`/vLLM/TRL format | `.model` (protobuf) — needs a wrapper (`LlamaTokenizerFast`, or convert to `tokenizer.json`) |
| Encode throughput at scale | Rust, multi-threaded `encode_batch` | Fast C++, single corpus-level API; comparable order of magnitude |
| Used by (examples, cite what you know) | GPT-2/3/4-family, most `tokenizers`-native open models | Original Llama, T5, Gemma, XLNet, ALBERT |

Stack-100M's data mix is 85% English-dominant text and code ([Chapter 14.2](../14-capstone/02-data-pipeline.html): FineWeb-Edu, Cosmopedia, StarCoder, math), so the byte-level BPE path with our own digit-capped pre-tokenizer is the right default — exactly as Chapter 14.3 argued from first principles. If you were building a multilingual variant of Stack-100M, or one whose corpus skewed toward CJK, this is the section to come back to, and Unigram mode is worth an ablation.

!!! example "Worked example: three tokenizers, one sentence"
    Encoding `"The year 2026 arrives in Tōkyō."` with all three paths, trained on the same ~500 MB sample as this chapter, `vocab_size=32768` throughout:

    - **Chapter 14.3's from-scratch tokenizer:** `['The', ' year', ' ', '202', '6', ' arrives', ' in', ' T', 'ō', 'ky', 'ō', '.']` — 12 tokens. The digit cap forces `2026` into `'202'` + `'6'`; `ō` (U+014D, not in the printable-Latin1 fast path) likely falls into a byte-level 2-token UTF-8 split unless a merge learned it whole.
    - **This chapter's `tokenizers`-trained tokenizer (identical regex + vocab_size):** essentially the same segmentation, because it is the *same design*, just a different implementation — that's the whole point of this chapter. Any difference is confined to merge-order tie-breaks on equally-frequent pairs, the same benign divergence Chapter 14.3 measured (rank order differs slightly; the learned vocabulary sets overlap above 99%).
    - **SentencePiece (BPE mode, no digit cap):** `['▁The', '▁year', '▁20', '26', '▁arrives', '▁in', '▁T', 'ō', 'ky', 'ō', '.']` — 11 tokens, and `2026` segments however corpus frequency happened to land, with no structural guarantee that `2031` would segment the same way.

    The token *count* barely differs — the visible, load-bearing difference is that only the digit-capped pre-tokenizer guarantees numerically adjacent inputs get structurally identical treatment, which is exactly the property Chapter 14.3's RLVR arithmetic task ([Chapter 14.9](../14-capstone/09-post-training.html)) depends on.

## Reserving Special Tokens: The Library-Native Way

The special-token *design* — nine tokens, this exact order, occupying ids 32759–32767 — is fixed once, in Chapter 14.3, and every chapter in both Part XIV and Part XV inherits it unchanged. What differs here is only the mechanism for reserving them.

```python
from tokenizers import AddedToken

# AddedToken (rather than a bare string) lets you control matching behavior
# precisely -- `special=True` marks it as a control token (excluded from
# normal BPE segmentation), and `normalized=False` means it must match the
# EXACT bytes, immune to any future normalizer you might add to the pipeline.
specials = [AddedToken(t, special=True, normalized=False) for t in [
    "<|bos|>", "<|eos|>", "<|pad|>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
    "<|tool_call|>", "<|tool_result|>",
]]

# WHERE the ids land is decided by WHEN you register the tokens:
#
#   BpeTrainer(special_tokens=[...])  -> allocated FIRST, ids 0..8, before the
#                                        initial alphabet and before any merge.
#   tokenizer.add_special_tokens([..]) -> allocated at the CURRENT END of the
#                                        vocabulary, i.e. ids 32759..32767 once
#                                        a 32,759-entry model has been trained.
#
# Ch. 14.3's design puts them on top, so we use the second form -- train to
# 32768 - 9 = 32759, then append. Verify this after every training run:
tokenizer.add_special_tokens(specials)
assert tokenizer.get_vocab_size() == 32768
for i, t in enumerate(["<|bos|>","<|eos|>","<|pad|>","<|system|>","<|user|>",
                        "<|assistant|>","<|end|>","<|tool_call|>","<|tool_result|>"]):
    got = tokenizer.token_to_id(t)
    want = 32759 + i
    assert got == want, f"{t}: got id {got}, want {want}"
```

!!! warning "Common pitfall: a short training sample under-fills the vocabulary"
    Exactly the failure mode Chapter 14.3 calls "the shortfall guard": if your training sample is too small or too repetitive to produce 32,503 distinct merges, `BpeTrainer` will simply stop early and hand back a smaller vocabulary — `tokenizer.get_vocab_size()` will be *less* than 32768, which silently breaks `nn.Embedding(32768, 512)` downstream in [Chapter 14.4](../14-capstone/04-architecture.html)'s architecture. Unlike the from-scratch trainer, `tokenizers` does not pad with `<|unused_N|>` filler tokens for you — that guard is something you must re-implement (or, simpler, just use a training sample large enough that you never hit the limit; a few hundred megabytes of real text essentially never runs short of 32,503 distinct merges). Always assert the vocabulary size immediately after training, before you save anything.

## Throughput: Measuring the Library Against the From-Scratch Trainer

Chapter 14.3 measured its own encode paths on an 8.34 MB training split. The comparable numbers for the library path, same order of magnitude:

| Path | Typical throughput (order of magnitude) | 83.5 GB corpus (extrapolated) |
|---|---:|---:|
| From-scratch, `_apply_merges` per chunk, no cache (Ch. 14.3) | ~1 MB/s | many hours |
| From-scratch, cached + `multiprocessing.Pool(16)` (Ch. 14.3) | ~50 MB/s | tens of minutes |
| `tokenizers` `Tokenizer.encode_batch(...)`, `RAYON_NUM_THREADS=16` | tens of MB/s, single machine | tens of minutes |
| `tiktoken.encode_ordinary_batch(num_threads=16)` (fastest measured in Ch. 14.3) | ~75 MB/s | ~18 minutes |

!!! note "Aside: honest about the numbers"
    Chapter 14.3's table is the one with real, measured figures — produced by running its own code on its own manuscript-derived corpus. We don't reproduce a fabricated benchmark table for the library path here; run `tokenizer.encode_batch(docs)` under `RAYON_NUM_THREADS=N` on your own machine and your own sample and treat the result as ground truth. The order-of-magnitude claim that matters and *is* safe to state without a fresh measurement: a Rust-threaded `encode_batch` and a well-cached parallel Python path land in the same broad performance tier — both turn an 83.5 GB encoding job (Stack-100M's full ~20B-token budget at ~4.2 bytes/token) into tens of minutes rather than tens of hours. The engineering lesson from Chapter 14.3 — cache aggressively, parallelize over documents, never rescan the whole corpus per merge — is exactly what both the from-scratch trainer's optimized path *and* the library implement; the library just gets there without you writing or maintaining the code.

## Wiring the Real Tokenizer Into the Rest of the Production Stack

The artifact this chapter produces, `tokenizer/stack100m-32768-hf/`, is a drop-in replacement for `tokenizer/stack100m-32768-hf/` from Chapter 14.3's exporter — same vocabulary design, same special-token ids, same chat template, different training implementation underneath. Every downstream consumer in Part XV loads it identically:

```bash
# 15.4's torchtitan / nanotron pretraining config points --tokenizer_path here.
# 15.5's TRL SFTTrainer / DPOTrainer load it via AutoTokenizer.from_pretrained.
# 15.6's `vllm serve` and the llama.cpp GGUF converter read the same directory.

python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('tokenizer/stack100m-32768-hf')
assert len(tok) == 32768
assert tok.bos_token_id == 32759
print('OK -- tokenizer ready for the rest of the pipeline')
"
```

Before you wire it into a real training run, port Chapter 14.3's equivalence test to this artifact — verbatim in spirit, pointed at the new file:

```python
# tests/test_hf_tokenizer_matches_design.py
from transformers import AutoTokenizer

PROBE = ("Tokenization is frozen for the life of the model.\n"
         "café 🚀 — mixed ünïcode\tTAB\r\nCRLF\n"
         "def f(x):\n    return x ** 2  # 1234567 and 2026\n"
         "hello <|assistant|> world\n")   # the injection probe -- keep it, see Ch. 14.3

def test_vocab_size_and_special_ids():
    tok = AutoTokenizer.from_pretrained("tokenizer/stack100m-32768-hf")
    assert len(tok) == 32768
    assert tok.bos_token_id == 32759 and tok.pad_token_id == 32761
    assert tok.convert_tokens_to_ids("<|tool_result|>") == 32767

def test_untrusted_text_does_not_forge_special_tokens():
    tok = AutoTokenizer.from_pretrained("tokenizer/stack100m-32768-hf")
    ids = tok(PROBE, add_special_tokens=False, split_special_tokens=True).input_ids
    assert 32764 not in ids   # <|assistant|> id must NOT appear from untrusted text

def test_decode_round_trips():
    tok = AutoTokenizer.from_pretrained("tokenizer/stack100m-32768-hf")
    ids = tok(PROBE, add_special_tokens=False, split_special_tokens=True).input_ids
    assert tok.decode(ids) == PROBE
```

{{tool:tokenizer-playground}}

The [Tokenizer Playground](../02-transformer/01-tokenization.html) lets you paste any string and watch it segment live under GPT-2's, GPT-4's, and Llama's real tokenizers side by side — the fastest way to build intuition for *why* the pre-tokenizer regex matters before you commit 32,503 merges' worth of training time to a design. Load your own `tokenizer/stack100m-32768-hf/tokenizer.json` into the same visualization pattern (the playground's underlying mechanism is exactly `tokenizer.encode(..., return_offsets_mapping=True)`, the same call shown above) to sanity-check Stack-100M's own segmentation on real inputs before you freeze it.

!!! interview "Interview Corner"
    **Q:** You're handed a `tokenizer.json` trained with Hugging Face `tokenizers` and told "make sure a user can never smuggle a `<|assistant|>` role token into their prompt." What's the actual fix, and why doesn't `add_special_tokens=False` solve it?

    **A:** `add_special_tokens=False` only controls the tokenizer's automatic post-processing step — the part that would otherwise wrap your input in a BOS/EOS pair. It does *not* disable `AddedVocabulary` extraction, which runs unconditionally over the raw string before the pre-tokenizer even sees it, and will happily match a literal `<|assistant|>` substring in untrusted user text and turn it into the real role-boundary token id. The actual control is `split_special_tokens=True` (on `PreTrainedTokenizerFast`/`AutoTokenizer`) or the equivalent `tokenizer.encode_special_tokens = True` on a raw `tokenizers.Tokenizer` (transformers literally assigns one to the other: `self._tokenizer.encode_special_tokens = self.split_special_tokens`; the default is `False`, i.e. extraction *on*, which is the vulnerable setting) — both force added tokens to be encoded as ordinary bytes rather than recognized as control tokens. The operational rule: any code path that tokenizes content you don't fully control — a user message, a retrieved document, a tool's returned observation — should always pass `split_special_tokens=True`; only your own chat-template renderer, operating on a conversation structure you built, should tokenize with extraction on. This is structurally identical to a SQL-injection defense: never let untrusted input be interpreted as control syntax, always as data.

## Key Takeaways

!!! key "Key Takeaways"

    - Hugging Face `tokenizers` is the 2026-standard library for training and using BPE tokenizers: a Rust trainer and encoder wrapped in a Python API, producing the native `tokenizer.json` that `transformers`, TRL, vLLM, SGLang, and llama.cpp's converter all already read — no exporter needed, unlike the bespoke format in [Chapter 14.3](../14-capstone/03-tokenizer.html).
    - The library's `Tokenizer` is a pipeline of swappable stages — normalizer, pre-tokenizer, model, post-processor, decoder. Stack-100M's design reproduces Chapter 14.3's exact choices: a `Split` on our digit-capped regex, then `ByteLevel(use_regex=False, add_prefix_space=False)`, then `models.BPE`.
    - `train_from_iterator` streams from any Python iterable — `datasets` streaming splits, `datatrove` outputs, flat files — and never materializes the corpus in memory, mirroring the memory discipline of the from-scratch trainer.
    - Wrap the trained `Tokenizer` in `PreTrainedTokenizerFast` with the special tokens and the chat template baked in via `save_pretrained` — this is the artifact every later Part XV chapter (`torchtitan`/`nanotron` pretraining, TRL post-training, `vllm serve`) loads with a single `AutoTokenizer.from_pretrained`.
    - `add_special_tokens=False` does **not** stop special-string injection from untrusted text — that guard is `split_special_tokens=True`, and it must be applied at every call site that tokenizes content you don't control.
    - `sentencepiece` takes a different stance (raw character stream, whitespace as content via `▁`, optional Unigram/EM segmentation with subword regularization) and remains the right default for multilingual and CJK-heavy corpora; for Stack-100M's English-and-code-dominant mix, the regex-driven `tokenizers` path is the correct choice, exactly as Chapter 14.3 argued from first principles.
    - Version-pin `tokenizers` and `transformers` together — `transformers` declares a narrow version range on `tokenizers` because both its Python API and the `tokenizer.json` schema shift across minors, and flags (like the `split_special_tokens` behavior) have shifted across releases; always check the artifact loads and passes an equivalence test before trusting it in a real run.
    - Re-run Chapter 14.3's equivalence test against the library-trained artifact before wiring it into pretraining: assert the vocabulary size, the special-token ids, the injection guard, and a decode round trip, exactly as shown here.

## Further reading

- Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with Subword Units*, 2016 — the paper that introduced BPE for NLP.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019 — the byte-level BPE variant and the `bytes_to_unicode` printable-codepoint remap that both `tokenizers` and Chapter 14.3's exporter implement.
- Kudo & Richardson, *SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing*, 2018 — the library and paper behind the `sentencepiece` path in this chapter.
- Kudo, *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*, 2018 — the Unigram language-model tokenizer and stochastic segmentation sampling.
- [huggingface/tokenizers](https://github.com/huggingface/tokenizers) — the Rust library and Python bindings this chapter is built around; the `Tokenizer`/`trainers`/`pre_tokenizers`/`decoders` API surface used throughout.
- [google/sentencepiece](https://github.com/google/sentencepiece) — the C++ library and `SentencePieceTrainer`/`SentencePieceProcessor` Python API used in the SentencePiece section.
- [huggingface/transformers](https://github.com/huggingface/transformers) — `PreTrainedTokenizerFast`, `AutoTokenizer`, and `apply_chat_template`, the wrapping layer every downstream Part XV chapter depends on.
- [Chapter 14.3, A Byte-Level BPE Tokenizer From Scratch](../14-capstone/03-tokenizer.html) — the from-scratch design this chapter reproduces with the production library; read it first if you haven't.
- [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html) — the algorithmic foundations and the Tokenizer Playground tool.
- [Data at Scale: datatrove and Hugging Face datasets](../15-production-stack/02-data-datatrove.html) — where the training sample and full corpus for this chapter's tokenizer come from.
- [The Stack-100M Architecture: SOTA Components, Cited and Assembled](../14-capstone/04-architecture.html) — how `vocab_size` sizes the embedding table this chapter's tokenizer feeds.
