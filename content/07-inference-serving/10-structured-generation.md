# 7.10 Structured & Constrained Generation

Every production LLM application eventually runs into the same wall: the model generates nearly-correct JSON, and one missing closing brace breaks the downstream parser. Or a function-calling system gets a plausible-sounding tool name that doesn't exist. Or a safety filter needs the model to fill a template with exactly four enumerated values. The *free-text* generation pipeline we've built in previous chapters simply wasn't designed to guarantee that the output satisfies a formal grammar. Structured and constrained generation is the discipline of enforcing these guarantees — not by prompting and hoping, but by hard-wiring the grammar into the decoding process itself.

This chapter covers the full machinery: finite-state machines (FSMs) and pushdown automata for grammar enforcement, logit masking as the enforcement mechanism, how tools like Outlines, XGrammar, and llguidance implement this at production speed, and what it means for tool-call and function-calling workflows. We derive the key algorithms, work through concrete examples, and give you runnable code you can drop into a real inference server.

The prerequisite for this chapter is a working understanding of the decoding loop, which we covered in [Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html). If you have not read that chapter, do so first — we will assume fluency with the concepts of token logits, temperature scaling, and top-k/top-p filtering.

## Why Free-Text Generation Breaks on Schemas

A language model trained on next-token prediction assigns probabilities to every token in its vocabulary at every step. Nothing in the training objective forces the sequence of chosen tokens to form valid JSON, match a regex, or satisfy a context-free grammar (CFG). The model learns implicit statistical tendencies from training data, but those tendencies are far from watertight.

The failure modes are well-documented:

- **Missing or extra braces/brackets** — the most common JSON failure; the model loses track of nesting depth.
- **Wrong type** — an integer field gets a quoted string.
- **Hallucinated enum values** — the model invents a value outside the allowed set.
- **Truncated output** — the model emits an EOS token mid-object, yielding a partial parse.
- **Key name drift** — field names shift under paraphrase pressure (`user_id` → `userId`).

Post-hoc repair (parsing with fallback, asking the model to fix its own output) adds latency and is unreliable. The only robust solution is to prevent invalid token sequences from being sampled in the first place.

The key insight that makes constrained generation tractable is this: **we do not need to forbid bad sequences globally; we only need to forbid the single next token that, given what has been emitted so far, would make the sequence irrecoverable.** This is exactly what an automaton state transition tells us.

## Finite-State Machines and Regular Languages

### FSM Primer

A deterministic finite-state machine (FSM) is a 5-tuple $M = (Q, \Sigma, \delta, q_0, F)$ where:

- $Q$ is a finite set of states.
- $\Sigma$ is the input alphabet (here, individual *bytes* or *characters*).
- $\delta: Q \times \Sigma \to Q$ is the transition function.
- $q_0 \in Q$ is the start state.
- $F \subseteq Q$ is the set of accepting states.

The FSM reads one symbol at a time. Given current state $q$ and next symbol $c$, it moves to $\delta(q, c)$ (or to a dead/error state if no transition exists). A string $s$ is accepted if the FSM ends in $F$ after reading all of $s$.

Regular expressions are exactly the languages recognizable by FSMs (Kleene's theorem). So any regex constraint — an ISO 8601 date, a UUID, a phone number — can be compiled into an FSM.

For constrained token generation we need a slightly different object: a **token-level FSM**. Because a tokenizer operates over multi-character tokens rather than single characters, we need to pre-compute, for each FSM state and each vocabulary token, whether consuming that token (i.e., advancing the FSM by each character in the token string) leads to a valid (non-dead) state. This yields a boolean matrix:

$$
\text{valid}[q][v] = \begin{cases} 1 & \text{if } \delta^*(q, \text{decode}(v)) \neq \perp \\ 0 & \text{otherwise} \end{cases}
$$

where $\delta^*$ is the extended transition function that applies $\delta$ character-by-character over a multi-character string, and $\perp$ denotes the dead state. The mask for state $q$ is simply the row $\text{valid}[q]$, a binary vector of length $|\mathcal{V}|$ (vocabulary size).

### Logit Masking

At each decoding step, after the model computes the raw logit vector $\ell \in \mathbb{R}^{|\mathcal{V}|}$, we apply the mask:

$$
\tilde{\ell}_v = \begin{cases} \ell_v & \text{if } \text{valid}[q_t][v] = 1 \\ -\infty & \text{otherwise} \end{cases}
$$

Feeding $\tilde{\ell}$ to softmax zeroes out the probability of every disallowed token. The sampler then operates on the masked distribution exactly as it would on an unconstrained one. Temperature scaling, top-p, and top-k all compose cleanly on top of the mask.

After sampling token $v_t$ and appending it to the output, we advance the FSM:

$$
q_{t+1} = \delta^*(q_t, \text{decode}(v_t))
$$

The loop continues until an EOS token is sampled *and* the FSM is in an accepting state, or until the FSM reaches an accepting state and we want to stop.

!!! example "Worked example: masking a 4-digit integer"

    Suppose our grammar requires exactly a 4-digit integer (regex `[0-9]{4}`).
    The vocabulary size is 32,000 tokens (roughly Llama-3 scale).

    The FSM has 5 states: $q_0$ (start, needs 4 digits), $q_1$ (1 digit seen), $q_2$ (2 digits),
    $q_3$ (3 digits), $q_4$ (accept, 4 digits). A dead state $\perp$ handles everything else.

    Suppose the model is at $q_0$ with logits $\ell$.  We need to allow only tokens
    whose decoded text is a single decimal digit `0`–`9`. In a BPE vocabulary these
    are exactly the 10 single-digit tokens (often at indices like 15, 16, ..., 24 — 
    exact positions depend on the tokenizer).

    Out of 32,000 tokens, at most 10 are allowed: masking ratio ≈ 99.97%.
    After one digit token, we're in $q_1$; after two more we're in $q_3$.
    At $q_3$, multi-character tokens like `"42"` or `"123"` are now *also* valid
    (they consume 2 and 3 characters respectively and land in $q_4$ or advance
    through intermediate states). This is why the token-level pre-computation
    matters: character-by-character simulation handles multi-char tokens correctly.

{{fig:structgen-mask-decode-loop}}

### FSM Compilation Complexity

For a regex with $s$ states and vocabulary size $V$:

- **Naive per-step mask computation:** For each token $v$, simulate the FSM over `decode(v)` starting from the current state. Cost is $O(V \cdot L_{\max})$ per step, where $L_{\max}$ is the max token length in bytes (typically 6–8 bytes). For $V = 128{,}000$ and $L_{\max} = 8$ this is roughly 1M operations per step — acceptable for a single sequence but a bottleneck for batched serving.

- **Pre-compiled transition table:** Pre-compute the full $|Q| \times V$ mask matrix once at load time. Lookup is $O(1)$ per step. Memory cost is $|Q| \cdot V$ bits. For 50 states and $V = 128{,}000$, this is about 800 KB — completely negligible. Outlines and XGrammar both take this approach.

## Beyond Regular Languages: Pushdown Automata and CFGs

JSON is not a regular language. It is a context-free language (CFL) because of its arbitrary nesting depth: `{"a": {"b": {"c": ...}}}` requires a stack to track the nesting level. An FSM has no memory beyond its current state, so it cannot handle unbounded nesting. A pushdown automaton (PDA) adds a stack to an FSM, enabling it to recognize all CFLs.

### The PDA State

At any point in JSON generation, the constraint automaton needs to track:

1. **What kind of value is expected next** (string, number, boolean, null, array, object).
2. **The nesting stack** — e.g., `[OBJECT, ARRAY, OBJECT]` means we're inside an object that's inside an array that's inside an object.
3. **Whether we're at the start, middle, or end of a structural element** (just opened an object, have written a key, awaiting a colon, etc.).

```text
PDA stack state example while generating:
  {"users": [{"id": 1, "name": |  ← cursor here

Stack (bottom to top):  OBJECT → ARRAY → OBJECT
Current sub-state:      AFTER_KEY_COLON (expecting a string value for "name")
Allowed next chars:     " (opening quote of a string)
```

The combined state space grows, but for any fixed grammar the number of reachable (stack prefix × sub-state) combinations is bounded. In practice, for JSON with a known schema (a JSON Schema object), the grammar is further restricted and the effective state count stays manageable.

### Grammar-Based Constrained Decoding

Tools like Outlines (Willard & Louf, 2023), XGrammar (MIT/CMU, 2024), and llguidance (Microsoft Research) convert a grammar specification into a token-level decision object:

1. **Parse the schema/grammar** — compile a JSON Schema, EBNF grammar, Pydantic model, or regex into an internal representation.
2. **Derive the token-level automaton** — expand the grammar into a state machine over the character-level alphabet, then lift it to the token level via pre-computation.
3. **At each decode step** — look up the current automaton state to get the valid-token bitmask; apply it as a logit mask; advance the automaton after sampling.

The critical engineering challenge is Step 2: lifting a character-level grammar to token-level transitions efficiently. A naive approach simulates the full grammar for every possible token string, which is expensive for a large LLM vocabulary and a complex grammar. XGrammar introduced several optimizations we'll cover in detail.

## Outlines: The Foundational Library

Outlines (Willard & Louf, 2023, *Efficient Guided Generation for Large Language Models*) was the first widely-adopted library to formalize regex and CFG-constrained LLM generation. Its core contribution is converting a regex or EBNF grammar into an **index**: a mapping from FSM state to the set of vocabulary tokens valid in that state, pre-computed once.

The Outlines index construction algorithm:

```python
# Conceptual pseudocode for Outlines-style FSM index construction
# The actual library uses interegular and the tokenizers package

import re
from typing import Dict, Set, List, Tuple

def build_fsm_index(
    regex_pattern: str,
    vocab: Dict[int, str],    # token_id -> decoded string
) -> Dict[int, Set[int]]:
    """
    Returns a dict: fsm_state -> set of valid token_ids
    Pre-computes the full transition table for constrained decoding.
    """
    # Step 1: compile regex to NFA, convert to DFA (standard automaton ops)
    fsm = regex_to_dfa(regex_pattern)  # returns (states, transitions, start, accepts)
    
    index: Dict[int, Set[int]] = {}
    dead_state = -1
    
    for state in fsm.states:
        valid_tokens: Set[int] = set()
        for token_id, token_str in vocab.items():
            # Simulate the DFA over the token string, starting from `state`
            current = state
            reachable = True
            for char in token_str:
                next_state = fsm.transitions.get((current, char), dead_state)
                if next_state == dead_state:
                    reachable = False
                    break
                current = next_state
            
            if reachable:
                # Token is valid: consuming it does not kill the FSM
                valid_tokens.add(token_id)
        
        index[state] = valid_tokens
    
    return index
```

At generation time, the guide maintains the current FSM state and calls `index[current_state]` to get the allowed token set, applies it as a logit mask, and advances the state.

A critical subtlety is handling *partial tokens* — a token like `"wh"` might be valid mid-pattern even if `"what"` is the only completing sequence. Outlines handles this by also allowing tokens that lead to a non-dead state even if that state isn't an accepting state, because generation will continue. Only at EOS must the state be accepting.

## XGrammar: Overlap-Aware and Context-Dependent Pushdown

XGrammar (Dong et al., 2024, from the MLC-LLM group) is the production-grade successor designed specifically for JSON Schema and EBNF grammars in batch serving environments. Its headline contribution is making constrained decoding nearly free at generation time by moving nearly all work into a preprocessing phase.

### The Key Innovation: Context-Independent Tokens

XGrammar's insight is that for most grammars, a large fraction of vocabulary tokens are **context-independent**: their validity does not depend on the exact PDA stack state, only on the "top-level" grammar rule currently active. For example, whitespace tokens are almost always valid inside a JSON object (between key-value pairs). Number digit tokens are valid whenever a number is expected, regardless of nesting depth.

XGrammar separates tokens into two classes:

1. **Context-independent tokens**: valid/invalid based only on the shallow grammatical position (which rule we're in, not the stack depth). These are pre-computed into a compact bitmask for each grammar rule.
2. **Context-dependent tokens**: require knowing the full stack state to determine validity (e.g., the closing `}` character, whose validity depends on whether there's a matching `{` on the stack).

For a typical JSON schema, on the order of 95%+ of tokens fall into the context-independent class. This means the per-step work reduces to a bitmap OR of the context-independent mask plus a small correction for the handful of context-dependent tokens.


{{fig:structgen-xgrammar-mask-architecture}}


The combined mask computation at each step is:

```
mask = precomputed_rule_mask[current_rule] | evaluate_context_dependent(stack)
```

The context-dependent evaluation traverses the stack but only touches a small set of tokens, keeping the amortized cost low.

### XGrammar Performance Characteristics

XGrammar reports that on standard JSON-constrained decoding benchmarks, the overhead of constrained decoding over unconstrained decoding is reduced to below 5% of total decode time for sufficiently large models (where the model's forward pass dominates). For smaller models where memory bandwidth is less of a bottleneck, the overhead is somewhat larger but still sub-10% with proper batching.

The key numbers to internalize:

- Mask application is a GPU kernel operating on the $|\mathcal{V}|$-dimensional logit vector. For $|\mathcal{V}| = 128{,}000$ tokens stored as fp16 (2 bytes each), one logit vector is 256 KB. Setting $-\infty$ on invalid indices is a simple write operation — a handful of microseconds.
- For batched inference with batch size $B$, each sequence has its own automaton state and its own mask. Mask application is embarrassingly parallel.
- FSM compilation from a typical JSON Schema takes on the order of tens of milliseconds on CPU, done once at request time or pre-cached per schema.

## llguidance: Microsoft's Production Engine

llguidance is an open-source Rust library (and Python bindings) developed at Microsoft Research for high-throughput constrained generation. It supports EBNF grammars with recursive rules, JSON Schemas, and Lark-style grammar specifications.

Key technical properties of llguidance:

- **Earley parser integration**: rather than compiling the full grammar to a DFA (which can exponentially explode in states for complex grammars), llguidance runs an incremental Earley parser that computes the set of valid next tokens on-the-fly. The Earley parser processes partial parses and returns, for each position, the set of "predict" items — grammar rules that could validly continue.
- **Byte-level tokenization compatibility**: properly handles tokenizers that merge bytes into tokens in non-obvious ways, a common source of bugs in regex-based approaches.
- **Integration with the llama.cpp and vLLM ecosystems**: designed as a backend that serving frameworks can call via a simple `get_mask()` / `advance(token)` API.

The Earley parser operates in $O(n^3)$ time in the worst case for ambiguous grammars, but for practical LLM grammars (JSON, SQL, function signatures) the parse is nearly linear. The dominant cost is the initial seeded chart construction.

## jsonformer: The Simple Early Approach

Before FSM-based libraries became mature, jsonformer (2023) took a simpler approach: **structural scaffolding**. Rather than computing valid-token masks, jsonformer:

1. Generates the structural tokens of the JSON (`{`, `}`, `[`, `]`, `:`, `,`, `"`, `"`, `true`, `false`, `null`) itself using deterministic rules based on the schema.
2. Only asks the LLM to generate the *value content* — the actual strings, numbers, and booleans — with minimal structural overhead.

This works well for simple flat schemas but fails for deeply nested or recursive structures. It also doesn't generalize beyond JSON. FSM-based approaches are strictly more general, and jsonformer's technique is largely superseded — though it is instructive as a motivation for why the automaton approach is needed.

## A Regex-Constrained Sampler From Scratch

Let's build a complete, working regex-constrained sampler in PyTorch. This implements the core FSM + logit-masking loop without external libraries.

```python
"""
regex_sampler.py — A complete regex-constrained token sampler from scratch.

Requires: torch, transformers (for tokenizer + model), regex (pip install regex)
Run:  python regex_sampler.py
"""

import re
import torch
import numpy as np
from typing import Dict, FrozenSet, Optional, Set, Tuple
from dataclasses import dataclass, field
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Regex → DFA using Python's `re` module (via simulation)
# We build a minimal DFA simulator by treating the regex as a string pattern
# and simulating NFA-style matching for the "is this a valid prefix?" question.
# ─────────────────────────────────────────────────────────────────────────────

def can_extend(pattern: str, prefix: str) -> bool:
    """
    Returns True if `prefix` is either a full match or a valid *prefix*
    of a string that could match `pattern`.

    We check prefix validity by attempting a partial match (ANCHORED start,
    partial end). Python's re module supports this via re.match with a
    modified pattern that allows trailing content.
    """
    # Wrap pattern: the prefix must match the *start* of the pattern.
    # We check if there exists some completion such that pattern matches.
    # Approximation: check if `re.match(pattern, prefix)` succeeds OR
    # if `re.match(pattern + '.*', prefix)` would succeed — but this is tricky.
    # Robust approach: use regex with partial matching via the `regex` library.
    try:
        import regex  # pip install regex
        m = regex.match(pattern, prefix, flags=regex.PARTIAL)
        return m is not None
    except ImportError:
        # Fallback: check if the full pattern matches prefix exactly
        # (handles only complete matches, not partial — use `regex` lib in prod)
        return bool(re.fullmatch(pattern, prefix))


@dataclass
class FSMState:
    """Tracks the current constrained generation state."""
    emitted_so_far: str = ""        # characters emitted since constraint start
    is_complete: bool = False       # True when pattern is fully matched


def build_token_mask(
    pattern: str,
    vocab: Dict[int, str],          # token_id -> decoded text
    current_prefix: str,            # text emitted so far under this constraint
    device: torch.device,
) -> torch.Tensor:
    """
    Builds a boolean mask (1 = allowed, 0 = forbidden) over the vocabulary
    for the next token, given the regex `pattern` and what has been `current_prefix`emitted.

    Returns a BoolTensor of shape [vocab_size].
    """
    # We need the maximum token id to size the mask correctly.
    max_id = max(vocab.keys())
    mask = torch.zeros(max_id + 1, dtype=torch.bool, device=device)

    for token_id, token_str in vocab.items():
        candidate = current_prefix + token_str
        # A token is valid if the candidate string is either:
        # (a) a full match of the pattern (we could stop here), or
        # (b) a valid prefix that could lead to a full match (we continue).
        if can_extend(pattern, candidate):
            mask[token_id] = True

    return mask


def apply_mask_to_logits(
    logits: torch.Tensor,           # shape [vocab_size] or [batch, vocab_size]
    mask: torch.Tensor,             # bool tensor, same last dim as logits
) -> torch.Tensor:
    """
    Sets logits of forbidden tokens to -inf. Operates in-place on a clone.
    """
    masked = logits.clone()
    masked[~mask] = float('-inf')
    return masked


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: The constrained generation loop
# ─────────────────────────────────────────────────────────────────────────────

def constrained_generate(
    model,
    tokenizer,
    prompt: str,
    pattern: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    device: str = "cpu",
) -> str:
    """
    Greedy/sampling generation that constrains output to match `pattern`.

    Args:
        model:          A HuggingFace CausalLM model (or any object with a
                        forward() that returns logits).
        tokenizer:      Corresponding HuggingFace tokenizer.
        prompt:         Input text prefix.
        pattern:        Python regex that the *entire generated portion* must match.
        max_new_tokens: Maximum tokens to generate.
        temperature:    Sampling temperature (1.0 = unscaled; set 0 for greedy).
        device:         "cpu" or "cuda".

    Returns:
        The generated text (not including the prompt).
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    # Build vocabulary mapping: id -> decoded string
    # We decode each single token to get its string representation.
    vocab: Dict[int, str] = {}
    for token_id in range(tokenizer.vocab_size):
        try:
            # Decode without special tokens; skip_special_tokens=True prevents
            # BOS/EOS from polluting the character strings.
            tok_str = tokenizer.decode([token_id], skip_special_tokens=True)
            vocab[token_id] = tok_str
        except Exception:
            vocab[token_id] = ""   # fallback for malformed tokens

    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)

    # State: track what we've generated so far under the constraint
    fsm = FSMState()

    generated_ids = []
    eos_token_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]    # shape: [vocab_size]

        # Apply temperature scaling
        if temperature > 0:
            logits = logits / temperature

        # Build and apply the constraint mask
        mask = build_token_mask(pattern, vocab, fsm.emitted_so_far, dev)

        # Handle vocabulary size mismatch (model logits may be larger than
        # tokenizer.vocab_size due to padding)
        if logits.shape[0] > mask.shape[0]:
            pad = torch.zeros(
                logits.shape[0] - mask.shape[0], dtype=torch.bool, device=dev
            )
            mask = torch.cat([mask, pad])

        masked_logits = apply_mask_to_logits(logits, mask)

        # Sample (greedy if temperature == 0)
        if temperature == 0:
            next_token_id = torch.argmax(masked_logits).item()
        else:
            probs = torch.softmax(masked_logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()

        # Check if generation is complete
        if next_token_id == eos_token_id:
            # Only allow EOS if the current prefix fully matches the pattern
            if re.fullmatch(pattern, fsm.emitted_so_far):
                break
            else:
                # Force continuation: mask out EOS was already done via the
                # grammar mask, so this branch should not be reached in
                # a correct implementation. Include as defensive guard.
                continue

        # Advance state
        next_token_str = vocab.get(next_token_id, "")
        fsm.emitted_so_far += next_token_str
        generated_ids.append(next_token_id)

        # Append token to input for next forward pass
        next_tensor = torch.tensor([[next_token_id]], device=dev)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)

        # Check if pattern is fully satisfied
        if re.fullmatch(pattern, fsm.emitted_so_far):
            fsm.is_complete = True
            break

    return fsm.emitted_so_far


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Demo — constrain a small model to generate a date in YYYY-MM-DD format
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Use a small model for demonstration; swap for any HF model.
    MODEL_NAME = "gpt2"
    print(f"Loading {MODEL_NAME}...")

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # Pattern: ISO date YYYY-MM-DD
    DATE_PATTERN = r"[12][0-9]{3}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"

    prompt = "The event is scheduled for "
    print(f"Prompt: '{prompt}'")
    print(f"Pattern: {DATE_PATTERN}")

    result = constrained_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        pattern=DATE_PATTERN,
        max_new_tokens=20,
        temperature=0,     # greedy for reproducibility
        device="cpu",
    )

    print(f"Generated: '{result}'")
    assert re.fullmatch(DATE_PATTERN, result), f"Pattern not satisfied! Got: {result}"
    print("Pattern constraint satisfied.")
```

!!! warning "Production note: vocabulary simulation is slow"

    The `build_token_mask` function above iterates over all vocabulary tokens per
    decode step. For GPT-2 (vocab_size ≈ 50,000) and a simple pattern this takes
    ~50–100 ms per step on CPU — unusable for production. Production systems
    (Outlines, XGrammar) pre-compute the full state × token mask matrix once,
    reducing per-step cost to a single table lookup costing microseconds.
    The code above is pedagogically correct; use Outlines or XGrammar in real deployments.

## Using Outlines in Production

Outlines provides a clean high-level API that wraps any HuggingFace or vLLM model with structured generation:

```python
"""
outlines_demo.py — Production-style structured generation with Outlines.
pip install outlines pydantic transformers torch
"""

import outlines
from pydantic import BaseModel, Field
from typing import Literal, List

# ─── Define the output schema with Pydantic ───────────────────────────────────

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    year: int = Field(description="Release year", ge=1888, le=2030)
    sentiment: Literal["positive", "negative", "neutral"]
    score: float = Field(description="Score from 0.0 to 10.0", ge=0.0, le=10.0)
    summary: str = Field(description="One-sentence summary", max_length=200)

# ─── Load model with Outlines ─────────────────────────────────────────────────

# Outlines wraps any HF model; it pre-compiles the grammar on first call.
model = outlines.models.transformers(
    "meta-llama/Llama-3.1-8B-Instruct",  # replace with any available model
    device="cuda",
)

# ─── Structured generation: Pydantic schema → guaranteed valid JSON ───────────

generator = outlines.generate.json(model, MovieReview)

prompt = """Review the following movie and return a structured JSON review.
Movie: Inception (2010)
Review text: Mind-bending thriller, visually stunning. 9/10.
Return only JSON:"""

# This call guarantees that `result` is a valid `MovieReview` instance.
result: MovieReview = generator(prompt)
print(result.model_dump_json(indent=2))

# ─── Regex-constrained generation ─────────────────────────────────────────────

# Generate exactly a US phone number
phone_generator = outlines.generate.regex(
    model,
    r"\([0-9]{3}\) [0-9]{3}-[0-9]{4}"
)

phone = phone_generator("Customer service number: ")
print(f"Phone: {phone}")   # Guaranteed to match the regex

# ─── Choice-constrained generation ───────────────────────────────────────────

# Force the model to output one of a fixed set of options.
classifier = outlines.generate.choice(
    model,
    ["positive", "negative", "neutral"]
)

sentiment = classifier("The food was absolutely fantastic! Sentiment: ")
print(f"Sentiment: {sentiment}")  # One of the three options, guaranteed.
```

## Tool-Call Enforcement and Function Calling

Function calling (tool use) is the highest-stakes application of structured generation. When an LLM selects and parameterizes a tool, the call must match the tool's schema exactly: the function name must be one of the registered tools, and the arguments must conform to the parameter schema.

The standard pattern in modern serving stacks (OpenAI, vLLM, SGLang) is:

1. **Build a tool-call grammar** at request time from the list of available tools and their JSON schemas. This is essentially a JSON Schema union:
   ```
   { "name": "<one of the tool names>", "arguments": <tool's parameter schema> }
   ```
2. **Activate the grammar** only when the model emits the tool-call start token (e.g., `<|tool_call|>` or a specific JSON prefix like `{"name":`).
3. **Allow free text** for the non-tool-call portion of the response (the conversational text before the tool call).

This is more complex than pure JSON generation because the grammar is *partially applied*: free text until a trigger, then constrained.


{{fig:structgen-toolcall-constrained-flow}}


SGLang (covered in [SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html)) implements this via its `constrained_decode` primitive, which can be composed with its programmatic generation API. The vLLM serving stack integrates Outlines or XGrammar as a backend, activated per-request when `guided_json` or `guided_regex` is set in the request parameters.

!!! note "Tool name as a vocabulary restriction"

    The function name constraint is the simplest possible constrained generation
    problem: it's a choice among a finite set of strings. This compiles to a
    trivially small FSM (one state per character position in each name, with
    branches for each valid name). Performance is negligible compared to
    generating the arguments, which may require the full JSON schema machinery.

## Performance Deep Dive: XGrammar's Overlap With Sampling

A naive mental model suggests that constrained decoding adds latency to every decode step: compute the logits, compute the mask, apply the mask, sample. But XGrammar's design exploits a crucial structural property to amortize the mask computation.

### Overlap With the Model Forward Pass

The model forward pass and the mask computation are on different hardware resources (GPU for the forward pass, CPU or GPU for the automaton logic). XGrammar is designed to run mask computation *concurrently* with the model forward pass using separate CPU threads. While the GPU executes the transformer layers for step $t$, the CPU is computing the mask for step $t+1$ by advancing the automaton from step $t-1$'s sampled token.

This overlap means the wall-clock cost of constrained decoding is approximately:

$$
T_{\text{constrained}} \approx \max(T_{\text{model\_fwd}}, T_{\text{mask\_compute}})
$$

rather than the naive:

$$
T_{\text{constrained, naive}} = T_{\text{model\_fwd}} + T_{\text{mask\_compute}}
$$

For large models where $T_{\text{model\_fwd}}$ dominates ($\gg T_{\text{mask\_compute}}$), the overhead of constrained generation is near-zero.

!!! example "Throughput numbers in context"

    For a 70B parameter model on a 4×H100 setup, a single decode step takes
    roughly 20–50 ms (model forward pass). XGrammar's mask computation for a
    complex JSON schema takes on the order of 0.1–1 ms on CPU. The ratio is
    20–500×, meaning the masking cost is completely hidden in the model's
    computation. For a 7B model on a single A100, the forward pass takes
    roughly 2–5 ms, and masking might be ~0.5 ms — still a small fraction.
    The overhead only becomes significant for very small models or trivially fast
    custom kernels.

### Batch Heterogeneity

In a batched serving scenario, different requests may have different grammar constraints. XGrammar handles this by maintaining a separate automaton state per sequence. Since mask application is per-sequence (each sequence has its own logit vector), there's no cross-sequence contention. The automaton state objects are lightweight CPU structures — a few hundred bytes each for typical JSON schemas.

One complication: when a batch contains a mix of constrained and unconstrained requests, the serving engine must apply masks only to the constrained ones. vLLM implements this by computing a "merged mask" that is all-ones (no restriction) for unconstrained sequences and the FSM-derived mask for constrained ones, then applying a single batched mask operation.

## Grammar Specification Formats

Different tools accept different input formats for expressing the constraint. Understanding these helps you choose the right tool.

| Format | Example | Tool Support |
|---|---|---|
| Python regex | `r"\d{4}-\d{2}-\d{2}"` | Outlines, llguidance, XGrammar |
| Pydantic model | `class Out(BaseModel): x: int` | Outlines, instructor |
| JSON Schema | `{"type": "object", "properties": {...}}` | Outlines, XGrammar, vLLM guided JSON |
| EBNF grammar | `rule ::= "a" rule "b" \| ""` | Outlines, llguidance, llama.cpp grammars |
| Lark grammar | Lark-style `.lark` syntax | llguidance |
| Choice list | `["opt1", "opt2", "opt3"]` | Outlines `.generate.choice()` |

JSON Schema is the most practically important format because it bridges directly to OpenAPI specs (tool schemas) and Pydantic model serialization.

!!! interview "Interview Corner"

    **Q:** A candidate claims that constrained decoding "doesn't change what the
    model would generate anyway" because well-trained models already output valid
    JSON. Why is this claim wrong, and when does constrained decoding most affect
    the output distribution?

    **A:** The claim is wrong for several reasons. First, even highly capable models
    fail at JSON validity with measurable frequency — especially for deeply nested
    schemas, long outputs, or schemas the model has not seen during training.
    Second, constrained decoding *does* change the probability distribution over
    valid completions: by zeroing out invalid tokens and renormalizing, the model
    effectively redistributes probability mass among only the valid options.
    This means constrained generation can produce different valid outputs than
    unconstrained generation (not just the same valid outputs minus the invalid ones).
    Third, constrained decoding matters most at structural boundaries — the opening
    of a nested object, the choice of enum value when several are plausible — where
    the model's probability mass is spread across valid and invalid options.
    At those points, the renormalization step meaningfully shifts which valid token
    wins. Fourth, constrained decoding provides a *hard guarantee* rather than a
    statistical tendency — critical for production systems where a single malformed
    output can crash a downstream service.

## Edge Cases, Pitfalls, and Practical Advice

### Tokenization Boundary Mismatches

The most insidious class of bugs in constrained generation arises from tokenization boundaries. Consider the pattern `r"true|false"` applied to a BPE tokenizer. The string `true` might be tokenized as a single token `true` (token ID 1234) or as the pair `tr` + `ue` depending on context. The FSM must be computed over the *decoded* token strings (bytes), not the token IDs directly. Libraries handle this, but custom implementations frequently get it wrong.

A specific failure mode: a regex that allows the character sequence `t-r-u-e` might incorrectly allow token `"truth"` because after consuming `t`, `r`, `u`, the FSM is in a state that also accepts `th` — if the FSM is character-level but the token is `"tr"`, the simulation must verify that `"tr"` leaves the FSM in a live state (it does) *and* that the remaining suffix `"ue"` can continue from that live state.

```python
# Bug: naive check that doesn't properly simulate multi-char tokens
def is_token_valid_BUGGY(fsm_state, token_str, fsm):
    """BUG: only checks if the token_str matches as a prefix of a word,
    not whether each character advances the FSM correctly."""
    return token_str.startswith(fsm.expected_prefix)   # WRONG

# Correct: simulate the FSM character by character over the token
def is_token_valid_CORRECT(fsm_state, token_str, fsm):
    """Simulates FSM transitions for every character in the token."""
    current = fsm_state
    for char in token_str:
        current = fsm.transition(current, char)
        if current is None:      # dead state
            return False
    return True  # reached a live (non-dead) state; token is valid
```

### Empty Mask (Infeasible Constraint)

If the grammar reaches a state where no token in the vocabulary is valid, the mask is all-zeros. After masking, `softmax` produces NaN because $\exp(-\infty)$ everywhere sums to zero. Production servers must detect this and either:

- Fall back to unconstrained generation (acceptable for non-critical constraints).
- Raise an error and return the partial output so far.
- Perform backtracking (expensive but possible with beam search or speculative approaches).

This situation can arise legitimately when a schema has constraints that the model genuinely cannot satisfy given the prefix it has emitted. The solution at the system level is to catch the all-zero mask condition before calling softmax:

```python
def safe_apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply mask with fallback if all tokens are forbidden."""
    if not mask.any():
        # All tokens forbidden: grammar is unsatisfiable at this point.
        # Log a warning and fall back to unconstrained sampling.
        import warnings
        warnings.warn(
            "Constrained generation: no valid tokens in current FSM state. "
            "Falling back to unconstrained sampling for this step.",
            RuntimeWarning,
            stacklevel=2,
        )
        return logits   # Return unmasked logits
    return apply_mask_to_logits(logits, mask)
```

### Schema Compilation Latency

For services with many distinct schemas (e.g., one per user-defined tool), schema compilation can become a bottleneck. Each unique JSON Schema or EBNF grammar requires a compilation pass (typically 10–100 ms for complex schemas). The solution is an LRU-cached schema compiler, keyed by schema hash. vLLM's XGrammar integration includes such a cache.

### Interaction With Speculative Decoding

Speculative decoding (covered in [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html)) speculatively generates multiple tokens via a draft model, then verifies them with the target model. Constrained generation complicates this: the draft model's speculative tokens must also be valid under the grammar, and the FSM state must be advanced tentatively during speculation and rolled back on rejection.

XGrammar and SGLang both support constrained speculative decoding. The FSM state is cloned before entering the speculative window and committed or rolled back based on the target model's acceptance decision. The overhead of state cloning is negligible compared to the draft inference.

## Relationship to Other Inference Techniques

Constrained generation interacts with several other inference optimizations discussed in this part of the book:

- **KV-cache and prefix caching** ([Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html)): The grammar state is deterministic given the output tokens, so it can be cached alongside the KV-cache key. If two requests share the same output prefix (unusual but possible for templated generation), the grammar state can be reused.

- **Continuous batching** ([Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)): Each request in a batch has an independent grammar state. Constrained requests may have sparser logit vectors (many tokens masked), which does not affect the batching logic but can affect the softmax kernel behavior.

- **Sampling** ([Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html)): Constrained generation composes with temperature, top-k, and top-p. The recommended order is: apply grammar mask first (hard constraint), then apply top-k/top-p to the allowed tokens (soft quality filter), then sample. Applying top-p before the grammar mask risks eliminating all valid tokens if the valid tokens collectively have less than $p$ probability mass.

- **Tool use and agents** ([Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html)): Constrained generation is the mechanism that makes reliable tool-call formatting possible. Without it, even good models occasionally emit malformed function calls.

## Summary: Choosing a Library

For a working engineer deploying constrained generation today, the decision tree is:


{{fig:structgen-library-decision-tree}}


!!! key "Key Takeaways"

    - Free-text generation does not guarantee output validity; constrained generation
      hard-wires a formal grammar into the decoding process via logit masking.
    - Finite-state machines (FSMs) handle regular languages (regexes, fixed schemas);
      pushdown automata (PDAs) are needed for context-free languages like JSON.
    - The core algorithm: pre-compute a `state × token_id` boolean mask matrix;
      at each decode step, look up the current state's row and zero out forbidden logits.
    - Outlines introduced the foundational token-level FSM index; XGrammar extended
      this to context-free grammars with the context-independent/context-dependent
      token split, achieving sub-5% overhead for large models.
    - llguidance uses an incremental Earley parser to avoid DFA state explosion for
      complex grammars while maintaining practical throughput.
    - Constrained decoding changes the conditional distribution over valid tokens
      (renormalization), not just the set of generated sequences.
    - Apply grammar masking first, then top-k/top-p, to avoid scenarios where all
      valid tokens fall below the top-p threshold.
    - Token boundary mismatches are the most common implementation bug: always
      simulate the FSM character-by-character over the decoded token string, not
      over the token ID.
    - XGrammar overlaps mask computation with the GPU forward pass, making the
      runtime overhead near-zero for model sizes ≥ 7B.

!!! sota "State of the Art & Resources (2026)"
    Structured and constrained generation has matured into a standard component of production LLM serving: FSM-based logit masking is now the default backend in every major inference engine (vLLM, SGLang, TensorRT-LLM, llama.cpp), and the engineering focus has shifted toward near-zero overhead via overlap with the GPU forward pass and dynamic grammar switching for agentic workflows.

    **Foundational work**

    - [Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023)](https://arxiv.org/abs/2307.09702) — introduces the token-level FSM index that underlies Outlines and most subsequent systems.
    - [Koo, Liu & He, *Automata-based Constraints for Language Model Decoding* (2024)](https://arxiv.org/abs/2407.08103) — rigorous automata-theoretic treatment; achieves ~7,000× faster grammar compilation than prior methods and extends coverage to deterministic CFLs.

    **Recent advances (2023–2026)**

    - [Dong et al., *XGrammar: Flexible and Efficient Structured Generation Engine for LLMs* (2024)](https://arxiv.org/abs/2411.15100) — context-independent/context-dependent token split reduces per-step masking cost to under 40 µs, now the default backend in vLLM and SGLang.
    - [Li et al., *XGrammar-2: Efficient Dynamic Structured Generation Engine for Agentic LLMs* (2026)](https://arxiv.org/abs/2601.04426) — adds TagDispatch for mid-response grammar switching and Cross-Grammar Cache for substructure reuse; over 6× faster compilation than XGrammar-1.

    **Open-source & tools**

    - [dottxt-ai/outlines](https://github.com/outlines-dev/outlines) — the original and most widely-used Python library for FSM-constrained generation; supports regex, JSON Schema, Pydantic models, and EBNF.
    - [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar) — production-grade structured generation engine; default backend in vLLM, SGLang, TensorRT-LLM, and MLC-LLM.
    - [guidance-ai/llguidance](https://github.com/guidance-ai/llguidance) — Rust-core Earley-parser-based engine (~50 µs/token for 128k-vocab tokenizers); integrated into llama.cpp, vLLM, SGLang, Chromium, and OpenAI's own structured outputs.
    - [ggml-org/llama.cpp GBNF grammar guide](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) — practical EBNF grammar format (GBNF) for the llama.cpp ecosystem; reference for writing JSON/SQL/custom grammars.

    **Go deeper**

    - [vLLM structured outputs documentation](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — official reference for `guided_json`, `guided_regex`, and `guided_grammar` parameters and XGrammar/guidance backend configuration.
    - [Outlines documentation](https://dottxt-ai.github.io/outlines/welcome/) — official docs covering the full API: `.generate.json()`, `.generate.regex()`, `.generate.choice()`, and multi-backend support.
    - [Cooper, *A Guide to Structured Generation Using Constrained Decoding* (2024)](https://www.aidancooper.co.uk/constrained-decoding/) — practitioner-focused explainer covering implementation patterns, pitfalls, and CFG extensions.

## Further Reading

- **Willard & Louf, "Efficient Guided Generation for Large Language Models" (2023)** — the foundational Outlines paper; introduces the token-level FSM index and the Outlines library.
- **Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" (2024)** — the context-independent/context-dependent token classification and overlap-with-sampling design.
- **Microsoft Research, llguidance (GitHub: microsoft/llguidance)** — open-source Earley-parser-based constrained generation engine with Rust core.
- **Peng et al., "LMQL: Programming Large Language Models" (2022)** — an early declarative constraint language for LLM queries; pioneered the idea of in-flight constraint enforcement.
- **llama.cpp GGUF grammar specification (GitHub: ggerganov/llama.cpp, `grammars/` directory)** — practical EBNF grammar format used by the llama.cpp ecosystem; good reference for JSON/SQL grammar construction.
- **Guidance (GitHub: guidance-ai/guidance)** — an alternative structured generation approach using a domain-specific templating language; predates Outlines and useful for understanding the design space.
- **Koo et al., "Automata-based Constraints for Language Model Decoding" (2024)** — a theoretical treatment of the expressiveness and complexity of automaton-based constraints for LLM decoding.
