"""
Runs the CPU-runnable Python blocks from:
    content/13-interp-safety-gov/03-privacy-memorization-dp.md

Chapter blocks (0-indexed, in document order) and disposition:
    #0 (line ~123, canary_exposure / sequence_logprob)      -> SKIP(needs-gpu):
        default device="cuda"; scores a real HF causal-LM's log-likelihood over
        thousands of sampled candidate bodies. Not a toy-shape CPU exercise.
    #1 (line ~202, PII scrubbing: PII_PATTERNS / scrub_pii)  -> TESTED (verbatim)
    #2 (line ~271, dp_sgd_step)                               -> SKIP(needs-gpu):
        default device="cuda"; written to run per-example backward passes
        against a real nn.Module supplied by the caller.
    #3 (line ~350, MemorizationGuard, Bloom-filter guard)    -> TESTED (verbatim,
        conditional on the optional `pybloom_live` third-party package being
        importable -- it is not in the guaranteed CI dependency set, so we guard
        the import per the hard rules and skip gracefully if absent).
    #4 (line ~385, audit_privacy.py harness)                  -> SKIP(needs-gpu):
        default device="cuda"; membership_scores/extract_canary call a real
        HF model's forward pass many times.
    #5 (line ~515, ```text sample output block)                -> SKIP(non-python):
        illustrative console output, not code.

This file executes blocks #1 and #3 verbatim (copied faithfully from the
chapter) with minimal glue so they run standalone on CPU.
"""

import sys

# --------------------------------------------------------------------------
# Optional third-party dependency used by block #3. Guard at module scope so
# this test file always loads in CI even though `pybloom_live` is not in the
# guaranteed dependency set (numpy, torch-cpu, einops, sklearn, stdlib).
# --------------------------------------------------------------------------
try:
    from pybloom_live import BloomFilter
except Exception:
    BloomFilter = None


# ==========================================================================
# Block #1 (chapter line ~202) -- data-level PII scrubbing.
# Copied verbatim from the chapter's "PII scrubbing" code block.
# ==========================================================================
import re

# High-recall regexes for the structured tail of PII. These over-redact on
# purpose: false positives (a redacted non-secret) are far cheaper than false
# negatives (a leaked SSN) when training a model whose weights are forever.
PII_PATTERNS = {
    "EMAIL":  re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE":  re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    "SSN":    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CCARD":  re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    # Common secret-key shapes (AWS, generic 32-64 hex, Slack/GitHub-ish tokens):
    "AWSKEY": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "HEX32":  re.compile(r"\b[0-9a-fA-F]{32,64}\b"),
    "GHPAT":  re.compile(r"\bghp_[0-9A-Za-z]{36}\b"),
}

def scrub_pii(text: str) -> tuple[str, dict]:
    """Replace structured PII with typed placeholders. Returns (clean, counts).

    Order matters: redact the most specific patterns first so a credit-card
    number is not partially eaten by the phone regex.
    """
    counts = {}
    for label in ("AWSKEY", "GHPAT", "SSN", "CCARD", "EMAIL", "PHONE", "HEX32"):
        pattern = PII_PATTERNS[label]
        text, n = pattern.subn(f"<{label}>", text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return text, counts

# For free-form PII (person names, locations, orgs) layer an NER model on top:
#   import spacy; nlp = spacy.load("en_core_web_trf")
#   doc = nlp(text)
#   for ent in reversed(doc.ents):        # reverse so offsets stay valid
#       if ent.label_ in {"PERSON", "GPE", "ORG"}:
#           text = text[:ent.start_char] + f"<{ent.label_}>" + text[ent.end_char:]

sample = "Contact john.doe@acme.io or 415-555-0199; key AKIAIOSFODNN7EXAMPLE; ssn 123-45-6789."
clean, counts = scrub_pii(sample)
print(clean)   # Contact <EMAIL> or <PHONE>; key <AWSKEY>; ssn <SSN>.
print(counts)  # {'AWSKEY': 1, 'SSN': 1, 'EMAIL': 1, 'PHONE': 1}


# ==========================================================================
# Block #3 (chapter line ~350) -- inference-level memorization guard.
# Copied verbatim from the chapter's "Output-side memorization guard" block,
# with only the module-level `from pybloom_live import BloomFilter` hoisted
# to the guarded try/except above (per the hard rules for optional 3rd-party
# imports).
# ==========================================================================
class MemorizationGuard:
    def __init__(self, n=8, capacity=10_000_000, error_rate=1e-6):
        self.n = n                                  # n-gram length (tokens/words)
        self.bf = BloomFilter(capacity=capacity, error_rate=error_rate)

    def index(self, sensitive_texts):
        for text in sensitive_texts:                # build the forbidden set
            toks = text.split()
            for i in range(len(toks) - self.n + 1):
                self.bf.add(" ".join(toks[i:i + self.n]))

    def violates(self, generated: str) -> bool:
        toks = generated.split()
        for i in range(len(toks) - self.n + 1):
            if " ".join(toks[i:i + self.n]) in self.bf:   # verbatim overlap
                return True                                # block / regenerate
        return False

# guard = MemorizationGuard(n=8)
# guard.index(known_secrets_and_copyrighted_corpus)
# if guard.violates(model_output): regenerate_with_higher_temp_or_refuse()


def main() -> None:
    # ---- exercise block #1 ---------------------------------------------
    assert clean == "Contact <EMAIL> or <PHONE>; key <AWSKEY>; ssn <SSN>.", clean
    assert counts == {"AWSKEY": 1, "SSN": 1, "EMAIL": 1, "PHONE": 1}, counts
    print("[block #1] scrub_pii OK")

    # ---- exercise block #3 (only if pybloom_live is actually available) --
    if BloomFilter is None:
        print(
            "[block #3] SKIP(optional-dep): `pybloom_live` is not installed "
            "in this environment (it is not in the guaranteed CI dependency "
            "set). MemorizationGuard is defined above but not instantiated."
        )
    else:
        secret_doc = "the quick brown fox jumps over the lazy dog near the river bank"
        guard = MemorizationGuard(n=4, capacity=1000, error_rate=1e-6)
        guard.index([secret_doc])

        # A generation that reproduces a 4-word (n=4) window verbatim -> should flag.
        verbatim_leak = "somehow the quick brown fox jumps over the lazy dog appeared"
        assert guard.violates(verbatim_leak) is True

        # A generation that shares no 4-gram with the indexed secret -> should not flag.
        clean_generation = "completely unrelated text about something else entirely today"
        assert guard.violates(clean_generation) is False

        print("[block #3] MemorizationGuard OK (verbatim n-gram flagged, "
              "unrelated text not flagged)")

    print("\nAll CPU-runnable blocks executed successfully.")


if __name__ == "__main__":
    main()
    sys.exit(0)
