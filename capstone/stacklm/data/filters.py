"""Domain-routed quality filters (Ch. 14.2).

A lean, dependency-free subset of the heuristics in Ch. 3.2, tuned so that
filtering compute never competes with training compute at a 20B-token budget.
Code and math are routed to their own gates: their character distributions are
so unlike prose that the generic web filter would reject nearly all of them.
"""
import hashlib
import json
import re

_WORD_RE = re.compile(r"\S+")

# Every threshold lives here so the whole filter config can be hashed into the
# corpus manifest (Ch. 14.12 reproducibility checklist).
FILTER_CONFIG = {
    "web": {"min_words": 50, "max_words": 100_000, "min_mean_word_len": 3.0,
            "max_mean_word_len": 10.0, "min_alpha_frac": 0.60, "max_digit_frac": 0.20,
            "max_repeat_line_frac": 0.30},
    "code": {"min_chars": 20, "max_chars": 200_000, "max_char_frac": 0.30},
    "math": {"min_words": 20, "min_digit_frac": 0.03, "min_markers": 3},
}


def filter_config_hash() -> str:
    """Stable 12-hex-char digest of FILTER_CONFIG, recorded in the manifest."""
    blob = json.dumps(FILTER_CONFIG, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=6).hexdigest()


def basic_stats(text: str) -> dict:
    words = _WORD_RE.findall(text)
    n_words = len(words) or 1
    n_chars = len(text) or 1
    alpha = sum(c.isalpha() for c in text)
    digit = sum(c.isdigit() for c in text)
    lines = text.splitlines() or [text]
    uniq = len(set(lines))
    return dict(
        n_words=n_words,
        alpha_frac=alpha / n_chars,
        digit_frac=digit / n_chars,
        mean_word_len=sum(len(w) for w in words) / n_words,
        dup_line_frac=1.0 - uniq / len(lines),
    )


def passes_web_filter(text: str) -> bool:
    """Generic prose gate for FineWeb-Edu / Cosmopedia documents."""
    c = FILTER_CONFIG["web"]
    s = basic_stats(text)
    return (
        c["min_words"] <= s["n_words"] <= c["max_words"]
        and c["min_mean_word_len"] <= s["mean_word_len"] <= c["max_mean_word_len"]
        and s["alpha_frac"] >= c["min_alpha_frac"]
        and s["digit_frac"] <= c["max_digit_frac"]
        and s["dup_line_frac"] <= c["max_repeat_line_frac"]
    )


def passes_code_filter(text: str) -> bool:
    """Loose gate for StarCoder: reject empty/binary/minified-looking files
    (dominated by one repeated character); keep everything else."""
    c = FILTER_CONFIG["code"]
    if not (c["min_chars"] <= len(text) <= c["max_chars"]):
        return False
    head = text[:2000]
    most_common_frac = max(head.count(ch) for ch in set(head)) / max(len(head), 1)
    return most_common_frac <= c["max_char_frac"]


def passes_math_filter(text: str) -> bool:
    """FineMath/OpenWebMath gate: require mathematical density, not prose fluency."""
    c = FILTER_CONFIG["math"]
    s = basic_stats(text)
    markers = sum(text.count(m) for m in ("=", "\\frac", "$", "^", "\\sum"))
    return s["n_words"] >= c["min_words"] and (
        s["digit_frac"] >= c["min_digit_frac"] or markers >= c["min_markers"]
    )


_FILTERS = {
    "web": passes_web_filter,
    "synthetic": passes_web_filter,
    "code": passes_code_filter,
    "math": passes_math_filter,
}


def quality_filter(doc: dict) -> bool:
    """Route a document to the filter for its source domain."""
    fn = _FILTERS.get(doc.get("domain", "web"), passes_web_filter)
    return fn(doc["text"])
