"""CI-tested extracts of runnable code blocks from
content/13-interp-safety-gov/04-watermarking-provenance.md

Only block #0 (the from-scratch KGW-style green-list watermark, line ~92) is
heuristically CPU-runnable. Blocks #1 (illustrative expected-output text) and
#2 (a C2PA JSON manifest example) are non-Python and are SKIPPED.

Run directly: `python3 tests/13-interp-safety-gov__04-watermarking-provenance.py`
"""

import hashlib
import math
import random
import struct


def block_green_list_watermark():
    # content lines ~92-305 ("From-Scratch Green-List Watermark: Code")
    #
    # NOTE ON A REAL BUG FOUND AND FIXED IN THE BOOK:
    # The original `detect_watermark` body contained a stray `import math`
    # statement placed AFTER an earlier use of `math.sqrt(...)` in the same
    # function. Because that `import` makes `math` a function-local name for
    # the *entire* function body (Python resolves names as local vs. global
    # at compile time based on any assignment/import anywhere in the
    # function), the earlier `math.sqrt(...)` call raised
    # `UnboundLocalError: cannot access local variable 'math' where it is
    # not associated with a value`. Fixed in the .md by deleting the
    # redundant inner `import math` (module-level `import math` already
    # covers it) — mirrored here.

    # ---------------------------------------------------------------------
    # Vocabulary and pseudo-logits (stand-in for a real LM)
    # ---------------------------------------------------------------------
    VOCAB_SIZE = 32_000
    VOCAB = list(range(VOCAB_SIZE))  # tokens are just integers

    def fake_lm_logits(prev_token: int, seed: int = 42) -> list[float]:
        """
        Returns random logits for demonstration. A real implementation
        would call model.forward() and extract the next-token logit vector.
        """
        rng = random.Random(seed ^ prev_token)
        return [rng.gauss(0, 1) for _ in VOCAB]

    # ---------------------------------------------------------------------
    # Green-list construction
    # ---------------------------------------------------------------------
    def get_green_list(
        prev_token: int,
        secret_key: bytes,
        gamma: float = 0.5,
    ) -> set[int]:
        """
        Derive the green list for position t given the previous token and key.
        Uses HMAC-SHA256 as the PRF; maps each vocabulary token to a hash value
        and selects the lowest gamma*|V| hashes as the green list.
        """
        green_size = int(gamma * VOCAB_SIZE)

        # Score every token by hashing (key || prev_token || token_id)
        scores = []
        for tok in VOCAB:
            # Pack prev_token and tok as little-endian 32-bit ints
            data = secret_key + struct.pack("<II", prev_token, tok)
            h = hashlib.sha256(data).digest()
            # Interpret first 8 bytes as a uint64 for a uniform [0, 2^64) score
            score = struct.unpack("<Q", h[:8])[0]
            scores.append((score, tok))

        # The green list is the gamma fraction with lowest hash scores
        scores.sort()
        green_set = {tok for _, tok in scores[:green_size]}
        return green_set

    # ---------------------------------------------------------------------
    # Watermarked sampler
    # ---------------------------------------------------------------------
    def softmax(logits: list[float]) -> list[float]:
        """Numerically stable softmax."""
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        s = sum(exps)
        return [e / s for e in exps]

    def sample_token(probs: list[float], rng: random.Random) -> int:
        """Sample a token index from a probability distribution."""
        r = rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return i
        return len(probs) - 1  # fallback

    def generate_watermarked(
        seed_token: int,
        length: int,
        secret_key: bytes,
        delta: float = 2.0,
        gamma: float = 0.5,
        generation_seed: int = 0,
    ) -> list[int]:
        """
        Generate a watermarked token sequence of given length.

        Args:
            seed_token: The token preceding the generation window (context).
            length: Number of tokens to generate.
            secret_key: The watermark secret key (kept by the operator).
            delta: Green-list logit boost.
            gamma: Fraction of vocabulary in the green list.
            generation_seed: Seed for token sampling (mimics temperature sampling).

        Returns:
            List of generated token ids.
        """
        rng = random.Random(generation_seed)
        tokens = []
        prev = seed_token

        for step in range(length):
            # Get base logits from the (fake) language model
            logits = fake_lm_logits(prev, seed=step)

            # Construct green list for this step
            green = get_green_list(prev, secret_key, gamma=gamma)

            # Boost green-list logits
            boosted_logits = [
                l + delta if tok in green else l
                for tok, l in enumerate(logits)
            ]

            probs = softmax(boosted_logits)
            tok = sample_token(probs, rng)
            tokens.append(tok)
            prev = tok

        return tokens

    # ---------------------------------------------------------------------
    # Detector
    # ---------------------------------------------------------------------
    def detect_watermark(
        tokens: list[int],
        seed_token: int,
        secret_key: bytes,
        gamma: float = 0.5,
    ) -> dict:
        """
        Compute the z-score for a sequence of tokens.

        Returns a dict with z_score, green_count, total_tokens, and p_value.
        The null hypothesis is that the text is human-written (each token is
        in the green list with probability gamma, independently).
        """
        T = len(tokens)
        if T == 0:
            return {"z_score": 0.0, "green_count": 0, "total": 0, "p_value": 1.0}

        green_count = 0
        prev = seed_token

        for tok in tokens:
            green = get_green_list(prev, secret_key, gamma=gamma)
            if tok in green:
                green_count += 1
            prev = tok

        # z-score under Binomial(T, gamma) null
        mu = gamma * T
        sigma = math.sqrt(T * gamma * (1 - gamma))
        z = (green_count - mu) / sigma

        # One-sided p-value (standard normal CDF approximation via erfc)
        # (book originally had a redundant `import math` here that shadowed
        # the module-level import and broke the `math.sqrt` call above --
        # fixed by deleting it, see note at top of this function.)
        p_value = 0.5 * math.erfc(z / math.sqrt(2))

        return {
            "z_score": round(z, 4),
            "green_count": green_count,
            "total_tokens": T,
            "gamma_expected": round(mu, 1),
            "p_value": round(p_value, 8),
            "flagged": z > 4.0,
        }

    # ---------------------------------------------------------------------
    # Demo (book's __main__ block, called directly here)
    # ---------------------------------------------------------------------
    KEY = b"supersecret-operator-key-2024"
    SEED_TOKEN = 1234
    N = 200  # tokens to generate

    print("=== Watermarked text ===")
    wm_tokens = generate_watermarked(SEED_TOKEN, N, KEY, delta=2.0, gamma=0.5)
    result_wm = detect_watermark(wm_tokens, SEED_TOKEN, KEY)
    print(result_wm)
    # Expected: z_score >> 4, flagged=True

    print("\n=== Human (random) text ===")
    rng = random.Random(99)
    human_tokens = [rng.randint(0, VOCAB_SIZE - 1) for _ in range(N)]
    result_human = detect_watermark(human_tokens, SEED_TOKEN, KEY)
    print(result_human)
    # Expected: z_score near 0, flagged=False

    print("\n=== Paraphrase attack: replace 40% tokens randomly ===")
    attacked = list(wm_tokens)
    rng2 = random.Random(7)
    for i in range(N):
        if rng2.random() < 0.4:
            attacked[i] = rng2.randint(0, VOCAB_SIZE - 1)
    result_atk = detect_watermark(attacked, SEED_TOKEN, KEY)
    print(result_atk)
    # Expected: z_score substantially reduced from the unattacked value

    # --- verify the book's claims about this exact (seeded, deterministic)
    #     demo, and the general statistical properties of the scheme ---

    # Watermarked text is flagged with a very high z-score.
    assert result_wm["total_tokens"] == N
    assert result_wm["green_count"] > result_wm["gamma_expected"]
    assert result_wm["z_score"] > 4.0
    assert result_wm["flagged"] is True

    # Unwatermarked ("human") random text should NOT be flagged: its
    # green-token rate should sit close to gamma with a small |z|.
    assert result_human["flagged"] is False
    assert abs(result_human["z_score"]) < 4.0

    # A 40% random-substitution attack must reduce the z-score relative to
    # the unattacked watermarked text (dilution effect), consistent with the
    # "Attack Robustness" table's claim that random substitution scales z by
    # roughly (1-p).
    assert result_atk["z_score"] < result_wm["z_score"]

    # p-values should be well-formed probabilities.
    for res in (result_wm, result_human, result_atk):
        assert 0.0 <= res["p_value"] <= 1.0

    # Detector agrees with itself: recomputing on the same tokens gives an
    # identical result (determinism of the hash-based green list).
    result_wm_again = detect_watermark(wm_tokens, SEED_TOKEN, KEY)
    assert result_wm_again == result_wm

    # Different secret key -> (with overwhelming probability) a different
    # green-list assignment, so the watermark should no longer be detected
    # as strongly. This demonstrates the cryptographic-key dependence of the
    # scheme (Section "Context Hashing and Key Security").
    WRONG_KEY = b"a-completely-different-key-000"
    result_wrong_key = detect_watermark(wm_tokens, SEED_TOKEN, WRONG_KEY)
    assert result_wrong_key["z_score"] < result_wm["z_score"]

    # Empty-sequence edge case (explicit in the book's code).
    empty_result = detect_watermark([], SEED_TOKEN, KEY)
    assert empty_result == {
        "z_score": 0.0,
        "green_count": 0,
        "total": 0,
        "p_value": 1.0,
    }


BLOCKS = [
    block_green_list_watermark,
]


def main():
    for fn in BLOCKS:
        print(f"\n===== {fn.__name__} =====")
        fn()
    print(f"\nAll {len(BLOCKS)} code blocks executed and verified.")
    print(
        "\nSKIPPED (non-Python, not code blocks): "
        "#1 (illustrative expected-output text block), "
        "#2 (C2PA JSON manifest example)."
    )


if __name__ == "__main__":
    main()
