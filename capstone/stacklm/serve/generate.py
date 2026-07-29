"""CPU generation (Ch. 14.11). Thin tokenizer-facing wrapper around the model's own
KV-cached `Stack100M.generate` (Ch. 14.4) -- prefill once, then one cached step per
token, so decode is O(T) rather than O(T^2). Quantized models keep the same path.
"""
import torch


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 32,
             temperature: float = 0.0, top_p: float = 1.0, stop_id=None,
             use_cache: bool = True) -> str:
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)], dtype=torch.long)
    if stop_id is None:
        stop_id = tokenizer.eos_id
    # keep the prompt inside the context window, leaving room for the completion
    budget = model.cfg.max_seq_len - max_new_tokens
    ids = ids[:, -budget:]
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_p=top_p, eos_id=stop_id, use_cache=use_cache)
    new = out[0, ids.shape[1]:].tolist()
    if stop_id in new:
        new = new[:new.index(stop_id)]
    return tokenizer.decode(new)


def generate_fn(model, tokenizer, prompt, max_new_tokens=8, temperature=0.0):
    """Adapter matching the eval probes' `generate_fn(model, tok, prompt=, ...)`."""
    return generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                    temperature=temperature)
