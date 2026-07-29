"""Token sampling for `Stack100M.generate` (Ch. 14.4; theory in Ch. 7.9)."""
import torch


def sample_next(logits, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0):
    """(B, V) logits -> (B, 1) sampled ids. temperature <= 0 => greedy."""
    logits = logits.float()
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(dim=-1)
    if top_p < 1.0:
        sp, si = probs.sort(dim=-1, descending=True)
        keep = (sp.cumsum(-1) - sp) < top_p     # smallest nucleus with mass >= top_p
        sp = sp * keep
        sp = sp / sp.sum(-1, keepdim=True)
        return si.gather(-1, torch.multinomial(sp, num_samples=1))
    return torch.multinomial(probs, num_samples=1)
