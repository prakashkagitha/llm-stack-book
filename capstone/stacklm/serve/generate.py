"""CPU generation (Ch. 14.11). A simple autoregressive sampler that runs on a
quantized model on a laptop. No KV cache here (kept minimal and correct); the
chapter discusses the KV-cache optimization separately.
"""
import torch


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 32,
             temperature: float = 0.0, stop_id=None) -> str:
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)], dtype=torch.long)
    if stop_id is None:
        stop_id = tokenizer.eos_id
    out = []
    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -model.cfg.max_seq_len:])
        nxt = _sample(logits[:, -1, :].float(), temperature)
        tok_id = int(nxt)
        if tok_id == stop_id:
            break
        out.append(tok_id)
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return tokenizer.decode(out)


def generate_fn(model, tokenizer, prompt, max_new_tokens=8, temperature=0.0):
    """Adapter matching the eval probes' `generate_fn(model, tok, prompt=, ...)`."""
    return generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                    temperature=temperature)
