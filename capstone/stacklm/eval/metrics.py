"""Evaluation (Ch. 14.11): held-out perplexity plus a couple of tiny capability
probes -- arithmetic exact-match and retrieval-QA. Honest framing: a 100M model
is a narrow tool, and these probes are sized to prove the code path, not to rank
the model.
"""
import math
import random
import re

import torch
import torch.nn.functional as F

from ..train.loop import autocast_ctx


@torch.no_grad()
def compute_perplexity(model, dataset, *, device="cpu", batch_size=4,
                       max_batches=None, use_seq_ids=True) -> dict:
    device = torch.device(device)
    model.to(device).eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)
    total_nll, total_tokens = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        seq_ids = batch["seq_ids"].to(device) if use_seq_ids else None
        with autocast_ctx(device):
            logits, _ = model(ids, seq_ids=seq_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   targets.reshape(-1), reduction="sum",
                                   ignore_index=-100)
        total_nll += loss.item()
        total_tokens += targets.numel()
    mean_nll = total_nll / max(1, total_tokens)
    return {"loss_nats_per_token": mean_nll, "perplexity": math.exp(min(mean_nll, 20.0)),
            "n_tokens_evaluated": total_tokens}


def make_arithmetic_probe(n=20, seed=0, max_digits=2):
    rng = random.Random(seed)
    problems = []
    for _ in range(n):
        a = rng.randint(0, 10 ** max_digits - 1)
        b = rng.randint(0, 10 ** max_digits - 1)
        op = rng.choice(["+", "-"])
        answer = a + b if op == "+" else a - b
        problems.append({"prompt": f"{a} {op} {b} = ", "answer": answer})
    return problems


_NUM_RE = re.compile(r"-?\d+")


@torch.no_grad()
def eval_arithmetic(model, tokenizer, generate_fn, problems) -> dict:
    n_correct = 0
    for p in problems:
        completion = generate_fn(model, tokenizer, prompt=p["prompt"],
                                 max_new_tokens=6, temperature=0.0)
        match = _NUM_RE.search(completion)
        predicted = int(match.group()) if match else None
        n_correct += int(predicted == p["answer"])
    return {"accuracy": n_correct / max(1, len(problems)), "n": len(problems)}


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


@torch.no_grad()
def eval_retrieval_qa(model, tokenizer, generate_fn, retriever, qa_pairs) -> dict:
    n_correct = 0
    for pair in qa_pairs:
        hits = retriever.search(pair["question"], k=1)
        passage = hits[0][0].text if hits else ""
        prompt = (f"Passage: {passage}\nQuestion: {pair['question']}\nAnswer: ")
        completion = generate_fn(model, tokenizer, prompt=prompt,
                                 max_new_tokens=8, temperature=0.0)
        n_correct += int(normalize_answer(completion) == normalize_answer(pair["gold_answer"])
                         or normalize_answer(pair["gold_answer"]) in normalize_answer(completion))
    return {"exact_match": n_correct / max(1, len(qa_pairs)), "n": len(qa_pairs)}
