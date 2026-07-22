"""
CI-tested extracts of runnable code blocks from
content/99-appendix/02-math-reference.md

Each block below is the book's ACTUAL code, executed and exercised
(functions called / classes instantiated), not merely imported.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Block 1: safe_softmax  (chapter lines ~23-44)
# ---------------------------------------------------------------------------
def test_safe_softmax():
    def safe_softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
        z_shifted = z - z.max(dim=dim, keepdim=True).values  # broadcast-safe
        exp_z = torch.exp(z_shifted)
        return exp_z / exp_z.sum(dim=dim, keepdim=True)

    logits = torch.tensor([2.0, 1.0, 0.1])
    probs = safe_softmax(logits)
    assert torch.allclose(probs.sum(), torch.tensor(1.0))
    # book print claim: tensor([0.6590, 0.2424, 0.0986])
    assert torch.allclose(probs, torch.tensor([0.6590, 0.2424, 0.0986]), atol=1e-4)
    # equivalence with reference softmax
    assert torch.allclose(probs, F.softmax(logits, dim=-1), atol=1e-6)


# ---------------------------------------------------------------------------
# Block 2: CrossEntropyLoss / perplexity  (lines ~72-86)
# ---------------------------------------------------------------------------
def test_cross_entropy_perplexity():
    torch.manual_seed(0)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    logits = torch.randn(4, 32_000)
    targets = torch.randint(0, 32_000, (4,))
    loss = criterion(logits, targets)
    perplexity = torch.exp(loss)
    assert loss.item() > 0
    assert torch.isfinite(perplexity)
    # equals manual log-sum-exp minus correct logit
    manual = (torch.logsumexp(logits, dim=-1) - logits[torch.arange(4), targets]).mean()
    assert torch.allclose(loss, manual, atol=1e-4)


# ---------------------------------------------------------------------------
# Block 3: rms_norm  (lines ~108-128)
# ---------------------------------------------------------------------------
def test_rms_norm():
    def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        return (x / rms) * weight

    d = 512
    x = torch.randn(2, 16, d)
    w = torch.ones(d)
    out_custom = rms_norm(x, w)
    out_builtin = torch.nn.functional.rms_norm(x, (d,), w, eps=1e-6)
    assert torch.allclose(out_custom, out_builtin, atol=1e-5)


# ---------------------------------------------------------------------------
# Block 4: scaled_dot_product_attention  (lines ~162-187)
# NOTE: book was missing `import math` (fixed in chapter).
# ---------------------------------------------------------------------------
def test_scaled_dot_product_attention():
    def scaled_dot_product_attention(q, k, v, causal: bool = True):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if causal:
            T = scores.size(-1)
            mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    B, H, T, dk, dv = 2, 3, 5, 8, 8
    q = torch.randn(B, H, T, dk)
    k = torch.randn(B, H, T, dk)
    v = torch.randn(B, H, T, dv)
    out = scaled_dot_product_attention(q, k, v, causal=True)
    assert out.shape == (B, H, T, dv)
    # matches PyTorch reference with a causal mask
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)
    # causal: first query position attends only to itself
    non_causal = scaled_dot_product_attention(q, k, v, causal=False)
    assert not torch.allclose(out, non_causal)


# ---------------------------------------------------------------------------
# Block 5: RoPE apply/build  (lines ~208-233)
# ---------------------------------------------------------------------------
def test_rope():
    def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
        x_rotated = x_complex * freqs_cis
        return torch.view_as_real(x_rotated).flatten(-2).type_as(x)

    def build_rope_freqs(seq_len: int, d_head: int, base: float = 10_000.0) -> torch.Tensor:
        i = torch.arange(0, d_head, 2).float()
        theta = 1.0 / (base ** (i / d_head))
        m = torch.arange(seq_len).float()
        freqs = torch.outer(m, theta)
        return torch.polar(torch.ones_like(freqs), freqs)

    B, T, H, d_head = 2, 6, 4, 16
    x = torch.randn(B, T, H, d_head)
    freqs = build_rope_freqs(T, d_head)
    assert freqs.shape == (T, d_head // 2)
    out = apply_rope(x, freqs)
    assert out.shape == x.shape
    # RoPE preserves per-vector norm (pure rotation)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)
    # position 0 rotation is identity (m=0 => angle 0)
    assert torch.allclose(out[:, 0], x[:, 0], atol=1e-4)

    # Relative-position property: <RoPE(q,m), RoPE(k,n)> depends only on m-n
    q = torch.randn(1, 1, 1, d_head)
    k = torch.randn(1, 1, 1, d_head)

    def dot_at(m, n):
        fm = build_rope_freqs(m + 1, d_head)[m:m + 1]
        fn = build_rope_freqs(n + 1, d_head)[n:n + 1]
        qm = apply_rope(q.expand(1, 1, 1, d_head).reshape(1, 1, 1, d_head), fm)
        kn = apply_rope(k.reshape(1, 1, 1, d_head), fn)
        return (qm * kn).sum().item()

    assert abs(dot_at(5, 3) - dot_at(4, 2)) < 1e-3  # both have m-n=2


# ---------------------------------------------------------------------------
# Block 6: adamw_step  (lines ~263-289)
# ---------------------------------------------------------------------------
def test_adamw_step():
    def adamw_step(param, grad, m, v, t, lr=3e-4, beta1=0.9, beta2=0.95,
                   eps=1e-8, wd=0.1) -> None:
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        param.mul_(1 - lr * wd)
        param.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

    # Reproduce a step against torch.optim.AdamW on a single tensor.
    torch.manual_seed(1)
    p_ref = torch.randn(10, requires_grad=True)
    g = torch.randn(10)
    p_custom = p_ref.detach().clone()

    opt = torch.optim.AdamW([p_ref], lr=3e-4, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.1)
    p_ref.grad = g.clone()
    opt.step()

    m = torch.zeros(10)
    v = torch.zeros(10)
    adamw_step(p_custom, g.clone(), m, v, t=1)

    assert torch.allclose(p_custom, p_ref.detach(), atol=1e-6)


# ---------------------------------------------------------------------------
# Block 7: LoRALinear  (lines ~360-417)
# ---------------------------------------------------------------------------
def test_lora_linear():
    class LoRALinear(nn.Module):
        def __init__(self, in_features, out_features, r=16,
                     lora_alpha=16.0, bias=False):
            super().__init__()
            self.r = r
            self.scaling = lora_alpha / r
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features), requires_grad=False)
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter("bias", None)
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def forward(self, x):
            base_out = nn.functional.linear(x, self.weight, self.bias)
            lora_out = nn.functional.linear(x, self.lora_A)
            lora_out = nn.functional.linear(lora_out, self.lora_B)
            return base_out + self.scaling * lora_out

        def merge(self) -> nn.Linear:
            merged_w = self.weight + self.scaling * (self.lora_B @ self.lora_A)
            lin = nn.Linear(self.weight.size(1), self.weight.size(0),
                            bias=self.bias is not None)
            lin.weight = nn.Parameter(merged_w)
            if self.bias is not None:
                lin.bias = nn.Parameter(self.bias.clone())
            return lin

        # freeze check
    layer = LoRALinear(64, 32, r=8, lora_alpha=16.0)
    x = torch.randn(4, 64)

    # B is zero at init -> ΔW = 0 -> output equals frozen base linear
    base = nn.functional.linear(x, layer.weight)
    assert torch.allclose(layer(x), base, atol=1e-6)
    assert not layer.weight.requires_grad
    assert layer.lora_A.requires_grad and layer.lora_B.requires_grad

    # After perturbing B, merged linear reproduces the LoRA forward pass
    with torch.no_grad():
        layer.lora_B.copy_(torch.randn_like(layer.lora_B))
    merged = layer.merge()
    assert torch.allclose(layer(x), merged(x), atol=1e-5)

    # parameter savings claim: r(d+k) trainable vs d*k
    trainable = layer.lora_A.numel() + layer.lora_B.numel()
    assert trainable == 8 * (64 + 32)


# ---------------------------------------------------------------------------
# Block 8: ppo_loss  (lines ~449-469)
# ---------------------------------------------------------------------------
def test_ppo_loss():
    def ppo_loss(log_probs_new, log_probs_old, advantages,
                 clip_eps=0.2, reduce=True):
        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
        loss = torch.min(surr1, surr2)
        return loss.mean() if reduce else loss

    lp_new = torch.tensor([0.0, 0.5, -0.5, 1.0])
    lp_old = torch.tensor([0.0, 0.0, 0.0, 0.0])
    adv = torch.tensor([1.0, 1.0, -1.0, 1.0])

    per = ppo_loss(lp_new, lp_old, adv, reduce=False)
    ratio = torch.exp(lp_new - lp_old)
    clipped = ratio.clamp(0.8, 1.2)
    expected = torch.min(ratio * adv, clipped * adv)
    assert torch.allclose(per, expected)
    # element with ratio=1 (idx 0) => loss = advantage
    assert torch.isclose(per[0], torch.tensor(1.0))
    # positive advantage with large ratio must be clipped down
    assert per[3] <= ratio[3] * adv[3]
    assert torch.isclose(ppo_loss(lp_new, lp_old, adv), per.mean())


# ---------------------------------------------------------------------------
# Block 9: grpo_advantages  (lines ~491-508)
# ---------------------------------------------------------------------------
def test_grpo_advantages():
    def grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
        mean = rewards.mean()
        std = rewards.std(unbiased=False).clamp(min=1e-8)
        return (rewards - mean) / std

    rewards = torch.tensor([1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0])
    adv = grpo_advantages(rewards)
    # zero-mean, unit-std (population) group-normalised advantages
    assert torch.isclose(adv.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(adv.std(unbiased=False), torch.tensor(1.0), atol=1e-5)
    # corrected book print claim
    expected = torch.tensor([1.15, -1.15, 1.15, 0.0, -1.15, 1.15, 0.0, -1.15])
    assert torch.allclose(adv, expected, atol=1e-2)


# ---------------------------------------------------------------------------
# Block 10: dpo_loss  (lines ~534-555)
# ---------------------------------------------------------------------------
def test_dpo_loss():
    def dpo_loss(log_probs_theta_w, log_probs_theta_l,
                 log_probs_ref_w, log_probs_ref_l, beta=0.1):
        log_ratio_w = log_probs_theta_w - log_probs_ref_w
        log_ratio_l = log_probs_theta_l - log_probs_ref_l
        gap = beta * (log_ratio_w - log_ratio_l)
        loss = -F.logsigmoid(gap)
        return loss.mean()

    # When policy == ref, gap = 0 => loss = -log(0.5) = ln 2
    z = torch.zeros(4)
    loss0 = dpo_loss(z, z, z, z)
    assert torch.isclose(loss0, torch.tensor(math.log(2.0)), atol=1e-6)

    # Preferred response gaining relative log-prob lowers the loss
    theta_w = torch.tensor([1.0, 1.0])
    theta_l = torch.tensor([-1.0, -1.0])
    ref = torch.zeros(2)
    loss_good = dpo_loss(theta_w, theta_l, ref, ref)
    loss_bad = dpo_loss(theta_l, theta_w, ref, ref)
    assert loss_good < loss0 < loss_bad


# ---------------------------------------------------------------------------
# Block 11: kl_estimators  (lines ~579-606)
# ---------------------------------------------------------------------------
def test_kl_estimators():
    def kl_estimators(log_pi, log_pi_ref) -> dict:
        log_r = log_pi - log_pi_ref
        r = log_r.exp()
        k1 = log_r
        k2 = r - 1.0 - log_r
        k3 = (r - 1.0).pow(2) / 2
        return {"k1": k1.mean().item(), "k2": k2.mean().item(), "k3": k3.mean().item()}

    lp = torch.full((4, 512), -3.0)
    out = kl_estimators(lp, lp)
    assert out == {"k1": 0.0, "k2": 0.0, "k3": 0.0}

    # k2 and k3 are non-negative; k1 can be negative
    torch.manual_seed(2)
    a = torch.randn(4, 512).log_softmax(-1)
    b = torch.randn(4, 512).log_softmax(-1)
    out2 = kl_estimators(a, b)
    assert out2["k2"] >= 0
    assert out2["k3"] >= 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} blocks passed.")
