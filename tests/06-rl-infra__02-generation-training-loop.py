"""
Runnable-code test harness for content/06-rl-infra/02-generation-training-loop.md

Blocks tested (executed for real, offline, on CPU):

  - Block #0 (~line 98, 42 lines): the vLLM rollout sketch -- `rollout()`.
    The real `vllm` package is never available in CI and instantiating a real
    `vllm.LLM(model="Qwen/Qwen2.5-7B", ...)` would try to download multi-GB model
    weights over the network, which is forbidden. We substitute a minimal fake
    `LLM`/`SamplingParams` pair with the same call surface vLLM exposes
    (`engine.generate(prompts, sampling)` -> list of request outputs, each with
    `.outputs` -> completions with `.token_ids` / `.logprobs`). `rollout()`'s own
    logic (extracting the behavior-policy logprob per token, assembling the
    experience dicts) is the book's verbatim code and executes for real against
    the fake engine's canned response.

  - Block #2 (~line 228, 34 lines): `build_experience_batch()` -- pure torch,
    runs verbatim, exercised with padding on one row so the pad path executes.

  - Block #6 (~line 385, 40 lines): the async rollout/trainer skeleton
    (`generator_worker`, `trainer_loop`, and the module-level `rollout_q` /
    `weight_version` / `shared_weights` state) -- defined verbatim (with one
    documented bug fix, see below) and `generator_worker` is run for real in a
    daemon thread; we assert its effect (rollouts landing in `rollout_q`,
    correctly tagged with the weight version, and `sync_weights_to_engine`
    getting invoked once a new version is pushed).

Included as necessary glue (officially a "fragment", not one of the three
assigned blocks, but block #6's `generator_worker` calls it directly, so it is
defined verbatim and exercised too rather than stubbed out):

  - Block #1 (~line 145): `sync_weights_to_engine()`.

BUG FOUND AND FIXED (mirrored in ../content/06-rl-infra/02-generation-training-loop.md):
  In block #6, `generator_worker` originally read:

      with shared_weights["lock"]:
          if shared_weights["sd"] is not None:
              sync_weights_to_engine(engine, shared_weights["sd"])
              local_v = weight_version["v"]
      ...
      for r in rollout(prompts):
          r["gen_version"] = local_v          # <-- UnboundLocalError on first pass

  `local_v` is assigned only inside the `if shared_weights["sd"] is not None:`
  branch, but `shared_weights["sd"]` starts as `None`. Because Python treats a
  name assigned anywhere in a function body as local to that function, the very
  first loop iteration (before any weight sync has happened) raises
  `UnboundLocalError: local variable 'local_v' referenced before assignment` the
  moment it's used to tag a rollout. The fix hoists `local_v = weight_version["v"]`
  out of the `if`, so it is always read fresh under the lock, and the engine's
  weights are only *synced* when a new state dict has actually been pushed:

      with shared_weights["lock"]:
          local_v = weight_version["v"]
          if shared_weights["sd"] is not None:
              sync_weights_to_engine(engine, shared_weights["sd"])

  This test exercises exactly the buggy path (weight_version untouched, sd=None
  on the first rollouts) to prove the fix is needed and correct.

Blocks intentionally SKIPPED (genuine fragments / non-standalone; not part of the
three assigned CPU-runnable blocks):

  - Block #3 (~line 268): `token_logprobs` + the "Phase 3" usage lines. The usage
    (`old_lp = token_logprobs(policy, input_ids)` etc.) references `policy`,
    `ref`, `input_ids` that only exist inside `rl_outer_step` (block #7) -- not
    standalone.
  - Block #4 (~line 289): `grpo_advantages` -- standalone-safe in isolation but
    not one of the three assigned blocks; out of scope for this harness.
  - Block #5 (~line 305): `minibatch_update_loop` -- needs a real trainable
    `nn.Module` policy + optimizer wired to Phase-3 outputs; out of scope here.
  - Block #7 (~line 436): `rl_outer_step` -- the full end-to-end orchestration
    calling the engine, a real policy/ref model, and the minibatch loop; also
    calls undefined-here `expand()`/`reward_fn()` helpers. `trainer_loop`
    (inside block #6) is likewise defined-but-not-called-to-completion: its body
    literally calls `build_experience_batch(batch, ...)` with a bare `Ellipsis`
    standing in for the elided `(pad_id, device)` arguments (per the chapter's
    own "intentionally minimal" comment), so it is not literally runnable code --
    it is architecture pseudocode, confirmed by the fact that calling it verbatim
    raises a TypeError on a missing argument.
"""

import itertools
import threading
import time

import torch
import torch.nn.functional as F


# =============================================================================
# Fake vLLM boundary (stands in for `from vllm import LLM, SamplingParams`).
# No network call, no model download -- CI has neither. The fake mimics just
# enough of vLLM's public shape for the book's own rollout logic to run for
# real against a canned response.
# =============================================================================

class _FakeLogprob:
    __slots__ = ("logprob",)

    def __init__(self, logprob):
        self.logprob = logprob


class _FakeCompletionOutput:
    def __init__(self, token_ids, logprobs):
        self.token_ids = token_ids
        self.logprobs = logprobs


class _FakeRequestOutput:
    def __init__(self, prompt, outputs):
        self.prompt = prompt
        self.outputs = outputs


class _FakeCollectiveRPC:
    """Records calls so tests can assert a weight sync actually happened."""

    def __init__(self):
        self.calls = []

    def __call__(self, method, args=None):
        self.calls.append((method, args))


class _FakeModelExecutor:
    def __init__(self):
        self.collective_rpc = _FakeCollectiveRPC()


class _FakeLLMEngineInner:
    def __init__(self):
        self.model_executor = _FakeModelExecutor()


class SamplingParams:  # stand-in for vllm.SamplingParams
    def __init__(self, n, temperature, top_p, max_tokens, logprobs):
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.logprobs = logprobs


class LLM:  # stand-in for vllm.LLM
    def __init__(self, model, dtype, gpu_memory_utilization, enable_prefix_caching, max_model_len):
        self.model = model
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching
        self.max_model_len = max_model_len
        self.llm_engine = _FakeLLMEngineInner()

    def generate(self, prompts, sampling):
        """Canned response shaped like vLLM's engine.generate()."""
        outs = []
        for p in prompts:
            comps = []
            for g in range(sampling.n):
                token_ids = [100 + g, 101 + g, 102 + g]  # tiny 3-token fake response
                logprobs = [
                    {tid: _FakeLogprob(-0.1 * (pos + 1) - 0.01 * g)}
                    for pos, tid in enumerate(token_ids)
                ]
                comps.append(_FakeCompletionOutput(token_ids, logprobs))
            outs.append(_FakeRequestOutput(p, comps))
        return outs


# =============================================================================
# Block #0 (~line 98): rollout call inside an RL trainer using vLLM. Verbatim
# from the chapter, except `LLM`/`SamplingParams` resolve to the fakes above.
# =============================================================================

# One persistent engine, created ONCE. We will hot-swap weights into it each step.
engine = LLM(
    model="Qwen/Qwen2.5-7B",
    dtype="bfloat16",
    gpu_memory_utilization=0.5,   # leave room for the trainer if COLOCATED
    enable_prefix_caching=True,   # reuse the shared prompt prefix across the group
    max_model_len=4096,
)

sampling = SamplingParams(
    n=8,                 # G: sample 8 completions per prompt in ONE call
    temperature=1.0,
    top_p=1.0,
    max_tokens=2048,
    logprobs=0,          # return the logprob of the SAMPLED token (behavior logprob)
    # ^ logprobs=0 means "0 extra besides the chosen token" -> we still get the
    #   chosen token's logprob, which is exactly the behavior-policy logprob we need.
)


def rollout(prompts):
    """Return, per prompt, G completions + their per-token sampling logprobs."""
    outs = engine.generate(prompts, sampling)         # batched, continuous-batched internally
    batch = []
    for req in outs:                                  # one req per prompt
        for comp in req.outputs:                      # G completions
            token_ids = comp.token_ids
            # vLLM gives a list of {token_id: Logprob} dicts, one per position.
            behavior_logprobs = [
                lp_dict[tid].logprob
                for tid, lp_dict in zip(token_ids, comp.logprobs)
            ]
            batch.append({
                "prompt": req.prompt,
                "response_ids": token_ids,
                "behavior_logprobs": behavior_logprobs,   # log pi_behavior(o_t|.)
            })
    return batch


# =============================================================================
# Block #1 (~line 145): weight swap into the engine. Fragment per the harness's
# heuristics, but block #6's generator_worker calls it directly, so it is
# defined verbatim as glue and exercised too.
# =============================================================================

def sync_weights_to_engine(engine, state_dict):
    """
    Push freshly-trained weights into the live vLLM engine without a restart.
    `state_dict` maps param name -> bf16 tensor (e.g. policy.state_dict()).
    In colocated setups this is an in-process update of the model's parameters;
    veRL/OpenRLHF wrap this so the named tensors map correctly onto vLLM's
    (possibly differently-sharded) internal layout.
    """
    # vLLM exposes a collective_rpc / load_weights path on its workers:
    engine.llm_engine.model_executor.collective_rpc(
        "load_weights", args=(list(state_dict.items()),)
    )
    # After this, the next engine.generate() samples from the NEW policy.


# =============================================================================
# Block #2 (~line 228): experience batch construction. Verbatim from the
# chapter.
# =============================================================================

def build_experience_batch(rollouts, pad_id, device):
    """
    rollouts: list of dicts with 'prompt_ids' (Lp,), 'response_ids' (Lg,),
              'behavior_logprobs' (Lg,), and 'reward' (scalar).
    Returns padded tensors + a response mask aligned to the next-token targets.
    """
    seqs, masks, beh_lp, rewards = [], [], [], []
    for r in rollouts:
        ids = torch.cat([r["prompt_ids"], r["response_ids"]])         # (Lp+Lg,)
        m = torch.zeros_like(ids, dtype=torch.float)
        m[len(r["prompt_ids"]):] = 1.0                                # mask = 1 on response
        seqs.append(ids); masks.append(m)
        # behavior logprobs are defined only on response tokens; left-pad with 0s
        beh = torch.zeros_like(ids, dtype=torch.float)
        beh[len(r["prompt_ids"]):] = torch.tensor(r["behavior_logprobs"])
        beh_lp.append(beh); rewards.append(r["reward"])

    maxlen = max(s.numel() for s in seqs)
    def pad(x, val):  # right-pad to maxlen
        return torch.stack([F.pad(t, (0, maxlen - t.numel()), value=val) for t in x])

    input_ids      = pad(seqs,  pad_id).to(device)        # (B, L)
    response_mask  = pad(masks, 0.0).to(device)           # (B, L)
    behavior_lp    = pad(beh_lp, 0.0).to(device)          # (B, L)
    rewards        = torch.tensor(rewards, device=device) # (B,)

    # IMPORTANT: logprobs/loss are computed on NEXT-token prediction, so when we
    # gather log pi(o_t | o_<t) from logits[:, :-1], the aligned mask/targets are
    # shifted by one. Keep the response_mask and shift it where you compute loss.
    return input_ids, response_mask, behavior_lp, rewards


# =============================================================================
# Block #6 (~line 385): skeleton of an async RL loop. Verbatim from the
# chapter, WITH the `local_v` bug fixed (see module docstring).
# =============================================================================

import queue, copy

rollout_q = queue.Queue(maxsize=256)     # bounded: backpressure if trainer lags
weight_version = {"v": 0}
shared_weights = {"sd": None, "lock": threading.Lock()}


def generator_worker(engine, prompt_stream):
    while True:
        # 1) refresh local weights if the trainer pushed a newer version
        with shared_weights["lock"]:
            local_v = weight_version["v"]                     # BUGFIX: always read fresh
            if shared_weights["sd"] is not None:
                sync_weights_to_engine(engine, shared_weights["sd"])
        # 2) generate a group, tag it with the weight version that produced it
        prompts = next(prompt_stream)
        for r in rollout(prompts):           # uses the engine (continuous-batched)
            r["gen_version"] = local_v
            rollout_q.put(r)                 # blocks if queue full -> backpressure


def trainer_loop(policy, opt, ref, max_staleness=4, batch_size=512):
    step = 0
    while True:
        # pull a batch of FRESH-ENOUGH rollouts
        batch = []
        while len(batch) < batch_size:
            r = rollout_q.get()
            if step - r["gen_version"] <= max_staleness:   # drop stale rollouts
                batch.append(r)
        # standard experience-prep + minibatch update (Phases 3-4)
        input_ids, resp_mask, beh_lp, rewards = build_experience_batch(batch, ...)
        # ... recompute old_lp/ref_lp, compute advantages, minibatch_update_loop ...
        step += 1
        # push new weights to generators every few steps (Phase 5)
        if step % 2 == 0:
            with shared_weights["lock"]:
                shared_weights["sd"] = copy.deepcopy(policy.state_dict())
                weight_version["v"] = step


# =============================================================================
# Execute the blocks.
# =============================================================================

def test_block0_rollout():
    prompts = ["what is 2+2?", "capital of France?"]
    batch = rollout(prompts)
    assert len(batch) == len(prompts) * sampling.n  # 2 prompts x G=8
    for item in batch:
        assert item["prompt"] in prompts
        assert len(item["response_ids"]) == 3
        assert len(item["behavior_logprobs"]) == 3
        assert all(isinstance(x, float) for x in item["behavior_logprobs"])
    print(f"[block #0] rollout() -> {len(batch)} completions "
          f"({len(prompts)} prompts x G={sampling.n}) -- OK")
    return batch


def test_block1_sync_weights_glue():
    sd = {"layer.weight": torch.zeros(2, 2)}
    sync_weights_to_engine(engine, sd)
    calls = engine.llm_engine.model_executor.collective_rpc.calls
    assert calls, "expected sync_weights_to_engine to invoke collective_rpc"
    name, args = calls[-1]
    assert name == "load_weights"
    assert args[0][0][0] == "layer.weight"
    print("[block #1 glue] sync_weights_to_engine() -> collective_rpc('load_weights', ...) -- OK")


def test_block2_build_experience_batch():
    rollouts = [
        {
            "prompt_ids": torch.tensor([1, 2, 3]),
            "response_ids": torch.tensor([4, 5]),
            "behavior_logprobs": [-0.5, -0.7],
            "reward": 1.0,
        },
        {
            # shorter total length -> exercises the right-padding path
            "prompt_ids": torch.tensor([1, 2]),
            "response_ids": torch.tensor([6, 7]),
            "behavior_logprobs": [-0.2, -0.3],
            "reward": 0.0,
        },
    ]
    input_ids, response_mask, behavior_lp, rewards = build_experience_batch(
        rollouts, pad_id=0, device=torch.device("cpu"))

    assert input_ids.shape == (2, 5)
    assert response_mask.shape == (2, 5)
    assert behavior_lp.shape == (2, 5)
    assert rewards.shape == (2,)

    assert torch.equal(input_ids[0], torch.tensor([1, 2, 3, 4, 5]))
    assert torch.equal(response_mask[0], torch.tensor([0., 0., 0., 1., 1.]))
    assert torch.allclose(behavior_lp[0], torch.tensor([0., 0., 0., -0.5, -0.7]))

    # row 1 has total length 4 < maxlen 5 -> right-padded with pad_id=0
    assert torch.equal(input_ids[1], torch.tensor([1, 2, 6, 7, 0]))
    assert torch.equal(response_mask[1], torch.tensor([0., 0., 1., 1., 0.]))
    assert torch.allclose(behavior_lp[1], torch.tensor([0., 0., -0.2, -0.3, 0.]))

    assert torch.allclose(rewards, torch.tensor([1.0, 0.0]))
    print(f"[block #2] build_experience_batch() -> input_ids{tuple(input_ids.shape)} "
          "with correct masking and right-padding -- OK")


def test_block6_async_skeleton():
    # A dedicated fake engine so this test doesn't share collective_rpc call
    # history with test_block1_sync_weights_glue.
    local_engine = LLM(
        model="Qwen/Qwen2.5-7B",
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        enable_prefix_caching=True,
        max_model_len=4096,
    )

    prompt_stream = itertools.cycle([["p1", "p2"]])
    worker = threading.Thread(
        target=generator_worker, args=(local_engine, prompt_stream), daemon=True)
    worker.start()

    # Drain the very first rollout. shared_weights["sd"] is still None here, so
    # this exercises exactly the path that used to raise UnboundLocalError on
    # `local_v` before the fix.
    first = rollout_q.get(timeout=10)
    assert first["gen_version"] == 0
    assert "response_ids" in first and "behavior_logprobs" in first
    assert not local_engine.llm_engine.model_executor.collective_rpc.calls, (
        "no weight sync should have happened yet (shared_weights['sd'] is still None)"
    )

    # Simulate what trainer_loop does every couple of steps: push a new weight
    # version, and confirm the worker (a) picks it up in `gen_version` and
    # (b) calls sync_weights_to_engine (block #1) before tagging the rollout.
    with shared_weights["lock"]:
        shared_weights["sd"] = {"layer.weight": torch.zeros(2)}
        weight_version["v"] = 7

    deadline = time.time() + 10
    seen_v7 = False
    while time.time() < deadline and not seen_v7:
        item = rollout_q.get(timeout=10)
        if item["gen_version"] == 7:
            seen_v7 = True
    assert seen_v7, "generator_worker never picked up the pushed weight version"
    assert local_engine.llm_engine.model_executor.collective_rpc.calls, (
        "sync_weights_to_engine (block #1) was not invoked after a weight push"
    )

    print("[block #6] generator_worker() tagged rollouts with gen_version 0 -> 7 "
          "and invoked sync_weights_to_engine on the version bump -- OK")

    # trainer_loop (also block #6) is defined and syntax/name-checked by the
    # import above, but is intentionally NOT called to completion: its body
    # calls `build_experience_batch(batch, ...)` where `...` is a bare
    # Ellipsis standing in for the chapter's elided (pad_id, device)
    # arguments -- confirmed non-literal pseudocode, not a bug. Calling it
    # verbatim raises TypeError (missing 'device'), which we assert narrowly
    # here to document that this is deliberate, not an oversight.
    class _DummyPolicy:
        def state_dict(self):
            return {}

    rollout_q.put({"gen_version": 0, "response_ids": [1], "behavior_logprobs": [0.0]})
    try:
        trainer_loop(_DummyPolicy(), opt=None, ref=None, max_staleness=100, batch_size=1)
        raise AssertionError("expected trainer_loop's elided build_experience_batch(...) call to fail")
    except TypeError as e:
        print(f"[block #6] trainer_loop() confirmed non-literal (elided-args) pseudocode: {e}")


if __name__ == "__main__":
    test_block0_rollout()
    test_block1_sync_weights_glue()
    test_block2_build_experience_batch()
    test_block6_async_skeleton()
    print("\nAll runnable blocks for 06-rl-infra/02-generation-training-loop.md executed OK.")
