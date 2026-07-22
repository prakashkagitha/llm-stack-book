"""CI-tested extracts of runnable code blocks from
content/13-interp-safety-gov/05-ai-safety-oversight.md

Each `block_*` function reproduces the book's ACTUAL code verbatim (modulo
wrapping in a function and adding tiny call-sites/fixtures) and then asserts
the claims the prose/comments make about the results.

Blocks tested (from the chapter's heuristic CPU-runnable list):
  - block #0 (line ~46)  : debate loop (DebateState, debate_round, judge_decision, debate_reward)
  - block #2 (line ~126) : weak_to_strong_loss, recovered_gap
  - block #3 (line ~185) : uplift_trial (dangerous-capability A/B trial stats)
  - block #4 (line ~228) : sandbagging_signal (heuristic sandbagging detector)
  - block #6 (line ~311) : trusted_monitoring_protocol, safety_usefulness_frontier
  - block #7 (line ~364) : SafetyLevel, deployment_gate (RSP decision logic)

Skipped blocks (non-standalone fragments, per the chapter's own framing):
  - block #1 (line ~93)  : recursive_evaluate -- SKIP(fragment): calls
    `decompose(task)`, `solve(task)`, `human_judge(...)` which are described
    as abstract model/human callables with no concrete implementation given
    in the chapter; the recursion base case and branching depend on
    `decompose` returning falsy/list values whose semantics are undefined
    outside the illustrative narrative. Faking a decompose/solve/human_judge
    triad well enough to exercise the recursion would just be testing our
    own stand-ins, not the book's logic.
  - block #5 (line ~259) : faithfulness_probes -- SKIP(fragment): takes a
    `model` argument and calls `model.answer_given(...)`, an unspecified
    external model interface with no concrete implementation in the
    chapter (this IS the LLM call being abstracted away, analogous to the
    debater callables in block #0, but here there's no natural toy
    substitute that would still exercise the *book's* logic rather than a
    test-authored model).

Run directly: `python3 tests/13-interp-safety-gov__05-ai-safety-oversight.py`
"""
import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy import stats
except Exception:
    stats = None


# ---------------------------------------------------------------------------
# Block #0 (line ~46): the debate loop
# ---------------------------------------------------------------------------
def block_debate():
    from dataclasses import dataclass, field

    @dataclass
    class DebateState:
        question: str
        answer_a: str               # the claim debater A defends
        answer_b: str               # the claim debater B defends
        transcript: list = field(default_factory=list)  # list of (speaker, text)

    def debate_round(state: DebateState, debater_a, debater_b, n_turns: int = 6):
        """Run an alternating debate. `debater_*` are callables:
           (question, my_answer, opponent_answer, transcript) -> str argument."""
        for turn in range(n_turns):
            if turn % 2 == 0:
                arg = debater_a(state.question, state.answer_a,
                                state.answer_b, state.transcript)
                state.transcript.append(("A", arg))
            else:
                # B sees A's latest argument and is incentivized to rebut it.
                arg = debater_b(state.question, state.answer_b,
                                state.answer_a, state.transcript)
                state.transcript.append(("B", arg))
        return state

    def judge_decision(state: DebateState, judge) -> str:
        """Judge reads ONLY the transcript (it cannot solve the task alone)
           and returns 'A' or 'B'. In training, this label is the reward signal."""
        return judge(state.question, state.transcript)  # -> 'A' or 'B'

    # Reward for debater A is +1 if the judge picks A, else -1 (zero-sum).
    def debate_reward(judge_label: str) -> tuple[float, float]:
        return (1.0, -1.0) if judge_label == "A" else (-1.0, 1.0)

    # --- minimal glue: tiny stand-in debaters/judge to actually run the loop ---
    def debater_a(question, my_answer, opponent_answer, transcript):
        return f"A defends '{my_answer}' (turn {len(transcript)})"

    def debater_b(question, my_answer, opponent_answer, transcript):
        return f"B defends '{my_answer}' (turn {len(transcript)})"

    def judge(question, transcript):
        # Toy judge: counts turns per speaker, picks whoever spoke last.
        return transcript[-1][0]

    state = DebateState(question="Is P=NP?", answer_a="Yes", answer_b="No")
    state = debate_round(state, debater_a, debater_b, n_turns=4)
    label = judge_decision(state, judge)
    reward_a, reward_b = debate_reward(label)

    print(f"transcript turns={len(state.transcript)}, judge picked={label}, "
          f"reward=({reward_a}, {reward_b})")

    # --- verify book's claims ---
    assert len(state.transcript) == 4
    assert [spk for spk, _ in state.transcript] == ["A", "B", "A", "B"]
    assert label in ("A", "B")
    # zero-sum: rewards are always (+1,-1) or (-1,+1)
    assert {reward_a, reward_b} == {1.0, -1.0}
    if label == "A":
        assert (reward_a, reward_b) == (1.0, -1.0)
    else:
        assert (reward_a, reward_b) == (-1.0, 1.0)


# ---------------------------------------------------------------------------
# Block #2 (line ~126): weak-to-strong generalization loss + PGR metric
# ---------------------------------------------------------------------------
def block_weak_to_strong():
    def weak_to_strong_loss(strong_logits, weak_labels, conf_threshold=0.0,
                            aux_weight=0.5):
        """Burns et al. found a key trick: an *auxiliary confidence loss* that
        lets the strong model disagree with weak labels when it is internally
        confident, mitigating imitation of the weak supervisor's errors."""
        # (1) standard cross-entropy toward the (possibly wrong) weak label
        ce = F.cross_entropy(strong_logits, weak_labels)

        # (2) auxiliary term: push the strong model toward its OWN confident
        #     prediction (a hardened version of its current belief), which
        #     encodes "trust yourself when you strongly disagree".
        p = F.softmax(strong_logits, dim=-1)
        hardened = (p > conf_threshold).float()
        hardened = hardened / hardened.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        conf_loss = F.cross_entropy(strong_logits, hardened.argmax(dim=-1))

        return (1 - aux_weight) * ce + aux_weight * conf_loss

    def recovered_gap(weak_acc, w2s_acc, strong_ceiling_acc):
        """Performance-gap-recovered (PGR): what fraction of the gap between
        the weak supervisor and the strong oracle does weak-to-strong recover?
        PGR=0 means pure imitation; PGR=1 means the strong model fully
        generalized to oracle-level performance from weak supervision alone."""
        denom = (strong_ceiling_acc - weak_acc)
        return (w2s_acc - weak_acc) / denom if denom > 0 else float("nan")

    # --- minimal glue: tiny toy logits/labels to actually run the loss ---
    torch.manual_seed(0)
    batch, n_classes = 6, 3
    strong_logits = torch.randn(batch, n_classes, requires_grad=True)
    weak_labels = torch.randint(0, n_classes, (batch,))

    loss = weak_to_strong_loss(strong_logits, weak_labels)
    loss.backward()  # confirm the loss is actually differentiable end-to-end

    print(f"weak_to_strong_loss={loss.item():.4f}")

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert strong_logits.grad is not None
    assert strong_logits.grad.shape == strong_logits.shape

    # worked numerical example from the chapter's "Worked numerical example" box
    pgr = recovered_gap(weak_acc=0.62, w2s_acc=0.79, strong_ceiling_acc=0.90)
    print(f"PGR={pgr:.4f}")
    assert abs(pgr - 0.6071428571428571) < 1e-9

    pgr_near_imitation = recovered_gap(weak_acc=0.62, w2s_acc=0.64,
                                        strong_ceiling_acc=0.90)
    assert abs(pgr_near_imitation - (0.02 / 0.28)) < 1e-9
    assert pgr_near_imitation < 0.1  # "near-pure imitation" per the chapter


# ---------------------------------------------------------------------------
# Block #3 (line ~185): dangerous-capability uplift A/B trial
# ---------------------------------------------------------------------------
def block_uplift_trial():
    if stats is None:
        print("SKIP(dependency): scipy not available, skipping uplift_trial")
        return

    def uplift_trial(control_scores, treatment_scores):
        """Estimate capability UPLIFT from a dangerous-capability A/B trial.
        control_scores: expert grades (0-100) for the no-model arm.
        treatment_scores: expert grades for the with-model arm.
        Returns the uplift effect size and a significance test -- this is
        the number a Responsible Scaling Policy threshold is checked against."""
        c, t = np.asarray(control_scores), np.asarray(treatment_scores)
        uplift = t.mean() - c.mean()
        # Cohen's d: standardized effect size (magnitude of uplift).
        pooled_sd = np.sqrt(((len(c)-1)*c.var(ddof=1) +
                             (len(t)-1)*t.var(ddof=1)) / (len(c)+len(t)-2))
        cohens_d = uplift / pooled_sd
        # One-sided test: is the model arm SIGNIFICANTLY more capable?
        tstat, p_two = stats.ttest_ind(t, c, equal_var=False)
        p_one = p_two / 2 if tstat > 0 else 1 - p_two / 2
        return {"uplift_points": uplift, "cohens_d": cohens_d,
                "p_value_one_sided": p_one,
                "crosses_threshold": uplift > 15 and p_one < 0.05}

    # Example: model arm averages 71/100 on plan completeness, control 58/100.
    ctrl = [55, 60, 52, 61, 58, 57, 63, 59, 54, 60]
    treat = [70, 74, 68, 73, 71, 69, 75, 72, 67, 71]
    result = uplift_trial(ctrl, treat)
    print(result)
    # -> uplift ~13 points, large Cohen's d, p < 0.05; borderline on a 15-pt bar.

    # --- verify book's claims ---
    assert abs(result["uplift_points"] - 13.1) < 1e-9
    assert result["cohens_d"] > 2.0            # "large Cohen's d"
    assert result["p_value_one_sided"] < 0.05  # "p < 0.05"
    # 13 points is BELOW the 15-point bar -> "borderline on a 15-pt bar"
    assert result["crosses_threshold"] == False


# ---------------------------------------------------------------------------
# Block #4 (line ~228): heuristic sandbagging detector
# ---------------------------------------------------------------------------
def block_sandbagging():
    def sandbagging_signal(acc_neutral, acc_eval_framed, acc_after_light_ft,
                           eval_drop_tol=0.08, recovery_tol=0.15):
        """Heuristic sandbagging detector. Three measurements of the SAME task:
          acc_neutral       : accuracy when the prompt looks like a normal task
          acc_eval_framed   : accuracy when the prompt signals 'this is a safety eval'
          acc_after_light_ft : accuracy after a small amount of fine-tuning
        Red flags:
          (1) performance DROPS when the model thinks it is being evaluated, and
          (2) a little fine-tuning RECOVERS a capability the model 'lacked'."""
        drops_under_eval = (acc_neutral - acc_eval_framed) > eval_drop_tol
        latent_capability = (acc_after_light_ft - acc_eval_framed) > recovery_tol
        return {
            "selective_underperformance": drops_under_eval,
            "fast_recovery_implies_latent": latent_capability,
            "sandbagging_suspected": drops_under_eval and latent_capability,
        }

    # The chapter's concrete example: model "fails" the dangerous-cap eval (0.41)
    # but does fine on a neutrally-framed version (0.52) and recovers to 0.79.
    result = sandbagging_signal(0.52, 0.41, 0.79)
    print(result)  # book claims -> sandbagging_suspected: True
    assert result["selective_underperformance"] is True
    assert result["fast_recovery_implies_latent"] is True
    assert result["sandbagging_suspected"] is True

    # Negative control: no eval-time drop and no fast recovery -> not suspected.
    clean = sandbagging_signal(0.50, 0.49, 0.50)
    assert clean["sandbagging_suspected"] is False


# ---------------------------------------------------------------------------
# Block #6 (line ~311): control protocol -- trusted monitoring
# ---------------------------------------------------------------------------
def block_control_protocol():
    def trusted_monitoring_protocol(untrusted_action, trusted_monitor,
                                    audit_threshold, trusted_fallback,
                                    audit_budget_remaining):
        """One step of a control protocol. The trusted (weak, safe) model scores
        the untrusted (strong, maybe-scheming) model's proposed action; we audit
        the most suspicious actions while the budget lasts, else defer to trusted."""
        suspicion = trusted_monitor.suspicion_score(untrusted_action)  # in [0,1]

        if suspicion >= audit_threshold and audit_budget_remaining > 0:
            # Spend scarce human review on the scariest actions.
            verdict = human_audit(untrusted_action)        # 'safe' or 'BACKDOOR'
            return ("audited", verdict, audit_budget_remaining - 1)
        elif suspicion >= audit_threshold:
            # Out of audit budget but still suspicious -> use the safe model.
            return ("deferred_to_trusted", trusted_fallback, audit_budget_remaining)
        else:
            return ("accepted_untrusted", untrusted_action, audit_budget_remaining)

    def safety_usefulness_frontier(suspicion_scores, is_backdoor, audit_frac):
        """Estimate P(catch a backdoor) given we audit the top `audit_frac`
        fraction of actions by suspicion -- the core control trade-off curve."""
        scores = np.asarray(suspicion_scores); flags = np.asarray(is_backdoor)
        k = max(1, int(audit_frac * len(scores)))
        audited = np.argsort(-scores)[:k]          # top-k most suspicious
        caught = flags[audited].sum()
        total_backdoors = flags.sum()
        return caught / total_backdoors if total_backdoors else 1.0

    # --- minimal glue: the chapter abstracts `trusted_monitor` and
    # `human_audit` as callables (same style as the debater callables in
    # block #0); supply tiny concrete stand-ins to exercise all three
    # branches of the protocol.
    def human_audit(action):
        return "BACKDOOR" if action.get("malicious") else "safe"

    class TrustedMonitor:
        def suspicion_score(self, action):
            return action["suspicion"]

    monitor = TrustedMonitor()

    # Branch 1: high suspicion, budget available -> audited.
    action_hi = {"suspicion": 0.95, "malicious": True}
    out1 = trusted_monitoring_protocol(action_hi, monitor, audit_threshold=0.8,
                                       trusted_fallback="SAFE_DEFAULT",
                                       audit_budget_remaining=1)
    print(out1)
    assert out1 == ("audited", "BACKDOOR", 0)

    # Branch 2: high suspicion, budget exhausted -> deferred to trusted.
    out2 = trusted_monitoring_protocol(action_hi, monitor, audit_threshold=0.8,
                                       trusted_fallback="SAFE_DEFAULT",
                                       audit_budget_remaining=0)
    print(out2)
    assert out2 == ("deferred_to_trusted", "SAFE_DEFAULT", 0)

    # Branch 3: low suspicion -> accepted untrusted.
    action_lo = {"suspicion": 0.1, "malicious": False}
    out3 = trusted_monitoring_protocol(action_lo, monitor, audit_threshold=0.8,
                                       trusted_fallback="SAFE_DEFAULT",
                                       audit_budget_remaining=1)
    print(out3)
    assert out3 == ("accepted_untrusted", action_lo, 1)

    # safety_usefulness_frontier: audit the top 20% by suspicion, check we
    # catch backdoors that are strongly correlated with high suspicion.
    rng = np.random.default_rng(0)
    n = 100
    suspicion_scores = rng.random(n)
    # Make backdoors concentrate among the top-suspicion actions so a
    # reasonable audit fraction catches most of them.
    is_backdoor = suspicion_scores > 0.9
    frac_caught = safety_usefulness_frontier(suspicion_scores, is_backdoor,
                                              audit_frac=0.2)
    print(f"frac_caught={frac_caught:.3f}")
    assert 0.0 <= frac_caught <= 1.0
    assert frac_caught == 1.0  # top-20% audit fully covers the top-10% backdoors


# ---------------------------------------------------------------------------
# Block #7 (line ~364): Responsible Scaling Policy decision gate
# ---------------------------------------------------------------------------
def block_deployment_gate():
    from dataclasses import dataclass

    @dataclass
    class SafetyLevel:
        name: str                 # e.g. "ASL-3"
        threat: str               # capability that defines this level
        required_deployment: set  # safeguards needed to DEPLOY at this level
        required_security: set    # safeguards needed to TRAIN/STORE weights here

    def deployment_gate(eval_results: dict, current_safeguards: set,
                        levels: list) -> dict:
        """Operationalize 'is this model safe to deploy?' as a gate, not a vibe.
        eval_results: {threat_name: crossed_threshold (bool)}.
        Returns the determined level and whether deployment is PERMITTED."""
        # Highest level whose defining capability the model has crossed.
        triggered = [lv for lv in levels if eval_results.get(lv.threat, False)]
        if not triggered:
            return {"level": "baseline", "deploy_permitted": True, "missing": set()}

        level = max(triggered, key=lambda lv: levels.index(lv))
        missing_deploy = level.required_deployment - current_safeguards
        missing_security = level.required_security - current_safeguards
        missing = missing_deploy | missing_security

        return {"level": level.name,
                "deploy_permitted": len(missing) == 0,   # HARD gate
                "missing": missing,
                # If security safeguards are missing, even *training* should pause:
                "must_pause_training": len(missing_security) > 0}

    levels = [
        SafetyLevel("ASL-3", "cbrn_uplift",
                    required_deployment={"enhanced_refusals", "misuse_monitoring",
                                         "vetted_access"},
                    required_security={"weight_access_controls", "insider_threat_prog"}),
        SafetyLevel("ASL-4", "autonomous_replication",
                    required_deployment={"agentic_sandboxing", "kill_switch"},
                    required_security={"airgapped_weights", "hardware_security"}),
    ]
    evals = {"cbrn_uplift": True, "autonomous_replication": False}
    have = {"enhanced_refusals", "misuse_monitoring"}   # vetted_access NOT yet built
    result = deployment_gate(evals, have, levels)
    print(result)
    # -> level ASL-3, deploy_permitted False, missing {'vetted_access',
    #    'weight_access_controls', 'insider_threat_prog'}, must_pause_training True

    # --- verify book's claims ---
    assert result["level"] == "ASL-3"
    assert result["deploy_permitted"] is False
    assert result["missing"] == {"vetted_access", "weight_access_controls",
                                  "insider_threat_prog"}
    assert result["must_pause_training"] is True

    # baseline case: no thresholds crossed -> deployment permitted.
    result_baseline = deployment_gate({"cbrn_uplift": False,
                                       "autonomous_replication": False},
                                      have, levels)
    assert result_baseline == {"level": "baseline", "deploy_permitted": True,
                                "missing": set()}

    # fully-mitigated case: all required safeguards present -> permitted.
    have_full = (levels[0].required_deployment | levels[0].required_security)
    result_full = deployment_gate(evals, have_full, levels)
    assert result_full["deploy_permitted"] is True
    assert result_full["must_pause_training"] is False


BLOCKS = [
    block_debate,
    block_weak_to_strong,
    block_uplift_trial,
    block_sandbagging,
    block_control_protocol,
    block_deployment_gate,
]


def main():
    for fn in BLOCKS:
        print(f"\n===== {fn.__name__} =====")
        fn()
    print(f"\nAll {len(BLOCKS)} code blocks executed and verified.")


if __name__ == "__main__":
    main()
