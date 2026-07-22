"""
Runs the CPU-runnable code from content/03-pretraining/06-distributed-model-parallel.md.

Block inventory (from the chapter scan):
  #0 (needs-gpu):    tensor-parallel ColumnParallelLinear/RowParallelLinear (dist.all_reduce,
                      real multi-GPU NCCL group) -- SKIP(needs-gpu)
  #1 (needs-gpu):    Megatron TP transformer block wiring with real process groups -- SKIP(needs-gpu)
  #2 (needs-gpu):    VocabParallelEmbedding with masked all-reduce over a real TP group -- SKIP(needs-gpu)
  #3 (CPU-runnable): run_1f1b() -- the minimal 1F1B pipeline-parallel driver. Pure Python
                      control flow (no torch/dist needed) -- TESTED below, verbatim.
  #4 (non-python):   ```text``` ASCII diagram of interleaved virtual stages -- SKIP(non-python)
  #5 (needs-gpu):    vocab_parallel_cross_entropy() -- dist.all_reduce over a real TP group -- SKIP(needs-gpu)
  #6 (needs-gpu):    (further TP/PP wiring in the chapter) -- SKIP(needs-gpu)
  #7 (fragment):     non-standalone code fragment -- SKIP(fragment)

Only block #3 is exercised here. It is a *driver* whose behavior only becomes visible when
wired to real send/recv/forward/backward callables, so we wire it to a tiny 2-stage pipeline
simulated in-process with two threads and Queue-based point-to-point channels -- exactly the
"single-stage view ... in reality each rank runs this with send/recv to its neighbors" setup
the chapter's comment describes. This exercises the real warmup/steady-state/cooldown control
flow of 1F1B (including the correct number of forwards-before-first-backward per stage) rather
than stubbing it out.

`run_1f1b` references `next_input` and `loss_grad` as free (module-global) names -- only the
true first stage calls `next_input` and only the true last stage calls `loss_grad`. Since we
run both simulated ranks as threads sharing one process, we can't just assign these as shared
module globals (that would race between the two concurrently-running "ranks"). Instead we bind
the *same, verbatim* function code object to two independent globals dicts, one per simulated
rank -- exactly mirroring how two real ranks are two separate processes each with their own
global namespace.
"""

import queue
import threading
import types


# ============================================================================
# Book block #3 (verbatim, line ~273-306 of the chapter)
# ============================================================================
def run_1f1b(stage, p, num_micro, fwd_step, bwd_step, recv_act, send_act,
             recv_grad, send_grad):
    warmup = p - stage - 1                 # how many forwards before first backward
    warmup = min(warmup, num_micro)
    steady = num_micro - warmup
    act_queue = []                         # activations awaiting their backward

    # ---- warmup: only forwards, prime the pipe ----
    for _ in range(warmup):
        x = recv_act() if stage > 0 else next_input()
        y, act = fwd_step(x)               # act = saved tensors for backward
        send_act(y) if stage < p - 1 else None
        act_queue.append(act)

    # ---- steady state: 1 forward then 1 backward, bounded memory ----
    for i in range(steady):
        x = recv_act() if stage > 0 else next_input()
        y, act = fwd_step(x)
        send_act(y) if stage < p - 1 else None
        act_queue.append(act)
        # immediately do a backward for the OLDEST in-flight microbatch
        g = recv_grad() if stage < p - 1 else loss_grad()
        gx = bwd_step(act_queue.pop(0), g)
        send_grad(gx) if stage > 0 else None

    # ---- cooldown: drain remaining backwards ----
    for _ in range(warmup):
        g = recv_grad() if stage < p - 1 else loss_grad()
        gx = bwd_step(act_queue.pop(0), g)
        send_grad(gx) if stage > 0 else None


# ============================================================================
# Glue: tiny toy "model" + a 2-thread, 2-stage pipeline simulation that wires
# real (blocking, Queue-based) send/recv channels between the two stages.
# ============================================================================

# scalar activations, y = 2*x per stage, so the local backward derivative is a
# fixed constant -- easy to check the 1F1B schedule actually produced the
# right number of forward/backward calls with the right values.
LOCAL_SLOPE = 2.0


def fwd_step(x):
    y = x * LOCAL_SLOPE
    act = x            # saved input, needed by the local backward
    return y, act


def bwd_step(act, g):
    # d(loss)/d(x) = d(loss)/d(y) * d(y)/d(x) = g * LOCAL_SLOPE
    return g * LOCAL_SLOPE


def _make_rank_fn(next_input_fn, loss_grad_fn):
    """
    Bind the verbatim run_1f1b code object to a fresh globals dict that
    supplies this rank's own `next_input`/`loss_grad` free-variable
    implementations -- mirrors two separate rank processes, each with its
    own global namespace, without mutating any shared state.
    """
    rank_globals = dict(globals())
    rank_globals["next_input"] = next_input_fn
    rank_globals["loss_grad"] = loss_grad_fn
    return types.FunctionType(run_1f1b.__code__, rank_globals, "run_1f1b")


def _run_two_stage_1f1b(num_micro):
    p = 2
    act_chan = queue.Queue()   # stage 0 -> stage 1 (activations)
    grad_chan = queue.Queue()  # stage 1 -> stage 0 (gradients)

    inputs = [float(i + 1) for i in range(num_micro)]
    input_iter = iter(inputs)

    grads_stage1_to_stage0 = []  # every gx stage 1 sends back, in completion order
    lock = threading.Lock()

    def stage0_next_input():
        return next(input_iter)

    def stage0_loss_grad():
        raise AssertionError("stage 0 (p=2) is not the last stage; loss_grad() must not be called")

    def stage1_next_input():
        raise AssertionError("stage 1 (p=2) is not the first stage; next_input() must not be called")

    def stage1_loss_grad():
        return 1.0  # toy identity loss: dL/dy_last = 1.0 for every microbatch

    def stage0_recv_act():
        raise AssertionError("stage 0 has no left neighbor; recv_act() must not be called")

    def stage0_send_act(y):
        act_chan.put(y)

    def stage0_recv_grad():
        return grad_chan.get()

    def stage0_send_grad(gx):
        raise AssertionError("stage 0 has no left neighbor; send_grad() must not be called (stage>0 is False)")

    def stage1_recv_act():
        return act_chan.get()

    def stage1_send_act(y):
        raise AssertionError("stage 1 has no right neighbor; send_act() must not be called (stage<p-1 is False)")

    def stage1_recv_grad():
        raise AssertionError("stage 1 has no right neighbor; recv_grad() must not be called (stage<p-1 is False)")

    def stage1_send_grad(gx):
        with lock:
            grads_stage1_to_stage0.append(gx)
        grad_chan.put(gx)

    run_1f1b_stage0 = _make_rank_fn(stage0_next_input, stage0_loss_grad)
    run_1f1b_stage1 = _make_rank_fn(stage1_next_input, stage1_loss_grad)

    errors = []

    def worker(fn, kwargs):
        try:
            fn(**kwargs)
        except Exception as e:  # surface thread exceptions to the main thread
            errors.append(e)

    t0 = threading.Thread(target=worker, args=(run_1f1b_stage0, dict(
        stage=0, p=p, num_micro=num_micro, fwd_step=fwd_step, bwd_step=bwd_step,
        recv_act=stage0_recv_act, send_act=stage0_send_act,
        recv_grad=stage0_recv_grad, send_grad=stage0_send_grad,
    )))
    t1 = threading.Thread(target=worker, args=(run_1f1b_stage1, dict(
        stage=1, p=p, num_micro=num_micro, fwd_step=fwd_step, bwd_step=bwd_step,
        recv_act=stage1_recv_act, send_act=stage1_send_act,
        recv_grad=stage1_recv_grad, send_grad=stage1_send_grad,
    )))

    t0.start()
    t1.start()
    t0.join(timeout=10)
    t1.join(timeout=10)
    assert not t0.is_alive() and not t1.is_alive(), "1F1B pipeline threads deadlocked"
    if errors:
        raise errors[0]

    return inputs, grads_stage1_to_stage0


def main():
    num_micro = 6
    inputs, grads_stage1_to_stage0 = _run_two_stage_1f1b(num_micro)

    # Toy model: y = 2*x at stage0, y2 = 2*y at stage1, loss_grad = 1 at stage1 -> at
    # stage1's own input x1 (= stage0's output), dL/dx1 = 1 * 2 = 2 for every microbatch.
    assert len(grads_stage1_to_stage0) == num_micro, (
        f"expected {num_micro} grads sent stage1->stage0, got {len(grads_stage1_to_stage0)}"
    )
    for g in grads_stage1_to_stage0:
        assert g == 2.0, f"expected local backward grad 2.0 (dy/dx of y=2x), got {g}"

    print(f"[block #3] run_1f1b: 2-stage pipeline processed {num_micro} microbatches "
          f"({inputs}), 1F1B schedule (warmup/steady/cooldown) completed without "
          f"deadlock, all {num_micro} stage1->stage0 gradients correct (=2.0).")

    # SKIP(needs-gpu): block #0 -- ColumnParallelLinear/RowParallelLinear over a real NCCL TP group
    # SKIP(needs-gpu): block #1 -- Megatron TP transformer block wired to real process groups
    # SKIP(needs-gpu): block #2 -- VocabParallelEmbedding masked all-reduce over a real TP group
    # SKIP(non-python): block #4 -- ASCII diagram of interleaved virtual pipeline stages
    # SKIP(needs-gpu): block #5 -- vocab_parallel_cross_entropy(), dist.all_reduce over a real TP group
    # SKIP(needs-gpu): block #6 -- further TP/PP wiring requiring real process groups
    # SKIP(fragment): block #7 -- non-standalone code fragment

    print("OK: 03-pretraining/06-distributed-model-parallel.md CPU-runnable block passed.")


if __name__ == "__main__":
    main()
