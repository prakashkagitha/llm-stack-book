# 3.12 Checkpointing, Fault Tolerance & Long-Running Jobs

Training a frontier LLM is one of the longest-running computational jobs in existence. Runs lasting weeks or months across thousands of GPUs are now routine. At that scale, hardware failure is not an exception — it is a certainty. This chapter is a systems-engineering deep dive into how you keep those jobs running, how you save state correctly, and how you resume training without losing work or introducing subtle bugs.

We build from the basics of what state must be saved, through sharded and asynchronous checkpointing for distributed training, to the mathematics of expected loss from hardware failures, elastic training, and deterministic reproducibility.

For the distributed training infrastructure that checkpointing sits on top of, see [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) and [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html). For the optimizer state that must be saved, see [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html). Memory-efficient techniques that reduce checkpoint sizes are covered in [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html). For the single-GPU specialisation of everything in this chapter — an atomic `torch.save`/`os.replace` checkpoint carrying weights, both optimizers, step, token cursor and RNG for the ~100M-parameter Stack-100M run — see [Chapter 14.7](../14-capstone/07-pretraining-run.html).

---

## 3.12.1 What State Must Be Saved?

A complete checkpoint for a training job contains four categories of state:

**1. Model parameters** — the weight tensors themselves. For a model with $P$ parameters stored in bf16, that is $2P$ bytes. A 70B-parameter model uses approximately 140 GB.

**2. Optimizer state** — for Adam, two additional copies of every parameter (the first and second moment estimates $m_t$ and $v_t$), typically kept in fp32 even when the model is in bf16. That is $4 \times P \times 4 = 16P$ bytes — for the 70B model, roughly 1.1 TB.

**3. Random-number generator (RNG) state** — the state of every RNG in the system: the CPU `torch` RNG, the CUDA RNG on each device, and potentially the data-loader's Python `random` and `numpy` RNG states. This is tiny (a few kilobytes per device) but critical for reproducibility.

**4. Training metadata** — the global step number, the current learning-rate schedule position, data shard cursors (which files and offsets have been consumed), and any other loop variables needed to resume identically.

Failing to save any of these correctly produces a run that resumes but silently diverges. A common mistake is saving only the model weights (sufficient for inference) while discarding optimizer state, which causes the resumed run to re-warm momentum from zero, effectively restarting learning-rate warmup.

!!! warning "Optimizer state mismatch is silent"
    If you save model weights but not optimizer moments, the resumed run will not crash. Loss will simply be higher than expected for hundreds of steps as the optimizer re-accumulates momentum. This is extremely hard to debug after the fact. Always save and reload optimizer state.

### What About Gradient Scaler State?

When training in fp16 with dynamic loss scaling (see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html)), the `GradScaler` object carries its own state: the current scale factor and a step counter. In bf16 training, which does not require dynamic scaling, there is no scaler to save.

---

## 3.12.2 Checkpoint Size and the Distributed Complication

On a single GPU, saving a checkpoint with `torch.save(state_dict, path)` is straightforward. The complication arises when model state is sharded across thousands of devices using FSDP, ZeRO-3, or tensor/pipeline parallelism.

### The Sharding Problem

In FSDP (Fully Sharded Data Parallel), each rank holds a disjoint shard of every parameter. Rank $r$ holds parameters indexed roughly as:

$$
\text{shard}_r = \left\{ w_i : i \bmod N_{\text{ranks}} = r \right\}
$$

If we naively call `torch.save` on each rank's local shard, we produce $N_{\text{ranks}}$ separate files that cannot be loaded without the same $N_{\text{ranks}}$ configuration. Changing cluster size during a resume becomes impossible.

There are two canonical approaches:

| Strategy | Description | Trade-offs |
|---|---|---|
| **Consolidated checkpoint** | Rank 0 gathers all shards, saves a single file | Load/save is serialized; bottleneck at all-gather; requires full model in rank-0 RAM |
| **Sharded checkpoint** | Each rank saves its own shard independently | Parallelizes I/O; format is topology-dependent; requires resharding at resume if topology changes |
| **Topology-agnostic sharded** | Save shards in a normalized layout (e.g., DTensor) that can be redistributed to any topology | Best of both; used by PyTorch Distributed Checkpoint (DCP) |

PyTorch's `torch.distributed.checkpoint` (DCP), introduced in PyTorch 2.x, implements the third strategy. It uses a uniform storage layout where parameters are saved as named chunks that can be remapped to any new device layout on load.

{{fig:ckpt-topology-agnostic-reshard}}

### Checkpoint Storage Layout

A well-designed sharded checkpoint directory looks like this:

{{fig:ckpt-sharded-dir-layout}}

Each `.distcp` file is a flat binary blob containing the tensor chunk bytes, with a companion index file mapping logical tensor names to physical byte ranges.

---

## 3.12.3 Saving and Loading: Concrete PyTorch Code

Let us build a minimal but production-grade checkpointing harness for FSDP training.

```python
"""
checkpoint.py — Production-grade FSDP checkpointing utilities.

Requirements: PyTorch >= 2.1, torchdata, FSDP model and optimizer.
"""

import os
import json
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
)
from torch.distributed.checkpoint import (
    save,
    load,
    FileSystemWriter,
    FileSystemReader,
)
from torch.distributed.checkpoint.metadata import BytesStorageMetadata
from pathlib import Path
import random
import numpy as np


# --------------------------------------------------------------------------
# RNG state helpers
# --------------------------------------------------------------------------

def get_rng_state() -> dict:
    """Capture all RNG states on the current rank."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),  # current device only
    }


def restore_rng_state(state: dict) -> None:
    """Restore RNG states — must be called on the same rank."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])


# --------------------------------------------------------------------------
# Sharded checkpoint save (topology-agnostic via DCP)
# --------------------------------------------------------------------------

def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    lr_scheduler,
    data_loader_state: dict,
    checkpoint_dir: str,
    rank: int,
) -> None:
    """
    Save a complete training checkpoint using PyTorch Distributed Checkpoint.
    All ranks participate; I/O is parallel across ranks.

    Args:
        model:            The FSDP-wrapped model.
        optimizer:        The optimizer (may also be FSDP-sharded).
        step:             Global training step.
        lr_scheduler:     LR scheduler object.
        data_loader_state: Dict with shard file index and byte offset.
        checkpoint_dir:   Root directory for all checkpoints.
        rank:             Current rank (for per-rank files).
    """
    ckpt_path = Path(checkpoint_dir) / f"step_{step:08d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # 1. Collect sharded model + optimizer state dict (stays sharded, no gather)
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
    ):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)

    # 2. Parallel distributed save — every rank writes its own shards
    save(
        {"model": model_state, "optimizer": optim_state},
        storage_writer=FileSystemWriter(ckpt_path / "model_optim"),
    )

    # 3. Per-rank RNG state (tiny, but critical for exact reproducibility)
    rng_path = ckpt_path / f"rng_state_rank{rank}.pt"
    torch.save(get_rng_state(), rng_path)

    # 4. Training metadata — only rank 0 writes the shared metadata file
    if rank == 0:
        metadata = {
            "step": step,
            "lr_scheduler": lr_scheduler.state_dict(),
            "data_loader": data_loader_state,
        }
        with open(ckpt_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    # 5. Synchronise: all ranks must finish before we consider the checkpoint
    #    "complete". Write an atomic sentinel file last.
    dist.barrier()
    if rank == 0:
        (ckpt_path / "COMPLETE").touch()
        print(f"[rank 0] Checkpoint saved: {ckpt_path}")


# --------------------------------------------------------------------------
# Sharded checkpoint load
# --------------------------------------------------------------------------

def load_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    rank: int,
) -> dict:
    """
    Load a checkpoint. Returns metadata dict (step, scheduler state, etc.).
    Works even if the current topology differs from the saved topology.
    """
    ckpt_path = Path(checkpoint_dir)
    assert (ckpt_path / "COMPLETE").exists(), \
        f"Checkpoint at {ckpt_path} is incomplete or corrupted!"

    # 1. Load model + optimizer via DCP (handles resharding automatically)
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
    ):
        state_dict = {"model": model.state_dict(), "optimizer": {}}
        load(
            state_dict,
            storage_reader=FileSystemReader(ckpt_path / "model_optim"),
        )
        model.load_state_dict(state_dict["model"])
        optim_state = FSDP.optim_state_dict_to_load(
            model, optimizer, state_dict["optimizer"]
        )
        optimizer.load_state_dict(optim_state)

    # 2. Restore per-rank RNG state.
    #    weights_only=False is required here: since PyTorch 2.6 `torch.load`
    #    defaults to weights_only=True, whose restricted unpickler rejects the
    #    numpy ndarray hidden inside np.random.get_state()'s tuple. Only ever
    #    pass weights_only=False for files you produced yourself.
    rng_path = ckpt_path / f"rng_state_rank{rank}.pt"
    if rng_path.exists():
        restore_rng_state(
            torch.load(rng_path, map_location="cpu", weights_only=False)
        )

    # 3. Load shared metadata (every rank reads it for the scheduler / step)
    with open(ckpt_path / "metadata.json") as f:
        metadata = json.load(f)

    dist.barrier()
    return metadata


# --------------------------------------------------------------------------
# Checkpoint manager: keeps last K checkpoints, rotates older ones
# --------------------------------------------------------------------------

class CheckpointManager:
    def __init__(self, root_dir: str, keep_last_k: int = 3):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k
        self._saved: list[Path] = []

    def save(self, model, optimizer, step, lr_scheduler,
             data_loader_state, rank):
        save_checkpoint(
            model, optimizer, step, lr_scheduler,
            data_loader_state, str(self.root_dir), rank
        )
        ckpt_path = self.root_dir / f"step_{step:08d}"
        self._saved.append(ckpt_path)
        self._saved.sort()

        # Remove checkpoints older than keep_last_k
        while len(self._saved) > self.keep_last_k:
            old = self._saved.pop(0)
            if rank == 0 and old.exists():
                import shutil
                shutil.rmtree(old)
                print(f"[rank 0] Removed old checkpoint: {old}")

    def latest(self) -> Path | None:
        """Find the most recent COMPLETE checkpoint under root_dir."""
        candidates = sorted(self.root_dir.glob("step_*"))
        for ckpt in reversed(candidates):
            if (ckpt / "COMPLETE").exists():
                return ckpt
        return None
```

The `COMPLETE` sentinel file pattern is critical: it prevents a partially written checkpoint from being mistakenly loaded after a crash mid-save. For a single file (rather than a directory of shards) the stronger idiom is to `torch.save` to `path + ".tmp"` and then `os.replace(tmp, path)`: rename is atomic on POSIX filesystems, so readers see either the old complete checkpoint or the new one, never a half-written mixture.

!!! warning "`torch.load` defaults changed: `weights_only=True` since PyTorch 2.6"
    PyTorch 2.6 flipped `torch.load`'s `weights_only` default to `True`. Its restricted unpickler accepts tensors and plain containers but rejects arbitrary objects — including the numpy `ndarray` buried inside the tuple returned by `np.random.get_state()`. A naively written RNG blob therefore raises `UnpicklingError` on your *first resume*, i.e. mid-run on GPUs you are paying for. Two honest fixes: pass `weights_only=False` for files you produced yourself (you are executing pickled code — never do this for a downloaded checkpoint), or store RNG state as tensors and plain Python types so the safe loader works unmodified. [Chapter 14.7](../14-capstone/07-pretraining-run.html) shows the plain-types version. `torch.serialization.add_safe_globals` is the middle path: allowlist exactly the types you need.

### The Modern DCP API: `Stateful`, `get_state_dict` and `async_save`

The harness above uses the FSDP1 `state_dict_type` context manager, which is the API most existing training code is written against and which you will still meet in the wild. Current PyTorch offers a cleaner, parallelism-agnostic surface that you should prefer for new code, and that FSDP2 (`fully_shard`) requires — FSDP2 parameters are `DTensor`s, which DCP consumes natively with no context manager at all:

- `torch.distributed.checkpoint.state_dict.get_state_dict(model, optimizers)` returns a matched `(model_state_dict, optimizer_state_dict)` pair that is correct under DDP, FSDP1, FSDP2 or plain single-GPU, and `set_state_dict(...)` writes them back.
- The `torch.distributed.checkpoint.stateful.Stateful` protocol (any object with `state_dict()` / `load_state_dict()`) lets you drop schedulers, data loaders and step counters into the *same* DCP save alongside the model, instead of maintaining a side-channel `metadata.json`.
- `dcp.save(...)` / `dcp.load(...)` take a `checkpoint_id` (a path or URI) instead of an explicit writer, and `dcp.async_save(...)` returns a future — the production-grade version of the hand-rolled background thread we build in the next section.

```python
"""
dcp_modern.py — The parallelism-agnostic PyTorch DCP idiom (PyTorch >= 2.4).
Works unchanged for single-GPU, DDP, FSDP1 and FSDP2 (`fully_shard`).
"""

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful


class TrainState(Stateful):
    """Bundles everything that must resume together into one DCP entry.

    Because it implements the Stateful protocol, DCP calls .state_dict() at
    save time and .load_state_dict() at load time — so the model, optimizer,
    LR schedule, step counter and data cursor are written as ONE consistent
    checkpoint, with no separate metadata file to fall out of sync.
    """

    def __init__(self, model, optimizer, scheduler, dataloader):
        self.model, self.optimizer = model, optimizer
        self.scheduler, self.dataloader = scheduler, dataloader
        self.step = 0

    def state_dict(self) -> dict:
        # get_state_dict() returns *sharded* DTensors under FSDP and plain
        # tensors otherwise; DCP handles both without special-casing.
        model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_sd,
            "optim": optim_sd,
            "sched": self.scheduler.state_dict(),
            "data": self.dataloader.state_dict(),   # StatefulDataLoader
            "step": self.step,
        }

    def load_state_dict(self, sd: dict) -> None:
        # set_state_dict() reshards the loaded tensors onto the CURRENT
        # topology — this is what makes a 512-GPU checkpoint loadable on 256.
        set_state_dict(
            self.model, self.optimizer,
            model_state_dict=sd["model"], optim_state_dict=sd["optim"],
        )
        self.scheduler.load_state_dict(sd["sched"])
        self.dataloader.load_state_dict(sd["data"])
        self.step = sd["step"]


def save(train_state, step: int, root: str, blocking: bool = False):
    """Synchronous or asynchronous save. Returns a future when async."""
    app_state = {"app": train_state}          # key must match on load
    ckpt_id = f"{root}/step_{step:08d}"
    if blocking:
        dcp.save(app_state, checkpoint_id=ckpt_id)
        return None
    # async_save stages tensors to CPU, then writes from a background
    # process; await the returned future before the NEXT async_save.
    return dcp.async_save(app_state, checkpoint_id=ckpt_id)


def restore(train_state, ckpt_id: str) -> None:
    """In-place load. The dict must be pre-populated with the same keys so
    DCP knows the destination shapes/shardings to resolve chunks into."""
    app_state = {"app": train_state}
    dcp.load(app_state, checkpoint_id=ckpt_id)
```

This is essentially what `torchtitan` — PyTorch's reference LLM pretraining stack — does in production, and it is the version to copy if you are starting fresh. Note the asymmetry that trips people up on their first DCP load: `dcp.load` fills a state dict *in place*, so you must construct the model and optimizer first and pass their (empty-valued but correctly shaped) state dict in. DCP never tells you what shapes to build.

!!! tip "At Stack-100M scale, none of this machinery is needed"
    A ~100M-parameter model in bf16 with fp32 AdamW state is on the order of a gigabyte of checkpoint, saved from one process in a couple of seconds. Plain `torch.save` to a temp file plus `os.replace`, keeping the last handful of steps, is entirely adequate — see [Chapter 14.7](../14-capstone/07-pretraining-run.html). Reach for DCP when the checkpoint no longer fits comfortably in one rank's host RAM, or when you need to resume on a different number of GPUs.

---

## 3.12.4 Asynchronous (Background) Checkpointing

Synchronous checkpointing blocks all GPUs while state is serialized and written to disk. For large models this pause can be measured in minutes — entirely "dead" compute time.

### The Async Pipeline

Asynchronous checkpointing runs the save in a background thread or process so that training continues immediately:

{{fig:ckpt-async-vs-sync-timeline}}

The key insight: if we can take a fast in-memory snapshot of all GPU tensors (copy to pinned CPU memory), we can resume training while the background thread writes to persistent storage.

```python
"""
async_checkpoint.py — Async checkpointing with background I/O thread.

The snapshot step (GPU -> CPU copy) is the only synchronisation point.
Disk I/O runs concurrently with the next training steps.
"""

import threading
import time
from copy import deepcopy
from pathlib import Path
import torch
import torch.distributed as dist


class AsyncCheckpointer:
    """
    Async checkpointing: copies state to CPU, then writes in a background
    thread so training can continue immediately.

    Usage:
        checkpointer = AsyncCheckpointer(save_fn=save_checkpoint, ...)
        # In training loop:
        checkpointer.save_if_due(step, model, optimizer, ...)
        # At the end of training:
        checkpointer.wait()
    """

    def __init__(self, save_fn, checkpoint_interval: int, root_dir: str):
        self.save_fn = save_fn
        self.interval = checkpoint_interval
        self.root_dir = root_dir
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._pending_error: Exception | None = None

    def _check_for_errors(self):
        """Propagate any exception from the background thread."""
        with self._lock:
            if self._pending_error is not None:
                raise self._pending_error

    @staticmethod
    def _to_cpu(obj):
        """Recursively copy every tensor in a nested state dict to CPU."""
        if torch.is_tensor(obj):
            return obj.detach().to("cpu", copy=True)
        if isinstance(obj, dict):
            return {k: AsyncCheckpointer._to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [AsyncCheckpointer._to_cpu(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(AsyncCheckpointer._to_cpu(v) for v in obj)
        return deepcopy(obj)

    def _snapshot_to_cpu(self, model, optimizer) -> tuple[dict, dict]:
        """
        Copy all tensors out to CPU RAM. This is the only step that blocks
        training, and it is fast because it is a device-to-host copy, not a
        disk write.

        Caveat: `.to("cpu")` allocates *pageable* memory. Production stagers
        (PyTorch DCP, DeepSpeed) copy into a pre-allocated *pinned* buffer,
        which both doubles achievable PCIe bandwidth and lets the copy run
        on a side CUDA stream, overlapping the next forward pass.
        """
        # model.state_dict() already returns CPU copies under
        # FSDP + SHARDED_STATE_DICT + offload_to_cpu=True; _to_cpu() makes
        # that unconditional so this class also works without that config.
        model_state = self._to_cpu(model.state_dict())
        # The optimizer state must ALSO leave the GPU. A plain
        # deepcopy(optimizer.state_dict()) clones Adam's moments *on device*,
        # transiently doubling optimizer VRAM at the worst possible moment.
        optim_state = self._to_cpu(optimizer.state_dict())
        return model_state, optim_state

    def _background_save(self, model_state, optim_state, step, metadata):
        """Runs in background thread: pure I/O, no GPU interaction."""
        try:
            start = time.time()
            ckpt_path = Path(self.root_dir) / f"step_{step:08d}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model_state, "optimizer": optim_state,
                 "metadata": metadata},
                ckpt_path / "checkpoint.pt"
            )
            (ckpt_path / "COMPLETE").touch()
            elapsed = time.time() - start
            print(f"[async ckpt] step {step} written in {elapsed:.1f}s")
        except Exception as e:
            with self._lock:
                self._pending_error = e

    def save_if_due(self, step, model, optimizer, metadata):
        """Call this every training step. Launches async save when due."""
        self._check_for_errors()
        if step % self.interval != 0:
            return

        # Wait for any previous background save to finish before starting new
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        self._check_for_errors()

        # Snapshot — this blocks, but is fast (GPU->CPU copy)
        t0 = time.time()
        model_state, optim_state = self._snapshot_to_cpu(model, optimizer)
        dist.barrier()  # All ranks finish snapshot before any continues
        print(f"[async ckpt] snapshot took {time.time() - t0:.1f}s, "
              f"background write starting...")

        # Launch background write thread
        self._thread = threading.Thread(
            target=self._background_save,
            args=(model_state, optim_state, step, metadata),
            daemon=True,
        )
        self._thread.start()

    def wait(self):
        """Block until any in-flight async checkpoint is written."""
        if self._thread is not None:
            self._thread.join()
        self._check_for_errors()
```

We wrote this out longhand because the mechanism is worth owning, but in real code you call `dcp.async_save(state, checkpoint_id=...)` from the previous section and get all of it — staging, background write, future — in one line. Modern training frameworks (PyTorch's async DCP path, DeepSpeed's `async_checkpoint_engine`) implement variations of this pattern. PyTorch DCP can run the write in a *separate process* rather than a background thread (`AsyncCheckpointerType.PROCESS`), which removes the Python GIL contention that otherwise slows the training steps overlapping the write. The `dist.barrier()` inside the snapshot step ensures all ranks have finished their CPU copy before training resumes, which is critical — you cannot have rank 0 already on step $N+1$ while rank 3 is still copying rank-$N$ tensors.

Two correctness rules govern any async checkpointer, hand-rolled or not. First, **do not mutate the staged tensors**: the snapshot must be a copy, because `optimizer.step()` on step $N+1$ writes in place over the very moment buffers the background thread is serialising. Second, **join the previous save before starting the next one** (as `save_if_due` does), or two writers race for the same directory and you can end up with a `COMPLETE` sentinel over a mixture of two steps' bytes.

### In-Memory Checkpointing

For the most aggressive fault tolerance, some systems keep the most recent checkpoint entirely in CPU DRAM across all nodes, writing to persistent storage (NFS, distributed filesystem) only for long-term retention. This is called **in-memory checkpointing**.

The trade-off: recovery from a GPU failure can proceed in seconds (reload from DRAM) rather than minutes (reload from disk), but a full node failure (including CPU DRAM) still requires loading from persistent storage.

PyTorch's `torch.distributed.checkpoint` supports a `StorageWriter` / `StorageReader` interface. You can implement an in-memory backend:

```python
"""
in_memory_storage.py — Minimal in-memory checkpoint storage backend.

Stores checkpoint bytes in a shared dict on each rank.
Production systems use more sophisticated shared-memory mechanisms
(e.g., mmap, Ray's plasma store).
"""

from torch.distributed.checkpoint.storage import StorageWriter, StorageReader
from torch.distributed.checkpoint.metadata import Metadata, StorageMeta
from io import BytesIO
import io


_IN_MEMORY_STORE: dict[str, bytes] = {}  # In a real system: cross-rank store


class InMemoryWriter(StorageWriter):
    def __init__(self):
        self._buffers: dict[str, bytes] = {}

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        pass  # no-op for in-memory

    def prepare_local_plan(self, plan):
        return plan

    def prepare_global_plan(self, global_plans):
        return global_plans

    def write_data(self, plan, planner):
        # Write each planned chunk to an in-memory buffer
        futures = []
        for bucket in plan.items:
            data = planner.resolve_data(bucket)
            buf = BytesIO()
            torch.save(data, buf)
            self._buffers[bucket.storage_index.fqn] = buf.getvalue()
        _IN_MEMORY_STORE.update(self._buffers)

    def finish(self, metadata, results):
        _IN_MEMORY_STORE["__metadata__"] = metadata


class InMemoryReader(StorageReader):
    def read_metadata(self) -> Metadata:
        return _IN_MEMORY_STORE["__metadata__"]

    def set_up_storage_reader(self, metadata, is_coordinator):
        pass

    def prepare_local_plan(self, plan):
        return plan

    def prepare_global_plan(self, global_plans):
        return global_plans

    def read_data(self, plan, planner):
        for req in plan.items:
            data = torch.load(BytesIO(_IN_MEMORY_STORE[req.storage_index.fqn]))
            planner.commit_tensor(req, data)
```

---

## 3.12.5 Hardware Failure Rates at Scale: The Math

Why does fault tolerance matter so much? Let us quantify the expected time between failures.

Let $\lambda$ be the hourly failure rate of a single node (GPU host). For modern GPU clusters, $\lambda$ is on the order of $10^{-3}$ failures per node per hour (i.e., a single node fails roughly every 40–50 days on average). For a cluster of $N$ nodes, the cluster-level failure rate is approximately $N\lambda$, and the expected time between any failure is:

$$
\mathbb{E}[\text{MTBF}_{\text{cluster}}] = \frac{1}{N \lambda}
$$

!!! example "Expected failure frequency for a 1024-node cluster"
    Suppose $\lambda = 1/1000$ failures per node per hour (each node fails on average once every 1000 hours, roughly 42 days).

    For $N = 1024$ nodes:

    $$
    \mathbb{E}[\text{MTBF}_{\text{cluster}}] = \frac{1}{1024 \times 10^{-3}} \approx 0.977 \text{ hours} \approx 59 \text{ minutes}
    $$

    That is, on a 1024-node cluster, you expect a failure somewhere in the cluster roughly every hour. Training runs lasting weeks will experience dozens to hundreds of failures. Without checkpointing, a single failure restarts training from scratch.

    The fraction of useful compute wasted by a failure with checkpoint interval $T_{\text{ckpt}}$ steps and checkpoint save time $T_{\text{save}}$ is approximately:

    $$
    \text{waste fraction} \approx \frac{T_{\text{ckpt}}/2 + T_{\text{save}}}{T_{\text{MTBF}}}
    $$

    For $T_{\text{ckpt}} = 1000$ steps at 10 steps/min (100 min of work), $T_{\text{save}} = 5$ min, and $T_{\text{MTBF}} = 60$ min:

    $$
    \text{waste} \approx \frac{50 + 5}{60} \approx 92\%
    $$

    This motivates more frequent checkpoints and faster (async) saves.

    Read that 92% as a red flag rather than a precise figure: the linear approximation assumes at most one failure per interval and is only accurate when $T_{\text{ckpt}} \ll T_{\text{MTBF}}$. Here $T_{\text{ckpt}}$ *exceeds* the MTBF, so the run frequently dies before reaching its next checkpoint, restarts from the same one, and — in the limit — never makes progress at all. The formula's job is to tell you the configuration is untenable, which it does.

{{fig:ckpt-failure-waste-and-daly}}

The optimal checkpoint interval $T^*$ that minimises expected wasted compute can be derived (Young 1974, commonly called "Daly's formula" in HPC):

$$
T^* \approx \sqrt{2 \cdot T_{\text{save}} \cdot T_{\text{MTBF}}}
$$

Plugging in $T_{\text{save}} = 5$ min and $T_{\text{MTBF}} = 60$ min: $T^* \approx \sqrt{600} \approx 24$ minutes. The lesson: frequent, fast checkpoints beat infrequent slow ones.

---

## 3.12.6 Resuming Training Correctly

Correctly resuming training is harder than it appears. The goal is that a run which was interrupted and resumed should produce identical model weights (given identical hardware) to an uninterrupted run. We call this **exact resumability**.

### Data Loader State

If the data loader does not restore its position, the resumed run will see data out of order or repeat data from earlier in the epoch. For a pre-shuffled dataset stored as shards, we need to track:

- Which shard files have been fully consumed.
- The offset (in tokens or samples) into the current shard.
- The random seed used for any in-flight shuffling.

```python
"""
stateful_dataloader.py — A minimal stateful data loader that saves/restores
its position within a sharded dataset.
"""

import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, IterableDataset


class ShardedTextDataset(IterableDataset):
    """
    Streams tokens from pre-tokenised shard files (.pt tensors).
    Saves and restores its cursor for exact reproducibility.
    """

    def __init__(
        self,
        shard_paths: list[str],
        seq_len: int,
        start_shard: int = 0,
        start_offset: int = 0,
    ):
        self.shard_paths = shard_paths
        self.seq_len = seq_len
        # Restored cursor positions
        self.start_shard = start_shard
        self.start_offset = start_offset

    def get_state(self) -> dict:
        """Call after every batch to get the current cursor state."""
        return {
            "current_shard": self._current_shard,
            "current_offset": self._current_offset,
        }

    def __iter__(self):
        self._current_shard = self.start_shard
        self._current_offset = self.start_offset

        for shard_idx in range(self.start_shard, len(self.shard_paths)):
            tokens = torch.load(self.shard_paths[shard_idx])  # 1D tensor
            start = self.start_offset if shard_idx == self.start_shard else 0
            self._current_shard = shard_idx

            pos = start
            while pos + self.seq_len + 1 <= len(tokens):
                x = tokens[pos : pos + self.seq_len]
                y = tokens[pos + 1 : pos + self.seq_len + 1]
                self._current_offset = pos
                yield x, y
                pos += self.seq_len

            # Move to next shard
            self.start_offset = 0  # only use start_offset for first shard
```

This hand-rolled cursor is honest but incomplete in one important way: with `num_workers > 0`, PyTorch's `DataLoader` forks worker processes that each hold their *own* copy of the dataset object and prefetch several batches ahead. The cursor you read on the main process is therefore stale by up to `num_workers × prefetch_factor` batches, and `dataset.get_state()` in the parent may not even reflect a live iterator at all.

The library that solves this is **`torchdata`**: `torchdata.stateful_dataloader.StatefulDataLoader` is a drop-in replacement for `torch.utils.data.DataLoader` that adds `state_dict()` / `load_state_dict()` and correctly aggregates the state of every worker, including in-flight prefetched batches. It works with map-style datasets (it records sampler position) and with iterable datasets that themselves expose `state_dict()`/`load_state_dict()`. It is what `torchtitan` uses for resumable pretraining data, and it satisfies the `Stateful` protocol, so it slots straight into the DCP `TrainState` above.

```python
"""
stateful_loader_usage.py — Resumable data loading with torchdata.
`pip install torchdata`
"""

from torchdata.stateful_dataloader import StatefulDataLoader


class ResumableTokenStream(ShardedTextDataset):
    """Add the Stateful hooks StatefulDataLoader looks for on the dataset."""

    def state_dict(self) -> dict:
        return self.get_state()

    def load_state_dict(self, sd: dict) -> None:
        self.start_shard = sd["current_shard"]
        self.start_offset = sd["current_offset"]


def build_loader(shard_paths, seq_len, batch_size, num_workers=4):
    ds = ResumableTokenStream(shard_paths, seq_len)
    # Identical signature to torch.utils.data.DataLoader.
    return StatefulDataLoader(ds, batch_size=batch_size,
                              num_workers=num_workers)


# At checkpoint time: loader.state_dict() -> goes into the checkpoint.
# At resume time:     loader.load_state_dict(saved) BEFORE iterating.
```

### Resuming Across Different Topologies

A practical need: a run crashes on 512 GPUs and is restarted on 256 GPUs because some nodes are under repair. This is called **elastic training**.

PyTorch DCP's topology-agnostic format makes this straightforward for the model and optimizer state. The data loader must also handle the change: if the number of data-parallel ranks changes, each rank's portion of the dataset changes, so we cannot simply restore the old per-rank shard cursor. The simplest approach is to track a global token offset and recompute each rank's starting position from it.

```python
def compute_rank_start_offset(
    global_token_offset: int,
    world_size: int,
    rank: int,
    tokens_per_rank_per_step: int,
) -> int:
    """
    Given the global number of tokens consumed so far, compute the
    starting offset for this rank in the new topology.

    global_token_offset: total tokens processed before the crash
    world_size: new number of data-parallel ranks
    rank: this rank's index in the new topology
    tokens_per_rank_per_step: seq_len * micro_batch_size
    """
    # Tokens consumed per global step
    tokens_per_step = tokens_per_rank_per_step * world_size
    # Completed steps so far
    steps_done = global_token_offset // tokens_per_step
    # This rank's start offset in the new topology
    return steps_done * tokens_per_rank_per_step + rank * tokens_per_rank_per_step
```

Truly elastic training — dynamically adding or removing nodes mid-run without restarting — is more complex. PyTorch's `torchrun` with `--nnodes=MIN:MAX` and `torch.distributed.elastic` (TorchElastic) support this. DeepSpeed also has elastic training support. The key mechanisms are:

1. **Rendezvous**: nodes join and leave a rendezvous barrier; membership changes trigger a re-initialisation of process groups.
2. **Checkpoint-on-membership-change**: a micro-checkpoint is taken whenever the membership changes, ensuring no work is lost.
3. **Rebalancing**: the optimizer state sharding is updated to reflect the new world size.

The frontier of this line of work is **`torchft`** (`meta-pytorch/torchft`), which pushes fault tolerance below the restart granularity entirely: a central *Lighthouse* coordinator tracks live replica groups, and when one dies the survivors continue the *current* step rather than tearing down and restarting from a checkpoint. It pairs with semi-synchronous algorithms — LocalSGD and DiLoCo, where replica groups train independently for $H$ inner steps and synchronise only occasionally — so that losing a replica group costs at most $H$ steps of its work instead of stalling the whole job. Checkpointing does not go away under this model; it becomes the slower backstop underneath a fast in-band recovery path.

---

## 3.12.7 Determinism and Reproducibility

A training run is **reproducible** if, given identical hardware and the same checkpoint, two resumed runs produce identical weight trajectories. This is harder to achieve than it sounds.

### Sources of Non-Determinism

| Source | Cause | Mitigation |
|---|---|---|
| RNG state not saved | Python/NumPy/PyTorch/CUDA RNG diverges | Save and restore all RNG states per rank |
| `cudnn.benchmark` mode | cuDNN picks fastest algorithm, which may vary run-to-run | Set `torch.backends.cudnn.deterministic = True` and `benchmark = False` |
| Non-deterministic kernels | Some CUDA kernels (e.g., atomics in scatter) are non-deterministic | `torch.use_deterministic_algorithms(True)` — may slow training |
| Data loader ordering | Workers pick up examples in different orders depending on timing | Use a fixed seed and deterministic data pipeline |
| Gradient accumulation float ordering | Floating-point addition is not associative; order of partial sums changes | Usually negligible numerically; exact bit-reproducibility requires fixed order |
| NCCL all-reduce ordering | Non-deterministic reduce ordering across rings | Set `NCCL_ALGO=Ring` and fixed chunk sizes (usually not worth it) |

For production pretraining, **strict bit-for-bit reproducibility** is often abandoned in favor of **statistical reproducibility**: the loss curve and final model quality match closely even if exact values differ. The RNG state and checkpoint integrity are preserved for resume correctness, but strict determinism in CUDA kernels is not enforced because it incurs a 10–30% performance penalty.

```python
"""
determinism_setup.py — Configure determinism at various strictness levels.
"""

import torch
import os


def configure_determinism(level: str = "soft") -> None:
    """
    Configure PyTorch determinism.

    level="soft"   — restore RNG, no strict kernel determinism.
                     Fast. Resumable runs track closely but not bit-exactly.
    level="strict" — full determinism. Slower; use for debugging.
    """
    if level == "strict":
        # Force deterministic CUDA kernels
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        # Required for some deterministic ops on CUDA
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        print("Strict determinism enabled (expect ~15% slowdown).")
    elif level == "soft":
        # Allow cuDNN to pick optimal (non-deterministic) algorithms
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        print("Soft determinism: RNG states saved, kernels non-deterministic.")
    else:
        raise ValueError(f"Unknown level: {level}")
```

### Seeding Strategy for Distributed Training

Each rank must have a unique but reproducible seed to avoid all ranks processing identical random augmentations:

```python
def seed_everything(base_seed: int, rank: int) -> None:
    """
    Set all RNG seeds deterministically for a given rank.
    The per-rank seed is derived from the base seed and rank index,
    ensuring different seeds across ranks but reproducibility from
    a given base seed.
    """
    import random
    import numpy as np

    rank_seed = base_seed + rank * 31337  # prime offset per rank
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**31))
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
```

---

## 3.12.8 Production Hardening and Operational Patterns

Beyond the core save/load mechanism, production-grade pretraining systems incorporate a range of hardening techniques.

### Checkpoint Integrity Verification

Disk I/O errors, network interruptions to a shared filesystem, or process crashes can produce silently corrupt checkpoint files. Compute a checksum at save time and verify it at load time:

```python
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB chunks
            h.update(chunk)
    return h.hexdigest()


def write_checksums(ckpt_dir: Path) -> None:
    """Write SHA-256 checksums for all checkpoint files."""
    checksums = {}
    for p in ckpt_dir.rglob("*"):
        if p.is_file() and p.name != "checksums.json":
            checksums[str(p.relative_to(ckpt_dir))] = sha256_file(p)
    with open(ckpt_dir / "checksums.json", "w") as f:
        json.dump(checksums, f, indent=2)


def verify_checksums(ckpt_dir: Path) -> bool:
    """Return True if all files match their saved checksums."""
    checksum_file = ckpt_dir / "checksums.json"
    if not checksum_file.exists():
        return False  # No checksums — cannot verify
    with open(checksum_file) as f:
        expected = json.load(f)
    for rel_path, expected_hash in expected.items():
        actual = sha256_file(ckpt_dir / rel_path)
        if actual != expected_hash:
            print(f"CHECKSUM MISMATCH: {rel_path}")
            return False
    return True
```

### From Training Checkpoint to Releasable Weights

A DCP directory of `.distcp` shards is a *training* artifact: it is topology-agnostic, but it is not something `transformers`, vLLM or llama.cpp can open. Converting it is the last mile of a pretraining run and a step people are surprised to find is not automatic.

```bash
# 1. Consolidate the sharded DCP directory into a single torch.save file.
#    (torch.distributed.checkpoint.format_utils also exposes the inverse,
#     torch_save_to_dcp, for importing a legacy single-file checkpoint.)
python -c "
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
dcp_to_torch_save('ckpt/step_00050000/model_optim', 'ckpt/consolidated.pt')
"

# 2. Strip optimizer state (~89% of the bytes) and re-serialise the weights
#    as safetensors — the format the HF ecosystem loads by default. It is a
#    zero-copy mmap-able layout with no pickle, so loading it cannot execute
#    code, and it is what you publish.
mkdir -p release && python export_safetensors.py
```

```python
# export_safetensors.py — weights only, in the format the ecosystem expects.
import torch
from safetensors.torch import save_file

sd = torch.load("ckpt/consolidated.pt", map_location="cpu",
                weights_only=True)["model"]
# safetensors requires contiguous, non-shared storage per tensor.
sd = {k: v.contiguous().clone() for k, v in sd.items()}
save_file(sd, "release/model.safetensors",
          metadata={"format": "pt"})   # HF loaders check this key
```

Keep the full training checkpoint too: an inference-only export cannot be resumed from, and continued pretraining ([Continual & Domain-Adaptive Pretraining](../03-pretraining/16-continual-pretraining.html)) or a long-context extension phase ([Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html)) needs the optimizer state back.

### Rotation and Retention Strategy

Keeping every checkpoint for a months-long run is prohibitively expensive. A typical rotation policy:

{{fig:ckpt-gfs-retention-timeline}}

This is analogous to grandfather-father-son (GFS) backup rotation schemes.

### Watchdog and Auto-Restart

At scale, manual restarts are unacceptable. A production training launcher includes a watchdog process that monitors for failures and automatically restarts the job from the latest checkpoint:

```bash
#!/bin/bash
# watchdog_launch.sh — Auto-restart training on failure.
# Uses torchrun with fault-tolerant options.

MAX_RESTARTS=20
RESTART_COUNT=0
CHECKPOINT_DIR="/mnt/checkpoints/my_run"

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo "Starting training attempt $((RESTART_COUNT + 1))..."

    torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node=8 \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:29500" \
        --max_restarts=0 \
        train.py \
            --checkpoint_dir "${CHECKPOINT_DIR}" \
            --resume_from_latest

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully."
        break
    else
        echo "Training failed (exit code $EXIT_CODE). Restarting..."
        RESTART_COUNT=$((RESTART_COUNT + 1))
        sleep 30  # Brief pause to allow failed nodes to be replaced
    fi
done

if [ $RESTART_COUNT -eq $MAX_RESTARTS ]; then
    echo "ERROR: Exceeded max restarts. Manual intervention required."
    exit 1
fi
```

### Training Heartbeat and Dead-Man Switch

A common pattern for detecting stuck (not crashed, just hung) jobs is a heartbeat: the training loop writes a timestamp to a file every N steps. A separate watchdog process kills and restarts the job if the timestamp is older than a threshold.

```python
class TrainingHeartbeat:
    """Write a heartbeat file periodically so external monitors can detect
    hung jobs (e.g., deadlocked collectives)."""

    def __init__(self, heartbeat_path: str, interval_steps: int = 10):
        self.path = Path(heartbeat_path)
        self.interval = interval_steps

    def beat(self, step: int) -> None:
        if step % self.interval == 0:
            self.path.write_text(
                json.dumps({"step": step, "ts": time.time()})
            )
```

A heartbeat tells you the job hung; it does not tell you *which rank* hung or *where*. For that, PyTorch ships the **NCCL Flight Recorder**: a ring buffer of recently enqueued collectives (op name, sizes, sequence number, start/completion state) that is dumped to disk when a collective watchdog times out. The dump makes the diagnosis mechanical — the ranks that logged sequence number $k$ but never completed it are the stragglers, and a rank whose buffer *lacks* an all-reduce every peer recorded is the one that took a divergent code path (a classic cause: a `if rank == 0:` branch that skips a collective).

```bash
# Enable the flight recorder on every rank before launching torchrun.
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000     # entries retained per rank
export TORCH_NCCL_DUMP_ON_TIMEOUT=1          # dump the ring buffer on hang
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/mnt/logs/nccl_trace_rank
# Fail fast instead of hanging forever on a dead peer:
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

Pair this with a collective timeout short enough that a hang becomes a *crash* — `dist.init_process_group(..., timeout=timedelta(minutes=10))` — because your watchdog and auto-restart loop can only recover from processes that actually exit. A job that hangs silently for six hours costs more than one that dies in ten minutes.

---

## 3.12.9 Connecting the Pieces: A Complete Training Loop

The following shows how all the components above fit together in a real training loop.

```python
"""
train_loop.py — Fault-tolerant pretraining main loop sketch.

Assumes: FSDP model, AdamW optimizer, cosine LR schedule,
         sharded data loader, CheckpointManager, AsyncCheckpointer.
"""

import os
import torch
import torch.distributed as dist
from pathlib import Path
from torchdata.stateful_dataloader import StatefulDataLoader


def main():
    # --- Distributed init ---
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % 8}")
    torch.cuda.set_device(device)

    # --- Build model, optimizer, scheduler (not shown for brevity) ---
    model = build_fsdp_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    scheduler = get_cosine_schedule(optimizer, warmup_steps=2000, total_steps=500_000)

    # --- Checkpoint manager ---
    ckpt_manager = CheckpointManager(
        root_dir="/mnt/checkpoints/my_run",
        keep_last_k=3,
    )
    heartbeat = TrainingHeartbeat(
        "/mnt/checkpoints/my_run/heartbeat.json",
        interval_steps=5,
    )

    # --- Resume from latest checkpoint if available ---
    start_step = 0
    latest_ckpt = ckpt_manager.latest()
    if latest_ckpt is not None:
        if rank == 0:
            print(f"Resuming from checkpoint: {latest_ckpt}")
        metadata = load_checkpoint(model, optimizer, str(latest_ckpt), rank)
        start_step = metadata["step"] + 1
        scheduler.load_state_dict(metadata["lr_scheduler"])
        # Restore data loader cursor
        data_loader_state = metadata.get("data_loader")
    else:
        data_loader_state = None

    # --- Data loader ---
    # StatefulDataLoader (torchdata) rather than DataLoader: with
    # num_workers > 0 only it can aggregate the per-worker cursors and the
    # in-flight prefetched batches. ResumableTokenStream is the Stateful
    # subclass from Section 3.12.6.
    dataset = ResumableTokenStream(shard_paths=get_shard_paths(), seq_len=2048)
    dl = StatefulDataLoader(dataset, batch_size=4, num_workers=4)
    if data_loader_state is not None:
        dl.load_state_dict(data_loader_state)   # must precede iteration
    loader = iter(dl)

    # --- Training loop ---
    model.train()
    for step in range(start_step, 500_000):
        x, y = next(loader)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = model(x, labels=y).loss
        loss.backward()

        # Gradient clipping — important for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Heartbeat so external watchdog knows we're alive
        heartbeat.beat(step)

        # Periodic logging
        if step % 100 == 0 and rank == 0:
            print(f"step={step:6d}  loss={loss.item():.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if step % 1000 == 0:
            ckpt_manager.save(
                model, optimizer, step, scheduler,
                dl.state_dict(), rank
            )

    # Ensure final async checkpoint write completes
    dist.barrier()
    if rank == 0:
        print("Training complete.")


if __name__ == "__main__":
    main()
```

---

!!! interview "Interview Corner"
    **Q:** You are training a 70B parameter model on 2048 GPUs and the job fails every 90 minutes on average. Your checkpoint takes 8 minutes to save synchronously. How would you design the checkpointing strategy to minimise wasted GPU hours, and what are the key correctness requirements?

    **A:** Several layers of improvement are available.

    First, apply Daly's formula for optimal checkpoint interval: $T^* \approx \sqrt{2 \cdot T_{\text{save}} \cdot T_{\text{MTBF}}} = \sqrt{2 \times 8 \times 90} \approx 38$ minutes. So checkpoint every ~38 minutes, not every 90.

    Second, switch to asynchronous checkpointing. The critical path is the GPU-to-CPU tensor snapshot (roughly 30–60 seconds for a 70B model, since the 1.1 TB of optimizer state must be copied to pinned CPU RAM). Once on CPU, disk write happens in the background while training continues. This reduces the hard blocking time from 8 minutes to ~1 minute.

    Third, use PyTorch DCP sharded checkpoints: all 2048 ranks write in parallel to a distributed filesystem, achieving near-linear I/O scaling versus serialised saves through rank 0.

    Key correctness requirements: (a) save optimizer state and LR schedule, not just weights; (b) save per-rank RNG state for reproducibility; (c) save data loader cursor (shard index and byte offset) so training resumes exactly where it stopped; (d) write a COMPLETE sentinel file atomically after all writes finish so a crashed mid-save is never loaded; (e) verify checkpoint integrity with checksums before releasing GPUs after a restore.

---

!!! key "Key Takeaways"
    - A complete checkpoint contains four components: model weights, optimizer state (including moments), per-rank RNG states, and training metadata (step, scheduler, data cursor). Omitting any one causes silent divergence or incorrect resumption. Use `torchdata`'s `StatefulDataLoader` for the cursor — a hand-rolled one is stale by `num_workers × prefetch_factor` batches.
    - At scale (1000+ GPUs), expect a hardware failure somewhere in the cluster every hour or less. Fault tolerance is not optional.
    - Daly's formula gives the optimal checkpoint interval: $T^* \approx \sqrt{2 \cdot T_{\text{save}} \cdot T_{\text{MTBF}}}$. More frequent, faster checkpoints reduce wasted work better than infrequent saves.
    - Async checkpointing decouples the GPU-to-CPU snapshot (fast, ~seconds) from the CPU-to-disk write (slow, minutes), dramatically reducing dead compute time.
    - PyTorch Distributed Checkpoint (DCP) uses a topology-agnostic sharded format, enabling resume with a different number of GPUs without manual resharding. The modern idiom is `get_state_dict`/`set_state_dict` plus a `Stateful` object handed to `dcp.save` / `dcp.async_save` — not the legacy FSDP1 `state_dict_type` context manager. Export to safetensors (via `dcp_to_torch_save`) only at release time.
    - Write a COMPLETE sentinel file last; never load a checkpoint that lacks it. Verify files with checksums.
    - Strict bit-for-bit CUDA determinism costs 10–30% performance; production runs typically opt for soft determinism (save RNG, allow non-deterministic kernels) and rely on statistical reproducibility instead.
    - Elastic training (TorchElastic, `torchrun --nnodes=MIN:MAX`) enables dynamic cluster resize; the global token offset is the currency for recalculating each rank's data cursor after a topology change.
    - Combine checkpointing with a heartbeat watchdog and auto-restart launcher to turn hardware failures from catastrophic events into short, automated interruptions.

---

!!! sota "State of the Art & Resources (2026)"
    Checkpointing and fault tolerance for LLM pretraining has become a first-class systems research area: with runs now spanning thousands to tens of thousands of GPUs over weeks or months, the field has moved from epoch-level saves to sub-minute asynchronous in-memory checkpoints, topology-agnostic sharded formats, and per-step fault tolerance with zero training interruption.

    **Foundational work**

    - [Zhao et al., *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel* (2023)](https://arxiv.org/abs/2304.11277) — canonical paper on FSDP state-dict types and the engineering decisions behind sharded checkpointing; basis for PyTorch DCP.
    - [Wang et al., *GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints* (SOSP 2023)](https://www.amazon.science/publications/gemini-fast-failure-recovery-in-distributed-training-with-in-memory-checkpoints) — landmark systems paper showing 13× faster recovery via near-optimal in-memory checkpoint placement across CPU DRAM of neighbor nodes.

    **Recent advances (2023–2026)**

    - [Wang et al., *Fault-Tolerant Hybrid-Parallel Training at Scale with Reliable and Efficient In-memory Checkpointing* (2023)](https://arxiv.org/abs/2310.12670) — hierarchical async snapshotting and intra-node redundancy for near-zero checkpoint overhead on 512-GPU Llama-2-34B runs.
    - [Lian et al., *Universal Checkpointing: A Flexible and Efficient Distributed Checkpointing System for Large-Scale DNN Training* (2024)](https://arxiv.org/abs/2406.18820) — decouples checkpoint structure from parallelism topology, enabling resume on arbitrary GPU counts; basis for DeepSpeed Universal Checkpoint.
    - [Maurya et al., *DataStates-LLM: Lazy Asynchronous Checkpointing for Large Language Models* (2024)](https://arxiv.org/abs/2406.10707) — exploits tensor immutability between optimizer steps for up to 48× faster checkpointing with minimal training interference.
    - [Liang et al., *TorchTitan: One-stop PyTorch Native Solution for Production-Ready LLM Pre-Training* (2024)](https://arxiv.org/abs/2410.06511) — describes the full production checkpointing pipeline (DCP + async + local storage) achieving <2 s overhead on Llama-3 405B at 432 H200 GPUs.

    **Open-source & tools**

    - [pytorch/torchtitan](https://github.com/pytorch/torchtitan) — PyTorch's reference LLM pretraining platform; includes production-grade DCP checkpointing, async saves, and fault-tolerance hooks.
    - [meta-pytorch/torchft](https://github.com/meta-pytorch/torchft) — per-step fault tolerance primitives (HSDP, DiLoCo, LocalSGD) with a Lighthouse coordinator for zero-interruption recovery.

    **Go deeper**

    - [*Distributed Checkpoint: Efficient Checkpointing in Large-Scale Jobs* (PyTorch Blog, 2025)](https://pytorch.org/blog/distributed-checkpoint-efficient-checkpointing-in-large-scale-jobs/) — official write-up of DCP optimizations (process-based async, pinned-memory staging, local checkpointing) with measured badput results on H200 clusters.

## Further Reading

- **Young, J.C. (1974)** — "A first order approximation to the optimum checkpoint interval." *Communications of the ACM*. The original derivation of optimal checkpoint interval (Daly's formula).
- **Daly, J.T. (2006)** — "A higher order estimate of the optimum checkpoint interval for restart dumps." *Future Generation Computer Systems*. Extended and widely cited version.
- **PyTorch Distributed Checkpoint documentation** — `torch.distributed.checkpoint` in the official PyTorch docs; covers DCP's topology-agnostic format, `FileSystemWriter/Reader`, and async save APIs.
- **Zhao et al. (2023)** — "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel." *VLDB*. Describes FSDP state dict types and the engineering decisions behind sharded checkpointing.
- **Rajbhandari et al. (2020)** — "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." *SC'20*. Covers ZeRO optimizer state sharding, which directly informs checkpoint design in DeepSpeed.
- **DeepSpeed `checkpoint_engine`** — DeepSpeed's async checkpoint engine and its `AsyncTensorSwapper` are described in the DeepSpeed GitHub repository and blog posts.
- **Wang et al. (2023)** — "GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints." *SOSP 2023*. A landmark systems paper on in-memory checkpointing with neighbor-replica redundancy.
- **Barham et al. (2022)** — "Pathways: Asynchronous Distributed Dataflow for ML." *MLSys 2022*. Describes Google's approach to fault tolerance in large-scale ML infrastructure.
- **`torchdata.stateful_dataloader`** — PyTorch's `StatefulDataLoader`, the drop-in `DataLoader` replacement with worker-aware `state_dict()`/`load_state_dict()`; the correct answer to "how do I resume my data pipeline?"

---

## Exercises

**1.** A colleague hands you a resume script that reloads only the model weights from a checkpoint (`model.load_state_dict(...)`) and rebuilds a fresh `AdamW` optimizer and cosine scheduler from scratch each time the job restarts. The job does not crash on resume, and the code "works." Explain what goes wrong, why it is hard to notice, and list the additional pieces of state the chapter says a complete checkpoint must contain.

??? note "Solution"
    Rebuilding a fresh optimizer discards Adam's first- and second-moment estimates $m_t$ and $v_t$. On resume, the optimizer must re-accumulate momentum from zero, so for hundreds of steps the effective update direction and magnitude are wrong — in practice loss jumps up and then slowly recovers, which looks like a partial learning-rate-warmup restart. Rebuilding the scheduler from scratch makes it worse: the scheduler's step counter resets to 0, so the learning rate re-enters its warmup ramp instead of continuing along the cosine decay it had reached.

    It is hard to notice precisely because nothing crashes. The run resumes, produces plausible-looking output, and the loss is merely "higher than expected" for a while before converging back — there is no exception, no error, and after the fact the transient is easy to mistake for normal noise. This is the "optimizer state mismatch is silent" warning from the chapter.

    A complete checkpoint must contain all four categories the chapter lists:

    1. **Model parameters** — the weight tensors ($2P$ bytes in bf16).
    2. **Optimizer state** — Adam's $m_t$ and $v_t$ (plus fp32 master state), the piece that was dropped here.
    3. **RNG state** — the Python, NumPy, `torch` CPU, and per-device CUDA RNG states, saved per rank.
    4. **Training metadata** — global step, LR-scheduler state, and the data-loader cursor (shard index and offset).

    (If training in fp16 with dynamic loss scaling, the `GradScaler` state — scale factor and step counter — must also be saved; bf16 training needs no scaler.)

**2.** You are checkpointing a 13B-parameter model trained in bf16 with the AdamW optimizer state kept in fp32. Using the byte accounting from Section 3.12.1, compute (a) the size of the model-weights portion, (b) the size of the optimizer-state portion, and (c) the total checkpoint size. Then explain in one sentence why the optimizer state, not the weights, dominates the I/O cost of every save.

??? note "Solution"
    Let $P = 13 \times 10^9$ parameters.

    (a) Model weights in bf16 are $2P$ bytes:
    $$
    2 \times 13 \times 10^9 = 26 \times 10^9 \text{ bytes} = 26 \text{ GB}.
    $$

    (b) The chapter accounts Adam optimizer state at $16P$ bytes (fp32 master/moment state at $4 \times P \times 4$):
    $$
    16 \times 13 \times 10^9 = 208 \times 10^9 \text{ bytes} = 208 \text{ GB}.
    $$

    (c) Total (RNG and metadata are negligible, a few KB per rank):
    $$
    26 + 208 = 234 \text{ GB}.
    $$

    The optimizer state is $16P/2P = 8\times$ larger than the weights, so roughly $208/234 \approx 89\%$ of the bytes written every checkpoint are optimizer state. That is why a checkpoint that saves only weights ("sufficient for inference") is not just incorrect for resume but also misleadingly cheap: the expensive-to-write portion is exactly the part it omits.

**3.** You are running on a cluster of $N = 512$ nodes, each with an hourly failure rate $\lambda = 10^{-3}$ failures per node per hour. (a) Compute the expected cluster-level mean time between failures. (b) A synchronous checkpoint save takes $T_{\text{save}} = 2$ minutes. Use the chapter's optimal-interval (Daly/Young) formula to find the checkpoint interval $T^*$ that minimises wasted compute. (c) Interpret the result relative to the MTBF.

??? note "Solution"
    (a) Cluster failure rate is $N\lambda = 512 \times 10^{-3} = 0.512$ failures/hour, so
    $$
    \mathbb{E}[\text{MTBF}_{\text{cluster}}] = \frac{1}{N\lambda} = \frac{1}{0.512} \approx 1.953 \text{ hours} \approx 117 \text{ minutes}.
    $$

    (b) With $T_{\text{save}} = 2$ min and $T_{\text{MTBF}} \approx 117$ min, Daly's formula gives
    $$
    T^* \approx \sqrt{2 \cdot T_{\text{save}} \cdot T_{\text{MTBF}}} = \sqrt{2 \times 2 \times 117} = \sqrt{468} \approx 21.6 \text{ minutes}.
    $$

    (c) The optimal interval ($\approx 22$ min) is far shorter than the MTBF ($\approx 117$ min) — you should checkpoint roughly every 22 minutes, about 5 times per expected failure. The formula's message is that checkpoint spacing scales with the *geometric mean* of save cost and failure time, so as the cluster grows (MTBF shrinks) or saves get cheaper (async), the optimal interval tightens.

**4.** Continuing from Exercise 3's cluster but now with a harder failure regime, take $T_{\text{MTBF}} = 90$ minutes and a checkpoint interval of $T_{\text{ckpt}} = 30$ minutes of work. Using the chapter's waste-fraction approximation
$$
\text{waste} \approx \frac{T_{\text{ckpt}}/2 + T_{\text{save}}}{T_{\text{MTBF}}},
$$
compute the wasted-compute fraction (a) for synchronous checkpointing with $T_{\text{save}} = 8$ min, and (b) after switching to asynchronous checkpointing that reduces the hard blocking time to $T_{\text{save}} = 1$ min. (c) Which term dominates, and what does that tell you about where to spend engineering effort?

??? note "Solution"
    (a) Synchronous, $T_{\text{save}} = 8$ min:
    $$
    \text{waste} \approx \frac{30/2 + 8}{90} = \frac{15 + 8}{90} = \frac{23}{90} \approx 25.6\%.
    $$

    (b) Asynchronous, $T_{\text{save}} = 1$ min (only the GPU-to-CPU snapshot blocks; disk I/O overlaps training):
    $$
    \text{waste} \approx \frac{15 + 1}{90} = \frac{16}{90} \approx 17.8\%.
    $$

    (c) The dominant term is $T_{\text{ckpt}}/2 = 15$ min — the expected lost work since a failure lands, on average, halfway through the interval. Async checkpointing shrinks only the $T_{\text{save}}$ term ($8 \to 1$ min), buying about 8 percentage points. To attack the larger $T_{\text{ckpt}}/2$ term you must checkpoint *more frequently*, which is only affordable once $T_{\text{save}}$ is small — so async saves and shorter intervals are complementary: async makes the frequent-checkpoint regime that Daly's formula recommends practical.

**5.** The chapter's `CheckpointManager.latest()` finds the newest directory that contains a `COMPLETE` sentinel, but it never checks *content* integrity — a checkpoint whose `COMPLETE` file was written yet whose shard bytes were later corrupted on disk would still be selected. Modify `latest()` so it (a) also runs the chapter's `verify_checksums()` on each candidate and (b) falls back to the next-most-recent good checkpoint when a candidate is incomplete or fails verification, returning `None` only if no valid checkpoint exists. Keep the chapter's style.

??? note "Solution"
    We reuse `verify_checksums(ckpt_dir) -> bool` from Section 3.12.8 (it returns `False` when the `checksums.json` file is missing or any file's SHA-256 does not match). We iterate candidates newest-first and return the first that is both `COMPLETE` and checksum-verified. (This assumes `write_checksums()` is called at save time, e.g. inside `save_checkpoint` before the `COMPLETE` sentinel is touched.)

    ```python
    def latest(self) -> Path | None:
        """Most recent checkpoint that is COMPLETE *and* passes checksum
        verification. Skips corrupt/incomplete ones and falls back to the
        next-most-recent good checkpoint. Returns None if none are valid."""
        candidates = sorted(self.root_dir.glob("step_*"))
        for ckpt in reversed(candidates):  # newest first
            if not (ckpt / "COMPLETE").exists():
                continue  # partially written / crashed mid-save
            if not verify_checksums(ckpt):
                print(f"[ckpt] skipping corrupt checkpoint: {ckpt}")
                continue  # bytes rotted after COMPLETE was written
            return ckpt
        return None
    ```

    Notes consistent with the chapter's design: the `COMPLETE` check comes first because it is a cheap `stat` and screens out mid-save crashes, while `verify_checksums()` reads every file and is only worth paying for on a candidate that already claims to be complete. Because the loop continues past a bad candidate rather than returning, a single corrupt checkpoint costs at most one interval of extra rolled-back work instead of failing the resume — which is why the `CheckpointManager` keeps `keep_last_k > 1` older checkpoints available as fallbacks.

**6.** A run crashes on `world_size = 512` data-parallel ranks after consuming a global total of `global_token_offset = 8,388,608` tokens. It is restarted on `world_size = 256` ranks (nodes under repair) with `tokens_per_rank_per_step = seq_len * micro_batch_size = 2048 * 4 = 8192`. (a) Explain why you cannot simply restore each rank's *old* per-rank shard cursor. (b) Using the chapter's `compute_rank_start_offset`, compute the number of completed global steps and the starting offset for rank 0 and rank 255 in the new topology.

??? note "Solution"
    (a) Each rank's cursor records where *that* rank was reading in the *old* topology. When the number of data-parallel ranks changes, the dataset is partitioned differently: the tokens that used to belong to old-rank $r$ are now split across a different set of ranks, and there are only 256 ranks instead of 512. Restoring old per-rank cursors would make some data be skipped and other data be replayed. The chapter's fix is to track a single **global token offset** and recompute each new rank's starting position from it.

    (b) Tokens consumed per global step in the new topology:
    $$
    \text{tokens\_per\_step} = 8192 \times 256 = 2{,}097{,}152.
    $$
    Completed global steps:
    $$
    \text{steps\_done} = \left\lfloor \frac{8{,}388{,}608}{2{,}097{,}152} \right\rfloor = \lfloor 4.0 \rfloor = 4.
    $$
    Then `compute_rank_start_offset` returns `steps_done * tokens_per_rank_per_step + rank * tokens_per_rank_per_step`:

    - Rank 0: $4 \times 8192 + 0 \times 8192 = 32{,}768$.
    - Rank 255: $4 \times 8192 + 255 \times 8192 = 32{,}768 + 2{,}088{,}960 = 2{,}121{,}728$.

    So every rank resumes at the boundary of the 5th global step (steps 0-3 done), with rank $r$ offset a further $r \times 8192$ tokens into the stream — reconstructing a clean, non-overlapping partition of the data at the new world size purely from the global offset, without trusting any stale per-rank cursor.
