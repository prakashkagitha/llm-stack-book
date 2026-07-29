"""MinHash deduplication of the Stack-100M corpus with HuggingFace `datatrove`
(Ch. 14.2, "Production path").

This is the version you run over the real ~90GB corpus. The from-scratch
`stacklm.data.dedup` module exists to make MinHash+LSH legible; it is roughly
5 ms/document single-core, i.e. on the order of a core-day for 20M documents,
with no parallelism, no checkpointing, and greedy (order-dependent) clustering.
`datatrove` fixes all three: four stages with an on-disk hand-off, N tasks per
stage, and a global union-find over candidate pairs.

NOT part of the hermetic CI smoke test -- it needs `pip install datatrove[all]`,
multiple processes, and real input/output paths.

Usage:
    python3 capstone/scripts/dedup_datatrove.py            # local, TASKS processes
Swap `LocalPipelineExecutor` for `SlurmPipelineExecutor` to run the same DAG on
a cluster; nothing else changes.
"""
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import (
    MinhashDedupSignature, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupFilter,
)
from datatrove.pipeline.dedup.minhash import MinhashConfig
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig

# FineWeb's settings: 5-grams, 112 permutations as 14 buckets x 8 hashes.
# The band/row S-curve of (1/14)^(1/8) ~ 0.72 targets documents ~75%+ similar --
# the same knob analyzed in Ch. 14.2, just at production defaults.
cfg = MinhashConfig(
    hash_config=HashConfig(precision=64),
    num_buckets=14,
    hashes_per_bucket=8,
    n_grams=5,
)

IN, WORK, OUT = "s3://.../filtered", "/scratch/minhash", "/scratch/deduped"
TASKS = 64  # one task per CPU core; SlurmPipelineExecutor scales this to a cluster

stage1 = LocalPipelineExecutor(
    pipeline=[JsonlReader(IN),
              MinhashDedupSignature(output_folder=f"{WORK}/signatures", config=cfg)],
    tasks=TASKS, logging_dir=f"{WORK}/logs/sig")

stage2 = LocalPipelineExecutor(          # one task per bucket: buckets are independent
    pipeline=[MinhashDedupBuckets(input_folder=f"{WORK}/signatures",
                                  output_folder=f"{WORK}/buckets", config=cfg)],
    tasks=cfg.num_buckets, logging_dir=f"{WORK}/logs/buckets", depends=stage1)

stage3 = LocalPipelineExecutor(          # union-find over all candidate pairs: single task
    pipeline=[MinhashDedupCluster(input_folder=f"{WORK}/buckets",
                                  output_folder=f"{WORK}/remove_ids", config=cfg)],
    tasks=1, logging_dir=f"{WORK}/logs/cluster", depends=stage2)

stage4 = LocalPipelineExecutor(          # re-read the corpus, drop the flagged ids
    pipeline=[JsonlReader(IN),
              MinhashDedupFilter(input_folder=f"{WORK}/remove_ids"),
              JsonlWriter(OUT)],
    tasks=TASKS, logging_dir=f"{WORK}/logs/filter", depends=stage3)

if __name__ == "__main__":
    stage4.run()   # `depends` chains the whole DAG; running the last stage runs all four
