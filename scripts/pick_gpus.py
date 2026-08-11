#!/usr/bin/env python3
"""Good-neighbor GPU picker for a SHARED box: print N free GPU indices (default 2), or exit 1
if fewer than N are free. 'Free' = memory.used < THRESH MiB and utilization < 10%.
Usage:  export CUDA_VISIBLE_DEVICES=$(python3 scripts/pick_gpus.py 2) || echo "defer"
"""
import subprocess, sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
THRESH = 2000
out = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
     '--format=csv,noheader,nounits']).decode()
free = []
for line in out.strip().splitlines():
    idx, mem, util = [x.strip() for x in line.split(',')]
    if int(mem) < THRESH and int(util) < 10:
        free.append(idx)
if len(free) < N:
    sys.stderr.write(f"only {len(free)} GPU(s) free ({free}); need {N}. Deferring.\n")
    sys.exit(1)
print(",".join(free[:N]))
