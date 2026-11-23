#!/usr/bin/env bash
# Track B2 grid. Usage: b2_grid.sh GPU DATA STEPS "K..." "SEED..." "ARM..."
# DATA is synthetic or clotho. Runs land in runs/b2/{syn,clo}-k{K}-s{SEED}-{ARM}; finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")"
gpu=$1; data=$2; steps=$3; ks=$4; seeds=$5; arms=$6
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
py=${MAI_PYTHON:-.venv/bin/python}
prefix=$([[ $data == synthetic ]] && echo syn || echo clo)
mkdir -p runs/b2
for k in $ks; do for seed in $seeds; do for arm in $arms; do
    out=runs/b2/$prefix-k$k-s$seed-$arm
    [[ -f $out/scores-dev.csv ]] && continue
    if [[ -e $out ]]; then echo "Incomplete run at $out; move it before retrying." >&2; exit 1; fi
    echo "$(date +%T) start $out" >&2
    "$py" -u b2_train.py train --data "$data" --arm "$arm" --k "$k" --seed "$seed" --steps "$steps" --out "$out" > "$out.log" 2>&1
done; done; done
