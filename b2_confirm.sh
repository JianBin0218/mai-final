#!/usr/bin/env bash
# Track B2.3 confirmation: score full audio and every finished synthetic checkpoint on the fold 2 confirmation half.
# Usage: b2_confirm.sh GPU "K..."   (run after b2_mixtures.py prepare --confirmation --out data/b2mix-confirm)
set -euo pipefail
cd "$(dirname "$0")"
gpu=$1; ks=$2
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
py=${MAI_PYTHON:-.venv/bin/python}
if [[ $ks == *full* ]]; then
    [[ -f runs/b2-confirm-control/scores.csv ]] || "$py" -u b2_mixtures.py control --plan phase2/b2-mixtures-v1/plan.json \
        --manifest data/b2mix-confirm/manifest.json --feature-cache cache/features --scores runs/b2-confirm-control/scores.csv
    ks=${ks//full/}
fi
for k in $ks; do for seed in 17 18 19; do for arm in audio gatedc route; do
    out=runs/b2/syn-k$k-s$seed-$arm
    [[ -f $out/scores-confirm.csv ]] && continue
    echo "$(date +%T) confirm $out" >&2
    "$py" -u b2_train.py score --data synthetic --split confirm --out "$out" >> "$out.confirm.log" 2>&1
done; done; done
