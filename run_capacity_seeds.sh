#!/usr/bin/env bash
# Run matched capacity trials sequentially from the prepared cml30 project.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Select one physical GPU first}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || { echo "Select one GPU" >&2; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
entry=${CAPACITY_ENTRY:-bottleneck.py}
if (($# == 0)); then set -- 23 41; fi
for seed in "$@"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || exit 2
  for k in 4 16; do
    out="capacity-k${k}-s${seed}-audio"
    if [[ -s "$out/train.json" && -s "$out/compressor.pt" ]]; then continue; fi
    .venv/bin/python -u "$entry" train --clotho --k "$k" --steps 192 --seed "$seed" --out "$out" > "$out.log" 2>&1
  done
done
