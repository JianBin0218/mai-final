#!/usr/bin/env bash
# Run from the prepared project directory; optional argument is a fresh output root.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Select one available GPU}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || exit 2
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
root=${1:-paired-rerun}
mkdir "$root"
py=.venv/bin/python
"$py" prepare_opposite.py
"$py" prepare_identity.py
"$py" prepare_counterbalanced.py
for spec in 'pairs opposite-v1' 'identity identity-v1' 'counterbalanced counterbalanced-v1'; do
    read -r manifest study <<< "$spec"
    dir="$root/$study"
    mkdir "$dir"
    cp "data/paired_evidence/$manifest.json" "$dir/pairs.json"
    "$py" -u bottleneck.py opposite --clotho --pair-manifest "$dir/pairs.json" --out "$dir" > "$dir/run.log" 2>&1
    "$py" analyze_opposite.py "$dir"
done
"$py" -u bottleneck.py level-probe --clotho --out "$root/level-probe-v1" > "$root/level-probe.log" 2>&1
