#!/usr/bin/env bash
# Question-guidance grid (track B). For each K and seed: train an audio-only parent, then warm-start
# four arms from it for the same number of steps: audio-only control, the original additive question
# term, a bounded gated term, and a gated term on centered question embeddings. All arms evaluate
# the original 64 validation questions and the disjoint val_extra split.
#
# Usage: guidance_grid.sh GPU "K..." "SEED..." [OUT_ROOT] [TRAIN_SPLIT] [STEPS] [ARMS]
# ARMS defaults to "audio add gated gatedc"; gatedh / gatedch use the LM's layer-16 hidden state
# instead of mean token embeddings for the question.
# Completed runs (train.json present) are skipped, so the script can be resumed.
# MODE=eval rescores finished arms on data/clotho/fresh_val.json (split val, shared with track A)
# and writes eval.json beside train.json; parents are not rescored.
set -euo pipefail
cd "$(dirname "$0")"
mode=${MODE:-train}
gpu=$1; ks=$2; seeds=$3; root=${4:-runs/guidance}; split=${5:-train}; steps=${6:-192}; arms=${7:-audio add gated gatedc}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
py=${MAI_PYTHON:-.venv/bin/python}
common=(--clotho --steps "$steps" --manifest data/clotho/manifest_extended.json --train-split "$split" --feature-cache cache/features)
mkdir -p "$root"
run() {
    local out=$1; shift
    if [[ $mode == eval ]]; then
        [[ -f $out/eval.json || $out == *-parent ]] && return
        [[ -f $out/compressor.pt ]] || { echo "Missing checkpoint: $out/compressor.pt" >&2; return 1; }
        echo "$(date +%T) eval $out" >&2
        "$py" -u bottleneck.py eval --out "$out" "$@" > "$out.eval.log" 2>&1
        return
    fi
    [[ -f $out/train.json ]] && return
    if [[ -e $out ]]; then
        echo "Incomplete run at $out; preserve it and choose a fresh output root or move it before retrying." >&2
        return 1
    fi
    echo "$(date +%T) start $out" >&2
    "$py" -u bottleneck.py train --out "$out" "$@" > "$out.log" 2>&1
}
for k in $ks; do
    for seed in $seeds; do
        base=$root/k$k-s$seed
        run "$base-parent" "${common[@]}" --k "$k" --seed "$seed"
        warm=("${common[@]}" --k "$k" --seed "$seed")
        if [[ $mode == eval ]]; then
            warm+=(--extra-manifest data/clotho/fresh_val.json --extra-split val)
        else
            warm+=(--init-from "$base-parent/compressor.pt" --extra-split val_extra)
        fi
        for arm in $arms; do
            case $arm in
                audio)   run "$base-audio" "${warm[@]}" ;;
                add)     run "$base-add" "${warm[@]}" --conditioned ;;
                gated)   run "$base-gated" "${warm[@]}" --conditioned --condition gated ;;
                gatedc)  run "$base-gatedc" "${warm[@]}" --conditioned --condition gated --center-question ;;
                gatedh)  run "$base-gatedh" "${warm[@]}" --conditioned --condition gated --question-repr hidden ;;
                gatedch) run "$base-gatedch" "${warm[@]}" --conditioned --condition gated --center-question --question-repr hidden ;;
                *) echo "unknown arm $arm" >&2; exit 2 ;;
            esac
        done
    done
done
