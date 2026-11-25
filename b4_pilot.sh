#!/bin/bash
# Track B4 pilot (seed 17, K=4, 3000 steps on AVQA-R1-6K train; scored on AVQA dev and MMAU test-mini).
# usage: bash b4_pilot.sh GPU "name:flags" ...
set -e
mkdir -p logs
gpu=$1; shift
for spec in "$@"; do
  name=${spec%%:*}; flags=${spec#*:}
  [ -f runs/b4/$name/scores-mmau.csv ] && continue
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python b4_think.py train $flags --seed 17 --steps 3000 --eval avqa-dev mmau \
    --out runs/b4/$name > logs/b4-$name.log 2>&1
done
