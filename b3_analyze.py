"""Compare track-B3 runs (the LLM writes the latents) with the B2 external compressors on the same items.

    python b3_analyze.py synthetic --runs runs/b3 [--split dev]
    python b3_analyze.py clotho --runs runs/b3

Synthetic: within-recording ranking (b2_analyze.ranking), mean over the seeds present for each (K, arm).
Clotho: fresh_val answers correct out of 340, original option order, mean over seeds.
External B2 runs come from runs/b2 or the gzipped copies in phase2/b2-v2/scores.
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from b2_analyze import bootstrap, full_reference, load, ranking

CONDITIONS = ["correct", "wrong", "generic", "none"]


def b3_runs(root, prefix, split):
    runs = defaultdict(dict)
    for path in sorted(Path(root).glob(f"{prefix}-k*-s*-*/scores-{split}.csv")):
        k, seed, arm = re.match(rf"{prefix}-k(\d+)-s(\d+)-(.+)", path.parent.name).groups()
        runs[int(k), arm][int(seed)] = list(csv.DictReader(path.open(encoding="utf-8")))
    return runs


def b2_runs(prefix, split, keys):
    runs = defaultdict(dict)
    for k in sorted({k for k, _ in keys}):
        for arm in ("audio", "gatedc"):
            for seed in (17, 18, 19):
                rows = load(prefix, k, seed, arm, split)
                if rows is not None:
                    runs[k, f"b2-{arm}"][seed] = rows
    return runs


def synthetic(root, split):
    runs = b3_runs(root, "syn", split)
    runs.update(b2_runs("syn", split, runs))
    full = full_reference(split)
    print(f"within-recording ranking ({split}); full audio:",
          ", ".join(f"N={n} {100*np.mean([v for (m, _), v in full.items() if m == n]):.1f}" for n in (1, 2, 4, 6)))
    tables = {}
    for (k, arm), seeds in sorted(runs.items()):
        for condition in CONDITIONS:
            ranks = [ranking(rows, condition) for rows in seeds.values()]
            if not ranks or not ranks[0]:
                continue
            table = {key: float(np.mean([r[key] for r in ranks])) for key in ranks[0]}
            tables[k, arm, condition] = table
            cells = [100 * np.mean([v for (m, _), v in table.items() if m == n]) for n in (1, 2, 4, 6)]
            print(f"K={k:<2} {arm:<14} {condition:<8} seeds {sorted(seeds)}" + "".join(f"  N={n}: {c:5.1f}" for n, c in zip((1, 2, 4, 6), cells)))
    print("contrasts over N in 1, 2, 4 (95% CI over recordings, descriptive):")
    for (k, arm, condition), table in sorted(tables.items()):
        if condition != "correct" or arm.startswith("b2"):
            continue
        for name, other in {"correct - none": (k, arm, "none"), "correct - wrong": (k, arm, "wrong"),
                            "correct - B2 audio": (k, "b2-audio", "correct"),
                            "correct - B2 gatedc": (k, "b2-gatedc", "correct")}.items():
            if other not in tables:
                continue
            values = [table[key] - tables[other][key] for key in table if key[0] in (1, 2, 4)]
            mean, low, high = bootstrap(values, 0.95)
            print(f"  K={k:<2} {arm:<14} {name:<20} {100*mean:+6.2f} [{100*low:+6.2f}, {100*high:+6.2f}]")


def clotho(root):
    runs = b3_runs(root, "clo", "dev")
    runs.update(b2_runs("clo", "dev", runs))
    for (k, arm), seeds in sorted(runs.items()):
        accuracy = {}
        for condition in CONDITIONS:
            hits = [sum((float(r["yes_minus_no"]) > 0) == (r["label"] == "0")
                        for r in rows if r["condition"] == condition and r["order"] == "0") for rows in seeds.values()]
            if any(hits):
                accuracy[condition] = np.mean(hits)
        print(f"K={k:<2} {arm:<14} seeds {sorted(seeds)} /340: " + ", ".join(f"{c} {v:.1f}" for c, v in accuracy.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", choices=["synthetic", "clotho"])
    parser.add_argument("--runs", default="runs/b3")
    parser.add_argument("--split", default="dev", choices=["dev", "confirm"])
    args = parser.parse_args()
    synthetic(args.runs, args.split) if args.data == "synthetic" else clotho(args.runs)
