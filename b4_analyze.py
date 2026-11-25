"""Summarize b4_think.py scores: accuracy by task, and what each latent intervention does to the answers.

    python b4_analyze.py runs/b4/*/scores-mmau.csv

Accuracy is averaged over the two choice orders per item; 95% intervals bootstrap items (b2_analyze.bootstrap).
For each intervention: the paired change in accuracy against the same reader with normal latents, and the share of
(item, order) answers that change.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from b2_analyze import bootstrap


def summarize(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    hit, pred, task = defaultdict(dict), defaultdict(dict), {}
    for r in rows:
        key = (r["condition"], r["audio"])
        hit[key][int(r["item"]), r["order"]] = r["pred"] == r["answer"]
        pred[key][int(r["item"]), r["order"]] = r["pred"]
        task[int(r["item"])] = r["task"]
    items = sorted(task)
    tasks = sorted(set(task.values()))
    print(f"\n{path}  ({len(items)} items; accuracy % averaged over both choice orders)")
    print(f"{'condition':<10} audio {'all':>17} " + " ".join(f"{t[:10]:>10}" for t in tasks) + "   change vs normal (paired)   answers changed")
    per_item = {key: np.array([np.mean([h[i, o] for o in ("0", "1")]) for i in items]) for key, h in hit.items()}
    for key in sorted(per_item, key=lambda k: (k[1] != "1", k[0] != "normal", k[0])):
        values = per_item[key]
        mean, low, high = bootstrap(values, 0.95)
        by_task = " ".join(f"{100*np.mean([v for i, v in zip(items, values) if task[i] == t]):10.1f}" for t in tasks)
        line = f"{key[0]:<10} {key[1]:>5} {100*mean:5.1f} [{100*low:4.1f},{100*high:4.1f}] {by_task}"
        base = ("normal", key[1])
        if key[0] != "normal" and base in per_item:
            d, dl, dh = bootstrap(values - per_item[base], 0.95)
            changed = np.mean([pred[key][k] != pred[base][k] for k in pred[key]])
            line += f"   {100*d:+5.1f} [{100*dl:+5.1f},{100*dh:+5.1f}]   {100*changed:5.1f}%"
        elif key == ("normal", "0") and ("normal", "1") in per_item:
            d, dl, dh = bootstrap(values - per_item["normal", "1"], 0.95)
            line += f"   {100*d:+5.1f} [{100*dl:+5.1f},{100*dh:+5.1f}] vs audio"
        print(line)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        if Path(path).exists():
            summarize(path)
