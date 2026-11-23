"""Analyze the track-B2 grids (phase2/b2-v2/plan.json) from the per-question scores.

    python b2_analyze.py synthetic --split dev        # or --split confirm
    python b2_analyze.py clotho

Reads runs/b2/*/scores-{split}.csv, or the gzipped copies under phase2/b2-v2/scores/ when runs/ is absent.
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ["audio", "gatedc", "route"]
SEEDS = [17, 18, 19]


def roots():
    return [Path("runs/b2"), Path("phase2/b2-v2/scores")]


def load(prefix, k, seed, arm, split):
    for root in roots():
        path = root / f"{prefix}-k{k}-s{seed}-{arm}" / f"scores-{split}.csv"
        if path.exists():
            return list(csv.DictReader(path.open(encoding="utf-8")))
        path = root / f"{prefix}-k{k}-s{seed}-{arm}-scores-{split}.csv.gz"
        if path.exists():
            return list(csv.DictReader(gzip.open(path, "rt", encoding="utf-8")))
    return None


def bootstrap(values, level, seed=17, replicates=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (replicates, len(values))).mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(values.mean()), float(np.percentile(draws, tail)), float(np.percentile(draws, 100 - tail))


def ranking(rows, condition):
    """Per recording (n, mixture): 1 if the present-category question outscores the absent one, orders averaged."""
    margins = defaultdict(list)
    for r in rows:
        if r["condition"] == condition:
            margins[int(r["n"]), int(r["mixture"]), r["label"]].append(float(r["yes_minus_no"]))
    keys = sorted({key[:2] for key in margins})
    return {key: float(np.mean(margins[(*key, "0")]) > np.mean(margins[(*key, "1")])) for key in keys}


def full_reference(split):
    path = {"dev": Path("phase2/b2-mixtures-v1/scores.csv"), "confirm": Path("phase2/b2-v2/confirm-control-scores.csv.gz")}[split]
    handle = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")
    rows = [dict(r, condition="correct") for r in csv.DictReader(handle) if r["model"] == "full"]
    return ranking(rows, "correct")


def synthetic(split):
    plan = json.loads(Path("phase2/b2-v2/plan.json").read_text())
    counts = plan["full_audio_gate"]["event_counts_used"]
    full = full_reference(split)
    table = {}
    for k in (1, 4, 16):
        for arm in ARMS:
            for condition in (["correct"] if arm == "audio" else ["correct", "wrong", "generic", "none"]):
                per_seed = [load("syn", k, s, arm, split) for s in SEEDS]
                if any(p is None for p in per_seed):
                    continue
                ranks = [ranking(p, condition) for p in per_seed]
                table[k, arm, condition] = {key: float(np.mean([r[key] for r in ranks])) for key in ranks[0]}
    out = {"split": split, "cells": {}, "contrasts": {}}
    print(f"within-recording ranking, {split}, mean of 3 seeds (full audio: "
          + ", ".join(f"N={n} {100*np.mean([v for (m, _), v in full.items() if m == n]):.1f}" for n in (1, 2, 4, 6)) + ")")
    for k in (1, 4, 16):
        for arm in ARMS:
            for condition in ("correct", "wrong", "generic", "none"):
                values = table.get((k, arm, condition))
                if values is None:
                    continue
                cells = {n: float(np.mean([v for (m, _), v in values.items() if m == n])) for n in (1, 2, 4, 6)}
                out["cells"][f"K{k}-{arm}-{condition}"] = cells
                print(f"K={k:<2} {arm:<6} {condition:<8}" + "".join(f"  N={n}: {100*v:5.1f}" for n, v in cells.items()))
    pressure = []
    for k in (1, 4, 16):
        audio = table.get((k, "audio", "correct"))
        if audio is None:
            continue
        for n in counts:
            keys = [key for key in audio if key[0] == n]
            gap = np.mean([full[key] - audio[key] for key in keys])
            if gap >= 0.05:
                pressure.append((k, n))
    out["pressure_cells_this_split"] = pressure
    if split == "confirm":
        # The plan fixes pressure cells on the development half; the confirmation reuses them.
        dev = json.loads(Path("phase2/b2-v2/synthetic-dev-summary.json").read_text())
        print("pressure cells recomputed on confirm (descriptive):", pressure)
        pressure = [tuple(c) for c in dev.get("pressure_cells_fixed", dev.get("pressure_cells"))]
    out["pressure_cells"] = out["pressure_cells_fixed"] = pressure
    print("pressure cells used (K, N):", pressure)
    for arm in ("gatedc", "route"):
        for name, (cond_a, arm_b, cond_b) in {"correct_minus_audio": ("correct", "audio", "correct"),
                                              "correct_minus_wrong": ("correct", arm, "wrong"),
                                              "correct_minus_none": ("correct", arm, "none")}.items():
            if not pressure or any((k, arm, cond_a) not in table or (k, arm_b, cond_b) not in table for k, _ in pressure):
                continue
            per_recording = defaultdict(list)
            for k, n in pressure:
                a, b = table[k, arm, cond_a], table[k, arm_b, cond_b]
                for key in a:
                    if key[0] == n:
                        per_recording[key].append(a[key] - b[key])
            values = [np.mean(v) for v in per_recording.values()]
            mean, low, high = bootstrap(values, 0.975)
            out["contrasts"][f"{arm}:{name}"] = [mean, low, high]
            print(f"{arm:<6} {name:<20} {100*mean:+6.2f} pts  97.5% CI [{100*low:+6.2f}, {100*high:+6.2f}]  ({len(values)} recordings)")
    print("secondary, per K over that K's pressure cells (95% CI, descriptive):")
    for k in (1, 4, 16):
        cells = [(kk, n) for kk, n in pressure if kk == k]
        for arm in ("gatedc", "route"):
            for name, (arm_b, cond_b) in {"correct_minus_audio": ("audio", "correct"),
                                          "correct_minus_wrong": (arm, "wrong")}.items():
                if not cells or (k, arm, "correct") not in table or (k, arm_b, cond_b) not in table:
                    continue
                a, b = table[k, arm, "correct"], table[k, arm_b, cond_b]
                values = [np.mean([a[key] - b[key] for kk, n in cells if key[0] == n]) for key in a
                          if any(key[0] == n for _, n in cells)]
                mean, low, high = bootstrap(values, 0.95)
                out["contrasts"][f"K{k}:{arm}:{name}"] = [mean, low, high]
                print(f"  K={k:<2} {arm:<6} {name:<20} {100*mean:+6.2f} pts  [{100*low:+6.2f}, {100*high:+6.2f}]  N in {[n for _, n in cells]}")
    Path("phase2/b2-v2").mkdir(parents=True, exist_ok=True)
    Path(f"phase2/b2-v2/synthetic-{split}-summary.json").write_text(json.dumps(out, indent=1))


def clotho():
    fresh = json.loads(Path("data/clotho/fresh_val.json").read_text())["val"]
    out = {}
    for k in (1, 4, 16):
        accuracy, hits = {}, {}
        for arm in ARMS:
            for condition in (["correct"] if arm == "audio" else ["correct", "wrong", "generic", "none"]):
                per_seed = [load("clo", k, s, arm, "dev") for s in SEEDS]
                if any(p is None for p in per_seed):
                    continue
                h = np.zeros(len(fresh))
                for rows in per_seed:
                    for r in rows:
                        if r["condition"] == condition and r["order"] == "0":
                            h[int(r["item"])] += (float(r["yes_minus_no"]) > 0) == (r["label"] == "0")
                hits[arm, condition] = h / len(per_seed)
                accuracy[f"{arm}-{condition}"] = float(h.sum() / len(per_seed))
        if not hits:
            continue
        print(f"\nK={k} fresh_val answers correct /340 (mean of seeds, original order A=yes):",
              ", ".join(f"{name} {v:.1f}" for name, v in accuracy.items()))
        clips = sorted({x["path"] for x in fresh})
        index = {c: i for i, c in enumerate(clips)}
        contrasts = {}
        for arm in ("gatedc", "route"):
            for name, (a, b) in {"correct_minus_audio": ((arm, "correct"), ("audio", "correct")),
                                 "correct_minus_wrong": ((arm, "correct"), (arm, "wrong")),
                                 "correct_minus_none": ((arm, "correct"), (arm, "none"))}.items():
                if a not in hits or b not in hits:
                    continue
                per_clip = defaultdict(list)
                for i, x in enumerate(fresh):
                    per_clip[index[x["path"]]].append(hits[a][i] - hits[b][i])
                mean, low, high = bootstrap([np.mean(v) for v in per_clip.values()], 0.95)
                contrasts[f"{arm}:{name}"] = [mean, low, high]
                print(f"  {arm:<6} {name:<20} {100*mean:+6.2f} pts  95% CI [{100*low:+6.2f}, {100*high:+6.2f}]")
        types = {(r["clip"], r["question"]): r["type"]
                 for r in csv.DictReader(open("phase2/fresh-capacity/question_types.csv", encoding="utf-8"))}
        by_type = defaultdict(dict)
        for name in sorted({types[x["path"], x["question"]] for x in fresh}):
            rows = [i for i, x in enumerate(fresh) if types[x["path"], x["question"]] == name]
            for (arm, condition), h in hits.items():
                if condition in ("correct", "wrong"):
                    by_type[name][f"{arm}-{condition}"] = float(h[rows].mean())
            print(f"  {name:<16} n={len(rows):<3}" + "".join(f" {key} {100*v:5.1f}" for key, v in by_type[name].items()))
        out[f"K{k}"] = {"accuracy": accuracy, "contrasts": contrasts, "by_type": by_type}
    Path("phase2/b2-v2/clotho-summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", choices=["synthetic", "clotho"])
    parser.add_argument("--split", default="dev", choices=["dev", "confirm"])
    args = parser.parse_args()
    synthetic(args.split) if args.data == "synthetic" else clotho()
