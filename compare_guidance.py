"""Summarize the track-B question-guidance grid from phase2/guidance-v1 (see export_guidance.py).

For each capacity K and arm, report accuracy averaged over seeds. For question-conditioned arms, report
paired contrasts on the same trained compressor (correct stem minus another question about the same
recording, minus a generic stem, minus no question) and against the audio-only control warm-started from
the same parent. Intervals resample recordings, averaging each recording's difference over seeds first.

    python compare_guidance.py                      # 96 training questions, fresh_val (shared with track A)
    python compare_guidance.py --grid large         # 768 training questions
    python compare_guidance.py --split val          # the 64 pilot questions
    python compare_guidance.py --probe              # fixed question-term magnitudes
"""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
from random import Random
import statistics

ROOT = Path(__file__).resolve().parent / "phase2" / "guidance-v1"
ARMS = ["audio", "add", "gated", "gatedc", "gatedh", "gatedch"]
LABEL = {"audio": "audio-only control", "add": "question, original additive", "gated": "question, gated",
         "gatedc": "question, gated + centered", "gatedh": "question, gated, LM hidden",
         "gatedch": "question, gated + centered, LM hidden"}


def contrast(pairs, clip_of):
    """pairs: list over seeds of (hits_a, hits_b) keyed by qid. Mean pp difference and recording bootstrap CI."""
    per_clip = defaultdict(list)
    for a, b in pairs:
        assert a.keys() == b.keys()
        grouped = defaultdict(list)
        for q in a:
            grouped[clip_of[q]].append(a[q] - b[q])
        for clip, values in grouped.items():
            per_clip[clip].append(statistics.mean(values))
    changes = [statistics.mean(v) for v in per_clip.values()]
    rng = Random(17)
    n = len(changes)
    draws = sorted(sum(rng.choices(changes, k=n)) / n for _ in range(10000))
    return 100 * sum(changes) / n, 100 * draws[249], 100 * draws[9749]


def probe_table():
    rows = list(csv.DictReader((ROOT / "probe.csv").open()))
    for split in ("fresh_val", "val_extra"):
        print(f"\n### fixed question-term norm, gated + centered arm, {split} (mean of seeds)")
        for k in ("1", "4", "16"):
            sel = [r for r in rows if r["k"] == k and r["split"] == split]
            print(f"K={k}: magnitude/|query|  correct stem  wrong stem  predicts yes")
            for m in sorted({float(r["magnitude"]) for r in sel}):
                g = [r for r in sel if float(r["magnitude"]) == m]
                mean = lambda f: statistics.mean(float(r[f]) for r in g)
                print(f"   {mean('ratio_to_query'):6.2f}  {mean('correct_stem'):7.1f}/{g[0]['n']}  "
                      f"{mean('wrong_stem'):7.1f}  {100*mean('yes_rate'):5.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=["small", "large"], default="small")
    parser.add_argument("--split", choices=["fresh_val", "val"], default="fresh_val")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        return probe_table()
    questions = list(csv.DictReader((ROOT / "questions.csv").open()))
    clip_of = {int(q["qid"]): q["path"] for q in questions if q["split"] == args.split}
    targets = {int(q["qid"]): int(q["target"]) for q in questions}
    hits = defaultdict(dict)  # (k, seed, arm, condition) -> qid -> hit
    yes = defaultdict(int)
    for r in csv.DictReader((ROOT / "predictions.csv").open()):
        if r["grid"] == args.grid and r["split"] == args.split:
            q = int(r["qid"])
            key = (int(r["k"]), int(r["seed"]), r["arm"], r["condition"])
            hits[key][q] = int(int(r["prediction"]) == targets[q])
            yes[key] += int(r["prediction"]) == 0
    gates = {(int(r["k"]), int(r["seed"]), r["arm"]): r["gate"]
             for r in csv.DictReader((ROOT / "runs.csv").open()) if r["grid"] == args.grid}
    n = len(clip_of)
    for k in sorted({key[0] for key in hits}):
        seeds = sorted({key[1] for key in hits if key[0] == k})
        arms = [a for a in ARMS if any((k, s, a, "heldout") in hits for s in seeds)]
        print(f"\n=== K={k}  seeds {seeds}  {args.split} ({n} questions)  training grid: {args.grid}")
        for ref in ("full_audio", "text_only"):
            print(f"reference {ref}: {statistics.mean(sum(hits[k, s, 'audio', ref].values()) for s in seeds):.1f}/{n}")
        for arm in arms:
            correct = [sum(hits[k, s, arm, "heldout"].values()) for s in seeds]
            yes_rate = statistics.mean(yes[k, s, arm, "heldout"] for s in seeds) / n
            line = f"{LABEL[arm]:<38} {statistics.mean(correct):6.1f}/{n} {correct}  predicts yes {100*yes_rate:4.1f}%"
            if gates.get((k, seeds[0], arm)):
                line += f"  tanh(gate) {[float(gates[k, s, arm]) for s in seeds]}"
            print(line)
            if arm == "audio":
                continue
            for name, other in [("vs audio control", None), ("vs wrong question", "wrong_question"),
                                ("vs generic stem", "generic_question"), ("vs no question", "no_question")]:
                if other and (k, seeds[0], arm, other) not in hits:
                    continue
                pairs = [(hits[k, s, arm, "heldout"], hits[k, s, "audio" if other is None else arm, other or "heldout"])
                         for s in seeds]
                mean, low, high = contrast(pairs, clip_of)
                print(f"    correct stem {name:<18} {mean:+6.2f} pp  95% CI [{low:+6.2f}, {high:+6.2f}]")


if __name__ == "__main__":
    main()
