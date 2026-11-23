"""Summarize the track-B question-guidance grid (guidance_grid.sh).

For each capacity K and arm, report accuracy on the original 64 validation questions and on the
disjoint val_extra split, averaged over seeds. For question-conditioned arms, report paired contrasts
on the same trained compressor (correct stem minus another question about the same recording, minus a
generic stem, minus no question) and against the audio-only control warm-started from the same
parent. Intervals resample recordings, averaging each recording's difference over seeds first.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from random import Random
import statistics

ARMS = ["audio", "add", "gated", "gatedc", "gatedh", "gatedch"]
LABEL = {"audio": "audio-only control", "add": "question, original additive", "gated": "question, gated",
         "gatedc": "question, gated + centered", "gatedh": "question, gated, LM hidden",
         "gatedch": "question, gated + centered, LM hidden"}


def hits(part):
    return {(r["path"], r["question"]): int(r["prediction"] == r["target"]) for r in part["rows"]}


def contrast(pairs):
    """pairs: list over seeds of (hits_a, hits_b). Returns mean pp difference and recording bootstrap CI."""
    per_clip = defaultdict(list)
    for a, b in pairs:
        assert a.keys() == b.keys()
        grouped = defaultdict(list)
        for key in a:
            grouped[key[0]].append(a[key] - b[key])
        for clip, values in grouped.items():
            per_clip[clip].append(statistics.mean(values))
    changes = [statistics.mean(v) for v in per_clip.values()]
    rng = Random(17)
    n = len(changes)
    draws = sorted(sum(rng.choices(changes, k=n)) / n for _ in range(10000))
    return 100 * sum(changes) / n, 100 * draws[249], 100 * draws[9749]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="phase2/guidance", type=Path)
    parser.add_argument("--json", type=Path, help="Write the summary table to this file")
    args = parser.parse_args()
    runs = defaultdict(dict)
    for path in sorted(args.root.glob("k*-s*-*/train.json")):
        k, seed, arm = path.parent.name.split("-", 2)
        if arm in ARMS:
            runs[int(k[1:])][int(seed[1:]), arm] = json.loads(path.read_text())
    summary = []
    for k in sorted(runs):
        seeds = sorted({s for s, _ in runs[k]})
        arms = [arm for arm in ARMS if any((s, arm) in runs[k] for s in seeds)]
        complete = [s for s in seeds if all((s, arm) in runs[k] for arm in arms)]
        print(f"\n=== K={k}  seeds {complete}" + ("" if complete == seeds else f" (incomplete: {sorted(set(seeds)-set(complete))})"))
        if not complete:
            continue
        get = lambda arm, s: runs[k][s, arm]
        ref = get("audio", complete[0])
        full = statistics.mean(get("audio", s)["extra"]["full_audio"]["correct"] for s in complete)
        text = statistics.mean(get("audio", s)["extra"]["text_only"]["correct"] for s in complete)
        n_val, n_extra = ref["heldout"]["n"], ref["extra"]["heldout"]["n"]
        print(f"reference on val_extra (/{n_extra}): full audio {full:.1f}, text only {text:.1f}")
        for arm in arms:
            val = [get(arm, s)["heldout"]["correct"] for s in complete]
            extra = [get(arm, s)["extra"]["heldout"]["correct"] for s in complete]
            yes = statistics.mean(sum(r["prediction"] == 0 for r in get(arm, s)["extra"]["heldout"]["rows"]) / n_extra
                                  for s in complete)
            row = dict(k=k, arm=arm, seeds=complete, val=val, val_extra=extra, yes_rate_extra=yes)
            line = (f"{LABEL[arm]:<28} val {statistics.mean(val):5.1f}/{n_val} {val}  "
                    f"val_extra {statistics.mean(extra):6.1f}/{n_extra} {extra}  predicts yes {100*yes:4.1f}%")
            if arm != "audio":
                if "gate" in get(arm, complete[0]):
                    row["gate"] = [round(get(arm, s)["gate"], 3) for s in complete]
                    line += f"  tanh(gate) {row['gate']}"
                print(line)
                for name, other in [("vs audio control", None), ("vs wrong question", "wrong_question"),
                                    ("vs generic stem", "generic_question"), ("vs no question", "no_question")]:
                    if other is not None and other not in get(arm, complete[0])["extra"]:
                        continue
                    pairs = [(hits(get(arm, s)["extra"]["heldout"]),
                              hits(get("audio", s)["extra"]["heldout"] if other is None else get(arm, s)["extra"][other]))
                             for s in complete]
                    mean, low, high = contrast(pairs)
                    row[name] = dict(pp=mean, ci95=[low, high])
                    print(f"    correct stem {name:<18} {mean:+6.2f} pp  95% CI [{low:+6.2f}, {high:+6.2f}]")
            else:
                print(line)
            summary.append(row)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
