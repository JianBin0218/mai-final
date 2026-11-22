"""Join fixed question annotations to matched predictions from each seed."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parent / "phase2"


def predictions(k, seed):
    run = f"capacity-k{k}-s{seed}-audio"
    if k == 16 and seed == 17:
        run = "clotho-k16-s17-audio"
    result = json.loads((ROOT / run / "train.json").read_text(encoding="utf-8"))
    assert (result["k"], result["seed"]) == (k, seed)
    scores = {}
    for r in result["heldout"]["rows"]:
        key = r["path"], r["question"].splitlines()[0]
        assert key not in scores
        scores[key] = (r["target"], int(r["prediction"] == r["target"]))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[17,23,41])
    seeds = parser.parse_args().seeds
    if len(set(seeds)) != len(seeds):
        parser.error("Seeds must be distinct")
    with (ROOT / "capacity_question_types.csv").open(encoding="utf-8", newline="") as f:
        types = {(r["clip"],r["question"]):r["type"] for r in csv.DictReader(f)}
    deltas = defaultdict(list)
    counts = Counter(types.values())
    by_type = defaultdict(list)
    for seed in seeds:
        a,b = predictions(4,seed), predictions(16,seed)
        assert a.keys() == b.keys() == types.keys()
        totals = Counter()
        for key, typ in types.items():
            assert a[key][0] == b[key][0]
            d = b[key][1]-a[key][1]
            totals[typ] += d
            deltas[key].append(d)
        print(seed, "K4", sum(v[1] for v in a.values()), "K16",sum(v[1] for v in b.values()),"of",len(types))
        for target, name in enumerate(("yes","no")):
            subset = [key for key in types if a[key][0] == target]
            print(" ",name, "K4",sum(a[key][1] for key in subset),"K16",sum(b[key][1] for key in subset),"of",len(subset))
        for typ in counts:
            by_type[typ].append(totals[typ])
    for typ,n in sorted(counts.items()):
        print(typ, "n",n,"net gains by seed",by_type[typ])
    # Repeated seeds share the same recordings. Resample recordings as whole clusters.
    by_clip = defaultdict(list)
    for key, runs in deltas.items():
        by_clip[key[0]].extend(runs)
    clip_diffs = [sum(vals)/len(vals) for vals in by_clip.values()]
    rng = Random(17)
    n = len(clip_diffs)
    draws = sorted(100*sum(rng.choices(clip_diffs,k=n))/n for _ in range(10000))
    print("Mean paired gain (pp):",100*sum(clip_diffs)/n, "clip bootstrap 95%:",draws[249],draws[9749])
    for sign in (1,-1):
        stable = [key for key,d in deltas.items() if all(v==sign for v in d)]
        print("All seeds favor", "K16" if sign==1 else "K4",len(stable))
        for key in stable:
            print(" ",types[key],key[1])


if __name__ == "__main__":
    main()
