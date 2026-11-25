"""Score B2 external compressors and B3 writers on track B's fixed-stem paired recordings (writer_pairs.py).

Each pair asks one question about two recordings that differ only in the target event; the primary metric is the
within-pair ranking (the yes-recording's yes-minus-no margin above the no-recording's), both option orders averaged,
95% bootstrap over pairs. Same prompt format as writer_pairs.measure on codex/ab-readout-followup.

    python b3_pairs.py --pairs runs/b3-pairs/stimuli/pairs.json --out runs/b3-pairs/scores.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from b2_analyze import bootstrap
from b2_mixtures import SUFFIX, Backbone
from b2_train import Arm as External
from b3_latent import Arm as Internal


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, default=Path("cache/features"))
    args = parser.parse_args()
    pairs = json.loads(args.pairs.read_text())["pairs"]
    model = Backbone(args.feature_cache)
    model.lm.eval()
    memory = {}

    def features(path):
        if path not in memory:
            memory[path] = model.features(path)
        return memory[path]

    arms = {"full": lambda audio, pair: [(None, audio)], "text": lambda audio, pair: [(None, None)]}
    for data in ("clo", "syn"):
        for seed in (17, 18, 19):
            path = Path(f"runs/b2/{data}-k4-s{seed}-audio/compressor.pt")
            if path.exists():
                ext = External(model, "audio", 4, set())
                ext.comp.load_state_dict(torch.load(path, weights_only=True))
                ext.comp.eval()
                arms[f"b2-{data}-k4-s{seed}-audio"] = lambda audio, pair, ext=ext: [(None, ext.latent(audio, None))]
    for run in sorted(Path("runs/b3-pilot").glob("*/state.pt")):
        config = json.loads((run.parent / "train.json").read_text())["config"]
        arm = Internal(model, config["arm"], config["k"])
        arm.load(torch.load(run, weights_only=True))
        if arm.blind:
            arms[run.parent.name] = lambda audio, pair, arm=arm: [(None, arm.latent(audio, None))]
        else:
            arms[run.parent.name] = lambda audio, pair, arm=arm: [(c, arm.latent(audio, s)) for c, s in
                                                                  (("correct", pair["stem"]), ("wrong", pair["wrong_stem"]), ("closed", None))]
    rows = []
    with torch.no_grad():
        for name, make in arms.items():
            for i, pair in enumerate(pairs):
                for clip in ("yes", "no"):
                    for condition, latent in make(features(pair[clip]), pair):
                        for order, suffix in SUFFIX.items():
                            logits = model.decode(latent, pair["stem"] + suffix)
                            margin = float(logits[0] - logits[1]) * (1 if order == 0 else -1)
                            rows.append(dict(model=name, condition=condition or "", pair=i, clip=clip, order=order, yes_minus_no=margin))
            print("scored", name, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report(rows, len(pairs))


def report(rows, n):
    margin = {(r["model"], r["condition"], r["pair"], r["clip"], r["order"]): r["yes_minus_no"] for r in rows}
    keys = sorted({(r["model"], r["condition"]) for r in rows}, key=lambda k: (k[0] != "full", k))
    ranks = {}
    for m, c in keys:
        per_order = np.array([[float(margin[m, c, i, "yes", o] > margin[m, c, i, "no", o]) + .5 * (margin[m, c, i, "yes", o] == margin[m, c, i, "no", o])
                               for o in (0, 1)] for i in range(n)])
        acc = np.mean([(margin[m, c, i, clip, o] > 0) == (clip == "yes") for i in range(n) for clip in ("yes", "no") for o in (0, 1)])
        ranks[m, c] = per_order.mean(1)
        mean, low, high = bootstrap(ranks[m, c], 0.95)
        print(f"{m:<32} {c:<8} ranking A/B order {100*per_order[:,0].mean():5.1f} / {100*per_order[:,1].mean():5.1f}"
              f"  mean {100*mean:5.1f} [{100*low:5.1f}, {100*high:5.1f}]  accuracy {100*acc:5.1f}")
    print("contrasts (95% over pairs):")
    for (m, c), r in ranks.items():
        for ref in ("b2-clo-k4-s17-audio", "b2-syn-k4-s17-audio"):
            if m.startswith("clo-" if "clo" in ref else "syn-") and (ref, "") in ranks and "k4" in m:
                mean, low, high = bootstrap(r - ranks[ref, ""], 0.95)
                print(f"  {m} {c} - {ref}: {100*mean:+5.1f} [{100*low:+5.1f}, {100*high:+5.1f}]")
        if c == "correct":
            for other in ("wrong", "closed"):
                mean, low, high = bootstrap(r - ranks[m, other], 0.95)
                print(f"  {m} correct - {other}: {100*mean:+5.1f} [{100*low:+5.1f}, {100*high:+5.1f}]")


if __name__ == "__main__":
    main()
