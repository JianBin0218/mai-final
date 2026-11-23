"""Compare native audio controls and fixed-capacity models on matched questions."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "phase2"
CAP = ROOT / "fresh-capacity"
CTRL = ROOT / "fresh-controls"


def main():
    with (CAP / "question_types.csv").open(encoding="utf-8",newline="") as f:
        labels = {(r["clip"],r["question"]):r["type"] for r in csv.DictReader(f)}
    keys = sorted(labels)
    targets = {}
    donors = json.loads((CTRL/"donors.json").read_text())
    clips = sorted({clip for clip,_ in keys})
    assert set(donors) == set(donors.values()) == set(clips) and all(c != donors[c] for c in clips)
    def load(path):
        rows = json.loads(path.read_text())["rows"]
        scores = {}
        for r in rows:
            key = r["path"],r["question"].splitlines()[0]
            assert key not in scores and targets.setdefault(key,r["target"]) == r["target"]
            scores[key] = int(r["prediction"]==r["target"])
        assert len(rows) == 340 and scores.keys() == labels.keys()
        return np.array([scores[key] for key in keys])
    models = {name:load(CTRL/(name+".json")) for name in ("full","text","unrelated")}
    for k in (4,16):
        runs = [load(CAP/("clotho-k16-s17-audio.json" if (k,s)==(16,17) else f"capacity-k{k}-s{s}-audio.json")) for s in (17,23,41)]
        models[f"K{k}"] = np.mean(runs,axis=0)
    target = np.array([targets[key] for key in keys])
    groups = ["all"] + sorted(set(labels.values()))
    rng = np.random.default_rng(17)
    intervals = []
    metrics = []
    for group in groups:
        mask = np.array([group=="all" or labels[key]==group for key in keys])
        for name,scores in models.items():
            yes = scores[mask & (target==0)].mean()
            no = scores[mask & (target==1)].mean()
            metrics.append(dict(group=group,method=name,n=int(mask.sum()),yes_n=int(sum(mask & (target==0))),
                                accuracy=round(100*scores[mask].mean(),2),yes_recall=round(100*yes,2),
                                no_recall=round(100*no,2),balanced_accuracy=round(50*(yes+no),2)))
        selected = [i for i in range(len(keys)) if mask[i]]
        _,clusters = np.unique([keys[i][0] for i in selected],return_inverse=True)
        n=len(set(clusters))
        weights=rng.multinomial(n,np.ones(n)/n,size=10000)
        den=weights @ np.bincount(clusters)
        for a,b in [("full","text"),("full","unrelated"),("full","K16"),("K16","K4")]:
            delta=(models[a]-models[b])[selected]
            boot=100*(weights @ np.bincount(clusters,weights=delta))/den
            lo,hi=np.quantile(boot,[.025,.975])
            intervals.append(dict(group=group,contrast=a+" minus "+b,gain_pp=round(100*delta.mean(),2),
                                  low=round(lo,2),high=round(hi,2)))
    for name,rows in [("metrics",metrics),("contrasts",intervals)]:
        with (CTRL/(name+".csv")).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=="__main__":
    main()
