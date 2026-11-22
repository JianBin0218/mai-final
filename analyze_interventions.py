"""Aggregate persisted latent interventions while keeping seeds paired by question."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent / "phase2"
out = ROOT / "interventions-v1"
rows = []
for f in sorted(out.glob("*.csv")):
    if "-d" not in f.stem:
        continue
    with f.open(encoding="utf-8", newline="") as handle:
        for r in csv.DictReader(handle):
            r["donor_seed"] = int(f.stem.split("-d")[-1])
            rows.append(r)
scores = defaultdict(dict)
donors = {}
for r in rows:
    seed,k,d = int(r["seed"]),int(r["k"]),r["donor_seed"]
    key = r["clip"],r["question"],int(r["target"])
    assert r["clip"]!=r["donor"]
    assert donors.setdefault((d,r["clip"]),r["donor"])==r["donor"]
    item = scores[d,k,r["condition"]].setdefault(key,{})
    assert seed not in item
    item[seed] = int(r["prediction"])==key[2]
def rate(v):
    return float(np.mean(list(v.values())))


def interval(differences):
    groups = defaultdict(list)
    for key,value in differences.items():
        groups[key[0]].append(value)
    groups = list(groups.values())
    sums, sizes = np.array([sum(v) for v in groups]), np.array([len(v) for v in groups])
    n = len(groups)
    weights = np.random.default_rng(17).multinomial(n,np.ones(n)/n,size=10000)
    return 100*sums.sum()/sizes.sum(), *np.quantile(100*(weights@sums)/(weights@sizes),[.025,.975])


def summarize():
    records = []
    contrasts = []
    donor_seeds = sorted({d for d,_,_ in scores})
    for k in (4,16):
        base = scores[donor_seeds[0],k,"own"]
        for seed in (17,23,41):
            folder = "clotho-k16-s17-audio" if (k,seed)==(16,17) else f"capacity-k{k}-s{seed}-audio"
            old = json.loads((ROOT/"fresh-capacity"/(folder+".json")).read_text())["rows"]
            hits = {(r["path"],r["question"].splitlines()[0],r["target"]):r["target"]==r["prediction"] for r in old}
            assert all(hits[key]==runs[seed] for key,runs in base.items())
        for d in donor_seeds:
            assert scores[d,k,"own"]==base
            for cond in ("own","donor","own_pool_donor_residual","donor_pool_own_residual"):
                part=scores[d,k,cond]
                assert len(part)==46 and part.keys()==base.keys() and all(set(v)=={17,23,41} for v in part.values())
                acc=100*np.mean([rate(v) for v in part.values()])
                records.append(dict(k=k,donor_seed=d,condition=cond,accuracy=round(acc,3)))
        for first,second in [("own","donor"),("donor_pool_own_residual","own_pool_donor_residual"),
                             ("own","donor_pool_own_residual"),("own","own_pool_donor_residual")]:
            # Repeated seed evaluations share the same 46 questions. Average first, then resample recordings.
            delta = {q:np.mean([rate(scores[d,k,first][q])-rate(scores[d,k,second][q]) for d in donor_seeds]) for q in base}
            estimate,lo,hi=interval(delta)
            contrasts.append(dict(k=k, first=first, second=second, gain_pp=round(estimate,3),
                                  ci_low=round(lo,3),ci_high=round(hi,3)))
            print(k,first,"minus",second,": %.2f [%.2f, %.2f]"%(estimate,lo,hi))
    for filename, table in (("summary.csv",records),("contrasts.csv",contrasts)):
        with (out/filename).open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)


if __name__=="__main__":
    summarize()
