"""Per-type paired accuracy; resample recordings, keeping seeds and questions together."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent / "phase2/fresh-capacity"


def main():
    with (ROOT/"question_types.csv").open(encoding="utf-8", newline="") as f:
        labels = {(r["clip"], r["question"]):r["type"] for r in csv.DictReader(f)}
    clips = sorted({c for c,_ in labels})
    types = sorted(set(labels.values()))
    totals = np.zeros((len(clips),len(types)))
    correct = np.zeros((3,2,*totals.shape))
    keys = list(labels)
    for clip,q in keys:
        totals[clips.index(clip), types.index(labels[clip,q])] += 1
    targets = {}
    for s,seed in enumerate([17,23,41]):
        for kidx,k in enumerate([4,16]):
            name = f"capacity-k{k}-s{seed}-audio" if (k,seed)!=(16,17) else "clotho-k16-s17-audio"
            rows = json.loads((ROOT/(name+".json")).read_text())["rows"]
            found = set()
            for r in rows:
                key = r["path"], r["question"].splitlines()[0]
                assert key not in found and targets.setdefault(key,r["target"])==r["target"]
                found.add(key)
                correct[s,kidx,clips.index(key[0]),types.index(labels[key])] += r["prediction"]==r["target"]
            assert found == set(keys)
    assert len(keys)==340 and len(clips)==170
    avg = correct.mean(axis=0).sum(axis=1)/totals.sum(axis=0)*100
    gain = (correct[:,1]-correct[:,0]).mean(axis=0)
    weights = np.random.default_rng(17).multinomial(len(clips),np.ones(len(clips))/len(clips),size=10000)
    resamples = 100*(weights @ gain)/(weights @ totals)
    for i,t in enumerate(types):
        ci=np.quantile(resamples[:,i],[.025,.975])
        print(t,int(totals[:,i].sum()),*(round(x,1) for x in [*avg[:,i],avg[1,i]-avg[0,i],*ci]),
              "seed net", (correct[:,1,:,i]-correct[:,0,:,i]).sum(axis=1).astype(int).tolist())
    i,j=types.index("count_or_relation"),types.index("source_or_event")
    print("relation minus source gain (pp)",round((avg[1]-avg[0])[i]-(avg[1]-avg[0])[j],1),
          np.quantile(resamples[:,i]-resamples[:,j],[.025,.975]).round(1))


if __name__=="__main__":
    main()
