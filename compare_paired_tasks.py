"""Compare paired behavioral discrimination across tasks and K on the same sources."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    root=Path(__file__).resolve().parent/"phase2"
    observations=defaultdict(list)
    for folder in ("opposite-v1","identity-v1","counterbalanced-v1"):
        for path in sorted((root/folder).glob("k*-s*.csv")):
            k,seed=map(int,path.stem[1:].split("-s"))
            pairs=defaultdict(dict)
            with path.open(encoding="utf-8",newline="") as f:
                for r in csv.DictReader(f):
                    if r["condition"]=="own":
                        pairs[r["task"],r["group"],r["pair"]][int(r["direction"])]=float(r["margin_yes"])
            for (task,group,_),v in pairs.items():
                observations[task,k,group].append((v[0]>=0 and v[1]<0,(int(v[0]>=0)+int(v[1]<0))/2))
    rows=[]
    for a,b in [(("natural",16),("natural",4)), (("level",16),("level",4))]+[
            ((t,k),("identity",k)) for k in (0,4,16) for t in ("second_event","second_other","anywhere_other")]+[
            (("anywhere_other",k),("second_other",k)) for k in (0,4,16)]:
        groups={g for task,k,g in observations if (task,k)==a}
        assert groups=={g for task,k,g in observations if (task,k)==b}
        changes=np.array([np.mean(observations[a+(g,)],axis=0)-np.mean(observations[b+(g,)],axis=0)
                          for g in sorted(groups)])
        n=len(changes)
        weights=np.random.default_rng(17).multinomial(n,np.ones(n)/n,size=10000)
        for i,metric in enumerate(["both","accuracy"]):
            lo,hi=np.quantile(100*(weights@changes[:,i])/n,[.025,.975])
            rows.append(dict(first=f"{a[0]}-K{a[1]}",second=f"{b[0]}-K{b[1]}",metric=metric,groups=n,
                             difference_pp=round(100*changes[:,i].mean(),2),low=round(lo,2),high=round(hi,2)))
    with (root/"paired_task_contrasts.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=="__main__":
    main()
