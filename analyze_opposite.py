"""Paired diagnostics. Bootstrap source groups after averaging repeated trials."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def interval(values):
    a=np.array(values)
    rng=np.random.default_rng(17)
    w=rng.multinomial(len(a),np.ones(len(a))/len(a),size=10000)
    return np.quantile((w@a)/len(a),[.025,.975])*100


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    root=parser.parse_args().directory
    stats=[]
    from collections import defaultdict
    by_group=defaultdict(list)
    for path in sorted(root.glob("k*-s*.csv")):
        k,seed=map(int,path.stem[1:].split("-s"))
        manifest_path=root/"pairs.json"
        manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        pairs=defaultdict(dict)
        with path.open(encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                ident=r["task"],r["group"],int(r["pair"])
                slot=int(r["direction"]),r["condition"]
                if manifest is not None:
                    spec=manifest[ident[2]]
                    assert (spec["task"],spec["group"],spec["q"])==(r["task"],r["group"],r["question"])
                    assert r["clip"]==spec[("yes","no")[slot[0]]] and r["donor"]==spec[("no","yes")[slot[0]]]
                assert slot not in pairs[ident]
                pairs[ident][slot]=float(r["margin_yes"])
        if manifest is not None:
            assert len(pairs)==len(manifest)
        by_task=defaultdict(list)
        for (task,group,_),vals in pairs.items():
            m=[vals[j,"own"] for j in (0,1)]
            # Ties predict A (yes), matching argmax on [A,B].
            own=[m[0]>=0,m[1]<0]
            row=dict(acc=sum(own)/2,both=float(all(own)),rank=float(m[0]>m[1])+0.5*(m[0]==m[1]))
            if k:
                # Two reciprocal directions provide both off-diagonal cells of the 2 x 2.
                # Source of the pooled component is indexed first, then source of residual.
                def yes_prob(margin):
                    return 1/(1+np.exp(-np.clip(margin,-80,80)))
                yy,nn = map(yes_prob,m)
                yn = yes_prob(vals[0,"donor_residual"])
                ny = yes_prob(vals[1,"donor_residual"])
                # Same hybrids assembled in the reverse direction must reproduce their score.
                assert abs(yn-yes_prob(vals[1,"donor_pool"]))<1e-7
                assert abs(ny-yes_prob(vals[0,"donor_pool"]))<1e-7
                row["residual_prob_effect"] = ((yy-yn)+(ny-nn))/2
                row["pool_prob_effect"] = ((yy-ny)+(yn-nn))/2
                target=[]
                for cond in ("donor_residual","donor_pool"):
                    switched=[vals[0,cond]<0,vals[1,cond]>=0]
                    row[cond]=sum(switched)/2
                    row[cond+"_eligible"]=sum(switched)/2 if all(own) else np.nan
                    target.append(row[cond])
                row["residual_minus_pool"]=target[0]-target[1]
            by_task[task].append(row)
            by_group[task,group,k].append(row)
        for task,rows in by_task.items():
            result=dict(task=task,k=k,seed=seed,n_pairs=len(rows),eligible_pairs=sum(r["both"] for r in rows))
            for key in rows[0]:
                vals=[r[key] for r in rows if np.isfinite(r[key])]
                result[key]=round(100*np.mean(vals),3) if vals else ""
            stats.append(result)
    assert stats, "No result files found"
    keys=list(dict.fromkeys(k for r in stats for k in r))
    with (root/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(stats)
    grouped=[]
    for task,k in sorted({(t,k) for t,_,k in by_group}):
        groups=[v for (t,_,kk),v in by_group.items() if t==task and k==kk]
        for key in groups[0][0]:
            all_values=[r[key] for g in groups for r in g if np.isfinite(r[key])]
            if not all_values:
                continue
            # Keep clusters intact; denominators can differ for eligibility-conditioned metrics.
            sums=np.array([sum(r[key] for r in g if np.isfinite(r[key])) for g in groups])
            sizes=np.array([sum(np.isfinite(r[key]) for r in g) for g in groups])
            weights=np.random.default_rng(17).multinomial(len(groups),np.ones(len(groups))/len(groups),size=10000)
            den=weights@sizes
            draws=(weights@sums)[den>0]/den[den>0]
            ci=np.quantile(draws,[.025,.975])*100
            grouped.append(dict(task=task,k=k,metric=key,value=round(100*np.mean(all_values),3),
                                ci_low=round(ci[0],3),ci_high=round(ci[1],3),source_groups=len(groups),
                                pair_seed_observations=len(all_values)))
    with (root/"grouped.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(grouped[0]));writer.writeheader();writer.writerows(grouped)


if __name__=="__main__":
    main()
