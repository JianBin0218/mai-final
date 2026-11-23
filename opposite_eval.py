"""Measure component dependence under known opposite donor labels in both directions."""
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F


@torch.no_grad()
def measure(args, prototype, features, decode):
    pairs=json.loads(args.pair_manifest.read_text())
    suffix="\nA. yes\nB. no\nAnswer with only A or B."
    for k,seed in [(0,0)]+[(k,s) for s in (17,23,41) for k in (4,16)]:
        dst=args.out/f"k{k}-s{seed}.csv"
        if dst.exists():
            continue
        if k:
            folder="clotho-k16-s17-audio" if (k,seed)==(16,17) else f"capacity-k{k}-s{seed}-audio"
            comp=type(prototype)(prototype.keys.in_features,k).cuda().eval()
            comp.load_state_dict(torch.load(Path(folder)/"compressor.pt",weights_only=True))
        tmp=dst.with_suffix(".partial")
        with tmp.open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["task","group","pair","direction","question","clip","donor","condition","margin_yes"])
            for i,p in enumerate(pairs):
                x=[features(p[t]) for t in ("yes","no")]
                if k:
                    h=[comp(v) for v in x]
                    avg=[F.adaptive_avg_pool1d(v.T.unsqueeze(0),k)[0].T for v in x]
                for j in (0,1):
                    other=1-j
                    conditions={"own": h[j] if k else x[j]}
                    if k:
                        conditions.update(donor_residual=avg[j]+(h[other]-avg[other]),
                                          donor_pool=avg[other]+(h[j]-avg[j]))
                    for name,v in conditions.items():
                        scores=decode(v,p["q"]+suffix)
                        w.writerow([p["task"],p["group"],i,j,p["q"],p[("yes","no")[j]],p[("no","yes")[j]],
                                    name,(scores[0]-scores[1]).item()])
        tmp.replace(dst)
        print(k,seed,flush=True)
