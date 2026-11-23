"""Interchange audio representations while retaining the original question and label."""
import csv
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F


@torch.no_grad()
def run_interventions(args, test, prototype, features, decode):
    with args.types.open(encoding="utf-8", newline="") as f:
        selected = {(r["clip"],r["question"]) for r in csv.DictReader(f) if r["type"]=="count_or_relation"}
    clips = sorted({p for p,_,_ in test})
    random.Random(args.donor_seed).shuffle(clips)
    donors = dict(zip(clips, clips[1:]+clips[:1]))
    rows = [(p,q,y) for p,q,y in test if (p,q.splitlines()[0]) in selected]
    assert len(rows) == len(selected) == 46
    output = args.out / f"{args.intervention}-d{args.donor_seed}.csv"
    if output.exists():
        raise FileExistsError(output)
    tmp = output.with_suffix(".partial")
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed","k","clip","donor","question","target","condition","prediction","logit_yes_minus_no"])
        for seed in (17,23,41):
            for k in (4,16):
                folder = "clotho-k16-s17-audio" if (k,seed)==(16,17) else f"capacity-k{k}-s{seed}-audio"
                comp = type(prototype)(prototype.keys.in_features,k).cuda().eval()
                comp.load_state_dict(torch.load(Path(folder)/"compressor.pt",weights_only=True))
                for p,q,y in rows:
                    donor = donors[p]
                    x,z = features(p),features(donor)
                    own,other = comp(x),comp(z)
                    inputs = {}
                    if args.intervention in ("swap", "factorial"):
                        inputs.update(own=own, donor=other)
                    if args.intervention in ("components", "factorial"):
                        px,pz = [F.adaptive_avg_pool1d(v.T.unsqueeze(0),k)[0].T for v in (x,z)]
                        inputs.update(own_pool_donor_residual=px+(other-pz),
                                      donor_pool_own_residual=pz+(own-px))
                    for condition, latent in inputs.items():
                        scores = decode(latent,q).float()
                        writer.writerow([seed,k,p,donor,q.splitlines()[0],y,condition,
                                         int(scores.argmax()),(scores[0]-scores[1]).item()])
                f.flush()
                print(seed,k,flush=True)
    tmp.replace(output)
