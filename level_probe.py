"""Read relative level from frozen representations, holding out both source categories per group."""
import csv
import json
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def probe(args, prototype, features):
    manifest=json.loads(args.pair_manifest.read_text())
    records=[r for r in manifest if r["task"]=="level" and " louder " in r["q"]]
    examples=[(r["group"],r[name],label) for r in records for name,label in (("yes",1),("no",-1))]
    assert len(examples)==100 and len({x[1] for x in examples})==100
    groups=np.array([r[0] for r in examples])
    labels=np.array([r[2] for r in examples])
    dest=args.out/"probe.csv"
    tmp=dest.with_suffix(".partial")
    null_results=[]
    with tmp.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["k","seed","heldout_group","path","target","score"])
        for k,seed in [(0,0)]+[(k,s) for s in (17,23,41) for k in (4,16)]:
            if k:
                folder="clotho-k16-s17-audio" if (k,seed)==(16,17) else f"capacity-k{k}-s{seed}-audio"
                comp=type(prototype)(prototype.keys.in_features,k).cuda().eval()
                comp.load_state_dict(torch.load(Path(folder)/"compressor.pt",weights_only=True))
            matrix=[]
            for _,path,_ in examples:
                v=features(path)
                if k:
                    v=comp(v)
                half=len(v)//2
                # Fixed readout: difference of mean vectors in the first and last halves.
                x=v[:half].mean(0)-v[-half:].mean(0)
                matrix.append(x.cpu().numpy())
            x=np.asarray(matrix,dtype=np.float64)
            x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
            gram=x@x.T
            # Null preserves all within-source correlations and refits each null readout.
            unique,cluster=np.unique(groups,return_inverse=True)
            signs=np.random.default_rng(17).choice([-1,1],size=(len(unique),999))
            shuffled_labels=labels[:,None]*signs[cluster]
            all_labels=np.column_stack((labels,shuffled_labels))
            all_scores=np.empty(all_labels.shape,dtype=float)
            for group in sorted(set(groups)):
                train=groups!=group; test=~train
                projection=np.linalg.solve(gram[np.ix_(train,train)]+np.eye(train.sum()),
                                           gram[np.ix_(train,test)]).T
                all_scores[test]=projection@all_labels[train]
            scores=all_scores[:,0]
            accuracy=np.mean((all_scores>0)==(all_labels>0),axis=0)
            null_results.append(dict(k=k, seed=seed, accuracy=float(accuracy[0]),
                null_mean=float(accuracy[1:].mean()),null_p95=float(np.quantile(accuracy[1:],.95)),
                permutation_p=(1+int(sum(accuracy[1:]>=accuracy[0])))/1000))
            for (group,path,label),value in zip(examples,scores):
                w.writerow([k,seed,group,path,label,value])
            f.flush()
            print("readout",k,seed,int(((scores>0)==(labels>0)).sum()),len(scores),flush=True)
    tmp.replace(dest)
    (args.out/"probe_null.json").write_text(json.dumps(null_results,indent=2))
