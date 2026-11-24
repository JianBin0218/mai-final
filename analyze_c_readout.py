"""Track C: paired differences in adapter gain between inputs (pair both-correct, group bootstrap)."""
import csv, json, sys
import numpy as np
root = sys.argv[1] if len(sys.argv) > 1 else "phase2/c-readout-v1"
def gains(name):
    rows = list(csv.DictReader(open(f"{root}/{name}.csv")))
    g = np.array([int(r["group"]) for r in rows]).reshape(-1, 2)[:, 0]
    out = {}
    for kind in ("base", "adapter"):
        per = []
        for o in ("yes_first", "no_first"):
            s = np.array([float(r[f"{kind}_{o}"]) for r in rows]).reshape(-1, 2)
            per.append((s[:, 0] >= 0) & (s[:, 1] < 0))
        out[kind] = np.mean(per, axis=0)
    return g, out["adapter"] - out["base"]
def ci(values, groups):
    u = np.unique(groups); m = np.array([values[groups == x].mean() for x in u])
    d = np.random.default_rng(29).integers(len(m), size=(10000, len(m)))
    s = m[d].mean(1); return [round(float(m.mean()), 3), round(float(np.quantile(s, .025)), 3), round(float(np.quantile(s, .975)), 3)]
g, k16 = gains("K16"); _, full = gains("full"); _, pool = gains("pool16")
res = {"K16_minus_full": ci(k16 - full, g), "K16_minus_pool16": ci(k16 - pool, g), "pool16_minus_full": ci(pool - full, g)}
print(json.dumps(res))
