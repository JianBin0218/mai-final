"""Swap the pooled and learned residual components on opposite-label natural audio pairs."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from count_confirmation import grouped_interval

SEEDS = (17, 23, 41)
ORDERS = {"yes_first": ("yes", "no", 1), "no_first": ("no", "yes", -1)}
BOOTSTRAP = {"bootstrap_seed": 17, "bootstrap_replicates": 10000}


def difference_preference(delta):
    return (delta > 0) + 0.5 * (delta == 0)


def estimate(values, pairs):
    return {"mean": float(values.mean()), "ci": grouped_interval(values, pairs, BOOTSTRAP)}


@torch.no_grad()
def measure(args, prototype, features, decode):
    manifest = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    original = json.loads((args.capacity_references / "summary.json").read_text(encoding="utf-8"))
    pairs = manifest["pairs"]
    if original["pairs"] != len(pairs) or original["plan_sha256"] != manifest["plan_sha256"]:
        raise ValueError("Reference run and manifest differ")
    expected = [(p["q"], p[label], label) for p in pairs for label in ("yes", "no")]
    summary = {"pairs": len(pairs), "model_seeds": SEEDS, "reference_plan_sha256": manifest["plan_sha256"],
               "capacities": {}}
    hybrid_directions = {}
    for k in (4, 16):
        by_seed = []
        for seed in SEEDS:
            name = f"K{k}-s{seed}"
            folder = "clotho-k16-s17-audio" if (k, seed) == (16, 17) else f"capacity-k{k}-s{seed}-audio"
            checkpoint = Path(folder) / "compressor.pt"
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            with (args.capacity_references / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
                refs = list(csv.DictReader(handle))
            if [(r["question"], r["clip"], r["label"]) for r in refs] != expected or any(
                r["checkpoint"] != digest for r in refs
            ):
                raise ValueError(f"References do not match the recordings or checkpoint for {name}")
            comp = type(prototype)(prototype.keys.in_features, k).cuda().eval()
            comp.load_state_dict(torch.load(checkpoint, weights_only=True))
            path = args.out / f"{name}.csv"
            tmp = path.with_suffix(".partial")
            run = []
            with tmp.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["pair", "question", "order", "own_yes", "own_no",
                                 "pool_yes_residual_no", "pool_no_residual_yes", "checkpoint"])
                for i, pair in enumerate(pairs):
                    inputs = [features(pair[label]) for label in ("yes", "no")]
                    pools = [F.adaptive_avg_pool1d(audio.T[None], k)[0].T for audio in inputs]
                    own = [comp(audio) for audio in inputs]
                    residuals = [latent - pool for latent, pool in zip(own, pools)]
                    hybrids = torch.stack((pools[0] + residuals[1], pools[1] + residuals[0]))
                    for order, (a, b, sign) in ORDERS.items():
                        question = f"{pair['q']}\nA. {a}\nB. {b}\nAnswer with only A or B."
                        logits = decode(hybrids, question)
                        hybrid = (sign * (logits[:, 0] - logits[:, 1])).cpu().tolist()
                        ref = [float(refs[2*i + j][order]) for j in (0, 1)]
                        if i < 3:
                            check = decode(torch.stack(own), question)
                            replicated = (sign * (check[:, 0] - check[:, 1])).cpu().tolist()
                            if not np.allclose(replicated, ref, rtol=0, atol=0.125):
                                raise ValueError(f"Reference logits changed for {name}, pair {i}, {order}")
                        values = [*ref, *hybrid]
                        writer.writerow([i, pair["q"], order, *values, digest])
                        run.append(values)
            tmp.replace(path)
            by_seed.append(np.array(run).reshape(len(pairs), len(ORDERS), 4))
            print("Completed", name, flush=True)
        yy, nn, yn, ny = np.moveaxis(np.array(by_seed), -1, 0)
        residual = ((yy - yn) + (ny - nn)) / 2
        pool = ((yy - ny) + (yn - nn)) / 2
        interaction = yy - yn - ny + nn
        average = lambda values: values.mean(axis=(0, 2))
        hybrid_directions[k] = average(difference_preference(ny - yn))
        summary["capacities"][str(k)] = {
            "residual_hybrid_preference": estimate(hybrid_directions[k], pairs),
            "residual_margin_effect": estimate(average(residual), pairs),
            "pool_margin_effect": estimate(average(pool), pairs),
            "interaction_margin": estimate(average(interaction), pairs),
            "preference_by_order": dict(zip(ORDERS, difference_preference(ny - yn).mean(axis=(0, 1)).tolist())),
        }
        print(k, summary["capacities"][str(k)], flush=True)
    summary["K16_minus_K4_residual_hybrid_preference"] = estimate(
        hybrid_directions[16] - hybrid_directions[4], pairs
    )
    print("Capacity difference", summary["K16_minus_K4_residual_hybrid_preference"], flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
