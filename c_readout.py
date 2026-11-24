"""Track C: train a residual readout adapter between fixed latents and the frozen Qwen decoder.

The compressor and the decoder stay frozen. A small residual MLP is applied to every latent row before the
rows enter the language model; its output layer starts at zero, so at initialisation it reproduces the
baseline logits exactly. The same adapter design is trained on three inputs:

* ``full``   – the uncompressed projected audio features,
* ``pool16`` – training-free adaptive average pooling to 16 slots,
* ``K16``    – a learned audio-only K=16 compressor (``--compressor``).

If only K16 (or pooling) improves, the adapter is reading compressed evidence. If all three improve
equally, the adapter is teaching the decoder the task. Training and evaluation are split by source group
(K-fold over ``group``), and every question is asked in both option orders so a letter bias cannot score.

Controls reported per input:
  * identity check: the zero-initialised adapter changes no logit;
  * donor latents: held-out questions answered from another held-out recording's latents; a gain that
    comes from audio should follow the donor's label rather than the original label;
  * readout ceiling: the grouped ridge readout from count_evidence on the same fixed latents;
  * optional natural regression (``--extra-manifest``): the trained adapters applied to Clotho questions.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from count_evidence import grouped_readout, statistics

ORDERS = {"yes_first": ("\nA. yes\nB. no\nAnswer with only A or B.", 0),
          "no_first": ("\nA. no\nB. yes\nAnswer with only A or B.", 1)}
INPUTS = ("full", "pool16", "K16")


class Adapter(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, hidden)
        self.up = nn.Linear(hidden, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, latent):
        return latent + self.up(F.gelu(self.down(self.norm(latent))))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folds_by_group(groups, count, seed):
    unique = np.unique(groups)
    order = np.random.default_rng(seed).permutation(unique)
    return {int(group): fold for fold, chunk in enumerate(np.array_split(order, count)) for group in chunk}


def derangement(indices, groups, seed):
    """Map each held-out example to a donor from another source group (deterministic)."""
    rng = np.random.default_rng(seed)
    donors = {}
    for i in indices:
        candidates = [j for j in indices if groups[j] != groups[i]]
        donors[i] = int(rng.choice(candidates))
    return donors


def group_interval(values, groups, seed=29, replicates=10000):
    unique = np.unique(groups)
    means = np.array([values[groups == g].mean() for g in unique])
    draws = np.random.default_rng(seed).integers(len(means), size=(replicates, len(means)))
    sample = means[draws].mean(axis=1)
    return [float(means.mean()), float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]


def measure(args, prototype, features, decode):
    manifest = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    examples = [(pair, label) for pair in manifest for label in ("yes", "no")]
    groups = np.array([pair["group"] for pair, _ in examples])
    labels = np.array([1 if label == "yes" else -1 for _, label in examples])
    fold_of = folds_by_group(groups, args.folds, args.seed)
    width = prototype.keys.in_features

    comp = type(prototype)(width, 16).cuda().eval().requires_grad_(False)
    comp.load_state_dict(torch.load(args.compressor, weights_only=True))
    with torch.no_grad():
        audio = [features(pair[label]) for pair, label in examples]
        latents = {"full": audio,
                   "pool16": [F.adaptive_avg_pool1d(a.T.unsqueeze(0), 16)[0].T for a in audio],
                   "K16": [comp(a) for a in audio]}

    natural = []
    if args.extra_manifest:
        rows = json.loads(args.extra_manifest.read_text(encoding="utf-8"))["val"]
        types = {}
        if args.types.is_file():
            with args.types.open(encoding="utf-8-sig", newline="") as handle:
                types = {(r["clip"], r["question"]): r["type"] for r in csv.DictReader(handle)}
        with torch.no_grad():
            for item in rows:
                a = features(item["path"])
                natural.append({"path": item["path"], "question": item["question"], "target": item["label"],
                                "type": types.get((item["path"], item["question"]), ""),
                                "latents": {"full": a, "pool16": F.adaptive_avg_pool1d(a.T.unsqueeze(0), 16)[0].T,
                                            "K16": comp(a)}})

    def yes_score(adapter, latent, question, order):
        suffix, yes_index = ORDERS[order]
        x = latent if adapter is None else adapter(latent)
        scores = decode(x, question + suffix)
        return scores[yes_index] - scores[1 - yes_index]

    summary = {"pairs": len(manifest), "groups": int(len(set(groups))), "folds": args.folds,
               "compressor": str(args.compressor), "compressor_sha256": digest(args.compressor),
               "pair_manifest_sha256": digest(args.pair_manifest), "steps": args.steps,
               "batch_size": args.batch_size, "adapter_hidden": args.adapter_hidden, "adapter_lr": args.adapter_lr,
               "seed": args.seed, "torch": torch.__version__, "inputs": {}}

    for name in INPUTS:
        destination = args.out / f"{name}.csv"
        if destination.exists():
            raise ValueError(f"{destination} exists; choose a fresh --out")
        base = {order: np.empty(len(examples)) for order in ORDERS}
        tuned = {order: np.empty(len(examples)) for order in ORDERS}
        donor_tuned = {order: np.empty(len(examples)) for order in ORDERS}
        donor_base = {order: np.empty(len(examples)) for order in ORDERS}
        donor_of = np.empty(len(examples), dtype=int)
        natural_rows = []
        identity_gap = 0.0
        history = []
        for fold in range(args.folds):
            heldout = [i for i in range(len(examples)) if fold_of[int(groups[i])] == fold]
            train = [i for i in range(len(examples)) if fold_of[int(groups[i])] != fold]
            torch.manual_seed(args.seed * 100 + fold)
            adapter = Adapter(width, args.adapter_hidden).cuda()
            with torch.no_grad():
                for i in heldout[:4]:
                    for order in ORDERS:
                        a = yes_score(adapter, latents[name][i], examples[i][0]["question"], order)
                        b = yes_score(None, latents[name][i], examples[i][0]["question"], order)
                        identity_gap = max(identity_gap, abs((a - b).item()))
            optimiser = torch.optim.AdamW(adapter.parameters(), lr=args.adapter_lr, weight_decay=0.01)
            rng = np.random.default_rng(args.seed * 100 + fold)
            for step in range(args.steps):
                batch = rng.choice(train, size=min(args.batch_size, len(train)), replace=False)
                loss = 0.0
                for i in batch:
                    order = list(ORDERS)[rng.integers(2)]
                    suffix, yes_index = ORDERS[order]
                    target = yes_index if labels[i] == 1 else 1 - yes_index
                    scores = decode(adapter(latents[name][i]), examples[i][0]["question"] + suffix)
                    loss = loss + F.cross_entropy(scores[None], torch.tensor([target], device="cuda"))
                loss = loss / len(batch)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                if step % 25 == 0 or step == args.steps - 1:
                    history.append({"fold": fold, "step": step, "loss": loss.item()})
                    print(name, "fold", fold, "step", step, "loss", round(loss.item(), 4), flush=True)
            adapter.eval()
            donors = derangement(heldout, groups, args.seed + fold)
            with torch.no_grad():
                for i in heldout:
                    donor_of[i] = donors[i]
                    question = examples[i][0]["question"]
                    for order in ORDERS:
                        base[order][i] = yes_score(None, latents[name][i], question, order).item()
                        tuned[order][i] = yes_score(adapter, latents[name][i], question, order).item()
                        donor_base[order][i] = yes_score(None, latents[name][donors[i]], question, order).item()
                        donor_tuned[order][i] = yes_score(adapter, latents[name][donors[i]], question, order).item()
                for item in natural:
                    # Each fold's adapter is applied to the natural set; scores are averaged over folds.
                    s0 = yes_score(None, item["latents"][name], item["question"], "yes_first").item()
                    s1 = yes_score(adapter, item["latents"][name], item["question"], "yes_first").item()
                    natural_rows.append((fold, item, s0, s1))
            torch.save(adapter.state_dict(), args.out / f"adapter-{name}-fold{fold}.pt")
            del adapter, optimiser
            torch.cuda.empty_cache()

        def aligned(table):
            return (table["yes_first"] + table["no_first"]) / 2

        result = {"identity_max_logit_gap": identity_gap, "history": history}
        for label, table in (("base", base), ("adapter", tuned)):
            result[label] = {order: statistics(table[order], threshold=0) for order in ORDERS}
            result[label]["mean_orders"] = statistics(aligned(table), threshold=0)
        # Pair-level both-correct per order, averaged over orders, compared within source groups.
        def both(table):
            per_order = [((table[o].reshape(-1, 2)[:, 0] >= 0) & (table[o].reshape(-1, 2)[:, 1] < 0)) for o in ORDERS]
            return np.mean(per_order, axis=0).astype(float)
        pair_groups = groups.reshape(-1, 2)[:, 0]
        result["both_correct_gain_ci"] = group_interval(both(tuned) - both(base), pair_groups)
        follows = {}
        for label, table in (("base", donor_base), ("adapter", donor_tuned)):
            predicted = np.mean([table[o] > 0 for o in ORDERS], axis=0)
            follows[label] = {"follows_original": float(np.mean((predicted > 0.5) == (labels == 1))),
                              "follows_donor": float(np.mean((predicted > 0.5) == (labels[donor_of] == 1))),
                              "same_label_donors": float(np.mean(labels[donor_of] == labels))}
        result["donor"] = follows
        if name == "full":
            result["readout"] = "see pool16"
        else:
            matrix = np.stack([x.flatten().cpu().numpy() for x in latents[name]])
            probe, p_value = grouped_readout(matrix, groups, labels)
            result["readout"] = {**statistics(probe, threshold=np.nextafter(0., 1.)), "permutation_p": p_value}
        if natural_rows:
            by_item = {}
            for fold, item, s0, s1 in natural_rows:
                entry = by_item.setdefault((item["path"], item["question"]), {"item": item, "base": s0, "tuned": []})
                entry["tuned"].append(s1)
            def correct(score, target):
                return (score > 0) == (target == 0)
            rows = [(e["item"], correct(e["base"], e["item"]["target"]),
                     np.mean([correct(s, e["item"]["target"]) for s in e["tuned"]])) for e in by_item.values()]
            natural_summary = {"n": len(rows), "base": float(np.mean([r[1] for r in rows])),
                               "adapter_mean_folds": float(np.mean([r[2] for r in rows])), "by_type": {}}
            for kind in sorted({r[0]["type"] for r in rows}):
                subset = [r for r in rows if r[0]["type"] == kind]
                natural_summary["by_type"][kind or "untyped"] = {
                    "n": len(subset), "base": float(np.mean([r[1] for r in subset])),
                    "adapter_mean_folds": float(np.mean([r[2] for r in subset]))}
            result["natural"] = natural_summary
        summary["inputs"][name] = result
        tmp = destination.with_suffix(".partial")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["fold", "group", "source", "path", "label", "donor_path", "donor_label"]
                            + [f"{kind}_{order}" for kind in ("base", "adapter", "donor_base", "donor_adapter")
                               for order in ORDERS])
            for i, (pair, label) in enumerate(examples):
                d_pair, d_label = examples[donor_of[i]]
                writer.writerow([fold_of[int(groups[i])], pair["group"], pair["source"], pair[label], label,
                                 d_pair[d_label], d_label]
                                + [table[order][i] for table in (base, tuned, donor_base, donor_tuned)
                                   for order in ORDERS])
        tmp.replace(destination)
        print(name, json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
