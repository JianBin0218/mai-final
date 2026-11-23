"""Compare ranking and a single yes/no threshold on saved relation-question scores."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "phase2"
SEEDS = (17, 23, 41)
DONOR_FILES = (
    ("swap-d17.csv", "components-d17.csv"),
    ("factorial-d23.csv",),
    ("factorial-d41.csv",),
)
INTERVENTIONS = {
    "own": "K16 direct",
    "donor": "K16 unrelated",
    "own_pool_donor_residual": "K16 own pool",
    "donor_pool_own_residual": "K16 own residual",
}


def read_evaluation(path, keys, targets=None):
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    by_key = {(row["path"], row["question"].splitlines()[0]): row for row in rows}
    selected = [by_key[key] for key in keys]
    labels = np.array([row["target"] for row in selected])
    if targets is not None and not np.array_equal(labels, targets):
        raise ValueError(f"Answer labels differ in {path}")
    losses = np.array([row["loss"] for row in selected])
    if np.any(losses <= 0):
        raise ValueError(f"Expected positive binary cross-entropy in {path}")
    # A binary target loss L equals log(1 + exp(-target_margin)).
    target_margin = -losses - np.log(-np.expm1(-losses))
    # Ignore float32 inversion noise when two scores are tied.
    margin = np.round(np.where(labels == 0, target_margin, -target_margin), 4)
    if not np.array_equal(margin < 0, [row["prediction"] for row in selected]):
        raise ValueError(f"Recovered margins disagree with saved predictions in {path}")
    return margin, labels


def auc_pairs(scores, yes):
    differences = scores[yes, None] - scores[None, ~yes]
    return 0.5 * ((differences > 0).astype(float) + (differences >= 0))


def threshold_fit(scores, yes):
    unique = np.unique(scores)
    candidates = np.r_[0, unique[0], (unique[1:] + unique[:-1]) / 2,
                       np.nextafter(unique[-1], np.inf)]
    correct = ((scores[None, :] >= candidates[:, None]) == yes).sum(axis=1)
    tied = candidates[correct == correct.max()]
    return tied[np.argmin(np.abs(tied))], correct.max() / len(scores)


def threshold_check(scores, yes, clips):
    held_out_correct = np.empty(len(scores), dtype=bool)
    for clip in np.unique(clips):
        holdout = clips == clip
        threshold, _ = threshold_fit(scores[~holdout], yes[~holdout])
        held_out_correct[holdout] = (scores[holdout] >= threshold) == yes[holdout]
    return held_out_correct.mean(), threshold_fit(scores, yes)[1]


def ranking_interval(delta, yes, clips):
    _, cluster = np.unique(clips, return_inverse=True)
    groups = int(cluster.max()) + 1
    weights = np.random.default_rng(17).multinomial(groups, np.ones(groups) / groups, size=10000)
    yes_weights, no_weights = weights[:, cluster[yes]], weights[:, cluster[~yes]]
    denominator = yes_weights.sum(axis=1) * no_weights.sum(axis=1)
    valid = denominator > 0
    numerator = ((yes_weights[valid] @ delta) * no_weights[valid]).sum(axis=1)
    return 100 * delta.mean(), *np.quantile(100 * numerator / denominator[valid], [0.025, 0.975])


def main():
    capacity = ROOT / "fresh-capacity"
    controls = ROOT / "fresh-controls"
    with (capacity / "question_types.csv").open(encoding="utf-8", newline="") as handle:
        keys = sorted((row["clip"], row["question"]) for row in csv.DictReader(handle)
                      if row["type"] == "count_or_relation")
    clips = np.array([clip for clip, _ in keys])
    full, targets = read_evaluation(controls / "full.json", keys)
    yes = targets == 0
    methods = {"full": [full]}
    for name in ("text", "unrelated"):
        methods[name] = [read_evaluation(controls / f"{name}.json", keys, targets)[0]]
    for k in (4, 16):
        stems = ["clotho-k16-s17-audio" if (k, seed) == (16, 17) else f"capacity-k{k}-s{seed}-audio"
                 for seed in SEEDS]
        methods[f"K{k}"] = [read_evaluation(capacity / f"{stem}.json", keys, targets)[0]
                             for stem in stems]

    methods.update({method: [] for method in INTERVENTIONS.values()})
    for files in DONOR_FILES:
        by_key = {}
        for name in files:
            with (ROOT / "interventions-v1" / name).open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["k"] == "16":
                        key = int(row["seed"]), row["condition"], row["clip"], row["question"]
                        by_key[key] = row
        for seed in SEEDS:
            for condition, method in INTERVENTIONS.items():
                rows = [by_key[seed, condition, clip, question] for clip, question in keys]
                if [int(row["target"]) for row in rows] != targets.tolist():
                    raise ValueError(f"Answer labels differ in {files} for {condition}, seed {seed}")
                scores = np.array([float(row["logit_yes_minus_no"]) for row in rows])
                if not np.array_equal(scores < 0, [int(row["prediction"]) for row in rows]):
                    raise ValueError(f"Margins disagree in {files} for {condition}, seed {seed}")
                methods[method].append(scores)

    print(f"{len(keys)} questions, {len(set(clips))} recordings, {yes.sum()} yes")
    print(f"{'method':<18} {'accuracy':>9} {'ROC AUC':>9} {'held-out*':>10} {'oracle*':>9}")
    rankings = {}
    for name, runs in methods.items():
        fixed = np.mean([((scores >= 0) == yes).mean() for scores in runs])
        rankings[name] = np.mean([auc_pairs(scores, yes) for scores in runs], axis=0)
        held_out, oracle = np.mean([threshold_check(scores, yes, clips) for scores in runs], axis=0)
        print(f"{name:<18} {100 * fixed:9.2f} {100 * rankings[name].mean():9.2f} "
              f"{100 * held_out:10.2f} {100 * oracle:9.2f}")
    print("* Threshold fitted on other recordings (held-out) or all evaluation questions (oracle).")
    both = rankings["K16 direct"]
    neither = rankings["K16 unrelated"]
    only_pool = rankings["K16 own pool"]
    only_residual = rankings["K16 own residual"]
    contrasts = {
        "full minus K16": rankings["full"] - rankings["K16"],
        "full minus K16 direct": rankings["full"] - both,
        "K16 minus K4": rankings["K16"] - rankings["K4"],
        "both matched minus neither matched": both - neither,
        "matched residual, holding pool matched": both - only_pool,
        "matched residual, holding pool unrelated": only_residual - neither,
        "matched pool, holding residual matched": both - only_residual,
        "matched pool, holding residual unrelated": only_pool - neither,
        "only residual matched minus only pool matched": only_residual - only_pool,
        "residual-by-pool interaction": (both - only_pool) - (only_residual - neither),
    }
    for label, delta in contrasts.items():
        estimate, low, high = ranking_interval(delta, yes, clips)
        print(f"ROC AUC {label}: {estimate:+.2f} points [{low:+.2f}, {high:+.2f}]")


if __name__ == "__main__":
    main()
