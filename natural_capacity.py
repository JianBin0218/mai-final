"""Compare frozen audio capacities on new recordings with the same question and opposite labels."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from count_confirmation import grouped_interval
from count_evidence import statistics
from prepare_clotho import CHECKSUM, md5


def questions_by_recording(path, members):
    votes = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            votes[row["file_name"], row["QuestionText"].strip()].append(row["answer"].strip().lower())
    questions = defaultdict(lambda: {"yes": [], "no": []})
    for (clip, question), labels in votes.items():
        if clip in members and len(labels) >= 3 and len(set(labels)) == 1 and labels[0] in ("yes", "no"):
            questions[clip][labels[0]].append(question)
    return questions


def select_pairs(plan, data, members):
    pilot = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    fresh = json.loads((data / "fresh_val.json").read_text(encoding="utf-8"))
    questions = {}
    eligible = {}
    for split in ("train", "val"):
        path = data / f"clotho_aqa_{split}.csv"
        if md5(path) != CHECKSUM[split]:
            raise ValueError(f"Unexpected Clotho-AQA annotation version: {path}")
        questions[split] = q = questions_by_recording(path, members)
        eligible[split] = sorted(clip for clip, labels in q.items() if labels["yes"] and labels["no"])

    # Replay the published pilot and train_large selection, checking the archived pilot.
    used = set()
    rng = random.Random(17)
    for split, n in (("train", 48), ("val", 32)):
        clips = rng.sample([clip for clip in eligible[split] if clip not in used], n)
        expected = []
        for clip in clips:
            used.add(clip)
            for label in ("yes", "no"):
                expected.append((clip, rng.choice(questions[split][clip][label]), int(label == "no")))
        actual = [(Path(r["path"]).name, r["question"], r["label"]) for r in pilot[split]]
        if expected != actual:
            raise ValueError(f"The archived {split} pilot differs from its published selection")
    rng = random.Random(plan["large_training_seed"])
    for clip in eligible["val"]:
        if clip not in used:
            rng.choice(questions["val"][clip]["yes"])
            rng.choice(questions["val"][clip]["no"])
    additions = set(rng.sample([clip for clip in eligible["train"] if clip not in used],
                               plan["large_training_recordings"] - 48))
    used.update(additions)
    used.update(Path(row["path"]).name for rows in fresh.values() for row in rows)

    stems = defaultdict(lambda: {"yes": set(), "no": set()})
    for clip, labels in questions[plan["source_split"]].items():
        if clip not in used:
            for label, texts in labels.items():
                for text in texts:
                    key = " ".join(text.lower().strip().rstrip("?.!").split())
                    if key:
                        stems[key][label].add(clip)
    for labels in stems.values():
        conflict = labels["yes"] & labels["no"]
        labels["yes"] -= conflict
        labels["no"] -= conflict
    order = sorted(key for key, labels in stems.items() if labels["yes"] and labels["no"])
    rng = random.Random(plan["pair_seed"])
    rng.shuffle(order)
    pairs = []
    for key in order:
        yes, no = (sorted(stems[key][label] - used) for label in ("yes", "no"))
        if yes and no:
            y, n = rng.choice(yes), rng.choice(no)
            used.update((y, n))
            pairs.append({"group": len(pairs), "q": key[0].upper() + key[1:] + "?", "yes": y, "no": n})
    return pairs, sorted(additions)


def prepare(plan_path, out):
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    data = Path("data/clotho")
    with zipfile.ZipFile(data / "audio.zip") as archive:
        members = {Path(name).name: name for name in archive.namelist()
                   if name.lower().endswith(".wav") and "__MACOSX" not in name}
        pairs, additions = select_pairs(plan, data, members)
        if len(pairs) < plan["minimum_pairs"]:
            raise ValueError(f"Only {len(pairs)} disjoint pairs, below the fixed minimum")
        audio = out / "audio"
        audio.mkdir(parents=True, exist_ok=True)
        for pair in pairs:
            for label in ("yes", "no"):
                name = pair[label]
                cached = data / name
                path = audio / name
                if not path.exists():
                    if cached.is_file():
                        path.symlink_to(cached.resolve())
                    else:
                        tmp = path.with_suffix(".partial")
                        with archive.open(members[name]) as source, tmp.open("wb") as target:
                            shutil.copyfileobj(source, target)
                        tmp.replace(path)
                pair[label] = str(path)
    manifest = {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                "reconstructed_train_large_additions": additions, "pairs": pairs}
    (out / "pairs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(pairs)} distinct question stems and {2 * len(pairs)} non-reused recordings", flush=True)


def preference(scores):
    paired = scores.reshape(-1, 2)
    return (paired[:, 0] > paired[:, 1]) + 0.5 * (paired[:, 0] == paired[:, 1])


@torch.no_grad()
def measure(args, prototype, features, decode):
    raw = args.capacity_plan.read_bytes()
    plan = json.loads(raw)
    manifest = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    digest = hashlib.sha256(raw).hexdigest()
    if manifest["plan_sha256"] != digest:
        raise ValueError("Prepared pairs do not match the frozen plan")
    pairs = manifest["pairs"]
    keys = [(pair["q"], pair[label], label) for pair in pairs for label in ("yes", "no")]
    if len(pairs) < plan["minimum_pairs"] or len({clip for _, clip, _ in keys}) != len(keys):
        raise ValueError("Pairs must satisfy the fixed sample size and recording disjointness")
    summary = {"plan_sha256": digest, "pairs": len(pairs), "conditions": {}}
    configs = {"full": (0, None), "pool4": (4, None), "pool16": (16, None)}
    for k in (4, 16):
        for seed in plan["model_seeds"]:
            folder = "clotho-k16-s17-audio" if (k, seed) == (16, 17) else f"capacity-k{k}-s{seed}-audio"
            configs[f"K{k}-s{seed}"] = (k, Path(folder) / "compressor.pt")
    order_options = {"yes_first": ("yes", "no", 1), "no_first": ("no", "yes", -1)}
    orders = plan["prompt_orders"]
    rankings = {}
    for name, (k, checkpoint) in configs.items():
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else ""
        path = args.out / f"{name}.csv"
        if path.exists():
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if [(r["question"], r["clip"], r["label"]) for r in rows] != keys or any(
                r["checkpoint"] != checkpoint_hash for r in rows
            ):
                raise ValueError(f"Saved scores do not match the paired experiment: {path}")
            margins = {order: np.array([float(r[order]) for r in rows]) for order in orders}
        else:
            comp = None
            if checkpoint:
                comp = type(prototype)(prototype.keys.in_features, k).cuda().eval()
                comp.load_state_dict(torch.load(checkpoint, weights_only=True))
            margins = {order: [] for order in orders}
            for pair in pairs:
                audio = [features(pair[label]) for label in ("yes", "no")]
                if comp is not None:
                    audio = [comp(v) for v in audio]
                elif k:
                    audio = [F.adaptive_avg_pool1d(v.T[None], k)[0].T for v in audio]
                for order in orders:
                    a, b, sign = order_options[order]
                    question = f"{pair['q']}\nA. {a}\nB. {b}\nAnswer with only A or B."
                    logits = (decode(torch.stack(audio), question) if audio[0].shape == audio[1].shape
                              else torch.stack([decode(v, question) for v in audio]))
                    margins[order].extend((sign * (logits[:, 0] - logits[:, 1])).cpu().tolist())
            tmp = path.with_suffix(".partial")
            with tmp.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["question", "clip", "label", *orders, "checkpoint"])
                for i, key in enumerate(keys):
                    writer.writerow([*key, *(margins[order][i] for order in orders), checkpoint_hash])
            tmp.replace(path)
            margins = {order: np.asarray(values) for order, values in margins.items()}
        results = {order: statistics(margins[order], threshold=0 if order_options[order][2] == 1 else np.nextafter(0., 1.))
                   for order in orders}
        rankings[name] = np.mean([preference(margins[order]) for order in orders], axis=0)
        summary["conditions"][name] = {"orders": results,
                                       "average_preference": float(rankings[name].mean()),
                                       "checkpoint": checkpoint_hash}
        print(name, json.dumps(summary["conditions"][name]), flush=True)

    fixed = plan["primary"]
    learned = {k: np.mean([rankings[f"K{k}-s{seed}"] for seed in plan["model_seeds"]], axis=0)
               for k in (4, 16)}
    delta = learned[16] - learned[4]
    interval = grouped_interval(delta, pairs, fixed)
    summary["primary"] = {"K4_preference": float(learned[4].mean()), "K16_preference": float(learned[16].mean()),
                          "gain": float(delta.mean()), "ci": interval,
                          "passes": bool(delta.mean() >= fixed["minimum_gain"]
                                         and interval[0] > fixed["minimum_lower_ci"])}
    pool_delta = rankings["pool16"] - rankings["pool4"]
    summary["secondary_pool_gain"] = {"gain": float(pool_delta.mean()),
                                      "ci": grouped_interval(pool_delta, pairs, fixed)}
    print("Fixed primary", summary["primary"], flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.plan, args.out)
