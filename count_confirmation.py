"""Confirm sound-burst readout on held-out sources under a fixed analysis plan."""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from count_evidence import CONFIGURATIONS, prepare, statistics
from prepare import ESC, download


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_heldout(plan_path, root):
    plan = load_json(plan_path)
    development = load_json(Path(plan["development_manifest"]))
    with Path("data/esc50.csv").open(encoding="utf-8", newline="") as handle:
        metadata = {row["filename"]: row for row in csv.DictReader(handle)}
    previous = [metadata[Path(item["source"]).name] for item in development]
    used = {row["src_file"] for row in previous}
    used.update(metadata[name]["src_file"] for name in plan["previously_used_esc_files"])
    candidates = defaultdict(list)
    for name, row in sorted(metadata.items()):
        if int(row["fold"]) == plan["evaluation_fold"] and row["src_file"] not in used:
            candidates[row["category"]].append(name)
    rng = random.Random(plan["selection_seed"])
    audio_root = root / "sources"
    audio_root.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    sources = []
    for old, meta in zip(development, previous):
        name = rng.choice(candidates[meta["category"]])
        row = metadata[name]
        if row["src_file"] in used:
            raise ValueError(f"Original recording reused across splits: {name}")
        used.add(row["src_file"])
        path = audio_root / name
        groups[old["group"]].append(str(path))
        sources.append({"group": old["group"], "development": meta["filename"], "evaluation": row})
    with ThreadPoolExecutor(max_workers=6) as workers:
        def acquire(item):
            name = item["evaluation"]["filename"]
            cached = Path("data") / name
            path = audio_root / name
            if cached.is_file() and not path.exists():
                path.symlink_to(cached.resolve())
            else:
                download(f"{ESC}/audio/{name}", path)
        list(workers.map(acquire, sources))
    items = [{"task": "event", "clips": clips} for clips in groups.values()]
    for name, parameters in plan["profiles"].items():
        target = root / name
        if not (target / "pairs.json").is_file():
            prepare(target, items, **parameters)
    (root / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")


def fit_readout(train, labels, test, ridge_lambda):
    for matrix in (train, test):
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    weights = np.linalg.solve(train @ train.T + ridge_lambda * np.eye(len(train)), labels)
    return (test @ train.T) @ weights


def grouped_interval(values, manifest, plan):
    groups = np.array([pair["group"] for pair in manifest])
    group_means = np.array([values[groups == group].mean() for group in np.unique(groups)])
    draws = np.random.default_rng(plan["bootstrap_seed"]).integers(
        len(group_means), size=(plan["bootstrap_replicates"], len(group_means))
    )
    return np.quantile(group_means[draws].mean(axis=1), [0.025, 0.975]).tolist()


def pair_hits(margin, positive_first):
    paired = np.asarray(margin).reshape(-1, 2)
    positive = paired >= 0 if positive_first else paired > 0
    return positive[:, 0] & ~positive[:, 1]


@torch.no_grad()
def measure(args, prototype, features, decode):
    plan_bytes = args.confirmation_plan.read_bytes()
    plan = json.loads(plan_bytes)
    source_rows = load_json(args.confirmation_stimuli / "sources.json")
    development = load_json(Path(plan["development_manifest"]))
    profiles = {key: load_json(args.confirmation_stimuli / key / "pairs.json") for key in plan["profiles"]}
    if len(development) != len(source_rows) or any(len(rows) != len(source_rows) for rows in profiles.values()):
        raise ValueError("Held-out profiles must have one pair for each development source")
    if any(int(row["evaluation"]["fold"]) != plan["evaluation_fold"] for row in source_rows):
        raise ValueError("Held-out fold differs from the fixed plan")
    development_files = {Path(pair["source"]).name for pair in development}
    evaluation_files = {row["evaluation"]["filename"] for row in source_rows}
    forbidden = development_files.union(plan["previously_used_esc_files"])
    if forbidden & evaluation_files or len(evaluation_files) != len(source_rows):
        raise ValueError("Expected unique recordings disjoint from development")
    for manifest in profiles.values():
        if [Path(pair["source"]).name for pair in manifest] != [
            row["evaluation"]["filename"] for row in source_rows
        ]:
            raise ValueError("Stimulus manifest differs from the held-out source list")

    summary = {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "n_pairs": len(source_rows),
               "n_source_groups": len({row["group"] for row in source_rows}),
               "readout": {}, "full_audio": {}, "compressed_native": {}}
    readout_cfg = plan["readout"]
    readout_test = profiles[readout_cfg["test_profile"]]
    models = {}
    checkpoints = {}
    for name in readout_cfg["checkpoint_names"]:
        checkpoint = Path(CONFIGURATIONS[name]) / "compressor.pt"
        comp = type(prototype)(prototype.keys.in_features, 16).cuda().eval()
        comp.load_state_dict(torch.load(checkpoint, weights_only=True))
        models[name] = comp
        checkpoints[name] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    labels = np.tile([1., -1.], len(development))
    readout_hits = []
    readout_path = args.out / "readout.csv"
    with readout_path.with_suffix(".partial").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "group", "source", "clip", "label", "score", "checkpoint"])
        for name, comp in {"pool16": None, **models}.items():
            matrices = []
            for manifest in (development, readout_test):
                vectors = []
                for pair in manifest:
                    for label in ("yes", "no"):
                        audio = features(pair[label])
                        latent = comp(audio) if comp is not None else F.adaptive_avg_pool1d(audio.T[None], 16)[0].T
                        vectors.append(latent.flatten().cpu().numpy())
                matrices.append(np.asarray(vectors, dtype=np.float64))
            scores = fit_readout(matrices[0], labels, matrices[1], readout_cfg["ridge_lambda"])
            hit = ((scores > 0) == (labels > 0)).reshape(-1, 2).mean(axis=1)
            if name in models:
                readout_hits.append(hit)
            summary["readout"][name] = {**statistics(scores, threshold=np.nextafter(0., 1.)),
                                        "accuracy_ci": grouped_interval(hit, readout_test, plan)}
            for idx, pair in enumerate(readout_test):
                for j, label in enumerate(("yes", "no")):
                    writer.writerow([name, pair["group"], pair["source"], pair[label], label,
                                     scores[2 * idx + j], checkpoints.get(name, "")])
            print("readout", name, summary["readout"][name], flush=True)
    readout_path.with_suffix(".partial").replace(readout_path)
    mean_hit = np.mean(readout_hits, axis=0)
    mean_ci = grouped_interval(mean_hit, readout_test, plan)
    summary["readout"]["K16_mean"] = {"accuracy": float(mean_hit.mean()), "accuracy_ci": mean_ci,
                                       "passes": bool(mean_hit.mean() >= readout_cfg["minimum_mean_accuracy"]
                                                      and mean_ci[0] >= readout_cfg["minimum_lower_ci"])}

    prompt = plan["prompt"]
    positive, negative = prompt["positive"], prompt["negative"]
    prompts = [f"{prompt['stem']}\nA. {a}\nB. {b}\nAnswer with only A or B."
               for a, b in ((positive, negative), (negative, positive))]
    gate = plan["native_gate"]
    profile = gate["profile"]
    native_path = args.out / "native.csv"
    with native_path.with_suffix(".partial").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "profile", "group", "source", "clip", "label",
                         "positive_first_margin", "negative_first_margin"])

        def native(name, profile_name, comp):
            manifest = profiles[profile_name]
            examples = [(pair, label) for pair in manifest for label in ("yes", "no")]
            margins = [[], []]
            for start in range(0, len(examples), args.batch_size):
                audio = [features(pair[label]) for pair, label in examples[start:start + args.batch_size]]
                batch = torch.stack(audio if comp is None else [comp(v) for v in audio])
                for j, question in enumerate(prompts):
                    logits = decode(batch, question)
                    margins[j].extend(((1 if j == 0 else -1) * (logits[:, 0] - logits[:, 1])).cpu().tolist())
            for i, (pair, label) in enumerate(examples):
                writer.writerow([name, profile_name, pair["group"], pair["source"], pair[label], label,
                                 margins[0][i], margins[1][i]])
            hits = [pair_hits(margin, j == 0) for j, margin in enumerate(margins)]
            result = {}
            for j, key in enumerate(("positive_first", "negative_first")):
                threshold = 0 if j == 0 else np.nextafter(0., 1.)
                result[key] = {**statistics(np.asarray(margins[j]), threshold=threshold),
                               "both_correct_ci": grouped_interval(hits[j], manifest, plan)}
            print("native", name, profile_name, result, flush=True)
            return result, np.mean(hits, axis=0)

        full_hits = {}
        for profile_name in profiles:
            summary["full_audio"][profile_name], full_hits[profile_name] = native("full", profile_name, None)
        passed = all(
            order["both_correct"] >= gate["minimum_both_correct_each_order"]
            and order["both_correct_ci"][0] > gate["minimum_lower_ci_each_order"]
            for order in summary["full_audio"][profile].values()
        )
        summary["native_gate_passed"] = passed
        if passed:
            compressed_hits = []
            for name, comp in models.items():
                result, hit = native(name, profile, comp)
                summary["compressed_native"][name] = result
                compressed_hits.append(hit)
            delta = full_hits[profile] - np.mean(compressed_hits, axis=0)
            summary["native_full_minus_K16"] = {"both_correct_gain": float(delta.mean()),
                                                 "ci": grouped_interval(delta, profiles[profile], plan)}
    native_path.with_suffix(".partial").replace(native_path)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Fixed gates", summary["readout"]["K16_mean"], "native gate", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepare_heldout(args.plan, args.out)
