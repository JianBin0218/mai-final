"""Confirm which compressed components carry a held-out sound event's identity."""

import argparse
import csv
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from baseline import waveform
from count_confirmation import grouped_interval
from count_evidence import RATE, fade
from natural_components import difference_preference
from prepare import ESC, download


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(plan_path, out):
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    with Path("data/esc50.csv").open(encoding="utf-8", newline="") as handle:
        metadata = {r["filename"]: r for r in csv.DictReader(handle)}
    original = [p for p in read(Path("data/pairs.json")) if p["task"] == "event"]
    prior_confirmation = read(Path("confirmation-stimuli/sources.json"))
    used = {r["evaluation"]["src_file"] for r in prior_confirmation}
    used.update(metadata[Path(clip).name]["src_file"] for p in original for clip in p["clips"])
    if len(original) != plan["target_pairs"]:
        raise ValueError("The earlier event category pairing does not match the plan")
    rng = random.Random(plan["source_selection_seed"])
    out.mkdir(exist_ok=True, parents=True)
    sources = out / "sources"
    sources.mkdir(exist_ok=True)
    pairs = []
    for group, p in enumerate(original):
        pair = {"group": group, "categories": p["options"], "original_sources": []}
        for old in p["clips"]:
            category = metadata[Path(old).name]["category"]
            eligible = [r for _, r in sorted(metadata.items()) if r["category"] == category
                        and int(r["fold"]) == plan["source_fold"] and r["src_file"] not in used]
            chosen = rng.choice(eligible)
            used.add(chosen["src_file"])
            pair["original_sources"].append(chosen)
        pairs.append(pair)
    tasks = [r["filename"] for pair in pairs for r in pair["original_sources"]]
    with ThreadPoolExecutor(max_workers=6) as workers:
        list(workers.map(lambda name: download(f"{ESC}/audio/{name}", sources / name), tasks))

    settings = plan["stimuli"]
    lengths = {key: round(value * RATE) for key, value in settings["event_seconds"].items()}
    total = round(settings["duration_seconds"] * RATE)
    for pair in pairs:
        prepared = {}
        center = round(rng.uniform(1.4, 2.6) * RATE)
        pair["center_samples"] = center
        for j, source in enumerate(pair["original_sources"]):
            x = waveform(sources / source["filename"])
            long_n = lengths["long"]
            step = RATE // 10
            windows = (x[i:i+long_n] for i in range(0, len(x)-long_n+1, step))
            long = max(windows, key=lambda v: np.quantile(np.mean(v.reshape(16, -1) ** 2, axis=1), 0.25))
            short_n = lengths["short"]
            sections = (long[i:i+short_n] for i in range(0, long_n-short_n+1, step))
            short = max(sections, key=lambda v: np.mean(v**2))
            for profile, view in (("long", long), ("short", short)):
                sound = fade(view)
                rms = np.sqrt(np.mean(sound**2))
                if rms <= 0:
                    raise ValueError(f"Silent source: {source['filename']}")
                prepared[profile, j] = sound / rms
        common_gain = min(settings["active_rms"], 0.95 / max(np.abs(v).max() for v in prepared.values()))
        pair["active_rms"] = float(common_gain)
        for profile, n in lengths.items():
            pair[profile] = []
            start = center - n // 2
            for j in (0, 1):
                recording = np.zeros(total, dtype=np.float32)
                recording[start:start+n] = prepared[profile, j] * common_gain
                path = out / f"{profile}-{pair['group']:02d}-{j}.wav"
                sf.write(path, recording, RATE, subtype="FLOAT")
                pair[profile].append(str(path))
    (out / "pairs.json").write_text(json.dumps({"plan_sha256": hashlib.sha256(raw).hexdigest(), "pairs": pairs},
                                                indent=2), encoding="utf-8")
    print(f"Prepared {len(pairs)} held-out category pairs with both event durations", flush=True)


@torch.no_grad()
def measure(args, prototype, features, decode):
    raw = args.event_plan.read_bytes()
    plan = json.loads(raw)
    manifest = read(args.pair_manifest)
    digest = hashlib.sha256(raw).hexdigest()
    pairs = manifest["pairs"]
    if digest != manifest["plan_sha256"] or len(pairs) != plan["target_pairs"]:
        raise ValueError("Stimuli do not match the frozen event plan")
    source_ids = {s["src_file"] for pair in pairs for s in pair["original_sources"]}
    if len(source_ids) != 2 * len(pairs) or any(int(s["fold"]) != plan["source_fold"]
                                             for pair in pairs for s in pair["original_sources"]):
        raise ValueError("Expected distinct held-out original recordings")
    configs = {"full": (0, None), "pool4": (4, None), "pool16": (16, None)}
    for k in (4, 16):
        for seed in plan["model_seeds"]:
            folder = "clotho-k16-s17-audio" if (k, seed) == (16, 17) else f"capacity-k{k}-s{seed}-audio"
            configs[f"K{k}-s{seed}"] = (k, Path(folder) / "compressor.pt")
    observations = {}
    summary = {"plan_sha256": digest, "pairs": len(pairs), "conditions": {}}
    output = args.out / "scores.csv"
    with output.with_suffix(".partial").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "profile", "group", "order", "category_a", "category_b", "own_a",
                         "own_b", "pool_a_residual_b", "pool_b_residual_a", "checkpoint"])
        for name, (k, checkpoint) in configs.items():
            comp = None
            sha = ""
            if checkpoint:
                comp = type(prototype)(prototype.keys.in_features, k).cuda().eval()
                comp.load_state_dict(torch.load(checkpoint, weights_only=True))
                sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            for profile in plan["stimuli"]["event_seconds"]:
                scores = []
                for pair in pairs:
                    x = [features(path) for path in pair[profile]]
                    pool = [F.adaptive_avg_pool1d(v.T[None], k)[0].T for v in x] if k else x
                    own = [comp(v) for v in x] if comp is not None else pool
                    encoded = own
                    if comp is not None:
                        encoded = [*own, pool[0] + (own[1] - pool[1]), pool[1] + (own[0] - pool[0])]
                    a, b = pair["categories"]
                    for j, (first, second) in enumerate(((a, b), (b, a))):
                        q = f"{plan['prompt']}\nA. {first}\nB. {second}\nAnswer with only A or B."
                        logits = decode(torch.stack(encoded), q)
                        score = ((1 - 2*j) * (logits[:, 0] - logits[:, 1])).cpu().tolist()
                        row = score if comp is not None else [*score, "", ""]
                        writer.writerow([name, profile, pair["group"], j, a, b, *row, sha])
                        scores.append(score)
                arr = np.asarray(scores).reshape(len(pairs), 2, -1)
                first_wins = (arr[:, :, 0] >= 0)
                second_wins = (arr[:, :, 1] < 0)
                first_wins[:, 1] = arr[:, 1, 0] > 0
                second_wins[:, 1] = arr[:, 1, 1] <= 0
                values = {"both_by_order": (first_wins & second_wins).mean(axis=0).tolist(),
                          "own_preference": float(difference_preference(arr[:, :, 0] - arr[:, :, 1]).mean())}
                if comp is not None:
                    hybrid = difference_preference(arr[:, :, 3] - arr[:, :, 2])
                    values["residual_preference"] = float(hybrid.mean())
                    observations[name, profile] = hybrid.mean(axis=1)
                summary["conditions"].setdefault(name, {})[profile] = values
                print(name, profile, values, flush=True)
    output.with_suffix(".partial").replace(output)
    primary = plan["primary"]
    profile, k = primary["profile"], primary["capacity"]
    values = np.mean([observations[f"K{k}-s{seed}", profile] for seed in plan["model_seeds"]], axis=0)
    ci = grouped_interval(values, pairs, primary)
    full_gate = min(summary["conditions"]["full"][profile]["both_by_order"]) >= primary["full_audio_both_correct_minimum_each_order"]
    summary["primary"] = {"full_audio_gate": full_gate, "residual_preference": float(values.mean()), "ci": ci,
                          "passes": bool(full_gate and values.mean() >= primary["residual_hybrid_preference_minimum"]
                                         and ci[0] > primary["residual_hybrid_preference_ci_lower_strictly_above"])}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Fixed primary", summary["primary"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parsed = parser.parse_args()
    prepare(parsed.plan, parsed.out)
