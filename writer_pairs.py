"""Evaluate locked writer checkpoints on opposite audio with the same question."""

import argparse
import csv
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from level_contrast import RATE, digest, estimate, fetch, pair_metrics, read_json, segment, summarize, write_json


def select(metadata, plan):
    used = set(plan["excluded_source_ids"])
    pools = defaultdict(dict)
    for row in sorted(metadata, key=lambda row: row["filename"]):
        if int(row["fold"]) in plan["folds"] and row["src_file"] not in used:
            pools[row["category"]].setdefault(row["src_file"], row)
    rng = random.Random(plan["seed"])
    groups = []
    for _ in range(plan["pairs"]):
        categories = rng.sample(sorted(k for k, items in pools.items() if items.keys() - used), plan["events"] + 1)
        group = []
        for cat in categories:
            key = rng.choice(sorted(pools[cat].keys() - used))
            group.append(pools[cat][key])
            used.add(key)
        groups.append((group, rng.randrange(plan["events"])))
    return groups


def prepare(plan_path, out):
    plan = read_json(plan_path)
    out.mkdir(parents=True, exist_ok=False)
    downloads = out / "sources"
    downloads.mkdir()
    with Path("data/esc50.csv").open(encoding="utf-8", newline="") as handle:
        groups = select(list(csv.DictReader(handle)), plan)

    def synthesize(item):
        index, (group, slot) = item
        clips = []
        for row in group:
            name = row["filename"]
            path = Path("data") / name
            if not path.exists():
                path = downloads / name
                fetch(f"{plan['esc50']}/audio/{name}", path)
            waveform = segment(path, plan["segment_seconds"])
            clips.append(waveform / np.sqrt(np.mean(waveform**2)))
        gap = np.zeros(round(RATE * plan["gap_seconds"]), dtype=np.float32)
        variants = []
        for target in clips[:2]:
            sounds = clips[2:].copy()
            sounds.insert(slot, target)
            variants.append(np.concatenate([np.concatenate((sound, gap)) for sound in sounds]))
        gain = 0.80 / max(np.abs(v).max() for v in variants)
        paths = [out / f"pair-{index:02d}-{label}.wav" for label in ("yes", "no")]
        for path, variant in zip(paths, variants):
            sf.write(path, variant * gain, RATE, subtype="FLOAT")
        target = group[0]["category"].replace("_", " ")
        distractor = group[2]["category"].replace("_", " ")
        return {"original_id": str(index), "source_ids": [row["src_file"] for row in group],
                "source_files": [row["filename"] for row in group], "target_slot": slot,
                "stem": f"Can you hear the sound of {target}?", "wrong_stem": f"Can you hear the sound of {distractor}?",
                "yes": str(paths[0]), "no": str(paths[1])}

    with ThreadPoolExecutor(max_workers=6) as pool:
        pairs = list(pool.map(synthesize, enumerate(groups)))
    write_json(out / "pairs.json", {"plan_sha256": digest(plan_path), "pairs": pairs})
    print(f"Prepared {len(pairs)} fixed-stem pairs from {len(pairs)*(plan['events']+1)} independent original recordings.", flush=True)


def measure(args, reference, model, encode, features, decode, decode_text):
    import torch

    plan = read_json(args.writer_pair_plan)
    manifest = read_json(args.pair_manifest)
    pairs = manifest["pairs"]
    ids = [source for pair in pairs for source in pair["source_ids"]]
    if manifest["plan_sha256"] != digest(args.writer_pair_plan) or len(pairs) != plan["pairs"]:
        raise ValueError("Paired data does not match the frozen plan")
    if len(set(ids)) != len(ids) or set(ids) & set(plan["excluded_source_ids"]):
        raise ValueError("Paired stimuli share an original recording or overlap known study sources")
    checkpoints = {"writer_audio": args.writer_load, "writer_question": args.writer_question_load}
    for name, path in checkpoints.items():
        if digest(path) != plan["checkpoints"][name]:
            raise ValueError(f"Checkpoint differs from the locked model for {name}")

    margins = defaultdict(list)
    orders = (("A. yes\nB. no", 1), ("A. no\nB. yes", -1))
    with torch.no_grad():
        for study in ("controls", *checkpoints):
            if study in checkpoints:
                model.load_state_dict(torch.load(checkpoints[study], weights_only=True))
            for pair in pairs:
                audio = [features(pair[key]) for key in ("yes", "no")]
                if study == "controls":
                    arms = {"full": audio, "external": [reference(v) for v in audio], "text": None}
                elif study == "writer_audio":
                    arms = {study: [encode(v, None) for v in audio]}
                else:
                    arms = {name: [encode(v, stem) for v in audio]
                            for name, stem in (("correct", pair["stem"]), ("wrong", pair["wrong_stem"]), ("closed", None))}
                for name, latents in arms.items():
                    scores = []
                    for options, sign in orders:
                        prompt = f"{pair['stem']}\n{options}\nAnswer with only A or B."
                        logits = torch.stack([decode(v, prompt) for v in latents]) if latents is not None else decode_text(prompt).repeat(2, 1)
                        scores.append(((logits[:, 0] - logits[:, 1]) * sign).cpu().tolist())
                    margins[name].append(scores)
            print(f"Scored {study} on {len(pairs)} pairs", flush=True)
    arrays = {name: np.asarray(values) for name, values in margins.items()}
    scores = {name: summarize(values, pairs, plan) for name, values in arrays.items()}
    full_rank = scores["full"]["ranking"]
    gate = all(value["mean"] >= plan["full_gate"] and value["ci"][0] > .5 for value in full_rank.values())
    ranking = {name: pair_metrics(values)["ranking"].mean(axis=1) for name, values in arrays.items()}
    contrasts = {f"{a}_minus_{b}": estimate(ranking[a]-ranking[b], pairs, plan)
                 for a, b in (("writer_audio", "external"), ("correct", "wrong"), ("correct", "closed"))}
    summary = {"protocol_sha256": digest(args.writer_pair_plan), "manifest_sha256": digest(args.pair_manifest),
               "pairs": len(pairs), "full_gate_passed": gate, "arms": scores, "contrasts": contrasts}
    with (args.out / "scores.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "pair", "AB_yes_clip", "AB_no_clip", "BA_yes_clip", "BA_no_clip"])
        for name, values in arrays.items():
            for pair, item in zip(pairs, values):
                writer.writerow([name, pair["original_id"], *item.ravel()])
    write_json(args.out / "summary.json", summary)
    print("Paired writer contrasts", gate, json.dumps(contrasts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.plan, args.out)
