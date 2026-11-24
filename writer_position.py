"""Move the target event across positions while holding each sound pair fixed."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from level_contrast import RATE, digest, estimate, pair_metrics, read_json, write_json


def prepare(plan_path, parent_path, scores_path, out):
    plan = read_json(plan_path)
    if digest(parent_path) != plan["parent_manifest_sha256"] or digest(scores_path) != plan["parent_scores_sha256"]:
        raise ValueError("Expected the exact completed opposite-audio replication artifacts")
    out.mkdir(parents=True, exist_ok=False)
    parent = read_json(parent_path)["pairs"]
    variants = []
    block = round(RATE * plan["segment_seconds"]) + round(RATE * plan["gap_seconds"])
    for pair in parent:
        audio = [sf.read(pair[tag], dtype="float32") for tag in ("yes", "no")]
        if any(rate != RATE or wave.shape != (block * plan["events"],) for wave, rate in audio):
            raise ValueError("Reference audio does not match the fixed event-block layout")
        for slot in range(plan["events"]):
            row = {**pair, "target_slot": slot, "reused": slot == pair["target_slot"]}
            if not row["reused"]:
                for tag, (wave, rate) in zip(("yes", "no"), audio):
                    path = out / f"pair-{pair['original_id']}-slot-{slot+1}-{tag}.wav"
                    sf.write(path, np.roll(wave, (slot - pair["target_slot"]) * block), rate, subtype="FLOAT")
                    row[tag] = str(path)
            variants.append(row)
    write_json(out / "pairs.json", {"plan_sha256": digest(plan_path), "parent_scores": str(scores_path), "pairs": variants})
    print(f"Prepared {len(variants)} position variants including {len(parent)} reused score pairs", flush=True)


def measure(args, reference, model, encode, features, decode, decode_text):
    import torch

    plan = read_json(args.writer_pair_plan)
    manifest = read_json(args.pair_manifest)
    scores_path = Path(manifest["parent_scores"])
    if manifest["plan_sha256"] != digest(args.writer_pair_plan) or digest(scores_path) != plan["parent_scores_sha256"]:
        raise ValueError("Position variants or existing logits differ from the plan")
    if digest(args.writer_load) != plan["checkpoints"]["writer_audio"]:
        raise ValueError("Writer checkpoint differs from the paired replication")
    model.load_state_dict(torch.load(args.writer_load, weights_only=True))
    arms = {"full": lambda audio: audio, "external": reference, "writer_audio": lambda audio: encode(audio, None)}
    orders = ("AB", "BA")
    names = [f"{order}_{label}_clip" for order in orders for label in ("yes", "no")]
    with scores_path.open(newline="", encoding="utf-8") as handle:
        previous = {(row["model"], row["pair"]): [float(row[key]) for key in names]
                    for row in csv.DictReader(handle) if row["model"] in arms}
    by_position = defaultdict(list)
    with torch.inference_mode(), (args.out / "scores.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "pair", "target_slot", "reused", *names])
        for i, pair in enumerate(manifest["pairs"]):
            if not pair["reused"]:
                audio = [features(pair[tag]) for tag in ("yes", "no")]
            for name, transform in arms.items():
                if pair["reused"]:
                    margins = previous[name, pair["original_id"]]
                else:
                    latents = [transform(v) for v in audio]
                    margins = []
                    for options, sign in (("A. yes\nB. no", 1), ("A. no\nB. yes", -1)):
                        prompt = f"{pair['stem']}\n{options}\nAnswer with only A or B."
                        for latent in latents:
                            logits = decode(latent, prompt)
                            margins.append(float((logits[0] - logits[1]) * sign))
                writer.writerow([name, pair["original_id"], pair["target_slot"], int(pair["reused"]), *margins])
                by_position[pair["target_slot"], name].append(margins)
            if (i + 1) % 48 == 0:
                handle.flush()
                print(f"Scored {i+1}/{len(manifest['pairs'])} position variants", flush=True)
    base = [row for row in manifest["pairs"] if row["target_slot"] == 0]
    ranks = {key: pair_metrics(np.asarray(value).reshape(-1, 2, 2))["ranking"].mean(axis=1)
             for key, value in by_position.items()}
    positions = {str(slot+1): {name: estimate(ranks[slot, name], base, plan) for name in arms}
                 for slot in range(plan["events"])}
    last = plan["events"] - 1
    shifts = {name: ranks[last, name] - ranks[0, name] for name in arms}
    summary = {"plan_sha256": digest(args.writer_pair_plan), "manifest_sha256": digest(args.pair_manifest),
               "positions": positions, "last_minus_first": {k: estimate(v, base, plan) for k, v in shifts.items()},
               "writer_minus_external_position_effect": estimate(shifts["writer_audio"]-shifts["external"], base, plan)}
    write_json(args.out / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--previous-scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.plan, args.previous, args.previous_scores, args.out)
