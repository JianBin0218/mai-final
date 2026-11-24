"""Explore the effect of relative loudness on the same audio at smaller level differences."""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from level_contrast import (PROTOCOL, RATE, append_scores, digest, estimate, level_pair,
                            pair_metrics, read_json, representation_readout, score_pairs,
                            summarize, write_json)


DOSE_PLAN = Path("phase2/a-level-dose-v1/protocol.json")


def prepare(previous, out):
    plan = read_json(DOSE_PLAN)
    protocol = read_json(PROTOCOL)
    if digest(previous) != plan["parent_manifest_sha256"]:
        raise ValueError("Expected the exact 24 dB pairs from the completed response experiment")
    original = read_json(previous)
    out.mkdir(parents=True, exist_ok=False)
    prefix = round(RATE * protocol["segment_seconds"])
    evaluation = []
    for pair in original["evaluation"]:
        first, rate = sf.read(pair["first"], dtype="float32")
        if rate != RATE or first.ndim != 1 or pair["db"] != 24:
            raise ValueError("Invalid reference stimulus")
        for db in plan["new_levels_db"]:
            row = {**pair, "db": db}
            for label, wave in zip(("first", "second"), level_pair(first[:prefix], db, protocol["gap_seconds"])):
                path = out / f"{pair['original_id']}-{db}db-{label}.wav"
                sf.write(path, wave, rate, subtype="FLOAT")
                row[label] = str(path)
            evaluation.append(row)
    write_json(out / "pairs.json", {"plan_sha256": digest(DOSE_PLAN), "development": original["development"],
                                     "evaluation": evaluation, "reference_24db": original["evaluation"]})
    print(f"Prepared {len(evaluation)} pairs by changing only relative gain.", flush=True)


def measure(args, prototype, features, decode):
    import torch
    from torch.nn import functional as F
    from count_evidence import CONFIGURATIONS

    plan, protocol, manifest = read_json(DOSE_PLAN), read_json(PROTOCOL), read_json(args.pair_manifest)
    if manifest["plan_sha256"] != digest(DOSE_PLAN):
        raise ValueError("Dose stimuli differ from the written plan")
    levels = plan["new_levels_db"]
    by_db = {db: [p for p in manifest["evaluation"] if p["db"] == db] for db in levels}
    by_db[24] = manifest["reference_24db"]
    reference_ids = [pair["original_id"] for pair in by_db[24]]
    if any([pair["original_id"] for pair in rows] != reference_ids for rows in by_db.values()):
        raise ValueError("Dose groups must match on original recordings")

    def pool(path):
        return F.adaptive_avg_pool1d(features(path).T[None], 16)[0].T

    results = {str(db): {"native": {}, "readout": {}} for db in by_db}
    rank, readouts = {}, {}
    with torch.inference_mode(), (args.out / "native.csv").open("x", encoding="utf-8", newline="") as handle, \
            (args.out / "readout.csv").open("x", encoding="utf-8", newline="") as readout_handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "db", "prompt", "original_id", "source", "AB_first_clip", "AB_second_clip",
                         "BA_first_clip", "BA_second_clip"])
        probe_writer = csv.writer(readout_handle)
        probe_writer.writerow(["model", "db", "original_id", "source", "first_louder_score", "second_louder_score"])
        for arm in ("full", "pool16", *protocol["compressors"]):
            if arm == "full":
                latent = features
            elif arm == "pool16":
                latent = pool
            else:
                model = type(prototype)(prototype.keys.in_features, 16).cuda().eval()
                model.load_state_dict(torch.load(Path(CONFIGURATIONS[arm]) / "compressor.pt", weights_only=True))

                def latent(path):
                    return model(features(path))

            if arm != "full":
                combined = [p for db in levels for p in by_db[db]] + (by_db[24] if arm == "pool16" else [])
                pscore, hits = representation_readout(manifest["development"], combined, latent, protocol)
                for db in {p["db"] for p in combined}:
                    mask = np.array([p["db"] == db for p in combined])
                    readouts[db, arm] = hits[mask]
                    results[str(db)]["readout"][arm] = estimate(hits[mask], by_db[db], protocol)
                probe_writer.writerows([arm, p["db"], p["original_id"], p["source"], *row]
                                       for p, row in zip(combined, pscore))
            for db in list(levels) + ([24] if arm == "pool16" else []):
                rows = by_db[db]
                margins = score_pairs(rows, latent, decode, plan["prompt"], args.batch_size)
                append_scores(writer, arm, plan["prompt"], rows, margins)
                results[str(db)]["native"][arm] = summarize(margins, rows, protocol)
                rank[db, arm] = pair_metrics(margins)["ranking"].mean(axis=1)
            print(f"Scored loudness dose profiles for {arm}", flush=True)
    for db in levels:
        mean_rank = np.mean([rank[db, arm] for arm in protocol["compressors"]], axis=0)
        mean_hit = np.mean([readouts[db, arm] for arm in protocol["compressors"]], axis=0)
        results[str(db)]["ranking"] = {
            "full": estimate(rank[db, "full"], by_db[db], protocol),
            "pool16": estimate(rank[db, "pool16"], by_db[db], protocol),
            "K16": estimate(mean_rank, by_db[db], protocol),
            "full_minus_K16": estimate(rank[db, "full"] - mean_rank, by_db[db], protocol),
        }
        results[str(db)]["readout"]["K16_mean"] = estimate(mean_hit, by_db[db], protocol)
    write_json(args.out / "summary.json", {"plan_sha256": digest(DOSE_PLAN), "pairs_per_level": len(reference_ids),
                                          "manifest_sha256": digest(args.pair_manifest), "levels": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.previous, args.out)
