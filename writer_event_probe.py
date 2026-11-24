"""Compare target-event readout from two fixed K=4 audio representations."""

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from level_contrast import digest, estimate, read_json, write_json


def measure(args, reference, model, encode, features, _decode, _decode_text):
    import torch
    from count_confirmation import fit_readout

    plan = read_json(args.writer_pair_plan)
    if digest(args.writer_load) != plan["checkpoints"]["writer_audio"]:
        raise ValueError("Wrong audio-only writer checkpoint")
    model.load_state_dict(torch.load(args.writer_load, weights_only=True))
    bank = read_json(args.pair_manifest)
    if bank["plan_sha256"] != plan["training_protocol_sha256"]:
        raise ValueError("Training mixtures differ from the B2 generation plan")
    train = list({row["path"]: row for row in bank["train_N4"]}.values())
    train_ids = {event["src_file"] for row in train for event in row["events"]}
    categories = sorted({event["category"] for row in train for event in row["events"]})
    index = {cat: i for i, cat in enumerate(categories)}
    labels = np.zeros((len(train), len(categories)))
    tests = {}
    for spec in plan["tests"]:
        path = Path(spec["manifest"])
        if digest(path) != spec["sha256"]:
            raise ValueError("Probe test manifest differs from the archived writer evaluation")
        pairs = read_json(path)["pairs"]
        if train_ids.intersection(src for pair in pairs for src in pair["source_ids"]):
            raise ValueError("Probe training and evaluation share an original recording")
        tests[spec["name"]] = pairs
    for i, row in enumerate(train):
        for event in row["events"]:
            labels[i, index[event["category"]]] = 1

    def matrix(paths, project):
        return np.asarray([project(features(path)).flatten().cpu().numpy() for path in paths], dtype=np.float64)

    table = {}
    with torch.no_grad(), (args.out / "scores.csv").open("x", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["split", "pair", "model", "category", "train_positives", "yes_score", "no_score"])
        projectors = {"external": reference, "writer_audio": lambda v: encode(v, None)}
        training = {arm: matrix([row["path"] for row in train], project) for arm, project in projectors.items()}
        for name, pairs in tests.items():
            table[name] = {}
            scores = {}
            for arm, project in projectors.items():
                test = matrix([pair[k] for pair in pairs for k in ("yes", "no")], project)
                values = fit_readout(training[arm].copy(), labels, test, plan["ridge_lambda"]).reshape(len(pairs), 2, -1)
                target_scores = []
                for i, row in enumerate(pairs):
                    cat = row["source_files"][0].split("-")[-1].split(".")[0]
                    key = plan["category_by_id"][cat]
                    target = index[key]
                    vals = values[i, :, target]
                    writer.writerow([name, row["original_id"], arm, key, int(labels[:, target].sum()), *vals])
                    target_scores.append(vals[0]-vals[1])
                gap = np.asarray(target_scores)
                scores[arm] = (gap > 0) + .5*(gap == 0)
                table[name][arm] = estimate(scores[arm], pairs, plan)
            table[name]["writer_minus_external"] = estimate(scores["writer_audio"]-scores["external"], pairs, plan)
            print("Event probes", name, table[name], flush=True)
    write_json(args.out / "summary.json", {"plan_sha256": digest(args.writer_pair_plan), "train_mixtures": len(train),
                                          "train_classes": len(categories), "results": table})


if __name__ == "__main__":
    from b2_mixtures import prepare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(SimpleNamespace(plan=Path("phase2/b2-mixtures-v1/plan.json"), out=args.out, confirmation=False, train_only=True))
