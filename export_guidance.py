"""Export the track-B guidance runs (runs/guidance*, kept on the GPU machine) to small CSVs in phase2/guidance-v1.

questions.csv    split, qid, path, stem, target            (val: pilot 64; fresh_val: Jianbin's 340)
predictions.csv  grid, k, seed, arm, split, condition, qid, prediction, loss
runs.csv         grid, k, seed, arm, train_split, steps, gate, parent_sha256, val_extra correct counts
probe.csv        k, seed, split, magnitude, ratio_to_query, correct_stem, wrong_stem, n, yes_rate, same_prediction

Conditions: heldout (correct stem / audio-only), wrong_question, generic_question, no_question, full_audio, text_only.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "phase2" / "guidance-v1"
GRIDS = {"small": ROOT / "runs" / "guidance", "large": ROOT / "runs" / "guidance-large"}
CONDITIONS = ["heldout", "wrong_question", "generic_question", "no_question", "full_audio", "text_only"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    questions, qid = [], {}
    sfx = "\nA. yes\nB. no\nAnswer with only A or B."
    for split, path, key in [("val", "data/clotho/manifest.json", "val"), ("fresh_val", "data/clotho/fresh_val.json", "val")]:
        for item in json.loads((ROOT / path).read_text())[key]:
            qid[split, item["path"], item["question"] + sfx] = len(questions)
            questions.append(dict(split=split, qid=len(questions), path=item["path"], stem=item["question"], target=item["label"]))
    with (OUT / "questions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(questions[0]))
        writer.writeheader()
        writer.writerows(questions)

    preds = csv.writer((OUT / "predictions.csv").open("w", newline=""))
    preds.writerow(["grid", "k", "seed", "arm", "split", "condition", "qid", "prediction", "loss"])
    runs = csv.writer((OUT / "runs.csv").open("w", newline=""))
    runs.writerow(["grid", "k", "seed", "arm", "train_split", "steps", "gate", "parent_sha256",
                   "val_extra_correct", "val_extra_wrong_question", "val_extra_no_question"])
    for grid, root in GRIDS.items():
        for run in sorted(root.glob("k*-s*-*/eval.json")):
            k, seed, arm = run.parent.name.split("-", 2)
            train = json.loads((run.parent / "train.json").read_text())
            evaluated = json.loads(run.read_text())
            extra = train.get("extra", {})
            runs.writerow([grid, k[1:], seed[1:], arm, evaluated["config"]["train_split"],
                           train["warm_start"]["steps"], round(evaluated["gate"], 4) if "gate" in evaluated else "",
                           train["warm_start"]["sha256"],
                           *(extra[c]["correct"] if c in extra else "" for c in ("heldout", "wrong_question", "no_question"))])
            for split, parts in [("val", evaluated), ("fresh_val", evaluated["extra"])]:
                for condition in CONDITIONS:
                    if condition in parts:
                        for r in parts[condition]["rows"]:
                            preds.writerow([grid, k[1:], seed[1:], arm, split, condition,
                                            qid[split, r["path"], r["question"]], r["prediction"], round(r["loss"], 4)])

    probe = csv.writer((OUT / "probe.csv").open("w", newline=""))
    probe.writerow(["k", "seed", "split", "magnitude", "ratio_to_query", "correct_stem", "wrong_stem", "n",
                    "yes_rate", "same_prediction"])
    for path in sorted(GRIDS["small"].glob("k*-s*-gatedc/probe*.json")):
        data = json.loads(path.read_text())
        split = "fresh_val" if "fresh_val" in path.name else "val_extra"
        k, seed, _ = path.parent.name.split("-", 2)
        for m in data["magnitudes"]:
            probe.writerow([k[1:], seed[1:], split, m["magnitude"], round(m["ratio_to_query"], 3), m["correct_stem"],
                            m["wrong_stem"], m["n"], round(m["yes_rate"], 4), m["same_prediction"]])


if __name__ == "__main__":
    main()
