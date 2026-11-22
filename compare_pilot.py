"""Report binary QA accuracy and paired uncertainty by resampling recordings."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("tag")
args = parser.parse_args()
read = lambda name: json.loads((Path("phase2") / (args.tag+"-"+name) / "train.json").read_text())
a, q = read("audio"), read("question")
parts = {"audio":a["heldout"], "question":q["heldout"], "wrong_question":q["wrong_question"],
         "full_audio":a["full_audio"], "text_only":a["text_only"]}
per_clip = {}
keys = [(r["path"],r["question"],r["target"]) for r in a["heldout"]["rows"]]
for name, part in parts.items():
    grouped = defaultdict(list)
    assert keys == [(r["path"],r["question"],r["target"]) for r in part["rows"]]
    for r in part["rows"]:
        grouped[r["path"]].append(int(r["prediction"] == r["target"]))
    assert all(len(hits)==2 for hits in grouped.values())
    per_clip[name] = {clip:sum(hits)/2 for clip,hits in grouped.items()}
    print(name, part["correct"], "/", len(keys), "both correct:",sum(all(hits) for hits in grouped.values()))
for control in ("audio","wrong_question"):
    differences = [value-per_clip[control][clip] for clip,value in per_clip["question"].items()]
    rng = random.Random(17)
    dist = sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(10000))
    print("question minus",control,"in pp:",100*statistics.mean(differences),"bootstrap 95%:",100*dist[249],100*dist[9749])
