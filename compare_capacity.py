"""Compare slot capacity and learning on identical Clotho-AQA validation questions."""
from collections import defaultdict
import json
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parent / "phase2"


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    fixed = load("capacity-fixed/pool.json")
    learned = load("clotho-k16-s17-audio/train.json")
    parts = {"learned4": load("capacity-k4-s17-audio/train.json")["heldout"],
             "learned16": learned["heldout"], "fixed4": fixed["4"], "fixed16": fixed["16"],
             "full": learned["full_audio"], "text": learned["text_only"]}
    scores = {}
    for name, part in parts.items():
        rows = part["rows"]
        scores[name] = {(r["path"], r["question"], r["target"]): int(r["prediction"] == r["target"]) for r in rows}
        assert len(scores[name]) == len(rows) == 64
        grouped = defaultdict(list)
        for (clip, _, _), hit in scores[name].items():
            grouped[clip].append(hit)
        assert len(grouped) == 32 and all(len(v) == 2 for v in grouped.values())
        print(name, "correct", sum(scores[name].values()), "both", sum(all(v) for v in grouped.values()),
              "loss", round(part["loss"], 3))
    for first, second in [("learned4","learned16"),("learned4","fixed4"),("learned16","fixed16")]:
        a, b = scores[first], scores[second]
        assert a.keys() == b.keys()
        grouped = defaultdict(list)
        for key in a:
            grouped[key[0]].append(a[key]-b[key])
        changes = [sum(v)/len(v) for v in grouped.values()]
        rng = Random(17)
        n = len(changes)
        draws = sorted(sum(rng.choices(changes,k=n))*100/n for _ in range(10000))
        print(first, "minus", second, "pp", 100*sum(changes)/n, "95% CI", draws[249], draws[9749],
              "gains", sum(a[k]>b[k] for k in a), "losses",sum(a[k]<b[k] for k in a))


if __name__ == "__main__":
    main()
