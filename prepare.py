"""Build paired pilot items on the server. Each pair has opposite correct choices."""
import argparse
import csv
import io
import json
import random
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("data")
ESC = "https://raw.githubusercontent.com/karolpiczak/ESC-50/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
COMPA = "https://huggingface.co/datasets/gijs/compa-order/resolve/0e2d826613b55e4b4ececa366192e86ac580e909/data/train-00000-of-00001.parquet"


def download(url, path):
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-order", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    rng = random.Random(20261122)
    download(COMPA, ROOT / "compa.parquet")
    table = pq.read_table(ROOT / "compa.parquet")
    indices = {name: i for i, name in enumerate(table["file_name"].to_pylist())}
    stems = sorted(name[:-4] for name in indices if name[:-4] + "_rev.wav" in indices)
    pairs = []
    for stem in rng.sample(stems, len(stems) if args.all_order else 32):
        clips, options = [], []
        for suffix in (".wav", "_rev.wav"):
            i = indices[stem + suffix]
            path = ROOT / ("compa_" + stem + suffix)
            path.write_bytes(table["audio"][i].as_py()["bytes"])
            clips.append(str(path))
            options.append(table["answer"][i].as_py().strip())
        pairs.append(dict(task="order", question="Which description matches the order of sounds?", clips=clips, options=options))

    meta = ROOT / "esc50.csv"
    download(ESC + "/meta/esc50.csv", meta)
    rows = list(csv.DictReader(io.StringIO(meta.read_text())))
    labels = sorted({row["category"] for row in rows})
    rng.shuffle(labels)
    for first, second in ([] if args.all_order else zip(labels[::2], labels[1::2])):
        clips = []
        for label in (first, second):
            row = rng.choice([r for r in rows if r["category"] == label and r["fold"] == "5"])
            path = ROOT / row["filename"]
            download(ESC + "/audio/" + row["filename"], path)
            clips.append(str(path))
        pairs.append(dict(task="event", question="Which sound is present?", clips=clips,
                          options=[s.replace("_", " ") for s in (first, second)]))
    for pair in pairs:
        if rng.randrange(2):
            pair["clips"].reverse()
            pair["options"].reverse()
    name = "all_order.json" if args.all_order else "pairs.json"
    (ROOT / name).write_text(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
