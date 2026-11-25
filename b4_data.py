"""Turn the multiple-choice audio reasoning sets into one manifest format for b4_think.py.

    python b4_data.py mmau     # data/reasoning/MMAU-test-mini/test_mini.parquet -> data/reasoning/mmau/
    python b4_data.py mmar     # data/reasoning/MMAR (MMAR-meta.json + mmar-audio.tar.gz) -> data/reasoning/mmar/
    python b4_data.py avqa     # data/reasoning/AVQA-R1-6K (AVQA_R1.tar.gz) -> data/reasoning/avqa/ (train/dev/valid)

Each item: {id, path, question, choices, answer (index), task, category, subcategory, source}.
Licences: MMAU test-mini and MMAR CC BY-NC 4.0, AVQA-R1-6K Apache-2.0 (course use, not redistributed in Git).
"""
import argparse
import io
import json
import random
import re
import tarfile
from pathlib import Path

ROOT = Path("data/reasoning")
PREFIX = re.compile(r"^\(?[A-H]\)?[.:]?\s+")


def strip(choice):
    return PREFIX.sub("", choice.strip())


def mmau():
    import pyarrow.parquet as pq
    import soundfile as sf
    out = ROOT / "mmau"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    items = []
    for row in pq.read_table(ROOT / "MMAU-test-mini/test_mini.parquet").to_pylist():
        meta = json.loads(row["other_attributes"])
        path = out / "audio" / f"{meta['id']}.wav"
        if not path.exists():
            samples, rate = sf.read(io.BytesIO(row["context"]["bytes"]), dtype="float32")
            sf.write(path, samples, rate)
        choices = [strip(c) for c in row["choices"]]
        answer = choices.index(strip(row["answer"])) if strip(row["answer"]) in choices else row["choices"].index(row["answer"])
        items.append(dict(id=meta["id"], path=str(path), question=row["instruction"], choices=choices, answer=answer,
                          task=meta["task"], category=meta["category"], subcategory=meta["sub-category"],
                          difficulty=meta.get("difficulty"), source=meta.get("dataset")))
    (out / "test.json").write_text(json.dumps(items, indent=1))
    print(len(items), "MMAU test-mini items")


def mmar():
    out = ROOT / "mmar"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / "MMAR/MMAR-meta.json").read_text())
    needed = {Path(row["audio_path"]).name for row in meta}
    with tarfile.open(ROOT / "MMAR/mmar-audio.tar.gz", "r|gz") as tar:  # one sequential pass over the archive
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name in needed and not (out / "audio" / name).exists():
                (out / "audio" / name).write_bytes(tar.extractfile(member).read())
    items = []
    for row in meta:
        name = Path(row["audio_path"]).name
        path = out / "audio" / name
        choices = [strip(c) for c in row["choices"]]
        if strip(row["answer"]) not in choices:
            print("skipped (answer not among choices):", row["id"], repr(row["answer"]))
            continue
        items.append(dict(id=row["id"], path=str(path), question=row["question"], choices=choices,
                          answer=choices.index(strip(row["answer"])), task=row["modality"], category=row["category"],
                          subcategory=row["sub-category"], source="MMAR"))
    (out / "test.json").write_text(json.dumps(items, indent=1))
    print(len(items), "MMAR items of", len(meta))


def avqa():
    """AVQA-R1-6K train file (4,491 items with audio) split 3,991 train / 500 dev; its valid file kept as avqa-valid."""
    out = ROOT / "avqa"
    if not (out / "AVQA_R1").exists():
        with tarfile.open(ROOT / "AVQA-R1-6K/AVQA_R1.tar.gz") as tar:
            tar.extractall(out, filter="data")

    def read(split):
        base = out / "AVQA_R1" / split
        rows = json.loads((base / f"omni_rl_format_{split}.json").read_text())
        items = []
        for row in rows:
            choices = [strip(c) for c in row["options"]]
            letter = re.search(r"<answer>\s*([A-H])", row["solution"]).group(1)
            path = base / row["path"]["audio"]
            if path.exists():
                items.append(dict(id=f"{split}-{row['problem_id']}", path=str(path), question=row["problem"], choices=choices,
                                  answer="ABCDEFGH".index(letter), task="sound", category="AVQA", subcategory="",
                                  source=row.get("data_source")))
        print(len(items), "of", len(rows), f"AVQA-R1-6K {split} items have audio")
        return items

    items = read("train")
    random.Random(17).shuffle(items)
    dev, train = items[:500], items[500:]
    (out / "train.json").write_text(json.dumps(train, indent=1))
    (out / "dev.json").write_text(json.dumps(dev, indent=1))
    (out / "valid.json").write_text(json.dumps(read("valid"), indent=1))
    print(len(train), "train,", len(dev), "dev AVQA items")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("set", choices=["mmau", "mmar", "avqa"])
    {"mmau": mmau, "mmar": mmar, "avqa": avqa}[parser.parse_args().set]()
