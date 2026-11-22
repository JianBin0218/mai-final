"""Select balanced, unanimous question pairs by recording from the official splits."""
import argparse
import csv
import hashlib
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from prepare import download

ROOT = Path("data/clotho")
CHECKSUM = {"train": "508513b91001ab17e641eaaa162ca045", "val": "41b8f6d906a57d7a5a4ee02bbcdefcc9"}
AUDIO_MD5 = "48c2c46ccd71dc06121228e2ed68d6b9"


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-val", action="store_true")
    fresh = parser.parse_args().fresh_val
    ROOT.mkdir(exist_ok=True, parents=True)
    audio = ROOT / "audio.zip"
    download("https://huggingface.co/datasets/danasone/clotho-aqa/resolve/main/audio_files.zip", audio)
    assert md5(audio) == AUDIO_MD5, "Audio archive does not match official release"
    rng = random.Random(17)
    manifest = {}
    used = set()
    if fresh:
        pilot = json.loads((ROOT / "manifest.json").read_text())
        used.update(Path(row["path"]).name for split in pilot.values() for row in split)
    with zipfile.ZipFile(audio) as archive:
        members = {Path(x).name: x for x in archive.namelist() if x.lower().endswith(".wav") and "__MACOSX" not in x}
        for split, count in ([("val", None)] if fresh else [("train", 48), ("val", 32)]):
            filename = f"clotho_aqa_{split}.csv"
            path = ROOT / filename
            download(f"https://zenodo.org/api/records/6473207/files/{filename}/content", path)
            assert md5(path) == CHECKSUM[split]
            votes = defaultdict(list)
            with path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    votes[row["file_name"],row["QuestionText"].strip()].append(row["answer"].strip().lower())
            questions = defaultdict(lambda: defaultdict(list))
            for (clip,q), labels in votes.items():
                if len(labels) >= 3 and len(set(labels)) == 1 and labels[0] in ("yes","no"):
                    questions[clip][labels[0]].append(q)
            eligible = sorted(c for c,v in questions.items() if c in members and v["yes"] and v["no"] and c not in used)
            items = []
            for clip in rng.sample(eligible, len(eligible) if fresh else count):
                used.add(clip)
                dest = ROOT / clip
                dest.write_bytes(archive.read(members[clip]))
                yes, no = (rng.choice(questions[clip][label]) for label in ("yes","no"))
                for q, donor, label in [(yes,no,0),(no,yes,1)]:
                    items.append(dict(path=str(dest), question=q, donor=donor, label=label))
            manifest[split] = items
    (ROOT / ("fresh_val.json" if fresh else "manifest.json")).write_text(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
