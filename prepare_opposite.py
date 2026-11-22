"""Construct opposite-label audio pairs without inspecting any model scores."""
import csv
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from baseline import waveform

DATA = Path("data/paired_evidence")
RATE = 16000


def normalized(path):
    x = waveform(path)
    x = x - x.mean()
    rms = np.sqrt(np.mean(x*x))
    assert rms > 0.00001
    return x / rms


def make_pairs():
    DATA.mkdir(exist_ok=True, parents=True)
    pairs = []
    pilot = json.loads(Path("data/clotho/manifest.json").read_text())
    excluded = {Path(row["path"]).name for split in pilot.values() for row in split}
    votes = defaultdict(list)
    with open("data/clotho/clotho_aqa_val.csv",encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            stem = " ".join(row["QuestionText"].lower().strip().rstrip("?.!").split())
            if row["file_name"] not in excluded:
                votes[stem,row["file_name"]].append(row["answer"].strip().lower())
    groups = defaultdict(lambda:defaultdict(list))
    for (q,p),v in votes.items():
        if len(v)>=3 and len(set(v))==1 and v[0] in ("yes","no"):
            groups[q][v[0]].append(p)
    rng = random.Random(17)
    with zipfile.ZipFile("data/clotho/audio.zip") as z:
        members = {Path(n).name:n for n in z.namelist() if n.lower().endswith(".wav") and "__MACOSX" not in n}
        used=set()
        for q in sorted(groups):
            ys,ns = [sorted(set(groups[q][label]) & members.keys() - used) for label in ("yes","no")]
            rng.shuffle(ys);rng.shuffle(ns)
            for y,n in zip(ys,ns):
                used.update([y,n])
                files=[]
                for fn in (y,n):
                    dest=Path("data/clotho")/fn
                    if not dest.exists(): dest.write_bytes(z.read(members[fn]))
                    files.append(str(dest))
                pairs.append(dict(task="natural",group=f"natural-{len(pairs)}",q=q.capitalize()+"?",yes=files[0],no=files[1]))
    esc=[p for p in json.loads(Path("data/pairs.json").read_text()) if p["task"]=="event"]
    for i,p in enumerate(esc):
        clips=[normalized(path) for path in p["clips"]]
        for order, (a,b) in enumerate([clips,clips[::-1]]):
            gap=np.zeros(RATE//2,dtype="float32")
            loud=np.concatenate([a,gap,b/8]); quiet=np.concatenate([a/8,gap,b])
            gain=0.9/max(abs(loud).max(),abs(quiet).max())
            paths=[]
            for tag,x in [("first",loud),("second",quiet)]:
                dest=DATA/f"level-{i}-{order}-{tag}.wav"
                sf.write(dest,x*gain,RATE,subtype="FLOAT")
                paths.append(str(dest))
            for word,y,n in [("louder",0,1),("quieter",1,0)]:
                pairs.append(dict(task="level",group=f"esc-{i}",q=f"Is the first sound {word} than the second sound?",
                                  yes=paths[y],no=paths[n], sources=p["clips"]))
    (DATA/"pairs.json").write_text(json.dumps(pairs,indent=2))
    return pairs


if __name__=="__main__":
    from collections import Counter
    print(Counter(p["task"] for p in make_pairs()))
