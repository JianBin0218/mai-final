"""Controlled source-identity relations and event probes on the same waveforms."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from prepare import ESC, download
from prepare_opposite import DATA, RATE, normalized


def main():
    DATA.mkdir(exist_ok=True,parents=True)
    rng=random.Random(17)
    with open("data/esc50.csv") as f:
        metadata=list(csv.DictReader(f))
    pilot=[p for p in json.loads(Path("data/pairs.json").read_text()) if p["task"]=="event"]
    pairs=[]
    for i,p in enumerate(pilot):
        second=[]
        for orig,label in zip(p["clips"],p["options"]):
            found=[r for r in metadata if r["filename"]==Path(orig).name]
            assert len(found)==1
            different=[r for r in metadata if r["category"]==label.replace(" ","_") and r["src_file"]!=found[0]["src_file"]]
            candidates=sorted(r["filename"] for r in different if r["fold"]=="5") or sorted(r["filename"] for r in different)
            assert candidates, label
            fn=rng.choice(candidates)
            path=Path("data")/fn
            download(ESC+"/audio/"+fn,path)
            second.append(str(path))
        a,b=[[normalized(c) for c in clips] for clips in [p["clips"],second]]
        gain=.9/max(np.abs(x).max() for x in a+b)
        gap=np.zeros(RATE//2,dtype="float32")
        for first in (0,1):
            paths=[]
            for last in (0,1):
                path=DATA/f"identity-{i}-{first}-{last}.wav"
                sf.write(path,gain*np.concatenate([a[first],gap,b[last]]),RATE,subtype="FLOAT")
                paths.append(str(path))
            for task,q,positive in [
                ("identity","Are the sounds before and after the pause from the same kind of source?",first),
                ("second_event",f"Can you hear the sound of {p['options'][first]} after the pause?",first)]:
                pairs.append(dict(task=task,group=f"esc-{i}",q=q,yes=paths[positive],no=paths[1-positive],
                                  sources=p["clips"]+second))
    (DATA/"identity.json").write_text(json.dumps(pairs,indent=2))


if __name__=="__main__":
    main()
