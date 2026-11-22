"""Counterbalance the queried category relative to the first event."""
import json
from prepare_opposite import DATA

base=json.loads((DATA/"identity.json").read_text())
events=[r for r in base if r["task"]=="second_event"]
out=[]
for start in range(0,len(events),2):
    a,b=events[start:start+2]
    assert a["group"]==b["group"]
    for original,other in [(a,b),(b,a)]:
        alternate=dict(original, task="second_other",q=other["q"],yes=original["no"],no=original["yes"])
        out.append(alternate)
        category=alternate["q"].removeprefix("Can you hear the sound of ").removesuffix(" after the pause?")
        out.append(dict(alternate,task="anywhere_other",q=f"Can you hear the sound of {category} anywhere in this recording?"))
(DATA/"counterbalanced.json").write_text(json.dumps(out,indent=2))
