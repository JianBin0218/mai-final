"""Test question wording and answer order on the existing matched sound-burst stimuli."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from count_evidence import CONFIGURATIONS, QUESTION, statistics

STEMS = {"count": QUESTION, "pause": "Is there a quiet pause with audible sound both before and after it?"}
DESCRIPTIONS = {
    "count": ("Two sound bursts separated by a quiet pause.", "One continuous sound burst."),
    "pause": ("The audible sound has a quiet pause inside it.", "The audible sound continues without a quiet pause."),
}


def prompt_set(answer_format):
    orders = ("yes_first", "no_first") if answer_format == "yes-no" else ("gap_first", "gap_second")
    prompts = {}
    for stem_key, stem in STEMS.items():
        positive, negative = ("yes", "no") if answer_format == "yes-no" else DESCRIPTIONS[stem_key]
        for (a, b), order, sign in zip(((positive, negative), (negative, positive)), orders, (1, -1)):
            prompts[f"{stem_key}_{order}"] = (f"{stem}\nA. {a}\nB. {b}\nAnswer with only A or B.", sign)
    return prompts, orders


def score_summary(scores, *, yes_on_tie=True):
    threshold = 0 if yes_on_tie else np.nextafter(0., 1.)
    return {**statistics(scores, threshold=threshold),
            "yes_rate": float((scores >= threshold).mean())}


@torch.no_grad()
def measure(args, prototype, features, decode):
    manifest = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    if any(pair["question"] != QUESTION for pair in manifest):
        raise ValueError("Expected the fixed sound-burst question in the pair manifest")
    examples = [(pair, label) for pair in manifest for label in ("yes", "no")]
    prompts, orders = prompt_set(args.count_prompt_format)
    summary = {"pairs": len(manifest), "groups": len({p["group"] for p in manifest}),
               "prompts": {key: question for key, (question, _) in prompts.items()},
               "answer_format": args.count_prompt_format,
               "batch_size": args.batch_size, "models": {}}
    for name, folder in CONFIGURATIONS.items():
        checkpoint_hash = None
        if folder is not None:
            checkpoint = Path(folder) / "compressor.pt"
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        destination = args.out / f"{name}.csv"
        keys = [(pair["source"], pair[label], label, checkpoint_hash or "") for pair, label in examples]
        if destination.exists():
            with destination.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            fields = ("source", "path", "label", "checkpoint")
            if [tuple(row[field] for field in fields) for row in rows] != keys:
                raise ValueError(f"Existing output belongs to a different experiment: {destination}")
            margins = {key: np.array([float(row[key]) for row in rows]) for key in prompts}
        else:
            comp = None
            if folder is not None:
                comp = type(prototype)(prototype.keys.in_features, 16).cuda().eval()
                comp.load_state_dict(torch.load(checkpoint, weights_only=True))
            margins = {key: [] for key in prompts}
            for start in range(0, len(examples), args.batch_size):
                batch = []
                for pair, label in examples[start:start + args.batch_size]:
                    audio = features(pair[label])
                    if comp is not None:
                        audio = comp(audio)
                    elif name == "pool16":
                        audio = F.adaptive_avg_pool1d(audio.T.unsqueeze(0), 16)[0].T
                    batch.append(audio)
                latents = torch.stack(batch)
                for key, (question, order) in prompts.items():
                    scores = decode(latents, question)
                    margins[key].extend((order * (scores[:, 0] - scores[:, 1])).cpu().tolist())
                    if name == "full" and key == f"count_{orders[0]}" and start == 0:
                        single = decode(latents[0], question)
                        error = (single - scores[0]).abs().max().item()
                        print(f"Batch versus scalar logit maximum difference: {error:.4f}", flush=True)
            tmp = destination.with_suffix(".partial")
            with tmp.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["group", "source", "path", "label", "checkpoint", *prompts])
                for index, (pair, _) in enumerate(examples):
                    writer.writerow([pair["group"], *keys[index], *(margins[key][index] for key in prompts)])
            tmp.replace(destination)
            margins = {key: np.asarray(values) for key, values in margins.items()}
        result = {key: score_summary(margins[key], yes_on_tie=sign > 0)
                  for key, (_, sign) in prompts.items()}
        for stem in STEMS:
            average = (margins[f"{stem}_{orders[0]}"] + margins[f"{stem}_{orders[1]}"]) / 2
            result[f"{stem}_balanced"] = score_summary(average)
        summary["models"][name] = result
        print(name, json.dumps(result, separators=(",", ":")), flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
