"""Prepare and evaluate one-versus-two sound-burst pairs from the cached ESC-50 sources."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from baseline import waveform

RATE = 16000
QUESTION = "Are there two separate bursts of sound with a quiet gap between them?"
SUFFIX = "\nA. yes\nB. no\nAnswer with only A or B."
CONFIGURATIONS = {"full": None, "pool16": None, "K16-s17": "clotho-k16-s17-audio",
                  "K16-s23": "capacity-k16-s23-audio", "K16-s41": "capacity-k16-s41-audio"}


def fade(samples):
    result = samples.copy()
    ramp = np.linspace(0, 1, RATE // 100, dtype=samples.dtype)
    result[:len(ramp)] *= ramp
    result[-len(ramp):] *= ramp[::-1]
    return result


def prepare(folder, sources=None, *, duration_seconds=3, gaps_seconds=(0.45, 0.65, 0.85)):
    folder.mkdir(parents=True, exist_ok=False)
    if sources is None:
        sources = json.loads(Path("data/pairs.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(17)
    duration = round(duration_seconds * RATE)
    half = 3 * RATE // 5
    manifest = []
    for group, item in enumerate(p for p in sources if p["task"] == "event"):
        for source in item["clips"]:
            audio = waveform(source)
            windows = [audio[i:i + 2 * half] for i in range(0, len(audio) - 2 * half + 1, RATE // 10)]
            # Choose a consistently active excerpt before inspecting any model outputs.
            excerpt = max(windows, key=lambda x: np.quantile(np.mean(x.reshape(12, -1) ** 2, axis=1), 0.25))
            gap = int(rng.choice(gaps_seconds) * RATE)
            start = int(rng.integers(RATE // 4, RATE * 3 // 4))
            start_one = start if len(manifest) % 2 == 0 else start + gap
            one = np.zeros(duration, dtype=np.float32)
            two = np.zeros_like(one)
            one[start_one:start_one + 2 * half] = fade(excerpt)
            two[start:start + half] = fade(excerpt[:half])
            two[start + half + gap:start + 2 * half + gap] = fade(excerpt[half:])
            one *= 0.05 / np.sqrt(np.mean(one ** 2))
            two *= 0.05 / np.sqrt(np.mean(two ** 2))
            common_gain = min(1, 0.95 / max(np.max(np.abs(one)), np.max(np.abs(two))))
            pair = {"group": group, "source": source, "question": QUESTION, "gap_samples": gap,
                    "start_one": start_one, "start_two": start}
            for label, audio in (("yes", two), ("no", one)):
                path = folder / f"{len(manifest):02d}-{label}.wav"
                sf.write(path, audio * common_gain, RATE, subtype="FLOAT")
                pair[label] = str(path)
            manifest.append(pair)
    if len({pair["source"] for pair in manifest}) != len(manifest):
        raise ValueError("ESC source recordings must be unique across groups")
    (folder / "pairs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest)} pairs from {len({p['group'] for p in manifest})} source groups")


def grouped_readout(matrix, groups, labels):
    matrix = matrix.astype(np.float64)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    gram = matrix @ matrix.T
    unique, cluster = np.unique(groups, return_inverse=True)
    # Refit on each group-wise label flip, preserving the matched pair structure.
    flips = np.random.default_rng(17).choice([-1, 1], size=(len(unique), 999))
    targets = labels[:, None] * np.column_stack((np.ones(len(unique)), flips))[cluster]
    scores = np.empty_like(targets)
    for group in unique:
        train = groups != group
        test = ~train
        weights = np.linalg.solve(gram[np.ix_(train, train)] + np.eye(train.sum()), targets[train])
        scores[test] = gram[np.ix_(test, train)] @ weights
    accuracy = ((scores > 0) == (targets > 0)).mean(axis=0)
    return scores[:, 0], (1 + int((accuracy[1:] >= accuracy[0]).sum())) / len(accuracy)


def statistics(scores, *, threshold):
    paired = scores.reshape(-1, 2)
    yes, no = paired[:, 0] >= threshold, paired[:, 1] >= threshold
    preference = (paired[:, 0] > paired[:, 1]) + 0.5 * (paired[:, 0] == paired[:, 1])
    return {"accuracy": float((yes.mean() + (~no).mean()) / 2),
            "both_correct": float((yes & ~no).mean()), "preference": float(preference.mean())}


@torch.no_grad()
def measure(args, prototype, features, decode):
    manifest = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    examples = [(pair, label) for pair in manifest for label in ("yes", "no")]
    groups = np.array([pair["group"] for pair, _ in examples])
    labels = np.array([1 if label == "yes" else -1 for _, label in examples])
    summary = {"pairs": len(manifest), "groups": len(set(groups)), "models": {}}
    for name, folder in CONFIGURATIONS.items():
        checkpoint_hash = None
        if folder is not None:
            checkpoint = Path(folder) / "compressor.pt"
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        destination = args.out / f"{name}.csv"
        if destination.exists():
            with destination.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            qa = np.array([float(row["qa_score"]) for row in rows])
            probe = np.array([float(row["probe_score"]) for row in rows])
            null_p = float(rows[0]["permutation_p"])
            expected = [(pair["source"], pair[label], label, checkpoint_hash or "") for pair, label in examples]
            found = [(row["source"], row["path"], row["label"], row["checkpoint"]) for row in rows]
            if found != expected:
                raise ValueError(f"Existing output belongs to a different experiment: {destination}")
        else:
            comp = None
            if folder is not None:
                comp = type(prototype)(prototype.keys.in_features, 16).cuda().eval()
                comp.load_state_dict(torch.load(checkpoint, weights_only=True))
            qa = []
            matrix = []
            for pair, label in examples:
                audio = features(pair[label])
                pooled = F.adaptive_avg_pool1d(audio.T.unsqueeze(0), 16)[0].T
                latent = comp(audio) if comp is not None else pooled
                scores = decode(audio if name == "full" else latent, pair["question"] + SUFFIX)
                qa.append((scores[0] - scores[1]).item())
                matrix.append(latent.flatten().cpu().numpy())
            qa = np.asarray(qa)
            probe, null_p = grouped_readout(np.asarray(matrix), groups, labels)
            tmp = destination.with_suffix(".partial")
            with tmp.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["group", "source", "path", "label", "qa_score", "probe_score",
                                 "permutation_p", "checkpoint"])
                for (pair, label), score, readout in zip(examples, qa, probe):
                    writer.writerow([pair["group"], pair["source"], pair[label], label, score, readout,
                                     null_p, checkpoint_hash or ""])
            tmp.replace(destination)
        summary["models"][name] = {"qa": statistics(qa, threshold=0),
                                   "probe": statistics(probe, threshold=np.nextafter(0., 1.)),
                                   "probe_permutation_p": null_p, "checkpoint": checkpoint_hash}
        print(name, summary["models"][name], flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="New stimulus directory")
    prepare(parser.parse_args().out)
