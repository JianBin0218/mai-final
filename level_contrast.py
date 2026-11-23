"""Screen a relative-loudness task, then test fixed models on unseen ESC-50 sources."""

import argparse
import csv
import hashlib
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


PROTOCOL = Path("phase2/a-level-contrast-v1/protocol.json")
PREVIOUS_PAIRS = Path("phase2/opposite-v1/pairs.json")
RATE = 16000
PROMPTS = {
    "which": ("Which sound is louder?", "The first sound.", "The second sound."),
    "describe": ("Which description matches the recording?",
                 "The first sound is louder than the second sound.",
                 "The second sound is louder than the first sound."),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def development_sources(metadata):
    names = {Path(source).name for pair in read_json(PREVIOUS_PAIRS) if pair["task"] == "level"
             for source in pair["sources"]}
    return [metadata[name] for name in sorted(names)]


def select_sources(metadata, development, protocol):
    used = {row["src_file"] for row in development}
    rng = random.Random(protocol["selection_seed"])
    chosen = []
    for category in sorted({row["category"] for row in development}):
        candidates = sorted((row for row in metadata.values()
                             if row["category"] == category
                             and int(row["fold"]) == protocol["evaluation_fold"]
                             and row["src_file"] not in used), key=lambda row: row["filename"])
        rng.shuffle(candidates)
        category_rows = []
        for row in candidates:
            if row["src_file"] not in used:
                used.add(row["src_file"])
                category_rows.append(row)
                if len(category_rows) == protocol["sources_per_category"]:
                    break
        if len(category_rows) != protocol["sources_per_category"]:
            raise ValueError(f"Insufficient independent original recordings for {category}")
        chosen.extend(category_rows)
    return chosen


def fetch(url, path):
    if not path.is_file():
        temp = path.with_suffix(".partial")
        with urllib.request.urlopen(url, timeout=25) as source:
            temp.write_bytes(source.read())
        temp.replace(path)


def segment(path, seconds):
    x, rate = sf.read(path, dtype="float32", always_2d=True)
    divisor = gcd(rate, RATE)
    x = resample_poly(x.mean(axis=1), RATE // divisor, rate // divisor)
    length = round(seconds * RATE)
    if len(x) < length:
        raise ValueError(f"Recording too short: {path}")
    windows = [x[start:start + length] for start in range(0, len(x) - length + 1, RATE // 4)]
    x = max(windows, key=np.std)
    x = x - x.mean()
    peak = np.abs(x).max()
    if not np.isfinite(peak) or peak <= 1e-5:
        raise ValueError(f"Unusable waveform segment: {path}")
    return (x * (0.75 / peak)).astype("float32")


def level_pair(x, step_db, gap_seconds):
    quiet = x * (10 ** (-step_db / 20))
    gap = np.zeros(round(RATE * gap_seconds), dtype="float32")
    return np.concatenate((x, gap, quiet)), np.concatenate((quiet, gap, x))


def prepare_stage(stage, root, lock_path):
    protocol = read_json(PROTOCOL)
    lock = verify_lock(lock_path) if stage == "confirm" else None
    levels = [lock["db"]] if lock else protocol["development_db"]
    if lock:
        development_manifest = Path(lock["development_manifest"])
        if digest(development_manifest) != lock["development_manifest_sha256"]:
            raise ValueError("Development stimulus manifest changed after locking")
        development_pairs = [pair for pair in read_json(development_manifest)["development"]
                             if pair["db"] == lock["db"]]
    root.mkdir(parents=True, exist_ok=False)
    sources = root / "sources"
    sources.mkdir()
    meta = Path("data/esc50.csv")
    meta.parent.mkdir(exist_ok=True)
    fetch(protocol["esc50"] + "/meta/esc50.csv", meta)
    with meta.open(encoding="utf-8", newline="") as handle:
        metadata = {row["filename"]: row for row in csv.DictReader(handle)}
    development = development_sources(metadata)
    evaluation = select_sources(metadata, development, protocol) if lock else []

    def synthesize(row):
        name = row["filename"]
        path = Path("data") / name
        if not path.is_file():
            path = sources / name
            fetch(f"{protocol['esc50']}/audio/{name}", path)
        audio = segment(path, protocol["segment_seconds"])
        pairs = []
        for db in levels:
            versions = level_pair(audio, db, protocol["gap_seconds"])
            clips = [root / f"{Path(name).stem}-{db}db-{order}.wav" for order in ("first", "second")]
            for clip, waveform in zip(clips, versions):
                sf.write(clip, waveform, RATE, subtype="FLOAT")
            pairs.append({"source": name, "original_id": row["src_file"], "category": row["category"],
                          "db": db, "first": str(clips[0]), "second": str(clips[1])})
        return pairs

    with ThreadPoolExecutor(max_workers=6) as workers:
        if not lock:
            development_pairs = [pair for pairs in workers.map(synthesize, development) for pair in pairs]
        evaluation_pairs = [pair for pairs in workers.map(synthesize, evaluation) for pair in pairs]
    manifest = {"development": development_pairs, "evaluation": evaluation_pairs}
    manifest["protocol_sha256"] = digest(PROTOCOL)
    manifest["lock_sha256"] = digest(lock_path) if lock else None
    write_json(root / "pairs.json", manifest)
    print({split: len(manifest[split]) for split in ("development", "evaluation")})


def order_prompts(name):
    stem, first, second = PROMPTS[name]
    return [f"{stem}\nA. {a}\nB. {b}\nAnswer with only A or B."
            for a, b in ((first, second), (second, first))]


def score_pairs(pairs, latent, decode, question, batch_size):
    import torch

    margins = []
    for start in range(0, len(pairs), batch_size):
        batch = torch.stack([latent(pair[key]) for pair in pairs[start:start + batch_size]
                             for key in ("first", "second")])
        scores = []
        for sign, prompt in zip((1, -1), order_prompts(question)):
            logits = decode(batch, prompt)
            scores.append((sign * (logits[:, 0] - logits[:, 1])).cpu().numpy().reshape(-1, 2))
        margins.append(np.stack(scores, axis=1))
    return np.concatenate(margins)


def pair_metrics(margins):
    """The last axis is the true louder segment, first or second, in that order."""
    gains = margins[..., 0] - margins[..., 1]
    # The actual forced choice breaks A/B logit ties in favor of A.
    first_choice = (margins > 0) | ((margins == 0) & np.array([True, False])[None, :, None])
    correct = first_choice == np.array([True, False])
    return {"ranking": (gains > 0) + 0.5 * (gains == 0), "margin_change": gains,
            "both_correct": correct.all(axis=-1), "accuracy": correct.mean(axis=-1)}


def estimate(values, pairs, protocol):
    groups = np.array([pair["original_id"] for pair in pairs])
    means = np.array([np.asarray(values)[groups == group].mean() for group in np.unique(groups)])
    draws = np.random.default_rng(protocol["bootstrap_seed"]).integers(
        len(means), size=(protocol["bootstrap_replicates"], len(means)))
    return {"mean": float(means.mean()), "ci": np.quantile(means[draws].mean(axis=1), [.025, .975]).tolist()}


def summarize(margins, pairs, protocol):
    return {metric: {order: estimate(values[:, i], pairs, protocol) for i, order in enumerate(("AB", "BA"))}
            for metric, values in pair_metrics(margins).items()}


def append_scores(writer, model, prompt, pairs, margins):
    for pair, row in zip(pairs, margins):
        writer.writerow([model, pair["db"], prompt, pair["original_id"], pair["source"], *row.ravel()])


def passes_full(summary, gate):
    return all(summary["both_correct"][order]["mean"] >= gate["both_correct"]
               and summary["ranking"][order]["mean"] >= gate["ranking"]
               and summary["both_correct"][order]["ci"][0] > gate.get("both_correct_ci_lower", 0)
               for order in ("AB", "BA"))


def lock_screen(screen_path, out):
    protocol = read_json(PROTOCOL)
    screen = read_json(screen_path)
    if screen["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Screen was generated with a different protocol")
    for db in protocol["development_db"]:
        for prompt in protocol["prompt_order"]:
            summary = screen["candidates"][f"{db}:{prompt}"]
            if passes_full(summary, protocol["screen_gate"]):
                write_json(out, {"db": db, "prompt": prompt, "screen_sha256": digest(screen_path),
                                 "protocol_sha256": digest(PROTOCOL), "development_manifest": screen["manifest"],
                                 "development_manifest_sha256": screen["manifest_sha256"]})
                print(f"Locked {db} dB and prompt {prompt}")
                return
    raise SystemExit("No candidate passed the full-audio development gate; no confirmation task was selected.")


def verify_lock(path):
    lock = read_json(path)
    if lock["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Confirmation uses a different protocol from the lock")
    return lock


def representation_readout(development, evaluation, latent, protocol):
    from count_confirmation import fit_readout

    def vectors(pairs):
        rows = []
        for pair in pairs:
            for key in ("first", "second"):
                x = latent(pair[key])
                half = len(x) // 2
                rows.append((x[:half].mean(0) - x[-half:].mean(0)).cpu().numpy())
        return np.asarray(rows, dtype=np.float64)

    y = np.tile([1.0, -1.0], len(development))
    scores = fit_readout(vectors(development), y, vectors(evaluation), protocol["ridge_lambda"]).reshape(-1, 2)
    accuracy = ((scores > 0) == np.array([True, False])).mean(axis=1)
    return scores, accuracy


def measure(args, prototype, features, decode):
    import torch
    from count_evidence import CONFIGURATIONS

    protocol = read_json(PROTOCOL)
    manifest = read_json(args.pair_manifest)
    lock = verify_lock(args.level_lock) if args.level_stage == "confirm" else None
    if manifest["protocol_sha256"] != digest(PROTOCOL) or (lock and manifest["lock_sha256"] != digest(args.level_lock)):
        raise ValueError("Stimulus manifest does not match the fixed protocol and lock")
    development, evaluation = manifest["development"], manifest["evaluation"]
    if (lock is None) != (not evaluation):
        raise ValueError("Expected held-out pairs only for the confirmation stage")
    if {pair["original_id"] for pair in development} & {pair["original_id"] for pair in evaluation}:
        raise ValueError("Development and confirmation contain the same original recording")
    summary = {"protocol_sha256": digest(PROTOCOL), "manifest": str(args.pair_manifest),
               "manifest_sha256": digest(args.pair_manifest),
               "development_sources": len({pair["original_id"] for pair in development}),
               "evaluation_sources": len({pair["original_id"] for pair in evaluation})}
    scores = args.out / "native.csv"
    with torch.inference_mode(), scores.with_suffix(".partial").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "db", "prompt", "original_id", "source",
                         "AB_first_clip", "AB_second_clip", "BA_first_clip", "BA_second_clip"])

        def native(name, rows, latent, prompt):
            margins = score_pairs(rows, latent, decode, prompt, args.batch_size)
            append_scores(writer, name, prompt, rows, margins)
            result = summarize(margins, rows, protocol)
            print(name, rows[0]["db"], prompt, result, flush=True)
            return margins, result

        if lock:
            summary["lock_sha256"] = digest(args.level_lock)
            confirm(args, summary, protocol, lock, development, evaluation, prototype, features, native, CONFIGURATIONS)
        else:
            summary["candidates"] = {}
            for db in protocol["development_db"]:
                rows = [pair for pair in development if pair["db"] == db]
                for prompt in protocol["prompt_order"]:
                    _, summary["candidates"][f"{db}:{prompt}"] = native("full", rows, features, prompt)
    scores.with_suffix(".partial").replace(scores)
    write_json(args.out / "summary.json", summary)


def confirm(args, summary, protocol, lock, development, evaluation, prototype, features, native, configurations):
    import torch

    prompt = lock["prompt"]
    if any(pair["db"] != lock["db"] for pair in development + evaluation):
        raise ValueError("Confirmation audio does not match the locked level difference")
    full_margins, summary["full"] = native("full", evaluation, features, prompt)
    summary["full_gate_passed"] = passes_full(summary["full"], protocol["confirmation_gate"])
    summary["models"] = {}
    model_rankings, model_hits, model_readouts = [], [], []
    out = args.out / "readout.csv"
    with out.with_suffix(".partial").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "original_id", "source", "first_louder_score", "second_louder_score"])
        for name in protocol["compressors"]:
            checkpoint = Path(configurations[name]) / "compressor.pt"
            comp = type(prototype)(prototype.keys.in_features, 16).cuda().eval()
            comp.load_state_dict(torch.load(checkpoint, weights_only=True))

            def latent(path):
                return comp(features(path))

            probe, readout_hits = representation_readout(development, evaluation, latent, protocol)
            for pair, row in zip(evaluation, probe):
                writer.writerow([name, pair["original_id"], pair["source"], *row])
            result = {"checkpoint_sha256": digest(checkpoint),
                      "readout_accuracy": estimate(readout_hits, evaluation, protocol)}
            model_readouts.append(readout_hits)
            if summary["full_gate_passed"]:
                margins, result["native"] = native(name, evaluation, latent, prompt)
                metrics = pair_metrics(margins)
                model_rankings.append(metrics["ranking"].mean(axis=1))
                model_hits.append(metrics["both_correct"].mean(axis=1))
            summary["models"][name] = result
    out.with_suffix(".partial").replace(out)
    readout = estimate(np.mean(model_readouts, axis=0), evaluation, protocol)
    summary["mean_readout_accuracy"] = readout
    summary["readout_gate_passed"] = (readout["mean"] >= protocol["readout_gate"]["accuracy"]
                                      and readout["ci"][0] > protocol["readout_gate"]["lower_ci"])
    if summary["full_gate_passed"]:
        full = pair_metrics(full_margins)
        ranking, both = np.mean(model_rankings, axis=0), np.mean(model_hits, axis=0)
        summary["mean_compressed_ranking"] = estimate(ranking, evaluation, protocol)
        summary["mean_compressed_both_correct"] = estimate(both, evaluation, protocol)
        gap = estimate(full["ranking"].mean(axis=1) - ranking, evaluation, protocol)
        summary["full_minus_compressed_ranking"] = gap
        summary["full_minus_compressed_both_correct"] = estimate(
            full["both_correct"].mean(axis=1) - both, evaluation, protocol)
        summary["ranking_gap_gate_passed"] = (gap["mean"] >= protocol["minimum_ranking_gap"] and gap["ci"][0] > 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--stage", choices=("screen", "confirm"), required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--lock", type=Path)
    freeze = commands.add_parser("lock")
    freeze.add_argument("--screen", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "lock":
        lock_screen(args.screen, args.out)
    else:
        if (args.stage == "confirm") != (args.lock is not None):
            parser.error("Provide --lock only for confirmation")
        prepare_stage(args.stage, args.out, args.lock)
