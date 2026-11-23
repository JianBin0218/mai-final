"""Prepare, score, and audit the frozen A-line Clotho-AQA evaluation."""

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from a_final_types import TYPES, classify, load_exemplars

EXEMPLARS = "question_types.csv"
ORDERS = {"yes_first": ("yes", "no", 1), "no_first": ("no", "yes", -1)}
CHECKPOINT_SEEDS = (17, 23, 41)
CONDITIONS = ("text", "K4", "K16", "full")


def digest(path, algorithm="sha256"):
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def require_digest(path, expected, algorithm="sha256"):
    if digest(path, algorithm) != expected:
        raise ValueError(f"Fingerprint differs for {path}")


def check_lock(plan_path):
    root = plan_path.parent
    lock = read(root / "lock.json")
    if not lock.get("smoke_passed"):
        raise ValueError("The locked validation smoke run did not pass")
    for name, expected in lock["files"].items():
        require_digest(root / name, expected)
    return lock


def questions(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def make_row(filename, path, stem, label, source, exemplars):
    qid = hashlib.sha256(f"{filename}\0{stem}".encode()).hexdigest()[:24]
    kind, origin = classify(stem, exemplars)
    return {"qid": qid, "recording": filename, "path": str(path), "source_id": source,
            "question": stem, "target": label, "type": kind, "type_method": origin}


def prepare(plan_path, out, *, split, inputs):
    plan = read(plan_path)
    exemplars = load_exemplars(plan_path.parent / EXEMPLARS)
    data = Path("data/clotho")
    out.mkdir(exist_ok=True, parents=True)
    audit = {}
    rows = []
    if split == "smoke":
        fresh = read(data / "fresh_val.json")["val"]
        films = set(sorted({Path(r["path"]).name for r in fresh})[:12])
        for row in fresh:
            clip = Path(row["path"]).name
            if clip in films:
                rows.append(make_row(clip, row["path"], row["question"], row["label"], None, exemplars))
    else:
        lock = check_lock(plan_path)
        provenance = plan["dataset"]
        test_path = inputs / "clotho_aqa_test.csv"
        require_digest(test_path, provenance["test_md5"], "md5")
        sources_path = inputs / "source_ids.csv"
        require_digest(sources_path, lock["source_ids_sha256"])
        ids = {r["file_name"]: r["source_id"] for r in questions(sources_path)}
        previous = set()
        for name in ("train", "val"):
            path = data / f"clotho_aqa_{name}.csv"
            require_digest(path, provenance[f"{name}_md5"], "md5")
            previous.update(r["file_name"] for r in questions(path))
        old_ids = {ids.get(name) for name in previous} - {None, ""}
        test_rows = questions(test_path)
        test_names = {r["file_name"] for r in test_rows}
        votes = defaultdict(list)
        for r in test_rows:
            votes[r["file_name"], r["QuestionText"].strip()].append(r["answer"].strip().lower())
        by_clip = defaultdict(lambda: {"yes": [], "no": []})
        for (clip, stem), answers in votes.items():
            if stem and len(answers) >= 3 and len(set(answers)) == 1 and answers[0] in ("yes", "no"):
                by_clip[clip][answers[0]].append(stem)
        archive_path = data / "audio.zip"
        require_digest(archive_path, provenance["audio_md5"], "md5")
        audio = out / "audio"
        audio.mkdir(exist_ok=True)
        excluded = Counter()
        with zipfile.ZipFile(archive_path) as archive:
            members = {Path(info.filename).name: info for info in archive.infolist()
                       if info.filename.lower().endswith(".wav") and "__MACOSX" not in info.filename}
            fingerprint = lambda item: (item.CRC, item.file_size)
            seen_audio = {fingerprint(members[name]) for name in previous if name in members}
            seen_ids = set(old_ids)
            for name in sorted(test_names):
                source = ids.get(name)
                labels = by_clip[name]
                why = None
                if not labels["yes"] or not labels["no"]:
                    why = "no_balanced_unanimous_questions"
                elif name not in members:
                    why = "no_audio"
                elif name in previous:
                    why = "previous_filename"
                elif not source:
                    why = "unknown_source_id"
                elif source in seen_ids:
                    why = "reused_original_source"
                elif fingerprint(members[name]) in seen_audio:
                    why = "duplicate_audio"
                if why:
                    excluded[why] += 1
                    continue
                seen_ids.add(source)
                seen_audio.add(fingerprint(members[name]))
                dest = audio / name
                tmp = dest.with_suffix(".partial")
                with archive.open(members[name]) as src, tmp.open("wb") as target:
                    shutil.copyfileobj(src, target)
                tmp.replace(dest)
                for label, target in (("yes", 0), ("no", 1)):
                    stem = min(labels[label], key=lambda q: (hashlib.sha256(
                        f"{plan['id']}\0{name}\0{label}\0{q}".encode()).digest(), q))
                    rows.append(make_row(name, dest, stem, target, source, exemplars))
        audit = {"official_recordings": len(test_names), "previous_recordings": len(previous),
                 "previous_recordings_without_source_id": sum(not ids.get(name) for name in previous),
                 "exclusions": dict(excluded)}
        if len(test_names) != len(rows) // 2 + sum(excluded.values()):
            raise ValueError("Official selection accounting does not balance")
    recordings = {row["recording"] for row in rows}
    counts = Counter((row["recording"], row["target"]) for row in rows)
    if not rows or len({row["qid"] for row in rows}) != len(rows) or any(
        counts[clip, label] != 1 for clip in recordings for label in (0, 1)
    ):
        raise ValueError("Expected unique question IDs and balanced recording pairs")
    manifest = {"split": split, "plan_sha256": digest(plan_path), "recordings": len(recordings),
                "question_types": dict(Counter(row["type"] for row in rows)),
                "type_methods": dict(Counter(row["type_method"] for row in rows)), "audit": audit, "rows": rows}
    write(out / "manifest.json", manifest)
    print({key: value for key, value in manifest.items() if key != "rows"}, flush=True)


def measure(args, prototype, features, decode, decode_text):
    import torch

    plan_path = args.final_plan
    plan = read(plan_path)
    manifest = read(args.pair_manifest)
    if manifest["plan_sha256"] != digest(plan_path) or manifest["split"] not in ("smoke", "test"):
        raise ValueError("Manifest was not prepared with the current evaluation plan")
    if manifest["split"] == "test":
        check_lock(plan_path)
    rows = manifest["rows"]
    scores_path = args.out / "scores.csv"
    if scores_path.exists():
        raise FileExistsError(f"Scores already exist: {scores_path}")
    compressor = {}
    checkpoint_hashes = {}
    for k in (4, 16):
        models = []
        for seed in CHECKPOINT_SEEDS:
            name = "clotho-k16-s17-audio" if (k, seed) == (16, 17) else f"capacity-k{k}-s{seed}-audio"
            checkpoint = Path(name) / "compressor.pt"
            module = type(prototype)(prototype.keys.in_features, k).cuda().eval()
            module.load_state_dict(torch.load(checkpoint, weights_only=True))
            checkpoint_hashes[f"K{k}-s{seed}"] = digest(checkpoint)
            models.append(module)
        compressor[k] = models
    if manifest["split"] == "test" and checkpoint_hashes != check_lock(plan_path)["compressor_sha256"]:
        raise ValueError("The compressor checkpoint differs from the frozen protocol")
    tmp = scores_path.with_suffix(".partial")
    with torch.no_grad(), tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        learned = [f"K{k}-s{s}" for k in (4, 16) for s in CHECKPOINT_SEEDS]
        writer.writerow(["qid", "recording", "order", "type", "target", "text", "full", *learned])
        previous = None
        for i, row in enumerate(rows):
            if row["path"] != previous:
                audio = features(row["path"])
                latents = {k: torch.stack([model(audio) for model in modules])
                           for k, modules in compressor.items()}
                previous = row["path"]
            for order, (a, b, sign) in ORDERS.items():
                question = f"{row['question']}\nA. {a}\nB. {b}\nAnswer with only A or B."
                text, full = decode_text(question), decode(audio, question)
                k4, k16 = (decode(latents[k], question) for k in (4, 16))
                raw = torch.cat((text[None], full[None], k4, k16))
                margins = (sign * (raw[:, 0] - raw[:, 1])).cpu().tolist()
                writer.writerow([row["qid"], row["recording"], order, row["type"], row["target"], *margins])
            if (i + 1) % 50 == 0:
                handle.flush()
                print(f"Scored {i + 1}/{len(rows)} questions", flush=True)
    tmp.replace(scores_path)
    write(args.out / "run.json", {"plan_sha256": digest(plan_path), "manifest_sha256": digest(args.pair_manifest),
                                  "split": manifest["split"], "compressor_sha256": checkpoint_hashes})
    report(plan_path, args.pair_manifest, args.out)


def score_rows(manifest, scores_path):
    with scores_path.open(encoding="utf-8", newline="") as handle:
        ordered = list(csv.DictReader(handle))
    rows = {(r["qid"], r["order"]): r for r in ordered}
    expected = {(r["qid"], order) for r in manifest["rows"] for order in ORDERS}
    if len(rows) != len(ordered) or rows.keys() != expected:
        raise ValueError("Score rows do not cover the manifest exactly")
    for row in manifest["rows"]:
        for order in ORDERS:
            found = rows[row["qid"], order]
            if (found["recording"], found["type"], int(found["target"])) != (
                row["recording"], row["type"], row["target"]
            ):
                raise ValueError("Score row does not match manifest metadata")
    return rows


def decisions(rows, order, method):
    values = np.array([float(r[method]) for r in rows])
    return values >= 0 if order == "yes_first" else values > 0


def interval(delta, recording_ids, bootstrap, level):
    _, index = np.unique(recording_ids, return_inverse=True)
    counts = np.bincount(index)
    totals = np.bincount(index, weights=delta)
    n = len(counts)
    weights = np.random.default_rng(bootstrap["seed"]).multinomial(
        n, np.ones(n) / n, size=bootstrap["replicates"]
    )
    tail = (1 - level) / 2
    return np.quantile(100 * (weights @ totals) / (weights @ counts), [tail, 1 - tail]).tolist()


def report(plan_path, manifest_path, out):
    plan = read(plan_path)
    manifest = read(manifest_path)
    run = read(out / "run.json")
    if run["plan_sha256"] != digest(plan_path) or run["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Run provenance differs from plan or manifest")
    source = manifest["rows"]
    scores = score_rows(manifest, out / "scores.csv")
    targets = np.array([r["target"] == 0 for r in source])
    recording_ids = np.array([r["recording"] for r in source])
    types = np.array([r["type"] for r in source])
    bootstrap = plan["bootstrap"]
    primary_order = plan["models"]["primary_order"]
    sec = plan["secondary_confirmatory"]
    summary = {"recordings": manifest["recordings"], "questions": len(source), "orders": {},
               "descriptive_types": list(TYPES), "cases": {}}
    for order in ORDERS:
        rows = [scores[r["qid"], order] for r in source]
        hits = {name: (decisions(rows, order, name) == targets) for name in ("text", "full")}
        per_seed = {f"K{k}": np.stack([decisions(rows, order, f"K{k}-s{s}") == targets
                                     for s in CHECKPOINT_SEEDS]) for k in (4, 16)}
        hits.update({name: values.mean(axis=0) for name, values in per_seed.items()})
        groups = {}
        for kind in ("all", *TYPES):
            mask = np.ones(len(source), dtype=bool) if kind == "all" else types == kind
            n = int(mask.sum())
            recs = recording_ids[mask]
            yes, no = mask & targets, mask & ~targets
            group = {"questions": n, "recordings": len(set(recs)), "yes_n": int(yes.sum()),
                     "no_n": int(no.sum()), "performance": {}, "contrasts": {}}
            if n:
                for name in CONDITIONS:
                    result = hits[name]
                    yr = float(result[yes].mean()) if yes.any() else None
                    nr = float(result[no].mean()) if no.any() else None
                    group["performance"][name] = {"accuracy": float(result[mask].mean()), "yes_recall": yr,
                                                  "no_recall": nr, "balanced_accuracy": (yr + nr) / 2
                                                  if yr is not None and nr is not None else None}
                for a, b in (("full", "text"), ("K16", "K4"), ("full", "K16")):
                    delta = (hits[a].astype(float) - hits[b])[mask]
                    level = (sec["confidence_level_each"] if order == primary_order and
                             kind in ("source_or_event", "count_or_relation") and (a, b) == ("full", "K16") else .95)
                    group["contrasts"][f"{a}_minus_{b}"] = {"pp": float(100 * delta.mean()),
                                                           "ci_pp": interval(delta, recs, bootstrap, level),
                                                           "level": level}
            groups[kind] = group
        summary["orders"][order] = groups
        if order == primary_order:
            patterns = {
                "source_preserved": (types == "source_or_event") & hits["full"] & per_seed["K16"].all(axis=0)
                                    & ~per_seed["K4"].any(axis=0),
                "relation_gap": (types == "count_or_relation") & hits["full"] & ~per_seed["K16"].any(axis=0),
                "timing_full_failure": (types == "time_pattern") & ~hits["full"] & ~per_seed["K16"].any(axis=0),
            }
            for name, matches in patterns.items():
                qids = sorted(r["qid"] for r, match in zip(source, matches) if match)
                summary["cases"][name] = qids[0] if qids else None
    groups = summary["orders"][primary_order]
    primary = groups["all"]["contrasts"]["K16_minus_K4"]
    summary["primary"] = {**primary, "supported": bool(primary["pp"] >= plan["primary"]["minimum_point_gain_pp"]
                                                      and primary["ci_pp"][0] > plan["primary"]["minimum_lower_pp_strict"])}
    for kind in ("source_or_event", "count_or_relation"):
        group = groups[kind]
        allowed = (group["questions"] >= sec["minimum_questions"]
                   and group["recordings"] >= sec["minimum_recordings"]
                   and min(group["yes_n"], group["no_n"]) >= sec["minimum_each_answer"])
        contrast = group["contrasts"].get("full_minus_K16")
        supported = allowed and ((contrast["ci_pp"][1] < sec["source_noninferiority_margin_pp"])
                                 if kind == "source_or_event" else
                                 (contrast["ci_pp"][0] > sec["relation_loss_minimum_lower_pp_strict"]))
        summary[kind] = {"minimum_support_met": allowed, "supported": supported, "contrast": contrast}
    write(out / "summary.json", summary)
    print("Primary", summary["primary"], "source", summary["source_or_event"], "relation", summary["count_or_relation"],
          flush=True)


def smoke_check(manifest_path, out, references):
    manifest = read(manifest_path)
    if manifest["split"] != "smoke":
        raise ValueError("Validation checks are only valid on the smoke manifest")
    scores = score_rows(manifest, out / "scores.csv")
    names = {"full": "fresh-controls/full.json", "text": "fresh-controls/text.json"}
    for k in (4, 16):
        for seed in CHECKPOINT_SEEDS:
            stem = "clotho-k16-s17-audio" if (k, seed) == (16, 17) else f"capacity-k{k}-s{seed}-audio"
            names[f"K{k}-s{seed}"] = f"fresh-capacity/{stem}.json"
    agreement = {}
    for model, filename in names.items():
        old = {(Path(r["path"]).name, r["question"].splitlines()[0]): (r["target"], r["prediction"])
               for r in read(references / filename)["rows"]}
        agree = []
        for item in manifest["rows"]:
            target, ref = old[item["recording"], item["question"]]
            if target != item["target"]:
                raise ValueError("Smoke target differs from the saved validation run")
            agree.append(int(float(scores[item["qid"], "yes_first"][model]) < 0) == ref)
        agreement[model] = float(np.mean(agree))
    passed = all(value >= (1 if name in ("text", "full") else .95) for name, value in agreement.items())
    write(out / "smoke_check.json", {"passed": passed, "questions": len(manifest["rows"]), "agreement": agreement})
    print("Smoke", passed, agreement, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "report", "smoke"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["smoke", "test"], default="smoke")
    parser.add_argument("--inputs", type=Path, default=Path("a-final-inputs"))
    parser.add_argument("--references", type=Path, default=Path("phase2"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.plan, args.out, split=args.split, inputs=args.inputs)
    elif args.command == "report":
        report(args.plan, args.manifest, args.out)
    else:
        smoke_check(args.manifest, args.out, args.references)
